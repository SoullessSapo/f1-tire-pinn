"""Background work for the GUI: data loading and the full training pipeline.

Streamlit reruns its script on every interaction, so anything that takes longer
than a click -- downloading a season from FastF1, training the PINN -- runs in a
worker thread. The thread never touches Streamlit; it only writes into a `Job`
object that the interface polls and draws once a second.

The pipeline is the one in `run_train.py`, reused function for function, so a
model trained from the interface is the same model the command line would give.
"""

from __future__ import annotations

import argparse
import io
import threading
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import run_train
import torch

from tirepinn.baselines import LinearDegBaseline, LSTMBaseline
from tirepinn.config import REAL_DATA_FREE_PARAMS, Config
from tirepinn.dataset import StintDataset, aggregate_context_by_race
from tirepinn.evaluate import evaluate, format_report, parameter_recovery
from tirepinn.physics import GROUND_TRUTH
from tirepinn.pinn import TirePINN, dde  # pinn selects the backend before importing deepxde

LOSS_LABELS = ["EDO térmica", "EDO desgaste", "Cota d ≤ d_max", "Datos: ritmo", "Datos: temperatura"]


@dataclass
class Settings:
    """Everything the sidebar collects. Mirrors the options of `run_train.py`."""

    source: str = "synthetic"
    seed: int = 42
    out_dir: str = "outputs/gui"
    test_fraction: float = 0.25

    # synthetic
    n_stints: int = 64
    noise_delta_s: float = 0.06

    # fastf1
    year: int = 2023
    gps: tuple[str, ...] = ("Monza",)
    session: str = "R"
    drivers: tuple[str, ...] = ()
    aggregate_context: tuple[str, ...] = ("q_fric", "load")
    free_params: tuple[str, ...] = REAL_DATA_FREE_PARAMS
    min_stint_laps: int = 8
    # "random": hold out a random fraction of stints. "practice": train on the
    # practice sessions below and test on the race of the same weekends.
    split_mode: str = "random"
    practice_sessions: tuple[str, ...] = ("FP2",)

    # training
    adam_iters: int = 15000
    lbfgs_iters: int = 3000
    lr: float = 1e-3
    hidden_width: int = 64
    hidden_depth: int = 4
    display_every: int = 250
    num_domain: int = 4000
    w_data_delta: float | None = None  # None = the default for the source
    train_baselines: bool = True
    lstm_epochs: int = 800

    def data_key(self) -> tuple:
        """Identity of the dataset these settings produce, to reuse it between runs."""
        if self.source == "synthetic":
            return ("synthetic", self.n_stints, self.noise_delta_s, self.seed)
        return (
            "fastf1", self.year, self.gps, self.session, self.drivers, self.aggregate_context,
            self.min_stint_laps, self.split_mode, self.practice_sessions,
        )

    @property
    def practice_to_race(self) -> bool:
        return self.source == "fastf1" and self.split_mode == "practice"

    def as_args(self) -> argparse.Namespace:
        """The same namespace `run_train.parse_args()` would produce."""
        return argparse.Namespace(
            source=self.source,
            out=self.out_dir,
            seed=self.seed,
            quick=False,
            stints=self.n_stints,
            year=self.year,
            gp=list(self.gps) or ["Monza"],
            session=self.session,
            drivers=list(self.drivers),
            adam=self.adam_iters,
            lbfgs=self.lbfgs_iters,
            lstm_epochs=self.lstm_epochs,
            no_baselines=not self.train_baselines,
            aggregate_context=list(self.aggregate_context),
            free_params=list(self.free_params) or None,
        )

    def to_config(self) -> Config:
        """`run_train.build_config`, plus the extra knobs only the GUI exposes."""
        cfg = run_train.build_config(self.as_args())
        cfg.data.noise_delta_s = self.noise_delta_s
        cfg.data.test_fraction = self.test_fraction
        cfg.pinn.lr = self.lr
        cfg.pinn.hidden = tuple([self.hidden_width] * self.hidden_depth)
        cfg.pinn.display_every = self.display_every
        cfg.pinn.num_domain = self.num_domain
        cfg.data.min_stint_laps = self.min_stint_laps
        if self.practice_to_race:
            cfg.data.session = "R"
            cfg.data.train_sessions = tuple(self.practice_sessions)
        if self.w_data_delta is not None:
            cfg.pinn.w_data_delta = self.w_data_delta
        return cfg


@dataclass
class Result:
    """Everything a finished training run leaves behind for the results tabs."""

    cfg: Config
    data: object
    train: object
    test: object
    models: dict
    metrics: list         # on the test stints
    train_metrics: list   # on the training stints, to compare against
    report: str
    recovery: list | None
    out_dir: Path
    seconds: float


class _PINNProgress(dde.callbacks.Callback):
    """Copies DeepXDE's loss history and the physical parameters into the job."""

    def __init__(self, job: Job, pinn: TirePINN):
        super().__init__()
        self.job = job
        self.pinn = pinn
        self.seen = 0

    def on_train_begin(self):
        # `TirePINN.train` calls `model.train` twice: Adam first, then L-BFGS.
        if isinstance(self.model.opt, torch.optim.LBFGS):
            s = self.job.settings
            self.job.lbfgs_start = int(self.model.train_state.step)
            self.job._set_stage(
                f"Afinando la PINN con L-BFGS ({s.lbfgs_iters:,} iteraciones tras {s.adam_iters:,} de Adam)",
                self.job.progress,
            )

    def on_epoch_end(self):
        job = self.job
        job.iteration = int(self.model.train_state.step)
        if job.stop_requested.is_set():
            self.model.stop_training = True
            # Skip the L-BFGS phase too if Adam is the one being stopped.
            self.pinn.cfg.pinn.lbfgs_iters = 0

        history = self.model.losshistory
        if len(history.steps) == self.seen:
            return
        params = self.pinn.learned_params().as_dict()
        phase = job.phase
        new = list(zip(history.steps[self.seen :], history.loss_train[self.seen :], strict=False))
        for k, (step, loss) in enumerate(new):
            terms = [float(v) for v in loss]
            # Parameters can only be read now, i.e. at the newest record. Older
            # records in the same batch -- such as the L-BFGS starting point, logged
            # before its first chunk ran -- keep the last values actually seen.
            last = k == len(new) - 1 or not job.param_trace
            job.loss_steps.append(int(step))
            job.loss_terms.append(terms)
            job.param_trace.append((int(step), params if last else job.param_trace[-1][1]))
            job.loss_phases.append(phase)
            job.log(
                f"{phase:<6} iter {job.phase_step(int(step)):>6d} | pérdida total {sum(terms):.4e} | "
                + " ".join(f"L{i + 1}={v:.2e}" for i, v in enumerate(terms))
            )
        self.seen = len(history.steps)


class Job(threading.Thread):
    """One unit of background work: `kind` is "data" or "train"."""

    def __init__(self, kind: str, settings: Settings, cached_data=None):
        super().__init__(daemon=True)
        self.kind = kind
        self.settings = settings
        self.cached_data = cached_data  # (key, dataset) from an earlier load

        self.stage = "En cola"
        self.progress = 0.0
        self.started = time.time()
        self.finished: float | None = None
        self.lines: list[str] = []
        self.error: str | None = None
        self.stop_requested = threading.Event()

        # live training traces
        self.iteration = 0  # DeepXDE's global step, Adam and L-BFGS together
        self.total_iterations = settings.adam_iters + settings.lbfgs_iters
        self.lbfgs_start: int | None = None  # global step at which L-BFGS began
        self.loss_steps: list[int] = []
        self.loss_phases: list[str] = []  # "Adam" or "L-BFGS", one per loss record
        self.loss_terms: list[list[float]] = []
        self.param_trace: list[tuple[int, dict]] = []
        self.lstm_loss: list[float] = []
        self.free_params: tuple[str, ...] = ()

        self.data = None
        self.result: Result | None = None
        self.harvested = False  # its output was moved into the session

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.is_alive()

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    @property
    def phase(self) -> str:
        return "Adam" if self.lbfgs_start is None else "L-BFGS"

    def phase_step(self, step: int) -> int:
        """Iterations into the current phase, counting each phase from zero."""
        return step if self.lbfgs_start is None else step - self.lbfgs_start

    @property
    def adam_iteration(self) -> int:
        return self.iteration if self.lbfgs_start is None else self.lbfgs_start

    @property
    def lbfgs_iteration(self) -> int:
        return 0 if self.lbfgs_start is None else self.iteration - self.lbfgs_start

    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        for line in str(message).rstrip().splitlines():
            self.lines.append(f"[{stamp}] {line}")

    def _set_stage(self, stage: str, progress: float) -> None:
        self.stage = stage
        self.progress = progress
        self.log(f"── {stage}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            if self.kind == "data":
                self._load_data()
            else:
                self._train()
            self._set_stage("Detenido por el usuario" if self._stopped() else "Terminado", 1.0)
        except Exception as exc:  # shown in the interface, not raised into the void
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(traceback.format_exc())
            self.stage = "Error"
        finally:
            self.finished = time.time()

    def _stopped(self) -> bool:
        return self.stop_requested.is_set()

    def _load_data(self):
        s = self.settings
        key = s.data_key()
        if self.cached_data is not None and self.cached_data[0] == key:
            self.log("Reutilizando los datos ya cargados")
            self.data = self.cached_data[1]
            return self.data

        what = "Generando datos sintéticos" if s.source == "synthetic" else (
            f"Descargando y procesando FastF1 {s.year}: {', '.join(s.gps)} (puede tardar minutos)"
        )
        self._set_stage(what, 0.02)
        cfg = s.to_config()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            if s.practice_to_race:
                data = _practice_to_race_data(cfg, s)
            else:
                data = run_train.load_data(cfg, s.as_args())
        self.log(buffer.getvalue())
        self.log(data.describe())
        self.data = data
        return data

    def _train(self) -> None:
        s = self.settings
        data = self._load_data()
        cfg = s.to_config()
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        if "test_ids" in data.meta:  # practice -> race: the split is fixed by session
            test_ids = set(data.meta["test_ids"])
            train = StintDataset([x for x in data.stints if x.stint_id not in test_ids], data.source)
            test = StintDataset([x for x in data.stints if x.stint_id in test_ids], data.source)
            self.log(f"Prácticas -> carrera: {len(train)} stints de práctica / {len(test)} de carrera")
        else:
            train, test = data.split(cfg.data.test_fraction, seed=s.seed)
            self.log(f"División por stint: {len(train)} entrenamiento / {len(test)} prueba")

        # ---- PINN
        self._set_stage("Construyendo la PINN", 0.05)
        pinn = TirePINN(cfg)
        pinn.build(train)
        self.free_params = tuple(pinn._raw_vars)
        self._set_stage(f"Entrenando la PINN con Adam ({cfg.pinn.adam_iters:,} iteraciones; luego L-BFGS {cfg.pinn.lbfgs_iters:,})", 0.08)
        pinn.train(out, callbacks=[_PINNProgress(self, pinn)])
        models = {"PINN": pinn}

        # ---- baselines
        if s.train_baselines and not self._stopped():
            self._set_stage("Ajustando la regresión lineal", 0.80)
            models["Linear (classic)"] = LinearDegBaseline(cfg.physics).fit(train)

            self._set_stage(f"Entrenando la LSTM ({s.lstm_epochs} épocas)", 0.82)

            def on_epoch(epoch: int, loss: float) -> bool:
                self.lstm_loss.append(loss)
                self.progress = 0.82 + 0.08 * epoch / max(s.lstm_epochs, 1)
                return self._stopped()

            models["LSTM (black box)"] = LSTMBaseline(
                cfg.physics, epochs=s.lstm_epochs, seed=s.seed
            ).fit(train, on_epoch=on_epoch)

        # ---- evaluation, figures, persistence
        self._set_stage("Evaluando en stints no vistos", 0.91)
        metrics = [evaluate(name, m.predict_stint, test, cfg.physics) for name, m in models.items()]
        train_metrics = [
            evaluate(name, m.predict_stint, train, cfg.physics) for name, m in models.items()
        ]
        report = format_report(metrics)
        self.log(report)

        recovery = None
        recovery_txt = ""
        if cfg.data.source == "synthetic":
            recovery = parameter_recovery(pinn.learned_params(), GROUND_TRUTH, cfg.pinn.free_params)
            recovery_txt = run_train.parameter_recovery_table(pinn, cfg.pinn.free_params)
            self.log(recovery_txt)

        self._set_stage("Generando figuras y guardando el modelo", 0.95)
        run_train.save_figures(models, pinn, test, cfg, out)
        pinn.save(out)
        cfg.to_json(str(out / "config.json"))
        with open(out / "report.txt", "w", encoding="utf-8") as fh:
            fh.write(data.describe() + "\n\n" + report + "\n" + recovery_txt + "\n")
        self.log(f"Modelo, figuras e informe en {out.resolve()}")

        self.result = Result(
            cfg=cfg,
            data=data,
            train=train,
            test=test,
            models=models,
            metrics=metrics,
            train_metrics=train_metrics,
            report=report,
            recovery=recovery,
            out_dir=out,
            seconds=time.time() - t0,
        )


def _practice_to_race_data(cfg: Config, s: Settings) -> StintDataset:
    """Practice stints plus race stints in one set, with the race ids in `meta`.

    Context aggregation runs on each half separately: pooling a race's practice
    and race stints into one median would leak the race's track temperature
    into practice and the other way round.
    """
    from tirepinn import data_fastf1

    train, test = data_fastf1.build_practice_to_race(
        cfg.data, cfg.physics, s.gps, s.practice_sessions
    )
    if s.aggregate_context:
        train = aggregate_context_by_race(train, s.aggregate_context)
        test = aggregate_context_by_race(test, s.aggregate_context)
    if test.meta.get("no_practice"):
        print(f"  Sin prácticas, se omiten en la prueba: {', '.join(test.meta['no_practice'])}")
    return StintDataset(
        train.stints + test.stints,
        f"{train.source} -> {test.source}",
        {"test_ids": [x.stint_id for x in test.stints], "sessions": list(s.practice_sessions)},
    )


def loss_labels(n_terms: int) -> list[str]:
    return LOSS_LABELS[:n_terms] + [f"L{i + 1}" for i in range(len(LOSS_LABELS), n_terms)]

