"""Past training runs: each one saved whole in its own folder, and reopened.

A run from the interface leaves three files next to the model and the figures:

    summary.json   one line of the history table: date, data, settings, RMSE
    traces.json    the loss and parameter curves and the log, as seen live
    run.pkl        the dataset, the split, the baselines and the metrics

`summary.json` and `traces.json` are small and read on every visit to the
history tab; `run.pkl` is read only when a run is opened, together with the
PINN weights, to rebuild the same `Result` the results tabs draw from.

`run.pkl` is a pickle: open only folders this interface wrote.

No Streamlit here.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import asdict
from pathlib import Path

from job import Job, Result

from tirepinn.config import Config
from tirepinn.dataset import StintDataset
from tirepinn.pinn import TirePINN

SUMMARY = "summary.json"
TRACES = "traces.json"
RUN = "run.pkl"
FORMAT = 1


def run_folder(base: str | Path, settings) -> Path:
    """A new folder under `base` for one run, named by date and data."""
    if settings.source == "synthetic":
        what = f"sintetico-{settings.n_stints}"
    else:
        what = "-".join(g[:3] for g in settings.gps) or "fastf1"
        what += "-practicas" if settings.practice_to_race else f"-{settings.session}"
    slug = re.sub(r"[^A-Za-z0-9-]+", "", what).lower()
    folder = Path(base) / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}"
    n = 1
    while folder.exists():  # two runs in the same second
        n += 1
        folder = folder.with_name(f"{folder.name}_{n}")
    return folder


def save_run(job: Job, result: Result) -> None:
    """Write the three history files of a finished (or stopped) run."""
    s, out = job.settings, Path(result.out_dir)
    summary = {
        "format": FORMAT,
        "saved_at": time.strftime("%Y-%m-%d %H:%M"),
        "source": s.source,
        "data": describe_data(s),
        "mode": "practicas" if s.practice_to_race else "azar",
        "test_fraction": None if s.practice_to_race else s.test_fraction,
        "adam_iters": s.adam_iters,
        "lbfgs_iters": s.lbfgs_iters,
        "activation": s.activation,
        "network": f"{s.hidden_depth}x{s.hidden_width}",
        "seed": s.seed,
        "free_params": list(result.cfg.pinn.free_params),
        "n_train": len(result.train),
        "n_test": len(result.test),
        "seconds": result.seconds,
        "stopped": job.stop_requested.is_set(),
        "rmse": {m.name: m.rmse for m in result.metrics},
        "train_rmse": {m.name: m.rmse for m in result.train_metrics},
    }
    traces = {
        "loss_steps": job.loss_steps,
        "loss_terms": job.loss_terms,
        "loss_phases": job.loss_phases,
        "param_trace": job.param_trace,
        "free_params": list(job.free_params),
        "lstm_loss": job.lstm_loss,
        "lbfgs_planned": s.lbfgs_iters > 0,
        "log": job.lines[-2000:],
        "settings": asdict(s),
    }
    payload = {
        "format": FORMAT,
        "data": result.data,
        "train_ids": [x.stint_id for x in result.train.stints],
        "test_ids": [x.stint_id for x in result.test.stints],
        "baselines": {k: m for k, m in result.models.items() if k != "PINN"},
        "metrics": result.metrics,
        "train_metrics": result.train_metrics,
        "report": result.report,
        "recovery": result.recovery,
        "seconds": result.seconds,
    }
    (out / TRACES).write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")
    with open(out / RUN, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    # Written last: a folder with a summary is a complete run.
    (out / SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def describe_data(s) -> str:
    if s.source == "synthetic":
        return f"Sintético · {s.n_stints} stints"
    races = ", ".join(s.gps)
    session = "+".join(s.practice_sessions) + " → R" if s.practice_to_race else s.session
    drivers = f" · {' '.join(s.drivers)}" if s.drivers else ""
    return f"{s.year} {races} · {session}{drivers}"


def list_runs(base: Path) -> list[dict]:
    """Every complete run under `base`, newest first, each with its `folder`."""
    if not base.exists():
        return []
    runs = []
    for path in base.rglob(SUMMARY):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (path.parent / RUN).exists():
            runs.append({**summary, "folder": path.parent})
    return sorted(runs, key=lambda r: (r.get("saved_at", ""), str(r["folder"])), reverse=True)


def load_traces(folder: Path) -> dict | None:
    path = Path(folder) / TRACES
    if not path.exists():
        return None
    traces = json.loads(path.read_text(encoding="utf-8"))
    traces["param_trace"] = [tuple(t) for t in traces["param_trace"]]
    return traces


def load_run(folder: Path) -> Result:
    """Rebuild a run's `Result`: the PINN from its weights, the rest from `run.pkl`."""
    folder = Path(folder)
    cfg = Config.from_json(str(folder / "config.json"))
    with open(folder / RUN, "rb") as fh:
        payload = pickle.load(fh)  # noqa: S301 -- a file this interface wrote
    data: StintDataset = payload["data"]
    by_id = {x.stint_id: x for x in data.stints}
    train = StintDataset([by_id[i] for i in payload["train_ids"]], data.source,
                         dict(data.meta, split="train"))
    test = StintDataset([by_id[i] for i in payload["test_ids"]], data.source,
                        dict(data.meta, split="test"))
    models = {"PINN": TirePINN.load(folder, cfg), **payload["baselines"]}
    return Result(
        cfg=cfg,
        data=data,
        train=train,
        test=test,
        models=models,
        metrics=payload["metrics"],
        train_metrics=payload["train_metrics"],
        report=payload["report"],
        recovery=payload["recovery"],
        out_dir=folder,
        seconds=payload["seconds"],
    )
