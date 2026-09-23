"""Train the degradation PINN and compare it against the baselines.

Examples
--------
    python run_train.py --source synthetic --stints 64
    python run_train.py --source synthetic --quick
    python run_train.py --source fastf1 --year 2023 --gp Monza Hungary
    python run_train.py --source fastf1 --year 2026     # every 2026 race run so far
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tirepinn import plots
from tirepinn.baselines import LinearDegBaseline, LSTMBaseline
from tirepinn.config import LEARNABLE_PARAMS, REAL_DATA_FREE_PARAMS, Config, DataConfig
from tirepinn.dataset import StintDataset, aggregate_context_by_race
from tirepinn.evaluate import evaluate, format_recovery, format_report, parameter_recovery
from tirepinn.physics import GROUND_TRUTH
from tirepinn.pinn import TirePINN

# Settings of --quick: a short run that validates the pipeline end to end.
QUICK_RUN = {"adam_iters": 1200, "lbfgs_iters": 300, "n_stints": 16, "lstm_epochs": 200}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", choices=["synthetic", "fastf1"], default="synthetic")
    p.add_argument("--out", default="outputs", help="output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="short run to validate the pipeline")

    g = p.add_argument_group("synthetic")
    g.add_argument("--stints", type=int, default=64)

    g = p.add_argument_group("fastf1")
    g.add_argument(
        "--year", type=int, default=date.today().year, help="season (default: the current one)"
    )
    g.add_argument(
        "--gp",
        nargs="+",
        default=None,
        help="one or more races. Default: every race of the season already run, "
        "from the API's event schedule. With several the context genuinely "
        "varies; with only one the model mostly sees the effect of compound alone.",
    )
    g.add_argument("--session", default="R")
    g.add_argument("--drivers", nargs="*", default=[], help="e.g. VER HAM LEC (default: all)")

    g = p.add_argument_group("training")
    g.add_argument("--adam", type=int, default=15000)
    g.add_argument("--lbfgs", type=int, default=3000)
    g.add_argument("--lstm-epochs", type=int, default=800)
    g.add_argument("--no-baselines", action="store_true")
    g.add_argument(
        "--aggregate-context",
        nargs="+",
        default=list(DataConfig.aggregate_context),
        metavar="FIELD",
        help="collapse these context proxies to their per-race median. The "
        "default pair is measured, not guessed: their within-circuit variation "
        "predicts nothing. See dataset.aggregate_context_by_race.",
    )
    g.add_argument(
        "--no-aggregate-context",
        action="store_const",
        const=[],
        dest="aggregate_context",
        help="keep every context proxy at its per-stint value",
    )
    g.add_argument(
        "--free-params",
        nargs="+",
        default=None,
        help="physical parameters to estimate on real data. Defaults to "
        "config.REAL_DATA_FREE_PARAMS; more data allows freeing more of them.",
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(out_dir=args.out)
    cfg.pinn.seed = args.seed
    cfg.pinn.adam_iters = QUICK_RUN["adam_iters"] if args.quick else args.adam
    cfg.pinn.lbfgs_iters = QUICK_RUN["lbfgs_iters"] if args.quick else args.lbfgs
    cfg.data.source = args.source
    cfg.data.n_stints = QUICK_RUN["n_stints"] if args.quick else args.stints
    cfg.data.year = args.year
    cfg.data.races = tuple(args.gp or ())
    cfg.data.session = args.session
    cfg.data.drivers = tuple(args.drivers)
    cfg.data.aggregate_context = tuple(args.aggregate_context)
    # Temperature is not observable in real data: only the synthetic bench can
    # give the network a thermal reference.
    cfg.pinn.use_theta_proxy = args.source == "synthetic"
    if args.source == "fastf1":
        configure_for_real_data(cfg, args.free_params)
    return cfg


def configure_for_real_data(cfg: Config, free_params: list[str] | None) -> None:
    """Settings that change when training on real telemetry instead of the synthetic bench."""
    # The thermo-mechanical law is calibrated on the physics bench, where ground
    # truth exists; on real telemetry only the quantities that change between
    # circuits or tire batches are fitted. See config.REAL_DATA_FREE_PARAMS for
    # the full argument.
    free = tuple(free_params) if free_params else REAL_DATA_FREE_PARAMS
    unknown = set(free) - set(LEARNABLE_PARAMS)
    if unknown:
        raise SystemExit(f"unknown parameters in --free-params: {sorted(unknown)}")
    for name in LEARNABLE_PARAMS:
        if name not in free:
            setattr(cfg.physics, f"{name}_init", float(getattr(GROUND_TRUTH, name)))
    cfg.pinn.free_params = free

    # The weight of the data term should scale with data quality. Synthetic
    # timing has sigma ~0.06 s, so its term drops to ~0.004 and stops pulling on
    # the gradient. A real lap has sigma ~0.5 s from traffic, wind and driving,
    # and its term plateaus around ~0.25: at the same weight it would dominate
    # forever and drag the network away from the ODE. The weight is lowered so
    # physics still counts once the noise floor is hit.
    cfg.pinn.w_data_delta = 5.0


def load_data(cfg: Config) -> StintDataset:
    """The stints to train and test on: the synthetic bench, or races from the FastF1 API."""
    if cfg.data.source == "synthetic":
        from tirepinn import data_synthetic

        return data_synthetic.generate(cfg.data, cfg.physics, cfg.ranges, seed=cfg.pinn.seed)

    from tirepinn import data_fastf1

    if not cfg.data.races:
        # Stored back in the config, so the saved config.json lists the races used.
        cfg.data.races = tuple(data_fastf1.season_races(cfg.data))
        if not cfg.data.races:
            raise SystemExit(f"The API lists no {cfg.data.year} race run yet: pass --year or --gp")
        print(f"  {len(cfg.data.races)} races of {cfg.data.year} taken from the API's event schedule")

    if len(cfg.data.races) == 1:
        data = data_fastf1.build_dataset(cfg.data, cfg.physics, cfg.data.races[0])
    else:
        data = data_fastf1.build_multi_dataset(cfg.data, cfg.physics, cfg.data.races)

    if cfg.data.aggregate_context:
        data = aggregate_context_by_race(data, cfg.data.aggregate_context)
        print(f"  Context collapsed to race medians: {', '.join(cfg.data.aggregate_context)}")
    return data


def fit_baselines(cfg: Config, train: StintDataset, args: argparse.Namespace) -> dict:
    """The two reference models: the classic linear fit and the black-box LSTM."""
    epochs = QUICK_RUN["lstm_epochs"] if args.quick else args.lstm_epochs
    return {
        "Linear (classic)": LinearDegBaseline(cfg.physics).fit(train),
        "LSTM (black box)": LSTMBaseline(cfg.physics, epochs=epochs, seed=args.seed).fit(train),
    }


def save_figures(models: dict, pinn: TirePINN, test: StintDataset, cfg: Config, out: Path) -> None:
    loss_labels = ["Thermal ODE", "Wear ODE", "Bound d<=dmax", "Data: pace"]
    if cfg.pinn.use_theta_proxy:
        loss_labels.append("Data: temperature")
    truth = GROUND_TRUTH if cfg.data.source == "synthetic" else None

    plots.plot_stint_grid(models, test, cfg.physics, out / "01_stints.png")
    plots.plot_extrapolation(models, test, cfg.physics, out / "02_extrapolation.png")
    plots.plot_latent_states(pinn, test, cfg.physics, out / "03_latent_states.png")
    plots.plot_parameter_convergence(pinn.var_history, truth, out / "04_parameters.png")
    plots.plot_loss_history(pinn.loss_history, out / "05_loss.png", loss_labels)
    plots.plot_cliff_map(pinn, cfg.physics, out / "06_cliff_map.png")


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("F1 tire degradation PINN")
    print("=" * 78)

    data = load_data(cfg)
    print(data.describe())
    train, test = data.split(cfg.data.test_fraction, seed=args.seed)
    print(f"  Split by stint: {len(train)} train / {len(test)} test\n")

    # ---------------------------------------------------------------- PINN --
    print("[1/4] Training the PINN ...")
    t0 = time.perf_counter()
    pinn = TirePINN(cfg)
    pinn.build(train)
    pinn.train(out)
    print(f"      done in {time.perf_counter() - t0:.1f} s\n")
    models = {"PINN": pinn}

    # ----------------------------------------------------------- baselines --
    if args.no_baselines:
        print("[2/4] Baselines skipped\n")
    else:
        print("[2/4] Fitting baselines ...")
        models.update(fit_baselines(cfg, train, args))
        print("      done\n")

    # ---------------------------------------------------------- evaluation --
    print("[3/4] Evaluating on unseen stints ...\n")
    metrics = [evaluate(name, m.predict_stint, test, cfg.physics) for name, m in models.items()]
    report = format_report(metrics)
    print(report)

    recovery = ""
    if cfg.data.source == "synthetic":
        rows = parameter_recovery(pinn.learned_params(), GROUND_TRUTH, cfg.pinn.free_params)
        recovery = format_recovery(rows)
        print(recovery)

    # ------------------------------------------------------------- outputs --
    print("\n[4/4] Generating figures ...")
    save_figures(models, pinn, test, cfg, out)
    pinn.save(out)
    cfg.to_json(out / "config.json")
    with open(out / "report.txt", "w", encoding="utf-8") as fh:
        fh.write(data.describe() + "\n\n" + report + "\n" + recovery + "\n")

    print(f"      figures, model and report in {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
