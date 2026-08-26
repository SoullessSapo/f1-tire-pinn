"""Train the degradation PINN and compare it against the baselines.

Examples
--------
    python run_train.py --source synthetic --stints 64
    python run_train.py --source synthetic --quick
    python run_train.py --source fastf1 --year 2023 --gp Monza Hungary
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from tirepinn import plots
from tirepinn.baselines import LinearDegBaseline, LSTMBaseline
from tirepinn.config import REAL_DATA_FREE_PARAMS, Config
from tirepinn.evaluate import evaluate, format_report, parameter_recovery
from tirepinn.physics import GROUND_TRUTH
from tirepinn.pinn import LEARNABLE, TirePINN


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
    g.add_argument("--year", type=int, default=2023)
    g.add_argument(
        "--gp",
        nargs="+",
        default=["Monza"],
        help="one or more races. With several the context genuinely varies; "
        "with only one the model mostly sees the effect of compound alone.",
    )
    g.add_argument("--session", default="R")
    g.add_argument("--drivers", nargs="*", default=[], help="e.g. VER HAM LEC")

    g = p.add_argument_group("training")
    g.add_argument("--adam", type=int, default=15000)
    g.add_argument("--lbfgs", type=int, default=3000)
    g.add_argument("--lstm-epochs", type=int, default=800)
    g.add_argument("--no-baselines", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(out_dir=args.out)
    cfg.pinn.seed = args.seed
    cfg.pinn.adam_iters = 1200 if args.quick else args.adam
    cfg.pinn.lbfgs_iters = 300 if args.quick else args.lbfgs
    cfg.data.source = args.source
    cfg.data.n_stints = 16 if args.quick else args.stints
    cfg.data.year = args.year
    cfg.data.gp = args.gp[0]
    cfg.data.session = args.session
    cfg.data.drivers = tuple(args.drivers)
    # Temperature is not observable in real data: only the synthetic bench can
    # give the network a thermal reference.
    cfg.pinn.use_theta_proxy = args.source == "synthetic"

    if args.source == "fastf1":
        # The thermo-mechanical law is calibrated on the physics bench, where
        # ground truth exists; on real telemetry only the quantities that change
        # between circuits or tire batches are fitted.
        # See config.REAL_DATA_FREE_PARAMS for the full argument.
        for name in LEARNABLE:
            if name not in REAL_DATA_FREE_PARAMS:
                setattr(cfg.physics, f"{name}_init", float(getattr(GROUND_TRUTH, name)))
        cfg.pinn.free_params = REAL_DATA_FREE_PARAMS

        # The weight of the data term should scale with data quality. Synthetic
        # timing has sigma ~0.06 s, so its term drops to ~0.004 and stops pulling
        # on the gradient. A real lap has sigma ~0.5 s from traffic, wind and
        # driving, and its term plateaus around ~0.25: at the same weight it
        # would dominate forever and drag the network away from the ODE. The
        # weight is lowered so physics still counts once the noise floor is hit.
        cfg.pinn.w_data_delta = 5.0
    return cfg


def load_data(cfg: Config, args: argparse.Namespace):
    if cfg.data.source == "synthetic":
        from tirepinn import data_synthetic

        return data_synthetic.generate(cfg.data, cfg.physics, cfg.ranges, seed=args.seed)
    from tirepinn import data_fastf1

    if len(args.gp) == 1:
        return data_fastf1.build_dataset(cfg.data, cfg.physics)
    return data_fastf1.build_multi_dataset(cfg.data, cfg.physics, args.gp)


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("F1 tire degradation PINN")
    print("=" * 78)

    data = load_data(cfg, args)
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
    if not args.no_baselines:
        print("[2/4] Fitting baselines ...")
        models["Linear (classic)"] = LinearDegBaseline(cfg.physics).fit(train)
        models["LSTM (black box)"] = LSTMBaseline(
            cfg.physics, epochs=200 if args.quick else args.lstm_epochs, seed=args.seed
        ).fit(train)
        print("      done\n")
    else:
        print("[2/4] Baselines skipped\n")

    # ---------------------------------------------------------- evaluation --
    print("[3/4] Evaluating on unseen stints ...\n")
    metrics = [evaluate(name, m.predict_stint, test, cfg.physics) for name, m in models.items()]
    report = format_report(metrics)
    print(report)

    recovery_txt = ""
    if cfg.data.source == "synthetic":
        rows = parameter_recovery(pinn.learned_params(), GROUND_TRUTH, cfg.pinn.free_params)
        lines = [
            "",
            "Physical parameter recovery (inverse problem)",
            f"{'Parameter':10s} {'Estimated':>10s} {'True':>10s} {'Rel. error':>11s}",
            "-" * 45,
        ]
        lines += [f"{n:10s} {est:10.4f} {ref:10.4f} {rel:10.1f}%" for n, est, ref, rel in rows]
        lines.append(
            f"{'':10s} {'':>10s} {'mean':>10s} {np.mean([r[3] for r in rows]):10.1f}%"
        )
        recovery_txt = "\n".join(lines)
        print(recovery_txt)

    # ------------------------------------------------------------- outputs --
    print("\n[4/4] Generating figures ...")
    loss_labels = ["Thermal ODE", "Wear ODE", "Bound d<=dmax", "Data: pace"]
    if cfg.pinn.use_theta_proxy:
        loss_labels.append("Data: temperature")

    plots.plot_stint_grid(models, test, cfg.physics, out / "01_stints.png")
    plots.plot_extrapolation(models, test, cfg.physics, out / "02_extrapolation.png")
    plots.plot_latent_states(pinn, test, cfg.physics, out / "03_latent_states.png")
    plots.plot_parameter_convergence(
        pinn.var_history,
        GROUND_TRUTH if cfg.data.source == "synthetic" else None,
        out / "04_parameters.png",
    )
    plots.plot_loss_history(pinn.loss_history, out / "05_loss.png", loss_labels)
    plots.plot_cliff_map(pinn, cfg.physics, out / "06_cliff_map.png")

    pinn.save(out)
    cfg.to_json(str(out / "config.json"))
    with open(out / "report.txt", "w", encoding="utf-8") as fh:
        fh.write(data.describe() + "\n\n" + report + "\n" + recovery_txt + "\n")

    print(f"      figures, model and report in {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
