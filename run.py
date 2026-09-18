"""
TRAIN AND COMPARE
=================

    python run.py                                 # synthetic data
    python run.py --quick                         # short run, just to check
    python run.py --source csv --csv data/2023.csv  # real downloaded data

    python run.py --width 96 --layers 5           # bigger network
    python run.py --stints 64 --iterations 12000  # longer training

The program does four things, in this order:

  1. get the stints (simulated, or from the CSV download_data.py produced)
  2. split them into train and test, BY WHOLE STINT
  3. train the PINN and fit the linear baseline on the same data
  4. measure both on stints neither has seen, and plot

With synthetic data it also does a fifth thing that is impossible with real
data: check whether the PINN recovered the true physical constants. That is the
proof that the method works.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

# Headless backend. It has to come BEFORE importing pyplot, because pyplot
# picks its backend on import. Without this the script fails on a server.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import data
from baseline import LinearBaseline
from evaluate import HEADER, evaluate, parameter_recovery
from physics import (
    GROUND_TRUTH,
    LAP_REF,
    LEARNABLE_PARAMS,
    STRATEGY_HORIZON,
    exact_solution,
)
from pinn import PINN

COLORS = {
    "pinn": "#B93A24",
    "linear": "#7C8593",
    "exact": "#2C6C8C",
    "measured": "#14181F",
}


# ---------------------------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define and read the command-line options, grouped by what they affect."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = p.add_argument_group("where the data comes from")
    g.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    g.add_argument("--csv", default="data/races.csv",
                   help="the file download_data.py produced")
    g.add_argument("--stints", type=int, default=48,
                   help="how many stints to simulate (only with --source synthetic)")
    g.add_argument("--noise", type=float, default=0.05,
                   help="timing noise in seconds (synthetic only)")
    g.add_argument("--min-laps", type=int, default=8,
                   help="discard shorter stints (only with --source csv)")

    g = p.add_argument_group("the network")
    g.add_argument("--width", type=int, default=64, help="neurons per layer")
    g.add_argument("--layers", type=int, default=4, help="number of hidden layers")
    g.add_argument("--iterations", type=int, default=8000)
    g.add_argument("--collocation", type=int, default=2000,
                   help="points where the equation is enforced each iteration")
    g.add_argument("--lr", type=float, default=3e-3, help="learning rate")

    g = p.add_argument_group("other")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="outputs")
    g.add_argument("--quick", action="store_true",
                   help="1500 iterations and 18 stints, just to check it runs")

    return p.parse_args()


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def _draw_stint_panel(ax, stint, models: dict, synthetic: bool, horizon) -> None:
    """Draw one compound's panel: the shaded data region, then every curve."""
    # Everything left of the dashed line is where data exists.
    ax.axvspan(1, stint.laps[-1], color="#000000", alpha=0.05, lw=0)
    ax.axvline(stint.laps[-1], color="#7C8593", ls="--", lw=1)

    ax.plot(stint.laps, stint.delta, "o", ms=3.5,
            color=COLORS["measured"], label="measured (with noise)")

    if synthetic:
        exact_d = exact_solution(horizon / LAP_REF, stint.context, GROUND_TRUTH)
        ax.plot(horizon, GROUND_TRUTH.gamma1 * exact_d,
                color=COLORS["exact"], lw=4, alpha=0.85, label="exact solution")

    ax.plot(horizon, models["PINN"].predict_stint(stint.context, horizon),
            color=COLORS["pinn"], lw=1.7, label="PINN")
    ax.plot(horizon, models["Linear"].predict_stint(stint.context, horizon),
            color=COLORS["linear"], lw=1.8, ls="-.", label="linear")

    ax.set_title(f"{stint.compound}  ({stint.stint_id})", fontsize=10)
    ax.set_xlabel("stint lap")
    ax.grid(alpha=0.15)


def plot_fit(models: dict, stints, synthetic: bool, path: Path) -> None:
    """One panel per compound: what was measured, the exact curve, both models.

    The shaded area is where data EXISTS. To its right every model is
    extrapolating, and that is where having physics inside starts to show.
    """
    # One representative stint per compound.
    by_compound: dict[str, object] = {}
    for stint in stints:
        by_compound.setdefault(stint.compound, stint)
    order = [c for c in ("SOFT", "MEDIUM", "HARD") if c in by_compound]
    if not order:
        order = list(by_compound)[:3]

    fig, axes = plt.subplots(
        1, len(order), figsize=(4.3 * len(order), 3.7), sharey=True, squeeze=False
    )
    horizon = np.arange(1, STRATEGY_HORIZON + 1)

    for ax, compound in zip(axes[0], order):
        _draw_stint_panel(ax, by_compound[compound], models, synthetic, horizon)

    axes[0][0].set_ylabel("pace loss [s]")
    axes[0][-1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "To the right of the dashed line, every model is extrapolating",
        fontsize=9.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_training(model: PINN, path: Path) -> None:
    """How the three terms of the loss come down."""
    history = model.history
    iterations = [h["iteration"] for h in history]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, label, color in [
        ("physics", "physics (equation residual)", COLORS["exact"]),
        ("data", "data (measured pace)", COLORS["pinn"]),
        ("ic", "initial condition d(0)=0", COLORS["linear"]),
    ]:
        ax.plot(iterations, [h[key] for h in history], lw=1.6,
                color=color, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("The three terms of the loss", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_parameters(model: PINN, synthetic: bool, path: Path) -> None:
    """The six physical constants while they are being estimated.

    With synthetic data the true value is drawn as well: if the curves land on
    the dashed lines, the inverse problem worked.
    """
    history = model.history
    iterations = [h["iteration"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), squeeze=False)

    for ax, name in zip(axes.ravel(), LEARNABLE_PARAMS):
        ax.plot(iterations, [h[name] for h in history],
                lw=1.8, color=COLORS["pinn"], label="estimated")
        if synthetic:
            ax.axhline(float(getattr(GROUND_TRUTH, name)),
                       color=COLORS["measured"], ls="--", lw=1.2, label="true value")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.15)

    axes[0][0].legend(fontsize=8)
    fig.suptitle(
        "Inverse problem: six physical constants estimated alongside the weights",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# THE FIVE STEPS
# ---------------------------------------------------------------------------
#
# main() below is a list of five calls. Each step lives in its own function so
# that reading main() tells you WHAT happens, and opening one function tells you
# HOW. Nothing here is clever; it is split up purely so no single function has
# to be held in your head all at once.


def get_stints(args) -> list | None:
    """Step 1. Fetch the stints, from the simulator or from the CSV.

    Returns None if the CSV could not be read, which main() turns into a clean
    exit rather than a traceback.
    """
    if args.source == "synthetic":
        n_stints = 18 if args.quick else args.stints
        stints = data.generate_synthetic(
            n_stints=n_stints, noise_s=args.noise, seed=args.seed
        )
        print(f"\nSource: synthetic bench (noise sigma = {args.noise} s)")
        return stints

    try:
        stints = data.load_csv(args.csv, min_laps=args.min_laps)
    except (FileNotFoundError, ValueError) as error:
        # A clear message beats a twenty-line traceback.
        print(f"\nCould not load the data:\n  {error}")
        return None

    print(f"\nSource: {args.csv}")
    return stints


def train_pinn(inputs, delta, args) -> PINN:
    """Step 2. Build the network and train it, printing what it is doing."""
    iterations = 1500 if args.quick else args.iterations

    print(f"\n[1/3] Training the PINN ({iterations} Adam iterations) ...")
    model = PINN(width=args.width, layers=args.layers, seed=args.seed)
    print(f"      network: {args.layers} layers of {args.width} neurons, "
          f"{model.n_weights()} weights")
    print(f"      plus {len(LEARNABLE_PARAMS)} physical constants to estimate: "
          f"{', '.join(LEARNABLE_PARAMS)}")

    started = time.perf_counter()
    model.train(
        inputs, delta,
        iterations=iterations,
        n_collocation=args.collocation,
        lr=args.lr,
    )
    print(f"      done in {time.perf_counter() - started:.1f} s")
    return model


def train_baseline(inputs, delta) -> LinearBaseline:
    """Step 3. Fit the classic rival on exactly the same data."""
    print("\n[2/3] Fitting the linear baseline ...")
    linear = LinearBaseline().fit(inputs, delta)
    print("      done")
    return linear


def _results_table(metrics) -> list[str]:
    """The accuracy table, as a list of lines."""
    return [HEADER, "-" * len(HEADER)] + [m.row() for m in metrics] + [
        "",
        "ViolIn / ViolExtrap = % of laps where the model predicts the tire",
        "REGAINING grip. That is physically impossible: the correct value is 0 %.",
    ]


def _recovery_table(model: PINN) -> list[str]:
    """The inverse-problem table. Synthetic data only, where truth exists."""
    rows = parameter_recovery(model.learned_params(), GROUND_TRUTH, LEARNABLE_PARAMS)
    mean_error = np.mean([r[3] for r in rows])

    return [
        "",
        "Inverse problem: recovery of the physical constants",
        f"{'Constant':12s} {'Estimated':>10s} {'True':>10s} {'Error':>9s}",
        "-" * 45,
    ] + [
        f"{name:12s} {estimated:10.4f} {true:10.4f} {error:8.1f}%"
        for name, estimated, true, error in rows
    ] + [
        f"{'':12s} {'':>10s} {'mean':>10s} {mean_error:8.1f}%"
    ]


def _estimates_list(model: PINN) -> list[str]:
    """Just the estimated constants, for real data where there is no truth."""
    learned = model.learned_params()
    return [
        "",
        "Estimated physical constants (with real data there is no ground",
        "truth to compare against):",
    ] + [
        f"  {name:8s} {float(getattr(learned, name)):8.4f}"
        for name in LEARNABLE_PARAMS
    ]


def build_report(model: PINN, linear: LinearBaseline, test, synthetic: bool) -> str:
    """Step 4. Measure both models and assemble the text report."""
    print("\n[3/3] Measuring on stints neither model has seen\n")

    metrics = [
        evaluate("PINN", model.predict_stint, test),
        evaluate("Linear (classic)", linear.predict_stint, test),
    ]

    lines = _results_table(metrics)
    lines += _recovery_table(model) if synthetic else _estimates_list(model)
    return "\n".join(lines)


def save_outputs(model, linear, stints, test, report, synthetic, out: Path) -> None:
    """Step 5. Write the three figures and the text report to disk."""
    models = {"PINN": model, "Linear": linear}

    plot_fit(models, test, synthetic, out / "01_fit.png")
    plot_training(model, out / "02_training.png")
    plot_parameters(model, synthetic, out / "03_parameters.png")

    (out / "report.txt").write_text(
        data.describe(stints) + "\n\n" + report + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the whole experiment end to end. Returns the process exit code."""
    args = parse_args()
    synthetic = args.source == "synthetic"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("Tire degradation PINN - v0 extended")
    print("=" * 74)

    # 1) the data
    stints = get_stints(args)
    if stints is None:
        return 1
    print(data.describe(stints))

    train, test = data.split(stints, seed=args.seed)
    print(f"  Split by whole stint: {len(train)} to train / {len(test)} to test")
    inputs, delta = data.flatten(train)

    # 2) and 3) the two models, on the same data
    model = train_pinn(inputs, delta, args)
    linear = train_baseline(inputs, delta)

    # 4) measure
    report = build_report(model, linear, test, synthetic)
    print(report)

    # 5) save
    save_outputs(model, linear, stints, test, report, synthetic, out)
    print(f"\nFigures and report in {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
