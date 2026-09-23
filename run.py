"""
TRAIN AND COMPARE
=================

    python run.py                                 # synthetic data
    python run.py --quick                         # short run, just to check
    python run.py --source csv --csv data/2023.csv  # real downloaded data
    python run.py --source csv --csv data/2023.csv --drivers VER HAM

    python run.py --width 96 --layers 5           # bigger network
    python run.py --stints 64 --iterations 12000  # longer training
    python run.py --lbfgs 600                     # add a refinement phase
    python run.py --w-thermal 0.05                # rebalance the loss

The program does four things, in this order:

  1. get the stints (simulated, or from the CSV download_data.py produced)
  2. split them into train and test, BY WHOLE STINT
  3. train the PINN and fit the linear baseline on the same data
  4. measure both on stints neither has seen, and plot

With synthetic data it also does a fifth thing that is impossible with real
data: check whether the PINN recovered the true physical constants. That is the
proof that the method works.

BEFORE TRAINING, CALIBRATE. The constants in physics.py describe the world the
synthetic bench simulates, and training against a world that does not behave
like a tire teaches the network nothing useful. `tune.py` grades them without
training anything, in seconds instead of minutes. TUNING.md is the guide.
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
    LEARNABLE_PARAMS,
    STRATEGY_HORIZON,
    Context,
    integrate_stint,
    pace_loss,
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
    g.add_argument("--drivers", nargs="+", default=[], metavar="CODE",
                   help="train and test only on these drivers' stints, e.g. "
                        "VER HAM (only with --source csv). Default: everyone")

    g = p.add_argument_group("the network")
    g.add_argument("--width", type=int, default=64, help="neurons per layer")
    g.add_argument("--layers", type=int, default=4, help="number of hidden layers")
    g.add_argument("--iterations", type=int, default=8000,
                   help="Adam iterations (phase 1)")
    g.add_argument("--lbfgs", type=int, default=0,
                   help="L-BFGS iterations after Adam (phase 2). Try 500 when "
                        "zeta, h0 or h1 refuse to move")
    g.add_argument("--collocation", type=int, default=2000,
                   help="points where the equations are enforced each iteration")
    g.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    g.add_argument("--ic", choices=["hard", "soft"], default="hard",
                   help="how the initial conditions are imposed. 'hard' makes "
                        "them exact by construction; 'soft' is the textbook "
                        "loss term, kept so the difference can be measured")

    g = p.add_argument_group(
        "the weights of the loss terms -- these are SCALES, not importances. "
        "See TUNING.md"
    )
    g.add_argument("--w-physics", type=float, default=1.0,
                   help="residual of the WEAR equation")
    g.add_argument("--w-thermal", type=float, default=1.0,
                   help="residual of the THERMAL equation. Its natural scale is "
                        "much larger, so this usually wants to be smaller")
    g.add_argument("--w-data", type=float, default=10.0,
                   help="fit to the measured pace")
    g.add_argument("--w-ic", type=float, default=10.0,
                   help="initial conditions. Ignored entirely when --ic hard")

    g = p.add_argument_group("other")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="outputs")
    g.add_argument("--quick", action="store_true",
                   help="1500 iterations and 18 stints, just to check it runs")

    args = p.parse_args()
    if args.drivers and args.source != "csv":
        p.error("--drivers needs --source csv: a simulated stint has no driver")
    return args


# ---------------------------------------------------------------------------
# THE TRUTH, WHICH ONLY EXISTS ON THE SYNTHETIC BENCH
# ---------------------------------------------------------------------------

def true_state(context: Context, laps: np.ndarray):
    """(theta, d) from the equations themselves, with the true constants.

    There is no closed form for the coupled system any more, so this really
    integrates. It is the reference every synthetic plot and the cliff-lap
    metric are measured against.
    """
    laps = np.asarray(laps, dtype=float).ravel()
    _, theta, d = integrate_stint(int(laps.max()), context, GROUND_TRUTH)
    index = laps.astype(int) - 1
    return theta[index], d[index]


def true_pace(context: Context, laps: np.ndarray) -> np.ndarray:
    """The noiseless pace loss. Same signature as any model's predict_stint."""
    return pace_loss(true_state(context, laps)[1], GROUND_TRUTH)


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
        ax.plot(horizon, true_pace(stint.context, horizon),
                color=COLORS["exact"], lw=4, alpha=0.85, label="true curve")

    ax.plot(horizon, models["PINN"].predict_stint(stint.context, horizon),
            color=COLORS["pinn"], lw=1.7, label="PINN")
    ax.plot(horizon, models["Linear"].predict_stint(stint.context, horizon),
            color=COLORS["linear"], lw=1.8, ls="-.", label="linear")

    ax.set_title(f"{stint.compound}  ({stint.stint_id})", fontsize=10)
    ax.set_xlabel("stint lap")
    ax.grid(alpha=0.15)


def _representative_stints(stints) -> list:
    """One stint per compound, in SOFT/MEDIUM/HARD order where possible."""
    by_compound: dict[str, object] = {}
    for stint in stints:
        by_compound.setdefault(stint.compound, stint)

    order = [c for c in ("SOFT", "MEDIUM", "HARD") if c in by_compound]
    if not order:
        order = list(by_compound)[:3]
    return [by_compound[c] for c in order]


def plot_fit(models: dict, stints, synthetic: bool, path: Path) -> None:
    """One panel per compound: what was measured, the true curve, both models.

    The shaded area is where data EXISTS. To its right every model is
    extrapolating, and that is where having physics inside starts to show.
    """
    chosen = _representative_stints(stints)
    fig, axes = plt.subplots(
        1, len(chosen), figsize=(4.3 * len(chosen), 3.7), sharey=True, squeeze=False
    )
    horizon = np.arange(1, STRATEGY_HORIZON + 1)

    for ax, stint in zip(axes[0], chosen):
        _draw_stint_panel(ax, stint, models, synthetic, horizon)

    axes[0][0].set_ylabel("pace loss [s]")
    axes[0][-1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "To the right of the dashed line, every model is extrapolating",
        fontsize=9.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_state(model: PINN, stints, synthetic: bool, path: Path) -> None:
    """The two LATENT states, which no data ever constrains directly.

    This is the most useful figure in the project for calibration, and the
    reason is worth stating: a model can produce the right seconds for entirely
    wrong reasons -- a temperature that never settles, or one that settles at
    200 degrees, with the wear curve bent to compensate. The seconds plot
    cannot show that. This one can.

    Only the equation holds theta in place. If theta is wrong here, the fit is
    a coincidence.
    """
    chosen = _representative_stints(stints)
    horizon = np.arange(1, STRATEGY_HORIZON + 1)

    fig, axes = plt.subplots(2, len(chosen), figsize=(4.3 * len(chosen), 6.2),
                             sharex=True, squeeze=False)

    for column, stint in enumerate(chosen):
        theta, d = model.predict_state(stint.context, horizon)

        axes[0][column].plot(horizon, theta, color=COLORS["pinn"], lw=1.8,
                             label="PINN")
        axes[1][column].plot(horizon, d, color=COLORS["pinn"], lw=1.8, label="PINN")

        if synthetic:
            true_theta, true_d = true_state(stint.context, horizon)
            axes[0][column].plot(horizon, true_theta, color=COLORS["exact"],
                                 lw=3, alpha=0.7, label="true")
            axes[1][column].plot(horizon, true_d, color=COLORS["exact"],
                                 lw=3, alpha=0.7, label="true")

        axes[0][column].set_title(f"{stint.compound}  ({stint.stint_id})", fontsize=10)
        axes[1][column].set_xlabel("stint lap")
        for row in (0, 1):
            axes[row][column].grid(alpha=0.15)

    axes[0][0].set_ylabel("theta  (temperature over reference)")
    axes[1][0].set_ylabel("d  (fraction of tread consumed)")
    axes[1][0].axhline(1.0, color=COLORS["measured"], ls="--", lw=1.0)
    axes[0][-1].legend(fontsize=8)

    fig.suptitle("The latent states: neither is ever measured", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_training(model: PINN, path: Path) -> None:
    """How the terms of the loss come down.

    Worth watching side by side rather than only as a total: a total that falls
    while one term stays flat means that term is being ignored, and the fix is
    its weight, not more iterations.
    """
    history = model.history
    iterations = [h["iteration"] for h in history]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, label, color, style in [
        ("physics", "wear equation residual", COLORS["exact"], "-"),
        ("thermal", "thermal equation residual", "#C98A2B", "-"),
        ("data", "data (measured pace)", COLORS["pinn"], "-"),
        ("ic", "initial conditions", COLORS["linear"], "--"),
    ]:
        values = [h[key] for h in history]
        # With hard initial conditions the IC term is identically zero, and a
        # log axis cannot draw that. Leaving it out says more than a flat line
        # at the bottom of the plot would.
        if max(values) <= 0.0:
            continue
        ax.plot(iterations, values, lw=1.6, color=color, ls=style, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("The terms of the loss", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_parameters(model: PINN, synthetic: bool, path: Path) -> None:
    """The ten physical constants while they are being estimated.

    With synthetic data the true value is drawn as well: if the curves land on
    the dashed lines, the inverse problem worked. A curve that is a flat line
    at its starting value is the signature described in TUNING.md section 6 --
    that constant never received a usable gradient.
    """
    history = model.history
    iterations = [h["iteration"] for h in history]

    columns = 5
    rows = -(-len(LEARNABLE_PARAMS) // columns)      # ceiling division
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.7 * rows),
                             squeeze=False)
    flat = axes.ravel()

    for ax, name in zip(flat, LEARNABLE_PARAMS):
        ax.plot(iterations, [h[name] for h in history],
                lw=1.8, color=COLORS["pinn"], label="estimated")
        if synthetic:
            ax.axhline(float(getattr(GROUND_TRUTH, name)),
                       color=COLORS["measured"], ls="--", lw=1.2, label="true value")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.15)

    # Hide the leftover cells rather than leaving empty axes with ticks on them.
    for ax in flat[len(LEARNABLE_PARAMS):]:
        ax.axis("off")

    flat[0].legend(fontsize=8)
    fig.suptitle(
        "Inverse problem: ten physical constants estimated alongside the weights",
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
        stints = data.load_csv(args.csv, min_laps=args.min_laps,
                               drivers=args.drivers)
    except (FileNotFoundError, ValueError) as error:
        # A clear message beats a twenty-line traceback.
        print(f"\nCould not load the data:\n  {error}")
        return None

    # The split needs one stint on each side. One driver in one race is
    # usually two or three stints, so this is easy to hit with --drivers.
    if len(stints) < 2:
        print(f"\nOnly {len(stints)} stint left, and at least 2 are needed: one "
              "to train, one to test. Add drivers, races, or lower --min-laps.")
        return None

    label = args.csv
    if args.drivers:
        label += f" (drivers: {' '.join(d.upper() for d in args.drivers)})"
    print(f"\nSource: {label}")
    return stints


def train_pinn(inputs, delta, args) -> PINN:
    """Step 2. Build the network and train it, printing what it is doing."""
    iterations = 1500 if args.quick else args.iterations

    print(f"\n[1/3] Training the PINN ...")
    model = PINN(width=args.width, layers=args.layers, ic=args.ic, seed=args.seed)
    print(f"      network: {args.layers} layers of {args.width} neurons, "
          f"{model.n_weights()} weights, 2 outputs (theta, d)")
    print(f"      plus {len(LEARNABLE_PARAMS)} physical constants to estimate: "
          f"{', '.join(LEARNABLE_PARAMS)}")
    print(f"      initial conditions: {args.ic}")
    print(f"      weights: physics {args.w_physics} thermal {args.w_thermal} "
          f"data {args.w_data} ic {args.w_ic}")

    started = time.perf_counter()
    model.train(
        inputs, delta,
        iterations=iterations,
        n_collocation=args.collocation,
        lr=args.lr,
        w_physics=args.w_physics,
        w_thermal=args.w_thermal,
        w_data=args.w_data,
        w_ic=args.w_ic,
        lbfgs_iterations=args.lbfgs,
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
        "Cliff = % of test stints where the model predicts a cliff within 45 laps.",
        "CliffErr = mean error in WHICH lap, in laps. Blank with real data,",
        "where there is no true cliff lap to compare against.",
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

    # The true curve is passed only on the synthetic bench. With real data
    # there is no cliff lap to be wrong about, so that column stays blank
    # rather than being filled with something invented.
    reference = true_pace if synthetic else None

    metrics = [
        evaluate("PINN", model.predict_stint, test, reference),
        evaluate("Linear (classic)", linear.predict_stint, test, reference),
    ]

    lines = _results_table(metrics)
    lines += _recovery_table(model) if synthetic else _estimates_list(model)
    return "\n".join(lines)


def save_outputs(model, linear, stints, test, report, synthetic, out: Path) -> None:
    """Step 5. Write the four figures and the text report to disk."""
    models = {"PINN": model, "Linear": linear}

    plot_fit(models, test, synthetic, out / "01_fit.png")
    plot_training(model, out / "02_training.png")
    plot_parameters(model, synthetic, out / "03_parameters.png")
    plot_state(model, test, synthetic, out / "04_state.png")

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
    print("Tire degradation PINN - coupled thermal + wear")
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
