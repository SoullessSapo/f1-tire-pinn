"""
CALIBRATION BENCH FOR THE PHYSICAL CONSTANTS
============================================

    python tune.py                                  # diagnose what is in physics.py
    python tune.py --Ea 2.8 --A_gen 2.1 --zeta 2.5  # try other values
    python tune.py --sweep zeta 0.5 4.0 8           # one knob, eight values
    python tune.py --check-integrator               # validate RK4 against exact
    python tune.py --plot outputs/tuning.png        # draw the curves

This script does NOT train anything. It answers one question: given these
twelve constants, what world do the equations describe? Because before asking a
network to recover constants from noisy lap times, the constants have to
describe a tire that behaves like a tire.

Nothing here writes to physics.py. When a set of values looks right, you copy
them into `TireParams` yourself -- the last block this prints is exactly the
text to paste.


THE ONE THING WORTH UNDERSTANDING FIRST
---------------------------------------
Whether a cliff can exist AT ALL is decided by one number, and you can work it
out on paper before running anything.

The temperature settles much faster than the tire wears, so to a good
approximation theta always sits at its steady value for the current d:

    theta_ss(d) = A_gen * q_fric * (1 + zeta*d) / (h0 + h1*speed)

Substitute that into the wear equation and everything collapses to

    dd/dtau = k0 * exp(beta * d) * (1 - d)

                         Ea * A_gen * q_fric * zeta
    with       beta  =  ----------------------------
                             h0 + h1 * speed

Now differentiate the right-hand side with respect to d. Wear ACCELERATES --
which is what a cliff is -- exactly while

    beta * (1 - d) > 1        i.e.        d  <  1 - 1/beta

Read off the two consequences, because they are the whole game:

    beta <= 1   THERE IS NO CLIFF. Not a faint one, not a late one: none.
                (1-d) always wins, the curve only ever flattens, and no amount
                of training will find a cliff that the equations cannot
                express.

    beta >  1   the curve steepens until d reaches 1 - 1/beta, then flattens.
                Bigger beta = earlier, sharper cliff.

So beta is the first number this script prints, and it is the one to move when
the phenomenon is missing entirely. The rest -- kw, gamma2 -- only change WHEN
and HOW BIG, never WHETHER.

The catch, and the reason this needs judgement rather than a search: beta goes
up with A_gen and zeta, and both of those also push theta up. Crank them and
you get a beautiful cliff on a tire running 200 degrees over the track. Both
numbers have to be right at once, which is what the DIAGNOSIS block checks.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace

import numpy as np

from evaluate import CLIFF_SUSTAINED, CLIFF_THRESHOLD, cliff_lap
from physics import (
    CLIFF_EXPONENT,
    COMPOUND_INDEX,
    CONTEXT_RANGES,
    FIXED_PARAMS,
    GROUND_TRUTH,
    LAP_REF,
    STRATEGY_HORIZON,
    Context,
    TireParams,
    exact_solution_isothermal,
    integrate_stint,
    pace_loss,
    steady_temperature,
)

# One normalised unit of temperature is 40 degrees C: `download_data.py` maps
# track temperature from 20..60 C onto 0..1. Everything thermal is reported in
# both, because 0.8 means nothing to anybody and 32 degrees does.
DEGREES_PER_UNIT = 40.0

# Three corners of the context space, chosen to bracket what a season contains.
SCENARIOS = {
    "worst  (soft, hot, heavy, slow)":
        Context(q_fric=1.45, load=1.35, speed=0.75, track_temp=0.90, compound=0.0),
    "middle (medium, at reference)":
        Context(q_fric=1.00, load=1.00, speed=1.00, track_temp=0.50, compound=0.5),
    "best   (hard, cold, light, fast)":
        Context(q_fric=0.55, load=0.65, speed=1.35, track_temp=0.10, compound=1.0),
}

# What a believable tire looks like. These are not laws; they are the bands
# this project decided to aim for, and they are here to be argued with.
TARGETS = {
    "beta at reference": (1.20, 4.00),
    "theta_ss cold [C]": (8.0, 30.0),
    "theta_ss worn [C]": (25.0, 70.0),
    "thermal time constant [laps]": (2.0, 10.0),
    "median d at lap 45": (0.40, 0.70),
    "fraction of dead tires (d>0.995)": (0.0, 0.05),
    "median pace loss at lap 30 [s]": (0.40, 1.20),
    "p95 pace loss at lap 30 [s]": (1.80, 3.50),
    "cliff within 45 laps": (0.20, 0.55),
    "cliff within 26 laps": (0.05, 0.30),
}

_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


# ---------------------------------------------------------------------------
# 1) THE TWO CLOSED-FORM DIAGNOSTICS
# ---------------------------------------------------------------------------

def feedback_strength(p: TireParams, context: Context) -> float:
    """beta: the number that decides whether a cliff can exist at all.

    Derived in the module docstring. Above 1 the wear accelerates for a while;
    at or below 1 it never does, whatever else is tuned.
    """
    cooling = p.h0 + p.h1 * context.speed
    return p.Ea * p.A_gen * context.q_fric * p.zeta / cooling


def acceleration_ends_at(beta: float) -> float:
    """The wear level where the steepening stops, 1 - 1/beta.

    Below beta = 1 there is no accelerating phase, so this is reported as 0.
    """
    return max(0.0, 1.0 - 1.0 / beta) if beta > 0 else 0.0


def thermal_time_constant(p: TireParams, context: Context) -> float:
    """How many LAPS the temperature takes to settle, 1/(h0 + h1*v).

    In tau units it is 1/(h0+h1*v); multiplying by LAP_REF puts it in laps,
    which is the only form anybody can sanity-check. A real tire takes a few
    laps to come up to temperature, so a value of 0.2 or of 40 is telling you
    the cooling constants are wrong.
    """
    return LAP_REF / (p.h0 + p.h1 * context.speed)


# ---------------------------------------------------------------------------
# 2) WHAT THE CONSTANTS DO TO ONE STINT
# ---------------------------------------------------------------------------

def scenario_rows(p: TireParams) -> list[str]:
    """One line per corner of the context space, over the full horizon."""
    horizon = STRATEGY_HORIZON
    lines = [
        f"{'scenario':34s} {'beta':>5s} {'thMax':>7s} {'d@45':>6s} "
        f"{'s@15':>6s} {'s@30':>6s} {'s@45':>6s} {'cliff':>6s}",
        "-" * 86,
    ]

    for label, context in SCENARIOS.items():
        _, theta, d = integrate_stint(horizon, context, p)
        delta = pace_loss(d, p)
        lap = cliff_lap(delta)
        lines.append(
            f"{label:34s} {feedback_strength(p, context):5.2f} "
            f"{theta.max() * DEGREES_PER_UNIT:6.0f}C {d[-1]:6.3f} "
            f"{delta[14]:6.2f} {delta[29]:6.2f} {delta[-1]:6.2f} "
            f"{('-' if lap is None else str(lap)):>6s}"
        )

    return lines


# ---------------------------------------------------------------------------
# 3) WHAT THEY DO ACROSS THE WHOLE CONTEXT SPACE
# ---------------------------------------------------------------------------

def sample_contexts(n: int, seed: int = 0) -> list[Context]:
    """Contexts drawn the way `data.generate_synthetic` draws them.

    Deliberately the same distribution: a set of constants that behaves well on
    three hand-picked corners and badly on the space the generator actually
    samples would produce a training set nothing can learn from.
    """
    rng = np.random.default_rng(seed)
    contexts = []
    for i in range(n):
        values = {k: float(rng.uniform(*r)) for k, r in CONTEXT_RANGES.items()}
        values["compound"] = COMPOUND_INDEX[_COMPOUNDS[i % len(_COMPOUNDS)]]
        contexts.append(Context(**values))
    return contexts


def population_statistics(p: TireParams, n: int, seed: int = 0) -> dict[str, float]:
    """Integrate `n` sampled stints and summarise the population.

    This is the slow part of the script and the honest one. The corners tell
    you the range; only this tells you what the BULK of the training set will
    look like, and the bulk is what the network actually learns from.
    """
    d_final, theta_peak, pace_30, deep, shallow = [], [], [], [], []

    for context in sample_contexts(n, seed):
        _, theta, d = integrate_stint(STRATEGY_HORIZON, context, p)
        delta = pace_loss(d, p)

        d_final.append(d[-1])
        theta_peak.append(theta.max())
        pace_30.append(delta[29])
        deep.append(cliff_lap(delta) is not None)
        # 26 laps is a typical real stint: a cliff nobody ever reaches inside
        # one is a cliff the training data never shows the network.
        shallow.append(cliff_lap(delta[:26]) is not None)

    percentile = lambda values, q: float(np.percentile(values, q))
    return {
        "median d at lap 45": percentile(d_final, 50),
        "fraction of dead tires (d>0.995)": float(np.mean(np.array(d_final) > 0.995)),
        "median pace loss at lap 30 [s]": percentile(pace_30, 50),
        "p95 pace loss at lap 30 [s]": percentile(pace_30, 95),
        "cliff within 45 laps": float(np.mean(deep)),
        "cliff within 26 laps": float(np.mean(shallow)),
        "_theta_p95": percentile(theta_peak, 95),
    }


# ---------------------------------------------------------------------------
# 4) THE VERDICT
# ---------------------------------------------------------------------------

def measurements(p: TireParams, n_samples: int) -> dict[str, float]:
    """Every number the diagnosis is graded on, in one dict."""
    reference = SCENARIOS["middle (medium, at reference)"]

    numbers = {
        "beta at reference": feedback_strength(p, reference),
        "theta_ss cold [C]": steady_temperature(
            0.0, reference.q_fric, reference.speed, p) * DEGREES_PER_UNIT,
        "theta_ss worn [C]": steady_temperature(
            1.0, reference.q_fric, reference.speed, p) * DEGREES_PER_UNIT,
        "thermal time constant [laps]": thermal_time_constant(p, reference),
    }
    numbers.update(population_statistics(p, n_samples))
    return numbers


def diagnosis_lines(numbers: dict[str, float]) -> list[str]:
    """Grade each measurement against its target band and say which way to move.

    OK / LOW / HIGH, nothing cleverer. The point is not the verdict -- it is
    that a failing row names the direction, so the next run is a decision
    rather than a guess.
    """
    lines = [
        f"{'quantity':34s} {'value':>9s} {'target band':>16s}  verdict",
        "-" * 78,
    ]

    for name, (low, high) in TARGETS.items():
        value = numbers[name]
        verdict = "OK" if low <= value <= high else ("LOW" if value < low else "HIGH")
        lines.append(
            f"{name:34s} {value:9.3f} {f'{low:.2f} .. {high:.2f}':>16s}  {verdict}"
        )

    return lines


def advice_lines(numbers: dict[str, float]) -> list[str]:
    """What to turn next, in order of how much it matters.

    Deliberately ordered rather than exhaustive: fixing beta first changes
    every other row, so suggesting six simultaneous edits would waste a run.
    """
    beta = numbers["beta at reference"]
    too_many_dead = numbers["fraction of dead tires (d>0.995)"] > 0.05
    advice: list[str] = []

    # beta first, always. It decides WHETHER the phenomenon exists, and every
    # other row is read differently depending on the answer, so advice about
    # the others before this is settled would just be noise.
    if beta <= 1.0:
        advice.append(
            f"beta = {beta:.2f} <= 1, so NO cliff can exist. Nothing else matters "
            "until this is fixed. Raise Ea, A_gen or zeta, or lower h0/h1 -- "
            "then re-read every row below, because they all move with it."
        )
        return advice

    advice.append(
        f"beta = {beta:.2f} > 1: wear accelerates up to d = "
        f"{acceleration_ends_at(beta):.2f}, then flattens."
    )

    if numbers["theta_ss cold [C]"] > TARGETS["theta_ss cold [C]"][1]:
        advice.append(
            "A FRESH tire already runs too hot. Lower A_gen, or raise h0 -- but "
            "note both also move beta, so check it again afterwards."
        )
    if numbers["theta_ss worn [C]"] > TARGETS["theta_ss worn [C]"][1]:
        advice.append(
            "A WORN tire runs hotter than any real one. Lower zeta: it is the "
            "only constant that moves the worn case without moving the fresh one."
        )
    if too_many_dead:
        advice.append(
            "Too many stints end with the tire completely gone; those curves are "
            "flat and carry no information. Lower kw."
        )
    # Only worth saying when kw is not ALREADY too high -- otherwise this and
    # the line above would ask for opposite edits in the same breath.
    if numbers["cliff within 26 laps"] < 0.05 and not too_many_dead:
        advice.append(
            "Almost no cliff falls inside a normal stint, so the training data "
            "will barely show one. Raise kw to bring it forward, or beta to "
            "sharpen it."
        )
    if numbers["p95 pace loss at lap 30 [s]"] > TARGETS["p95 pace loss at lap 30 [s]"][1]:
        advice.append(
            "The worst stints lose implausibly much time by lap 30. Lower kw, "
            "or gamma2 if the excess is all in the last few laps."
        )

    return advice


def paste_block(p: TireParams) -> list[str]:
    """The exact text to paste into `TireParams` in physics.py.

    A copyable block rather than an instruction, because transcribing twelve
    numbers by hand is how one of them ends up different from what was tested.
    """
    lines = ["    # --- paste into TireParams in physics.py ---"]
    for field in fields(p):
        value = float(getattr(p, field.name))
        fixed = "    # FIXED" if field.name in FIXED_PARAMS else ""
        lines.append(f"    {field.name}: float = {value:.4g}{fixed}")
    return lines


# ---------------------------------------------------------------------------
# 5) THE OTHER MODES
# ---------------------------------------------------------------------------

def sweep(p: TireParams, name: str, low: float, high: float, steps: int) -> list[str]:
    """Move ONE constant across a range and tabulate what it does.

    The whole reason to sweep instead of guessing: in a coupled system almost
    every constant moves almost every output, so the only way to build an
    intuition for which knob does what is to move one at a time and watch.
    """
    reference = SCENARIOS["middle (medium, at reference)"]
    lines = [
        f"Sweeping {name} from {low} to {high}",
        "",
        f"{name:>10s} {'beta':>6s} {'thWorn':>8s} {'d@45':>7s} {'s@30':>7s} "
        f"{'s@45':>7s} {'cliff':>6s}",
        "-" * 56,
    ]

    for value in np.linspace(low, high, steps):
        trial = replace(p, **{name: float(value)})
        _, theta, d = integrate_stint(STRATEGY_HORIZON, reference, trial)
        delta = pace_loss(d, trial)
        lap = cliff_lap(delta)
        worn = steady_temperature(1.0, reference.q_fric, reference.speed,
                                  trial) * DEGREES_PER_UNIT
        lines.append(
            f"{value:10.3f} {feedback_strength(trial, reference):6.2f} "
            f"{worn:7.0f}C {d[-1]:7.3f} {delta[29]:7.2f} {delta[-1]:7.2f} "
            f"{('-' if lap is None else str(lap)):>6s}"
        )

    return lines


def check_integrator(p: TireParams, n_contexts: int = 20) -> list[str]:
    """Validate RK4 against the one exact answer that survives.

    Setting A_gen = 0 kills the generation term, so theta decays to zero and
    stays there, the wear constant stops depending on time, and the system
    collapses back to the single equation that DOES have a closed form. If RK4
    reproduces it to machine precision there, the integrator is sound and any
    disagreement in the coupled case is the coupling, not the solver.

    This is the only validation left in the project that compares against
    something exact, which is why it is worth running after any change to
    `integrate_stint`.
    """
    isothermal = replace(p, A_gen=0.0)
    worst = 0.0

    for context in sample_contexts(n_contexts, seed=7):
        laps, theta, d = integrate_stint(STRATEGY_HORIZON, context, isothermal)
        exact = exact_solution_isothermal(laps / LAP_REF, context, isothermal)
        worst = max(worst, float(np.max(np.abs(d - exact))), float(np.max(np.abs(theta))))

    verdict = "PASS" if worst < 1e-8 else "FAIL"
    return [
        f"RK4 vs the exact isothermal solution, over {n_contexts} contexts:",
        f"  worst absolute discrepancy: {worst:.3e}   {verdict}",
        "",
        "  (with A_gen = 0 the coupling is off, theta stays at 0 and",
        "   d(tau) = 1 - exp(-k*tau) is exact. Anything above 1e-8 means",
        "   the integrator itself is wrong, not the physics.)",
    ]


def plot_curves(p: TireParams, path: str) -> None:
    """Three panels: the temperature, the wear, and the seconds they produce.

    Worth drawing because the numbers in the tables can all look reasonable
    while the SHAPE is wrong -- a temperature that never settles, a wear curve
    with a kink, a cliff that is really a corner. Shape is the thing tables
    hide.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colours = ["#B93A24", "#2C6C8C", "#7C8593"]

    for (label, context), colour in zip(SCENARIOS.items(), colours):
        laps, theta, d = integrate_stint(STRATEGY_HORIZON, context, p)
        delta = pace_loss(d, p)
        short = label.split("(")[0].strip()

        axes[0].plot(laps, theta * DEGREES_PER_UNIT, color=colour, lw=1.8, label=short)
        axes[1].plot(laps, d, color=colour, lw=1.8, label=short)
        axes[2].plot(laps, delta, color=colour, lw=1.8, label=short)

        lap = cliff_lap(delta)
        if lap is not None:
            axes[2].axvline(lap, color=colour, ls=":", lw=1.2)

    for ax, title, ylabel in [
        (axes[0], "theta: temperature over reference", "degrees C"),
        (axes[1], "d: fraction of tread consumed", "d"),
        (axes[2], f"delta = gamma1*d + gamma2*d^{CLIFF_EXPONENT}", "pace loss [s]"),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("stint lap")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.15)

    axes[1].axhline(1.0, color="#14181F", ls="--", lw=1.0)
    axes[2].legend(fontsize=8)
    fig.suptitle(
        f"beta at reference = "
        f"{feedback_strength(p, SCENARIOS['middle (medium, at reference)']):.2f}"
        f"   (dotted line = cliff, {CLIFF_THRESHOLD} s/lap for "
        f"{CLIFF_SUSTAINED} laps)",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6) COMMAND LINE
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Build one --flag per physical constant, plus the alternative modes."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_argument_group(
        "the constants (any you leave out keep the value in physics.py)"
    )
    for field in fields(TireParams):
        note = " [held FIXED during training]" if field.name in FIXED_PARAMS else ""
        group.add_argument(
            f"--{field.name}", type=float, default=None,
            help=f"default {field.default}{note}",
        )

    group = parser.add_argument_group("modes")
    group.add_argument("--sweep", nargs=4, metavar=("NAME", "LOW", "HIGH", "STEPS"),
                       help="move one constant across a range and tabulate it")
    group.add_argument("--check-integrator", action="store_true",
                       help="validate RK4 against the exact isothermal solution")
    group.add_argument("--plot", metavar="PATH",
                       help="draw theta, d and delta to this PNG")
    group.add_argument("--samples", type=int, default=240,
                       help="stints to integrate for the population stats")

    return parser.parse_args()


def params_from_args(args) -> tuple[TireParams, list[str]]:
    """Apply the flags on top of GROUND_TRUTH. Returns (params, what changed)."""
    overrides = {
        field.name: getattr(args, field.name)
        for field in fields(TireParams)
        if getattr(args, field.name) is not None
    }
    changes = [
        f"{name}: {float(getattr(GROUND_TRUTH, name)):g} -> {value:g}"
        for name, value in overrides.items()
    ]
    return replace(GROUND_TRUTH, **overrides), changes


def main() -> int:
    """Diagnose one set of constants, or run one of the other modes."""
    args = parse_args()
    p, changes = params_from_args(args)

    print("=" * 86)
    print("Calibration bench - nothing here is trained, and nothing is written back")
    print("=" * 86)
    print("\n".join(f"  changed  {c}" for c in changes) if changes
          else "  using the constants currently in physics.py")

    if args.check_integrator:
        print()
        print("\n".join(check_integrator(p)))
        return 0

    if args.sweep:
        name, low, high, steps = args.sweep
        if not hasattr(p, name):
            print(f"\nNo such constant: {name}")
            return 1
        print()
        print("\n".join(sweep(p, name, float(low), float(high), int(steps))))
        return 0

    print("\nTHE THREE CORNERS")
    print("\n".join(scenario_rows(p)))

    numbers = measurements(p, args.samples)

    print(f"\nDIAGNOSIS  (over {args.samples} sampled stints)")
    print("\n".join(diagnosis_lines(numbers)))

    print("\nWHAT TO TURN NEXT")
    for i, line in enumerate(advice_lines(numbers), start=1):
        print(f"  {i}. {line}")

    print()
    print("\n".join(paste_block(p)))

    if args.plot:
        plot_curves(p, args.plot)
        print(f"\nCurves written to {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
