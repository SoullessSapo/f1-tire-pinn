"""
WHERE THE STINTS COME FROM
==========================

A "stint" is one set of tires: from the moment the car leaves the pits until it
comes back in. It is the unit of learning for the whole project. The network
never sees isolated laps, because the physics imposed on it is an equation in
time WITHIN a stint.

This file knows how to build stints from two places:

  generate_synthetic()  simulates stints by solving the equation with known
                        constants and adding timing noise.

  load_csv()            reads the CSV produced by `download_data.py` from real
                        FastF1 telemetry.

Both return exactly the same thing: a list of `Stint` objects. From there on,
the rest of the project neither knows nor cares where they came from.


WHY START WITH SYNTHETIC DATA
-----------------------------
Because the answer is known. With synthetic data you know what the physical
constants really are and what the wear d really is on every lap, so you can
check whether the model gets it right. With real data that is impossible:
nobody publishes the state of the tread.

If the model cannot recover the constants on the synthetic bench, pointing it
at real data only produces wrong numbers with more effort.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physics import (
    COMPOUND_INDEX,
    CONTEXT_RANGES,
    GROUND_TRUTH,
    LAP_REF,
    Context,
    TireParams,
    integrate_stint,
    pace_loss,
)

_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


# ---------------------------------------------------------------------------
# 1) THE UNIT OF DATA
# ---------------------------------------------------------------------------

@dataclass
class Stint:
    """One set of tires, pits to pits."""

    stint_id: str
    context: Context          # the 5 conditions, constant across the stint
    laps: np.ndarray          # lap within the stint: 1, 2, 3, ...
    delta: np.ndarray         # MEASURED pace loss [s] (carries noise)

    # Only real stints have one: a simulated stint was driven by nobody.
    driver: str | None = None

    # These only exist on synthetic stints, where the truth is known.
    # They are used for checking, never for training. `theta_true` is the one
    # the coupled model added: the temperature curve nobody ever measures, kept
    # so a run can be asked not just whether it got the seconds right but
    # whether it got them right FOR THE RIGHT REASON.
    d_true: np.ndarray | None = None
    theta_true: np.ndarray | None = None
    delta_clean: np.ndarray | None = None

    @property
    def n_laps(self) -> int:
        """How many laps this stint lasted."""
        return int(self.laps.size)

    @property
    def tau(self) -> np.ndarray:
        """The dimensionless time of each lap."""
        return self.laps / LAP_REF

    @property
    def compound(self) -> str:
        """The compound name (SOFT/MEDIUM/HARD), for labels and grouping."""
        return self.context.compound_name


# ---------------------------------------------------------------------------
# 2) SYNTHETIC STINTS
# ---------------------------------------------------------------------------

def _sample_context(rng: np.random.Generator, compound: str) -> Context:
    """Draw plausible conditions for a given compound."""
    values = {}
    for name, (low, high) in CONTEXT_RANGES.items():
        values[name] = float(rng.uniform(low, high))
    # The compound is not drawn: the caller decides it.
    values["compound"] = COMPOUND_INDEX[compound]
    return Context(**values)


def generate_synthetic(
    n_stints: int = 48,
    p: TireParams = GROUND_TRUTH,
    noise_s: float = 0.05,
    min_laps: int = 12,
    max_laps: int = 30,
    seed: int = 0,
) -> list[Stint]:
    """Generate `n_stints` simulated stints, cycling through the three compounds.

    The noise matters and is not decoration. Without it the fit is trivial and
    the physics term of the loss has nothing to do. With it, the difference
    between a model that chases the noise and one held down by an equation
    shows up immediately.
    """
    rng = np.random.default_rng(seed)
    stints: list[Stint] = []

    for i in range(n_stints):
        compound = _COMPOUNDS[i % len(_COMPOUNDS)]
        context = _sample_context(rng, compound)
        n_laps = int(rng.integers(min_laps, max_laps + 1))

        # 1) solve BOTH equations -> the true temperature and wear, lap by lap.
        #    There is no closed form for the coupled system, so this really is
        #    a numerical integration now, not a convenience wrapper over one.
        laps, theta, d = integrate_stint(n_laps, context, p)

        # 2) translate it into the only observable: seconds lost
        clean_delta = pace_loss(d, p)

        # 3) dirty it the way reality does: traffic, wind, the driver
        measured_delta = clean_delta + rng.normal(0.0, noise_s, size=clean_delta.shape)

        stints.append(
            Stint(
                stint_id=f"SYN{i:02d}",
                context=context,
                laps=laps,
                delta=measured_delta,
                d_true=d,
                theta_true=theta,
                delta_clean=clean_delta,
            )
        )

    return stints


# ---------------------------------------------------------------------------
# 3) REAL STINTS, FROM THE DOWNLOADED CSV
# ---------------------------------------------------------------------------

# Columns `load_csv` needs. `download_data.py` writes these plus a few more
# (lap time, race lap...) so the file can be inspected by hand in a spreadsheet.
_REQUIRED_COLUMNS = (
    "stint_id", "stint_lap", "delta",
    "q_fric", "load", "speed", "track_temp", "compound",
)


def _read_rows(path: Path) -> list[dict]:
    """Read the CSV and check it has the columns we need."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Download the data first:\n"
            f"  python download_data.py --year 2023 --races Monza --out {path}"
        )

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"{path} is empty")

    missing = [c for c in _REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    return rows


def _stint_from_rows(stint_id: str, rows: list[dict]) -> Stint:
    """Build one Stint from its already-sorted rows.

    The context is taken from the first row. `download_data.py` writes the same
    value on every row of a stint (it uses the stint median), so any row will do.
    """
    first = rows[0]
    return Stint(
        stint_id=stint_id,
        context=Context(
            q_fric=float(first["q_fric"]),
            load=float(first["load"]),
            speed=float(first["speed"]),
            track_temp=float(first["track_temp"]),
            compound=float(first["compound"]),
        ),
        laps=np.array([float(r["stint_lap"]) for r in rows]),
        delta=np.array([float(r["delta"]) for r in rows]),
        driver=first.get("driver") or None,
    )


def _only_drivers(rows: list[dict], drivers, path: Path) -> list[dict]:
    """Keep the rows of these drivers, or say exactly why that is impossible.

    A driver that is not in the file is an error, not something to skip: with
    `--drivers VER HMA` (a typo) silently training on VER alone would give
    results that look fine and answer a different question than the one asked.
    """
    if "driver" not in rows[0]:
        raise ValueError(f"{path} has no 'driver' column, so it cannot be filtered")

    wanted = {d.upper() for d in drivers}
    present = {r["driver"].upper() for r in rows}
    missing = sorted(wanted - present)
    if missing:
        raise ValueError(
            f"{path} has no laps from {', '.join(missing)}. "
            f"The drivers in it are: {' '.join(sorted(present))}"
        )

    return [r for r in rows if r["driver"].upper() in wanted]


def load_csv(path: str | Path, min_laps: int = 8, drivers=()) -> list[Stint]:
    """Read the CSV from `download_data.py` and turn it into stints.

    The CSV has one row per lap. Here they are grouped by `stint_id`, since the
    whole model rests on the context being constant within a stint.

    Stints with fewer than `min_laps` valid laps are discarded: with four
    points you cannot tell a curve from a straight line.

    `drivers` (three-letter codes, any case) keeps only those drivers' stints.
    Empty keeps everyone. Filtering happens BEFORE the length check, so the
    discarded count and the errors below speak about the drivers asked for.
    """
    path = Path(path)
    rows = _read_rows(path)
    if drivers:
        rows = _only_drivers(rows, drivers, path)

    # Group by stint, keeping the order they appeared in.
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["stint_id"], []).append(row)

    stints: list[Stint] = []
    discarded = 0

    for stint_id, stint_rows in groups.items():
        stint_rows.sort(key=lambda r: float(r["stint_lap"]))
        if len(stint_rows) < min_laps:
            discarded += 1
            continue
        stints.append(_stint_from_rows(stint_id, stint_rows))

    if not stints:
        raise ValueError(
            f"No stint in {path} reaches {min_laps} valid laps. "
            "Lower --min-laps or download more races."
        )

    if discarded:
        print(f"  {discarded} stints discarded for having fewer than {min_laps} laps")

    # A driver can be in the file and still lose every stint to the length
    # check. That is not an error, but it must not go unnoticed.
    kept = {s.driver.upper() for s in stints if s.driver}
    for driver in sorted({d.upper() for d in drivers} - kept):
        print(f"  WARNING: every {driver} stint is shorter than {min_laps} laps, "
              "none kept")

    return stints


# ---------------------------------------------------------------------------
# 4) SPLITTING INTO TRAIN AND TEST
# ---------------------------------------------------------------------------

def split(
    stints: list[Stint], test_fraction: float = 0.25, seed: int = 0
) -> tuple[list[Stint], list[Stint]]:
    """Split by WHOLE STINT, never by lap.

    Splitting by lap would put laps 5 and 6 of the same set on opposite sides
    of the split. The model would then be evaluated on a curve it has already
    partly seen: that is information leakage, and it is the worst kind, because
    it improves ALL models equally and therefore cannot be spotted by comparing
    them against each other.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(stints))
    n_test = max(1, round(test_fraction * len(stints)))
    test_indices = set(order[:n_test].tolist())

    train = [s for i, s in enumerate(stints) if i not in test_indices]
    test = [s for i, s in enumerate(stints) if i in test_indices]
    return train, test


def flatten(stints: list[Stint]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a list of stints into the two matrices that train the network.

    Returns:
      inputs  (N, 6)  each row is [tau, and the 5 context variables]
      delta   (N, 1)  the pace loss measured on that lap

    The context is constant within the stint, so it is REPEATED on each of its
    laps. That is what lets the network learn the effect of time and the effect
    of conditions at the same time.
    """
    input_blocks = []
    delta_blocks = []

    for s in stints:
        tau = s.tau.reshape(-1, 1)
        repeated_context = np.tile(s.context.vector().reshape(1, -1), (tau.shape[0], 1))
        input_blocks.append(np.hstack([tau, repeated_context]))
        delta_blocks.append(s.delta.reshape(-1, 1))

    return np.vstack(input_blocks), np.vstack(delta_blocks)


def describe(stints: list[Stint]) -> str:
    """A paragraph describing the set, to print before training."""
    if not stints:
        return "Empty dataset"

    lengths = np.array([s.n_laps for s in stints])
    deltas = np.concatenate([s.delta for s in stints])

    counts: dict[str, int] = {}
    for s in stints:
        counts[s.compound] = counts.get(s.compound, 0) + 1
    by_compound = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))

    text = (
        f"{len(stints)} stints | {int(lengths.sum())} laps\n"
        f"  Stint length:  min={lengths.min()} "
        f"median={np.median(lengths):.0f} max={lengths.max()}\n"
        f"  Pace loss:     min={deltas.min():.2f}s "
        f"median={np.median(deltas):.2f}s max={deltas.max():.2f}s\n"
        f"  Compounds: {by_compound}"
    )

    drivers: dict[str, int] = {}
    for s in stints:
        if s.driver:
            drivers[s.driver] = drivers.get(s.driver, 0) + 1
    if drivers:
        text += "\n  Drivers:   " + ", ".join(
            f"{k}:{v}" for k, v in sorted(drivers.items())
        )
    return text
