"""Data structures shared by the synthetic generator and the FastF1 loader.

A `Stint` is the unit of learning: one set of tires from leaving the pits to
coming back in. The PINN never sees isolated laps, only whole stints, because
the physics it enforces is an ODE in time *within* a stint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import CONTEXT_NAMES, PhysicsConfig


@dataclass
class Stint:
    """One stint: laps, observed pace loss and constant context."""

    stint_id: str
    driver: str
    compound: str
    laps: np.ndarray              # lap within the stint, 1..n
    delta: np.ndarray             # observed pace loss [s]
    context: np.ndarray           # (5,) q_fric, load, speed, track_temp, compound

    # Only available on the synthetic bench (reference ground truth).
    theta_true: np.ndarray | None = None
    d_true: np.ndarray | None = None
    delta_true: np.ndarray | None = None   # noise-free pace curve
    # Only available with real data: the race lap number.
    race_laps: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.laps = np.asarray(self.laps, dtype=float).ravel()
        self.delta = np.asarray(self.delta, dtype=float).ravel()
        self.context = np.asarray(self.context, dtype=float).ravel()
        if self.laps.shape != self.delta.shape:
            raise ValueError(
                f"{self.stint_id}: laps {self.laps.shape} and delta {self.delta.shape} do not match"
            )
        if self.context.size != len(CONTEXT_NAMES):
            raise ValueError(
                f"{self.stint_id}: context must have {len(CONTEXT_NAMES)} components"
            )

    @property
    def n_laps(self) -> int:
        return int(self.laps.size)

    def tau(self, phys: PhysicsConfig) -> np.ndarray:
        """Dimensionless time tau = lap / L_ref."""
        return self.laps / phys.lap_ref

    def inputs(self, phys: PhysicsConfig) -> np.ndarray:
        """(n_laps, 6) network input matrix: tau plus the replicated context."""
        tau = self.tau(phys).reshape(-1, 1)
        ctx = np.tile(self.context.reshape(1, -1), (tau.shape[0], 1))
        return np.hstack([tau, ctx])


@dataclass
class StintDataset:
    """A collection of stints, with tensor-assembly and splitting helpers."""

    stints: list[Stint]
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.stints)

    @property
    def n_laps(self) -> int:
        return int(sum(s.n_laps for s in self.stints))

    def inputs(self, phys: PhysicsConfig) -> np.ndarray:
        """(N, 6) with every lap of every stint."""
        return np.vstack([s.inputs(phys) for s in self.stints])

    def delta(self) -> np.ndarray:
        """(N, 1) observed pace loss."""
        return np.concatenate([s.delta for s in self.stints]).reshape(-1, 1)

    def contexts(self) -> np.ndarray:
        """(n_stints, 5) context of each stint."""
        return np.vstack([s.context.reshape(1, -1) for s in self.stints])

    def theta_observations(self, phys: PhysicsConfig) -> tuple[np.ndarray, np.ndarray] | None:
        """Weak temperature supervision, if the source provides it.

        Returns (X, theta) or None. With real data this is normally None: the
        tire's internal temperature is not public.
        """
        usable = [s for s in self.stints if s.theta_true is not None]
        if not usable:
            return None
        X = np.vstack([s.inputs(phys) for s in usable])
        theta = np.concatenate([s.theta_true for s in usable]).reshape(-1, 1)
        return X, theta

    def split(self, test_fraction: float, seed: int = 0) -> tuple[StintDataset, StintDataset]:
        """Split by whole stint, never by lap.

        Splitting by lap would leak information from the same stint between
        train and test, and would inflate every model's apparent performance
        equally.
        """
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.stints))
        n_test = max(1, round(test_fraction * len(self.stints)))
        test_idx = set(idx[:n_test].tolist())
        train = [s for i, s in enumerate(self.stints) if i not in test_idx]
        test = [s for i, s in enumerate(self.stints) if i in test_idx]
        return (
            StintDataset(train, self.source, dict(self.meta, split="train")),
            StintDataset(test, self.source, dict(self.meta, split="test")),
        )

    def describe(self) -> str:
        if not self.stints:
            return "Empty dataset"
        lens = np.array([s.n_laps for s in self.stints])
        deltas = np.concatenate([s.delta for s in self.stints])
        comps: dict[str, int] = {}
        for s in self.stints:
            comps[s.compound] = comps.get(s.compound, 0) + 1
        comp_txt = ", ".join(f"{k}:{v}" for k, v in sorted(comps.items()))
        return (
            f"Source: {self.source} | stints: {len(self.stints)} | laps: {self.n_laps}\n"
            f"  Stint length:  min={lens.min()} med={np.median(lens):.0f} max={lens.max()}\n"
            f"  Pace loss:     min={deltas.min():.2f}s med={np.median(deltas):.2f}s max={deltas.max():.2f}s\n"
            f"  Compounds: {comp_txt}"
        )


def aggregate_context_by_race(
    data: StintDataset, fields: Sequence[str], key: Callable[[Stint], str] | None = None
) -> StintDataset:
    """Replace per-stint context values with the median over their race.

    Motivated by a measurement, not a hunch. Decomposing the variance of the
    context proxies over the 2026 season:

        q_fric  53% of its variance is WITHIN a circuit
        load    61% within
        speed    7% within
        track_temp  1% within

    and correlating the within-circuit deviation of each proxy against the
    within-circuit deviation of the measured degradation rate gives r = 0.001 for
    q_fric and 0.029 for load, against a 5% critical value of 0.095. So more than
    half the variance of those two proxies is variation that **predicts nothing**
    -- it is measurement noise from double-differentiated GPS, not a real
    difference between one driver's stint and another's.

    Between circuits the same proxies do carry signal (track_temp r = 0.664,
    q_fric 0.248, load 0.201), which is the variation worth keeping. Collapsing
    to the race median keeps it and discards the noise.

    `speed` is deliberately not aggregated by default: its within-circuit
    deviation does correlate with degradation (r = -0.203, significant), so
    averaging it would throw away real signal.
    """
    key = key or (lambda s: s.stint_id[:3])
    idx = [CONTEXT_NAMES.index(f) for f in fields]

    groups: dict[str, list[Stint]] = {}
    for stint in data.stints:
        groups.setdefault(key(stint), []).append(stint)

    for members in groups.values():
        medians = np.median(np.vstack([s.context for s in members]), axis=0)
        for stint in members:
            for i in idx:
                stint.context[i] = medians[i]

    meta = dict(data.meta, context_aggregated=list(fields))
    return StintDataset(data.stints, data.source, meta)


def input_bounds(
    dataset: StintDataset, phys: PhysicsConfig, pad: float = 0.05
) -> tuple[list[float], list[float]]:
    """Hypercube enclosing the observed data, with a relative margin.

    This is the domain where the ODE residuals are enforced: physics regularises
    the network in context combinations that never appear in the data too, which
    is precisely a PINN's advantage over a plain fit.
    """
    X = dataset.inputs(phys)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    lo = lo - pad * span
    hi = hi + pad * span
    lo[0] = 0.0  # tau always starts at the beginning of the stint
    # tau is stretched out to the full decision horizon, beyond the longest
    # observed stint: that is where physics alone has to sustain the
    # extrapolation towards the cliff, with no data to guide it.
    hi[0] = max(hi[0] * 1.20, phys.strategy_horizon / phys.lap_ref)
    return lo.tolist(), hi.tolist()
