"""Synthetic stint generator.

Real telemetry is not part of v0 on purpose. Building the dataset from FastF1
is a substantial piece of work in its own right -- reconstructing lateral
acceleration from GPS, separating fuel burn from degradation, deciding where a
stint really begins -- and none of it can be debugged until the model that
consumes it is known to work. So the model is validated first, against data
whose true answer is known.

Each stint is one set of tires: integrate the ODE, read off the pace loss, add
timing noise. The noise matters. Without it the fit is trivial and the physics
term in the loss has nothing to do; with it, the difference between a model
that follows the noise and one that is constrained shows up immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import COMPOUND_INDEX, LAP_REF, GROUND_TRUTH, TireParams, integrate_stint, pace_loss

_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass
class Stint:
    """One set of tires, from leaving the pits to coming back in."""

    stint_id: str
    compound: str
    laps: np.ndarray      # lap within the stint, 1..n
    delta: np.ndarray     # OBSERVED pace loss [s], carries noise
    d_true: np.ndarray    # true wear state -- for checking only, never trained on
    delta_true: np.ndarray  # noise-free pace curve

    @property
    def c(self) -> float:
        return COMPOUND_INDEX[self.compound]

    @property
    def tau(self) -> np.ndarray:
        return self.laps / LAP_REF


def generate(
    n_stints: int = 24,
    p: TireParams = GROUND_TRUTH,
    noise_s: float = 0.05,
    min_laps: int = 12,
    max_laps: int = 30,
    seed: int = 0,
) -> list[Stint]:
    """Generate `n_stints` stints, cycling through the three compounds."""
    rng = np.random.default_rng(seed)
    stints = []

    for i in range(n_stints):
        compound = _COMPOUNDS[i % len(_COMPOUNDS)]
        n_laps = int(rng.integers(min_laps, max_laps + 1))

        laps, d = integrate_stint(n_laps, COMPOUND_INDEX[compound], p)
        delta_true = pace_loss(d, p)
        delta_obs = delta_true + rng.normal(0.0, noise_s, size=delta_true.shape)

        stints.append(
            Stint(
                stint_id=f"S{i:02d}",
                compound=compound,
                laps=laps,
                delta=delta_obs,
                d_true=d,
                delta_true=delta_true,
            )
        )
    return stints


def split(stints: list[Stint], test_fraction: float = 0.25, seed: int = 0):
    """Split by whole stint, never by lap.

    Splitting by lap would put laps 5 and 6 of the same set on opposite sides of
    the split, which leaks: the model would be tested on a curve it has already
    partly seen. Every model in the comparison would look better than it is.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(stints))
    n_test = max(1, round(test_fraction * len(stints)))
    test_idx = set(idx[:n_test].tolist())
    train = [s for i, s in enumerate(stints) if i not in test_idx]
    test = [s for i, s in enumerate(stints) if i in test_idx]
    return train, test


def as_arrays(stints: list[Stint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a list of stints into (tau, c, delta) column vectors."""
    tau = np.concatenate([s.tau for s in stints]).reshape(-1, 1)
    c = np.concatenate([np.full(s.laps.size, s.c) for s in stints]).reshape(-1, 1)
    delta = np.concatenate([s.delta for s in stints]).reshape(-1, 1)
    return tau, c, delta
