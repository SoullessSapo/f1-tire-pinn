"""Synthetic test bench with known physical ground truth.

Integrates the (E1)-(E2) system with the parameters in `physics.GROUND_TRUTH`
and adds measurement noise. It serves two purposes:

1. The whole pipeline runs without depending on the network or the FastF1 API.
2. It validates the inverse problem: the PINN starts from deliberately
   different initial values and must *recover* the true physical parameters
   from the observed pace loss alone. With real data that check is impossible,
   because no reference ground truth exists.

The generator imitates a strategist's decision: the stint is cut a couple of
laps after the cliff, because no team runs a destroyed tire.
"""

from __future__ import annotations

import numpy as np

from .config import COMPOUND_INDEX, ContextRanges, DataConfig, PhysicsConfig
from .dataset import Stint, StintDataset
from .physics import GROUND_TRUTH, TireParams, cliff_lap, integrate_stint, pace_loss

# Compounds simulated, in order.
_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def _sample_context(rng: np.random.Generator, ranges: ContextRanges, compound: str) -> np.ndarray:
    """Sample a plausible context for a given compound."""
    q = rng.uniform(*ranges.q_fric)
    load = rng.uniform(*ranges.load)
    speed = rng.uniform(*ranges.speed)
    trk = rng.uniform(*ranges.track_temp)
    comp = COMPOUND_INDEX[compound]
    return np.array([q, load, speed, trk, comp])


def generate(
    cfg: DataConfig,
    phys: PhysicsConfig,
    ranges: ContextRanges | None = None,
    params: TireParams = GROUND_TRUTH,
    seed: int = 0,
) -> StintDataset:
    """Generate `cfg.n_stints` synthetic stints."""
    ranges = ranges or ContextRanges()
    rng = np.random.default_rng(seed)
    stints: list[Stint] = []

    for i in range(cfg.n_stints):
        compound = _COMPOUNDS[i % len(_COMPOUNDS)]
        context = _sample_context(rng, ranges, compound)

        # Integrate to the maximum length, then decide where the team would
        # actually have pitted.
        laps_full, theta_full, d_full = integrate_stint(cfg.max_stint, context, params, phys)
        delta_full = pace_loss(d_full, params)
        cliff = cliff_lap(laps_full, delta_full, phys)

        if cliff is not None:
            # Leave a few laps *after* the cliff. A team does not pit at the
            # exact instant: it loses laps deciding, waiting for a pit window or
            # covering a rival. Those are also the only laps that carry
            # information about the d -> 1 regime, which is what gamma2 and the
            # scale of kw depend on (see README, identifiability).
            n_laps = int(min(cfg.max_stint, cliff + rng.integers(2, 6)))
        else:
            n_laps = int(rng.integers(cfg.min_stint, cfg.max_stint + 1))
        n_laps = int(np.clip(n_laps, cfg.min_stint, cfg.max_stint))

        laps = laps_full[:n_laps]
        theta = theta_full[:n_laps]
        d = d_full[:n_laps]
        delta = delta_full[:n_laps]

        # Measurement noise: real timing carries traffic, wind and driver error
        # that have nothing to do with the tire.
        delta_obs = delta + rng.normal(0.0, cfg.noise_delta_s, size=delta.shape)
        theta_obs = theta + rng.normal(0.0, cfg.noise_theta, size=theta.shape)

        stints.append(
            Stint(
                stint_id=f"SYN{i:03d}",
                driver=f"SYN{i % 10:02d}",
                compound=compound,
                laps=laps,
                delta=delta_obs,
                context=context,
                theta_true=theta_obs,
                d_true=d,
                delta_true=delta,
            )
        )

    return StintDataset(
        stints=stints,
        source="synthetic",
        meta={
            "ground_truth": params.as_dict(),
            "noise_delta_s": cfg.noise_delta_s,
            "noise_theta": cfg.noise_theta,
            "seed": seed,
        },
    )
