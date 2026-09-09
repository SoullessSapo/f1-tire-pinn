"""Physical model of tire wear, v0: one ODE.

    (E)   dd/dtau = kw * exp(-kappa * c) * (1 - d)

    obs   delta(tau) = gamma1 * d

    tau   dimensionless stint time, lap / lap_ref
    d     fraction of tread consumed, 0 = new, 1 = gone   <- LATENT, never observed
    c     compound hardness index, 0 = soft ... 1 = hard
    delta pace loss in seconds against the stint's best lap  <- the ONLY observable

Two modelling choices carry the weight here.

`(1 - d)` is a saturation term. It bounds d to [0, 1] structurally, because you
cannot consume more tread than exists, and it keeps the rate non-negative, so
monotonic wear falls out of the equation itself instead of being a constraint
bolted on afterwards.

`exp(-kappa * c)` is the compound: a harder tire resists wear.

What this model deliberately does NOT have is temperature. There is no thermal
state, so there is no feedback from a thinning tread to a hotter surface, and
therefore no cliff -- the pace curve can only bend one way. That is the first
thing v1 has to fix; see ROADMAP.md, step 2.

A useful property of a model this small: it has a closed-form solution. That is
what makes it the right first step. The PINN can be checked against an exact
answer here, before the method is pointed at a system where no exact answer
exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch is only needed when the PINN evaluates the residual
    import torch
except ImportError:  # pragma: no cover
    torch = None

# Laps per unit of tau. A stint of 30 laps spans tau = 0 .. 1.
LAP_REF = 30.0

# Compound hardness index used everywhere in the project.
COMPOUND_INDEX = {"SOFT": 0.0, "MEDIUM": 0.5, "HARD": 1.0}


@dataclass(frozen=True)
class TireParams:
    """The three physical constants of the v0 model."""

    kw: float = 1.50      # wear rate coefficient
    kappa: float = 0.85   # compound resistance
    gamma1: float = 1.35  # seconds of pace lost per unit of wear [s]


# The values the synthetic generator uses. The PINN starts from a deliberately
# wrong kw and has to recover this one from noisy lap times alone.
GROUND_TRUTH = TireParams()


def wear_rate(d, c, kw: float, kappa: float):
    """Right-hand side of (E).

    Takes kw and kappa loose rather than a TireParams, because the PINN passes a
    torch tensor for kw: during training it is being estimated, not known. The
    only array op used is exp, so numpy arrays and torch tensors both work.
    """
    exp = torch.exp if torch is not None and torch.is_tensor(c) else np.exp
    return kw * exp(-kappa * c) * (1.0 - d)


def pace_loss(d, p: TireParams):
    """Observation operator: from the latent wear state to seconds per lap."""
    return p.gamma1 * d


def exact_solution(tau, c, p: TireParams) -> np.ndarray:
    """Closed-form solution of (E) with d(0) = 0.

    Separating variables gives d(tau) = 1 - exp(-k(c) * tau) with
    k(c) = kw * exp(-kappa * c). This is the reference the PINN is checked
    against, and the reason a one-ODE model is the right place to start.
    """
    k = p.kw * np.exp(-p.kappa * np.asarray(c, dtype=float))
    return 1.0 - np.exp(-k * np.asarray(tau, dtype=float))


def integrate_stint(n_laps: int, c: float, p: TireParams, steps_per_lap: int = 8):
    """Integrate (E) over a stint with Runge-Kutta 4.

    The exact solution above makes this redundant for v0, and that is precisely
    the point: it is here so the integrator can be validated against the closed
    form now, while it is still cheap to check. v1 couples a second ODE and the
    closed form disappears; the integrator is what remains.

    Returns (laps, d) with laps numbered 1..n_laps.
    """
    dt = 1.0 / (LAP_REF * steps_per_lap)
    rate = lambda d: p.kw * np.exp(-p.kappa * c) * (1.0 - d)  # noqa: E731

    d = 0.0
    out = []
    for _ in range(n_laps):
        for _ in range(steps_per_lap):
            k1 = rate(d)
            k2 = rate(d + 0.5 * dt * k1)
            k3 = rate(d + 0.5 * dt * k2)
            k4 = rate(d + dt * k3)
            d = d + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(d)

    return np.arange(1, n_laps + 1, dtype=float), np.asarray(out)
