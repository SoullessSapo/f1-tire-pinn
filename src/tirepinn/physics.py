"""Physical model of an F1 tire.

Two coupled dimensionless ODEs describe the life of a stint:

    (E1)  dtheta/dtau = A_gen * q * (1 + zeta * d)  -  (h0 + h1 * v) * theta
    (E2)  dd/dtau     = kw * lam^m * exp(Ea * (theta + T_trk) - kappa * c) * (1 - d)

where

    tau   = stint_lap / L_ref                  dimensionless time
    theta = (T_surface - T_track) / dT_ref     thermal excess of the tread
    d     = fraction of tread consumed (0 = new, 1 = gone)
    q     = specific frictional energy per lap (telemetry)
    lam   = mean mechanical load in g (telemetry)
    v     = mean speed (forced convective cooling)
    T_trk = normalised track temperature
    c     = compound hardness index (0 soft .. 1 hard)

(E1) is a lumped-capacitance heat balance: frictional generation minus
exponential surface decay (baseline cooling + forced convection). (E2) combines
Archard's law (wear proportional to load) with an Arrhenius thermal activation
linearised around the working window: wear grows exponentially with temperature.

The (1 - d) factor in (E2) is a saturation term: it bounds d to [0, 1]
structurally, because you cannot consume more tread than exists. It is an
explicit modelling assumption rather than a numerical trick, and it has the
advantage that the physical bound is enforced by the ODE itself.

The (1 + zeta * d) factor in (E1) is the coupling that produces the cliff: as
the tread thins, the same frictional energy is deposited into less rubber mass,
surface temperature rises, and by Arrhenius wear accelerates. It is positive
feedback, so the cliff emerges from the coupled dynamics instead of being
imposed by hand. This is exactly the kind of thermodynamic constraint an LSTM
has no way of knowing.

The measurable observable is not d but the pace loss:

    delta(tau) = gamma1 * d  +  gamma2 * d^p

The first term is linear degradation; the second, with p large, is negligible
until d approaches 1 and then explodes: that is the cliff.

These functions accept both numpy arrays and torch tensors, so the same code
defines the reference ground truth (numerical integration) and the PINN residual.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import PhysicsConfig

try:  # torch is only needed when evaluating the residual inside DeepXDE
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _is_tensor(x) -> bool:
    return torch is not None and torch.is_tensor(x)


def _exp(x):
    return torch.exp(x) if _is_tensor(x) else np.exp(x)


def _pow(x, p):
    if _is_tensor(x):
        return torch.pow(torch.clamp(x, min=1e-3), p)
    return np.power(np.maximum(x, 1e-3), p)


def _clamp_max(x, hi):
    if _is_tensor(x):
        return torch.clamp(x, max=hi)
    return np.minimum(x, hi)


def _relu(x):
    if _is_tensor(x):
        return torch.relu(x)
    return np.maximum(x, 0.0)


@dataclass
class TireParams:
    """Physical model parameters, estimated by the PINN as an inverse problem."""

    A_gen: float
    zeta: float
    h0: float
    h1: float
    kw: float
    m: float
    Ea: float
    kappa: float
    gamma1: float
    gamma2: float
    p: float = 8.0

    @classmethod
    def from_config(cls, cfg: PhysicsConfig) -> TireParams:
        """Initial values declared in the configuration."""
        return cls(
            A_gen=cfg.A_gen,
            zeta=cfg.zeta_init,
            h0=cfg.h0_init,
            h1=cfg.h1_init,
            kw=cfg.kw_init,
            m=cfg.m_init,
            Ea=cfg.Ea_init,
            kappa=cfg.kappa_init,
            gamma1=cfg.gamma1_init,
            gamma2=cfg.gamma2_init,
            p=cfg.cliff_exponent,
        )

    def as_dict(self) -> dict:
        return asdict(self)


# Reference ground truth used by the synthetic generator. It differs from the
# initial values in PhysicsConfig on purpose: the PINN has to *recover* it from
# the data, and that recovery is what validates the inverse problem.
GROUND_TRUTH = TireParams(
    A_gen=8.0,
    zeta=0.90,
    h0=6.0,
    h1=4.0,
    kw=0.55,
    m=1.50,
    Ea=0.95,
    kappa=0.85,
    gamma1=1.35,
    gamma2=2.60,
    p=8.0,
)


# --------------------------------------------------------------------------
# Terms of the system
# --------------------------------------------------------------------------
def theta_rhs(theta, d, q, v, p: TireParams):
    """Right-hand side of (E1): frictional generation minus thermal decay.

    The (1 + zeta * d) factor concentrates the frictional energy into an ever
    thinner tread: it is what drives the cliff.
    """
    return p.A_gen * q * (1.0 + p.zeta * d) - (p.h0 + p.h1 * v) * theta


def wear_rate(theta, d, lam, track_temp, compound, p: TireParams, exp_clamp: float = 6.0):
    """Right-hand side of (E2): Archard times Arrhenius wear, with saturation.

    It is non-negative while d <= 1, so monotonic wear (dd/dtau >= 0) and the
    bound d <= 1 are both implied by the ODE residual itself, with no extra
    constraints on the network.
    """
    expo = _clamp_max(p.Ea * (theta + track_temp) - p.kappa * compound, exp_clamp)
    return p.kw * _pow(lam, p.m) * _exp(expo) * _relu(1.0 - d)


def pace_loss(d, p: TireParams):
    """Observable: pace loss in seconds relative to the stint's best lap.

    With d bounded to [0, 1] by (E2), the maximum loss is gamma1 + gamma2.
    """
    d = _clamp_max(_relu(d), 1.0)
    return p.gamma1 * d + p.gamma2 * _pow(d, p.p)


# --------------------------------------------------------------------------
# Reference integration (numpy): generates the ground truth of a stint
# --------------------------------------------------------------------------
def integrate_stint(
    n_laps: int,
    context: np.ndarray,
    p: TireParams,
    phys: PhysicsConfig,
    steps_per_lap: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate (E1)-(E2) with Runge-Kutta 4 over a full stint.

    Args:
        n_laps: number of laps in the stint.
        context: vector (q_fric, load, speed, track_temp, compound).
        p: physical parameters.
        phys: non-dimensionalisation scales.
        steps_per_lap: time subdivision of the integrator.

    Returns:
        laps: laps 1..n_laps.
        theta: thermal excess at the end of each lap.
        d: accumulated degradation at the end of each lap.
    """
    q, lam, v, trk, comp = (float(c) for c in context)
    dt = 1.0 / (phys.lap_ref * steps_per_lap)

    def f(state):
        th, dd = state
        return np.array(
            [
                theta_rhs(th, dd, q, v, p),
                wear_rate(th, dd, lam, trk, comp, p, phys.exp_clamp),
            ]
        )

    state = np.array([phys.theta_init, 0.0])
    theta_out, d_out = [], []
    for _ in range(n_laps):
        for _ in range(steps_per_lap):
            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        theta_out.append(state[0])
        d_out.append(state[1])

    laps = np.arange(1, n_laps + 1, dtype=float)
    return laps, np.asarray(theta_out), np.asarray(d_out)


# --------------------------------------------------------------------------
# Post-processing: cliff and remaining useful life
# --------------------------------------------------------------------------
def cliff_lap(
    laps: np.ndarray,
    delta: np.ndarray,
    phys: PhysicsConfig,
    smooth: bool = True,
) -> float | None:
    """First lap at which the pace curve enters the cliff.

    Operational criterion: the slope of the pace loss stays above
    cliff_slope_s_per_lap for at least cliff_min_run consecutive laps. Returns
    None if the stint ends before the cliff.

    Both halves of that criterion are load-bearing, and the second one was added
    after measuring the failure of the first. The curve is smoothed with a
    three-lap moving average before differentiating, but smoothing alone is not
    enough: against a purely linear curve with no cliff at all, plus the ~0.3-0.5 s
    of timing noise a real lap carries, a single-point test fires on essentially
    every stint. Demanding a sustained run is what separates a genuine knee from
    a noise excursion.

    The same criterion is applied to the observed ground truth and to any model's
    prediction, which is what makes the metric comparable across models.
    """
    laps = np.asarray(laps, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if laps.size < 3:
        return None
    if smooth and delta.size >= 5:
        kernel = np.ones(3) / 3.0
        padded = np.pad(delta, 1, mode="edge")
        delta = np.convolve(padded, kernel, mode="valid")

    hot = np.gradient(delta, laps) >= phys.cliff_slope_s_per_lap
    run = 0
    for i, is_hot in enumerate(hot):
        run = run + 1 if is_hot else 0
        if run >= phys.cliff_min_run:
            return float(laps[i - phys.cliff_min_run + 1])
    return None


def wear_lap(laps: np.ndarray, d: np.ndarray, phys: PhysicsConfig) -> float | None:
    """First lap at which the tire is worn past `d_crit`.

    Use this whenever the latent wear state is available -- that is, for the
    model's own predictions. It is the *physical* criterion, and unlike
    `cliff_lap` it needs no noise-robustness machinery, because `d` is a
    predicted state rather than a differentiated noisy measurement.

    `cliff_lap` exists for the case where `d` is not available: comparing
    against observed lap times, or against baselines that only predict pace.
    """
    hit = np.nonzero(np.asarray(d, dtype=float) >= phys.d_crit)[0]
    if hit.size == 0:
        return None
    return float(np.asarray(laps, dtype=float)[hit[0]])


def remaining_useful_life(current_lap: float, cliff: float | None, horizon: float) -> float:
    """RUL in laps. If no cliff is detected, it is capped by the given horizon."""
    if cliff is None:
        return float(max(horizon - current_lap, 0.0))
    return float(max(cliff - current_lap, 0.0))
