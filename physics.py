"""
PHYSICAL MODEL OF TIRE WEAR
===========================

HOW TO READ THIS FILE
---------------------
Everything here revolves around ONE differential equation.

A differential equation does not say how much something is worth. It says how
fast it CHANGES. It is the difference between "the tank holds 40 litres" and
"the tank loses 2 litres per minute". With the second sentence, plus a starting
level, you can reconstruct the level at any future instant: that is integrating.

Here what changes is tire wear, and the equation says how fast the tire is
being consumed given the conditions it is running in.


THE VARIABLES, ONE BY ONE
-------------------------
There are three groups and they should never be mixed up:

1) TIME
   tau .............. dimensionless time = stint_lap / 30
                      A normal 30-lap stint runs from tau=0 to tau=1.
                      Dividing by 30 keeps the numbers entering the network
                      close to 1. A network trained on inputs of wildly
                      different scales converges badly.

2) THE STATE (what evolves)
   d ................ fraction of the tread that has been consumed.
                      0 = brand-new tire, 1 = tire gone.
                      *** IT IS LATENT: it is NEVER measured, not on the test
                      bench and not in a race. The model reconstructs it. ***

3) THE CONTEXT (5 variables, constant within a stint)
   q_fric ........... frictional energy per lap, normalised (~1)
                      How much energy the track puts into the rubber.
   load ............. mean mechanical load in g, normalised (~1)
                      How much apparent weight the tire carries in a corner.
   speed ............ mean speed, normalised (~1)
                      More speed = more airflow = more cooling.
   track_temp ....... track temperature, normalised from 0 to 1
   compound ......... hardness: 0 = soft, 0.5 = medium, 1 = hard

And one more thing, the only one that is actually measured:

   delta ............ pace loss in seconds relative to the stint's best lap.
                      *** THIS IS THE ONLY OBSERVABLE. ***


THE EQUATION
------------
    dd/dtau = k(context) * (1 - d)

    with  k(context) = kw * (load/load_ref)^m
                          * exp( Ea*(track_temp - temp_ref)
                               + Eq*(q_fric     - q_ref)
                               - Ev*(speed      - speed_ref)
                               - kappa*(compound - compound_ref) )

Read in plain words: "the tire wears at a rate that depends on the conditions,
and that slows down as there is less rubber left".

Every piece of the exponent is a physical claim, and every sign is chosen:

    + Ea*track_temp .... hotter track, more wear (thermal activation)
    + Eq*q_fric ........ more frictional energy, more wear
    - Ev*speed ......... more speed, more cooling, LESS wear
    - kappa*compound ... harder compound, LESS wear
    load^m ............. Archard's law: wear grows with load


WHY EACH TERM SUBTRACTS A REFERENCE
-----------------------------------
Notice that every variable enters as (variable - its_reference). That makes the
exponent vanish at reference conditions, so there k = kw. In other words, kw
comes to mean literally "the wear rate under normal conditions", which is a
sentence you can argue about with an engineer.

But it is not cosmetic. Without subtracting the reference, kw and the other
coefficients step on each other: raising kw and lowering Ev at the same time
leaves mean wear unchanged, so there are infinitely many near-equivalent
combinations and the optimiser cannot tell which to pick. It was measured in
this very project: uncentred, kw came out with 10 % error and Ev with 33 %, BUT
the combination log(kw) - Ev was recovered to within 0.0098. That is, the model
knew perfectly well how fast the tire was wearing; what it did not know was
which of the two constants to attribute it to.

Centring the regressors is the classic remedy for that problem in regression,
and it does exactly the same job here.


WHAT THIS MODEL DOES NOT HAVE
-----------------------------
It has no tire temperature as a state of its own. Temperature enters only
approximately, through track_temp, q_fric and speed, but it does not evolve.

That has an important consequence: there is no feedback between wear and heat,
and therefore THERE IS NO CLIFF. The pace curve can only bend one way. Adding
that feedback is the jump to the `main` branch, where a second differential
equation carries the temperature. See ROADMAP.md.


THE PROPERTY THAT MAKES THIS MODEL USEFUL
-----------------------------------------
Because the context is constant within a stint, k(context) is a CONSTANT, and
then the equation has an EXACT hand-written solution:

    d(tau) = 1 - exp(-k * tau)

That is precisely what is needed to start: you can check that the network is
right by comparing it against the exact answer, before pointing the method at a
system where no exact answer exists to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

try:
    # torch is only needed when the PINN evaluates the residual during
    # training. Generating data or integrating only needs numpy.
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------------
# 1) SCALE CONSTANTS
# ---------------------------------------------------------------------------

# How many laps make up one unit of tau. A 30-lap stint -> tau from 0 to 1.
LAP_REF = 30.0

# How far ahead we look when extrapolating: a strategy decision horizon.
STRATEGY_HORIZON = 45

# Translation from compound name to a number between 0 (soft) and 1 (hard).
COMPOUND_INDEX = {
    "SOFT": 0.0,
    "MEDIUM": 0.5,
    "HARD": 1.0,
    # The ones that can show up in real data and are not slicks:
    "INTERMEDIATE": 0.75,
    "WET": 1.0,
}


# ---------------------------------------------------------------------------
# 2) THE CONTEXT OF A STINT
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Context:
    """The 5 conditions a stint runs in.

    They are constant within the stint: one set of tires always runs the same
    circuit, at roughly the same temperature, on the same compound.

    A class with names is used rather than a list of 5 numbers because
    `context.track_temp` can be understood while reading and `context[3]`
    cannot.
    """

    q_fric: float = 1.0
    load: float = 1.0
    speed: float = 1.0
    track_temp: float = 0.5
    compound: float = 0.0

    def vector(self) -> np.ndarray:
        """The 5 numbers in canonical order, ready to feed the network."""
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=float)

    @classmethod
    def from_vector(cls, v) -> Context:
        """Inverse of `vector()`."""
        return cls(*(float(x) for x in np.asarray(v, dtype=float).ravel()))

    @property
    def compound_name(self) -> str:
        """The nearest compound, for labelling plots."""
        return min(COMPOUND_INDEX, key=lambda k: abs(COMPOUND_INDEX[k] - self.compound))


# Names in canonical order. THE WHOLE project assumes this order: the network,
# the baseline, the generator and the CSV reader. It is the project's contract.
CONTEXT_NAMES = tuple(f.name for f in fields(Context))
N_CONTEXT = len(CONTEXT_NAMES)          # 5
N_INPUTS = 1 + N_CONTEXT                # 6 = tau + context

# Physically sensible range of each context variable.
# It is used for two things: generating synthetic stints, and deciding WHERE
# the network is required to satisfy the equation (see pinn.py, collocation).
CONTEXT_RANGES = {
    "q_fric": (0.40, 1.60),
    "load": (0.50, 1.50),
    "speed": (0.60, 1.40),
    "track_temp": (0.00, 1.00),
    "compound": (0.00, 1.00),
}

# Reference conditions: the centre of each range. At this point the exponent of
# the equation is zero, so kw is exactly "the wear rate under normal
# conditions". See the explanation above on why centring matters.
REFERENCE = Context(
    q_fric=1.0,
    load=1.0,
    speed=1.0,
    track_temp=0.5,
    compound=0.5,
)


# ---------------------------------------------------------------------------
# 3) THE PHYSICAL CONSTANTS OF THE MODEL
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TireParams:
    """The 7 constants of the equation.

    Six of them are ESTIMATED by the PINN while it trains (the inverse
    problem). The seventh, gamma1, stays fixed, and there is an important
    reason for that spelled out just below the class.
    """

    kw: float = 0.55        # base wear rate
    m: float = 1.50         # load exponent (Archard's law)
    Ea: float = 0.95        # how much track temperature accelerates wear
    Eq: float = 0.40        # how much frictional energy accelerates it
    Ev: float = 0.35        # how much cooling by speed slows it down
    kappa: float = 0.85     # how much a harder compound resists wear
    gamma1: float = 1.35    # seconds lost per unit of wear [s]


# The "true" values the synthetic generator uses. The PINN starts from
# deliberately wrong values and has to RECOVER these from noisy lap times.
# Succeeding is the proof that the method works, and it is the one check that
# is impossible with real data, because no ground truth exists there.
GROUND_TRUTH = TireParams()

# The six the PINN estimates. gamma1 is NOT in the list, on purpose.
#
# Why gamma1 stays fixed
# ----------------------
# The only thing measured is delta = gamma1 * d. If d and gamma1 are both left
# free, infinitely many combinations produce exactly the same delta: doubling d
# and halving gamma1 changes nothing observable. The optimiser has no way to
# prefer one, and slides along that direction until it overflows. Fixing gamma1
# anchors the scale of d and closes the problem.
#
# In the `main` branch this same issue appears at full size and was the worst
# bug of the project: training diverged to an RMSE of billions of seconds while
# the training loss looked low.
LEARNABLE_PARAMS = ("kw", "m", "Ea", "Eq", "Ev", "kappa")


# ---------------------------------------------------------------------------
# 4) THE EQUATION
# ---------------------------------------------------------------------------

def _is_tensor(x) -> bool:
    """True if x is a torch tensor (rather than a numpy array)."""
    return torch is not None and torch.is_tensor(x)


def wear_constant(q_fric, load, speed, track_temp, compound, p):
    """k(context): everything in the equation that does NOT depend on d.

    It is split into its own function because it appears in two places that
    have to agree exactly: the wear rate and the exact solution.

    It works identically with numpy numbers and with torch tensors. That
    duality is deliberate: it lets the SAME function define the ground truth
    (when generating data) and the residual the network minimises (when
    training). With two copies, sooner or later one gets fixed and the other
    does not.
    """
    if _is_tensor(q_fric):
        exp, power = torch.exp, torch.pow
        relative_load = torch.clamp(load / REFERENCE.load, min=1e-6)
    else:
        exp, power = np.exp, np.power
        relative_load = np.maximum(load / REFERENCE.load, 1e-6)

    # Load is never negative physically, but during training the network can
    # be handed odd points, and `power` with a negative base gives NaN.

    # Each variable enters as "how far it deviates from normal". That way kw
    # keeps a meaning of its own and does not step on the other coefficients.
    exponent = (
        p.Ea * (track_temp - REFERENCE.track_temp)
        + p.Eq * (q_fric - REFERENCE.q_fric)
        - p.Ev * (speed - REFERENCE.speed)
        - p.kappa * (compound - REFERENCE.compound)
    )
    return p.kw * power(relative_load, p.m) * exp(exponent)


def wear_rate(d, q_fric, load, speed, track_temp, compound, p):
    """The right-hand side of the equation: dd/dtau.

    It is non-negative while d <= 1, so monotonic wear (the tire can only get
    worse) and the bound d <= 1 both fall out of the equation itself.
    """
    k = wear_constant(q_fric, load, speed, track_temp, compound, p)
    return k * (1.0 - d)


def pace_loss(d, p: TireParams):
    """Turns the latent state d into the only observable: seconds.

    This is the "observation operator". The network predicts d, which nobody
    has ever measured; this function translates it into something that can be
    compared against the data. That is how d can be reconstructed without ever
    appearing in the loss function.
    """
    return p.gamma1 * d


# ---------------------------------------------------------------------------
# 5) SOLVING THE EQUATION (two ways)
# ---------------------------------------------------------------------------

def exact_solution(tau, context: Context, p: TireParams) -> np.ndarray:
    """The hand-written solution, with nothing approximated.

    Because the context is constant within the stint, k is constant too, and
    separating variables in dd/dtau = k*(1-d) with d(0)=0 gives directly:

        d(tau) = 1 - exp(-k * tau)

    This is the reference everything else is checked against. Having an exact
    answer is the reason this model is the right starting point: the method is
    validated where it CAN be validated.
    """
    k = wear_constant(
        context.q_fric, context.load, context.speed,
        context.track_temp, context.compound, p,
    )
    return 1.0 - np.exp(-k * np.asarray(tau, dtype=float))


def integrate_stint(
    n_laps: int,
    context: Context,
    p: TireParams,
    steps_per_lap: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the equation step by step, with 4th-order Runge-Kutta.

    Here this is redundant: the exact solution is right above. And that is
    exactly why it is here. The integrator can be checked against the exact
    answer NOW, while checking is cheap. In `main` there are two coupled
    equations, the exact solution disappears, and this integrator is all that
    is left: it had better be validated.

    Runge-Kutta 4 rather than Euler because Euler accumulates a systematic
    bias, and the inverse problem would read that bias as if it were physics.

    Returns (laps, d) with laps numbered 1..n_laps.
    """
    dt = 1.0 / (LAP_REF * steps_per_lap)

    def rate(current_d: float) -> float:
        """The slope at a given wear level, with this stint's context."""
        return wear_rate(
            current_d, context.q_fric, context.load, context.speed,
            context.track_temp, context.compound, p,
        )

    d = 0.0            # brand-new tire
    history = []

    for _ in range(n_laps):
        for _ in range(steps_per_lap):
            # Runge-Kutta 4: instead of trusting the slope at a single point,
            # it averages four slopes taken across the step.
            k1 = rate(d)
            k2 = rate(d + 0.5 * dt * k1)
            k3 = rate(d + 0.5 * dt * k2)
            k4 = rate(d + dt * k3)
            d = d + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        history.append(d)

    laps = np.arange(1, n_laps + 1, dtype=float)
    return laps, np.asarray(history)
