"""
PHYSICAL MODEL OF TIRE WEAR
===========================

HOW TO READ THIS FILE
---------------------
Everything here revolves around TWO coupled differential equations.

A differential equation does not say how much something is worth. It says how
fast it CHANGES. It is the difference between "the tank holds 40 litres" and
"the tank loses 2 litres per minute". With the second sentence, plus a starting
level, you can reconstruct the level at any future instant: that is integrating.

Here two things change at once, and each one feeds the other:

    the TEMPERATURE of the rubber above its resting state
    the WEAR of the tread

That mutual feedback is the whole point, and it is explained under THE CLIFF.


THE VARIABLES, ONE BY ONE
-------------------------
There are four groups and they should never be mixed up:

1) TIME
   tau .............. dimensionless time = stint_lap / 30
                      A normal 30-lap stint runs from tau=0 to tau=1.
                      Dividing by 30 keeps the numbers entering the network
                      close to 1. A network trained on inputs of wildly
                      different scales converges badly.

2) THE STATE (what evolves: two numbers now, not one)
   theta ............ excess temperature of the rubber over the reference
                      thermal state, in the SAME normalised units as
                      track_temp (1 unit = 40 degrees C).
                      theta = 0 means "as cool as the reference condition".
   d ................ fraction of the tread that has been consumed.
                      0 = brand-new tire, 1 = tire gone.

   *** BOTH ARE LATENT: neither is ever measured, not on the test bench and
   *** not in a race. The model reconstructs both from lap times alone.

3) THE CONTEXT (5 variables, constant within a stint)
   q_fric ........... frictional energy per lap, normalised (~1)
                      How much energy the track puts into the rubber.
   load ............. mean mechanical load in g, normalised (~1)
                      How much apparent weight the tire carries in a corner.
   speed ............ mean speed, normalised (~1)
                      More speed = more airflow = more cooling.
   track_temp ....... track temperature, normalised from 0 to 1
   compound ......... hardness: 0 = soft, 0.5 = medium, 1 = hard

4) THE OBSERVABLE, the only thing that is actually measured:

   delta ............ pace loss in seconds relative to the stint's best lap.
                      *** THIS IS THE ONLY OBSERVABLE. ***


THE TWO EQUATIONS
-----------------

    (E1)  dtheta/dtau = A_gen * q_fric * (1 + zeta * d)  -  (h0 + h1*speed) * theta

    (E2)  dd/dtau     = k(context, theta) * (1 - d)

    with  k = kw * (load/load_ref)^m
                 * exp( Ea*(track_temp - track_temp_ref + theta)
                      + Eq*(q_fric     - q_fric_ref)
                      - Ev*(speed      - speed_ref)
                      - kappa*(compound - compound_ref) )

    and the observation operator

    (E3)  delta = gamma1 * d  +  gamma2 * d^8


E1 IN PLAIN WORDS: A HEATING ELEMENT WITH A FAN
-----------------------------------------------
This is the lumped-capacitance heat balance, the same equation as a capacitor
charging through a resistor. Two terms fight each other:

    GENERATION   A_gen * q_fric * (1 + zeta*d)
                 Friction dumps energy into the rubber. The more energy the
                 track demands (q_fric), the more heat.

    COOLING      (h0 + h1*speed) * theta
                 Heat leaves to the air, and it leaves faster the hotter the
                 tire already is (that is why theta multiplies). h0 is the
                 part that does not depend on speed; h1*speed is the airflow.

Left alone, the two balance out and the temperature settles at

    theta_steady = A_gen * q_fric * (1 + zeta*d) / (h0 + h1*speed)

reached with a time constant of 1/(h0 + h1*speed). With the default constants
that is about 5 laps, which is roughly how long a real tire takes to come up to
temperature.


THE CLIFF, AND WHY IT NEEDS THE SECOND EQUATION
-----------------------------------------------
The decisive piece of E1 is the factor `(1 + zeta*d)`.

When the tread thins out, the SAME frictional energy is deposited into LESS
mass of rubber, so the temperature rises. And by the Arrhenius term in E2,
more temperature means faster wear. Faster wear means a thinner tread, which
means more temperature again.

That is POSITIVE FEEDBACK, and it is the entire reason this file has two
equations instead of one:

    *** THE CLIFF IS NOT WRITTEN DOWN ANYWHERE. IT EMERGES from the loop  ***
    *** between E1 and E2. Nothing in the code says "fall off after lap N". ***

The previous version of this file had only E2 with theta frozen at zero. With
one equation and a linear observable, the pace curve can only bend ONE way:
it flattens out and never steepens. A real tire does not do that. It degrades
gently and then, somewhere, falls off a precipice.

E3 sharpens the same phenomenon in the observable. `gamma2 * d^8` is utterly
negligible while d is moderate (at d=0.6 it contributes 1.7% of what it will
contribute at d=1) and then takes over as d approaches 1. The exponent 8 is
not fitted; it is chosen to be large enough that the term is invisible until
the tire is nearly gone.


WHY THETA SHARES Ea WITH track_temp
-----------------------------------
Look at the exponent of E2: theta is added INSIDE the same bracket as
track_temp, multiplied by the same Ea. That is deliberate and it does two jobs.

PHYSICS: Arrhenius activation depends on the ABSOLUTE temperature of the
rubber, which is the track temperature plus whatever the tire has generated on
top of it. Splitting them into two coefficients would be claiming that a degree
of heat from the track wears the tire differently from a degree of heat from
friction. It does not.

IDENTIFIABILITY: it is also what pins down the SCALE of theta. Nobody measures
theta, so on its own the model could shrink theta by a factor and grow Ea by
the same factor with no observable change. Because Ea is also tied to
track_temp, which IS measured, that escape route is closed: the data fixes Ea,
and Ea fixes theta. This is the single most important structural decision in
the file.


WHY EACH CONTEXT TERM SUBTRACTS A REFERENCE
-------------------------------------------
Every variable enters as (variable - its_reference). That makes the exponent
vanish at reference conditions with a cold tire, so there k = kw. In other
words, kw means literally "the wear rate under normal conditions at the start
of the stint", which is a sentence you can argue about with an engineer.

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


WHAT WAS LOST BY ADDING E1
--------------------------
The exact solution.

With one equation and a constant k, separating variables gave
d(tau) = 1 - exp(-k*tau) and every result could be checked against it by hand.
With theta in the loop, k is no longer constant in time and no closed form
exists. `exact_solution_isothermal` below is what survives: it is exact ONLY
when the thermal coupling is switched off (A_gen = 0, so theta stays at 0
forever), and it exists precisely so the RK4 integrator can still be validated
against something known. See the recipe in TUNING.md.
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

# The exponent of the cliff term in the observation operator (E3).
# It is FIXED, never fitted: its only job is to be big enough that the term
# stays invisible until d is close to 1. See TUNING.md if you want to move it.
CLIFF_EXPONENT = 8

# Numerical guard on the Arrhenius exponent of E2. exp(12) is already 1.6e5,
# far past anything physical. This is NOT physics: it only stops a runaway
# theta early in training from producing inf and then NaN. If your training
# runs sit pinned against this cap, the thermal constants have run away and
# TUNING.md section 5 tells you what to do about it.
EXPONENT_CAP = 12.0

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

# The network now predicts TWO numbers instead of one: (theta, d).
N_STATES = 2

# Physically sensible range of each context variable.
# It is used for two things: generating synthetic stints, and deciding WHERE
# the network is required to satisfy the equations (see pinn.py, collocation).
CONTEXT_RANGES = {
    "q_fric": (0.40, 1.60),
    "load": (0.50, 1.50),
    "speed": (0.60, 1.40),
    "track_temp": (0.00, 1.00),
    "compound": (0.00, 1.00),
}

# Reference conditions: the centre of each range. At this point, and with a
# cold tire, the exponent of E2 is zero, so kw is exactly "the wear rate under
# normal conditions". See the explanation above on why centring matters.
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
    """The 12 constants of the system.

    Ten of them are ESTIMATED by the PINN while it trains (the inverse
    problem). Two are held FIXED, and the reason is not convenience: they are
    the two directions along which the system is degenerate. See
    FIXED_PARAMS below.
    """

    # --- E2, the wear equation -------------------------------------------
    kw: float = 0.45        # base wear rate, cold tire, reference conditions
    m: float = 1.50         # load exponent (Archard's law)
    Ea: float = 0.95        # Arrhenius: how much TOTAL heat accelerates wear
    Eq: float = 0.40        # how much frictional energy accelerates it
    Ev: float = 0.35        # how much cooling by speed slows it down
    kappa: float = 0.85     # how much a harder compound resists wear

    # --- E1, the thermal equation ----------------------------------------
    A_gen: float = 5.00     # FIXED. heat generated per unit of q_fric
    zeta: float = 0.90      # the feedback: how much a worn tire overheats
    h0: float = 4.00        # cooling that does not depend on speed
    h1: float = 2.00        # extra cooling from airflow

    # --- E3, the observation operator -------------------------------------
    gamma1: float = 1.35    # FIXED. seconds lost per unit of wear [s]
    gamma2: float = 2.20    # seconds lost to the cliff [s]


# The "true" values the synthetic generator uses. The PINN starts from
# deliberately wrong values and has to RECOVER these from noisy lap times.
# Succeeding is the proof that the method works, and it is the one check that
# is impossible with real data, because no ground truth exists there.
GROUND_TRUTH = TireParams()


# The two constants that are NOT estimated, and why each one has to be nailed
# down. Both are DEGENERATE DIRECTIONS: moving along them changes the latent
# state but leaves the observable untouched, so no amount of data can decide
# where on the line to sit, and the optimiser slides along it until it
# overflows.
#
# gamma1 -- anchors the scale of d.
#   The observable starts as delta = gamma1*d. Double d and halve gamma1 and
#   nothing observable changes.
#
# A_gen -- anchors the scale of theta.
#   Multiply theta by e and divide Ea by e and E1 still balances. That escape
#   is ALMOST closed by Ea being shared with track_temp (see the module
#   docstring), but "almost" is doing real work there: it only holds as far as
#   the data actually spreads over track_temp. Fixing A_gen closes it outright.
#
# This is not numerical caution. In the `main` branch the same issue at full
# size was the worst bug of the project: one run finished with gamma2 = 2.5e13
# and a test RMSE of 5.8e9 seconds WHILE THE TRAINING LOSS LOOKED LOW (0.117),
# because along a degenerate direction the fit really is perfect.
FIXED_PARAMS = ("A_gen", "gamma1")

# The ten the PINN estimates.
LEARNABLE_PARAMS = (
    "kw", "m", "Ea", "Eq", "Ev", "kappa",   # E2
    "zeta", "h0", "h1",                     # E1
    "gamma2",                               # E3
)

# gamma2 gets a hard box rather than a log-parametrisation. It is the most
# weakly identified constant in the system -- it only shows up once d is close
# to 1, and teams pit before that -- so it is the one most likely to wander.
# The box is a physical statement: a destroyed tire costs a few seconds a lap,
# not millions.
GAMMA2_RANGE = (0.20, 6.00)


# ---------------------------------------------------------------------------
# 4) THE TWO EQUATIONS
# ---------------------------------------------------------------------------

def _is_tensor(x) -> bool:
    """True if x is a torch tensor (rather than a numpy array or a float)."""
    return torch is not None and torch.is_tensor(x)


def _backend(*values):
    """`torch` or `numpy`, whichever matches the values being handed in.

    Every equation below is written ONCE and runs on both. That duality is
    deliberate: the SAME function defines the ground truth (numpy, when
    generating data) and the residual the network minimises (torch, when
    training). With two copies, sooner or later one gets fixed and the other
    does not, and the bench would then be validating a model that is not the
    one being trained.

    Only `exp` and `clip` are needed, and both libraries spell those the same
    way. Powers use the `**` operator, which both also understand.
    """
    return torch if any(_is_tensor(v) for v in values) else np


def wear_constant(theta, q_fric, load, speed, track_temp, compound, p):
    """k(context, theta): everything in E2 that does NOT depend on d.

    This is where the coupling lives. `theta` is added to the centred track
    temperature inside the same Arrhenius bracket, so the tire's own heat and
    the track's heat are treated as the same physical quantity -- which they
    are. See "WHY THETA SHARES Ea WITH track_temp" in the module docstring.

    With theta = 0 this reduces EXACTLY to the single-equation model this
    project started from, which is what makes the two comparable.
    """
    xp = _backend(theta, q_fric, load, speed, track_temp, compound)

    # Load is never negative physically, but during training the network can
    # be handed odd points, and a negative base raised to a real power is NaN.
    relative_load = xp.clip(load / REFERENCE.load, 1e-6, None)

    # Each variable enters as "how far it deviates from normal". That way kw
    # keeps a meaning of its own and does not step on the other coefficients.
    exponent = (
        p.Ea * (track_temp - REFERENCE.track_temp + theta)
        + p.Eq * (q_fric - REFERENCE.q_fric)
        - p.Ev * (speed - REFERENCE.speed)
        - p.kappa * (compound - REFERENCE.compound)
    )

    # Numerical guard only -- see EXPONENT_CAP.
    exponent = xp.clip(exponent, -EXPONENT_CAP, EXPONENT_CAP)

    return p.kw * relative_load ** p.m * xp.exp(exponent)


def wear_rate(d, theta, q_fric, load, speed, track_temp, compound, p):
    """The right-hand side of E2: dd/dtau.

    It is non-negative while d <= 1, so monotonic wear (the tire can only get
    worse) and the bound d <= 1 both fall out of the equation itself.

    Note what this means for the cliff: d stays MONOTONE even with the thermal
    feedback switched on. The cliff is not a reversal, it is a steepening. Any
    model that predicts the tire getting faster is wrong, and `evaluate.py`
    measures exactly that.
    """
    k = wear_constant(theta, q_fric, load, speed, track_temp, compound, p)
    return k * (1.0 - d)


def thermal_rate(theta, d, q_fric, speed, p):
    """The right-hand side of E1: dtheta/dtau.

    Generation minus cooling:

        generation = A_gen * q_fric * (1 + zeta*d)
                     Friction heats the rubber. `(1 + zeta*d)` is THE CLIFF
                     TERM: as the tread thins, the same energy goes into less
                     mass, so the same work produces more temperature.

        cooling    = (h0 + h1*speed) * theta
                     Newton's law of cooling. Proportional to how hot the tire
                     already is, which is what makes the temperature settle
                     instead of running away.

    Note that `d` enters here and `theta` enters E2: that mutual dependence is
    what makes the pair COUPLED, and it is the reason the exact solution is
    gone.
    """
    generation = p.A_gen * q_fric * (1.0 + p.zeta * d)
    cooling = (p.h0 + p.h1 * speed) * theta
    return generation - cooling


def steady_temperature(d, q_fric, speed, p):
    """The temperature E1 settles at, if d were held still at this value.

    Not used in training: it is a reading instrument. It answers "how hot does
    this tire end up?" without integrating anything, which is the quickest way
    to tell whether a set of thermal constants is physically sane. TUNING.md
    leans on it heavily.
    """
    return p.A_gen * q_fric * (1.0 + p.zeta * d) / (p.h0 + p.h1 * speed)


def pace_loss(d, p: TireParams):
    """E3. Turns the latent state d into the only observable: seconds.

    This is the "observation operator". The network predicts theta and d, which
    nobody has ever measured; this function translates one of them into
    something that can be compared against the data. That is how both latent
    states get reconstructed without either ever appearing in the loss.

        gamma1 * d        the gentle, everyday part of degradation
        gamma2 * d^8      the cliff: nothing at all, and then everything

    CLIFF_EXPONENT is even, so a slightly negative d (which the network can
    produce early in training, unless the hard initial condition is on) gives a
    small positive contribution rather than a NaN.
    """
    return p.gamma1 * d + p.gamma2 * d ** CLIFF_EXPONENT


# ---------------------------------------------------------------------------
# 5) SOLVING THE SYSTEM
# ---------------------------------------------------------------------------

def exact_solution_isothermal(tau, context: Context, p: TireParams) -> np.ndarray:
    """The hand-written solution of E2 ALONE, with the thermal coupling off.

    *** THIS IS ONLY THE TRUTH WHEN A_gen = 0. ***

    With A_gen = 0 the generation term of E1 vanishes, theta decays to 0 and
    stays there, k becomes constant, and separating variables in
    dd/dtau = k*(1-d) with d(0)=0 gives directly:

        d(tau) = 1 - exp(-k*tau)

    That is the only exact answer left in this file, and it exists for one
    reason: `integrate_stint` is now the only way to solve the real system, so
    it had better be validated against something known. Setting A_gen = 0 turns
    the coupled system back into the one that does have a closed form, and the
    integrator can be checked there. The recipe is in TUNING.md.

    It raises rather than returning a wrong curve quietly, because a plausible
    wrong reference is worse than no reference.
    """
    if p.A_gen != 0.0:
        raise ValueError(
            "exact_solution_isothermal is only exact when the thermal coupling "
            f"is off, i.e. A_gen = 0, but A_gen = {p.A_gen}. For the coupled "
            "system use integrate_stint()."
        )

    k = wear_constant(
        0.0, context.q_fric, context.load, context.speed,
        context.track_temp, context.compound, p,
    )
    return 1.0 - np.exp(-k * np.asarray(tau, dtype=float))


def integrate_stint(
    n_laps: int,
    context: Context,
    p: TireParams,
    steps_per_lap: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve E1 and E2 together, step by step, with 4th-order Runge-Kutta.

    This is now the ONLY way to get the truth: with the two equations coupled
    there is no closed form to fall back on. That is also why the integrator
    was written and validated back when the model had one equation and an exact
    answer to check against -- validating it then was cheap, and it is the same
    code path today.

    Runge-Kutta 4 rather than Euler because Euler accumulates a systematic
    bias, and the inverse problem would read that bias as if it were physics.
    Both states are advanced TOGETHER within each stage: E1 needs d and E2
    needs theta, so stepping one and then the other would silently make the
    scheme first-order.

    Returns (laps, theta, d) with laps numbered 1..n_laps.
    """
    dt = 1.0 / (LAP_REF * steps_per_lap)

    def rates(theta: float, d: float) -> tuple[float, float]:
        """Both slopes at a given state, with this stint's context."""
        return (
            thermal_rate(theta, d, context.q_fric, context.speed, p),
            wear_rate(
                d, theta, context.q_fric, context.load, context.speed,
                context.track_temp, context.compound, p,
            ),
        )

    theta = 0.0        # a tire leaving the pits sits at the reference state
    d = 0.0            # brand-new tire
    theta_history: list[float] = []
    d_history: list[float] = []

    for _ in range(n_laps):
        for _ in range(steps_per_lap):
            # Runge-Kutta 4: instead of trusting the slope at a single point,
            # it averages four slopes taken across the step.
            a1, b1 = rates(theta, d)
            a2, b2 = rates(theta + 0.5 * dt * a1, d + 0.5 * dt * b1)
            a3, b3 = rates(theta + 0.5 * dt * a2, d + 0.5 * dt * b2)
            a4, b4 = rates(theta + dt * a3, d + dt * b3)

            theta = theta + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
            d = d + (dt / 6.0) * (b1 + 2 * b2 + 2 * b3 + b4)

        theta_history.append(theta)
        d_history.append(d)

    laps = np.arange(1, n_laps + 1, dtype=float)
    return laps, np.asarray(theta_history), np.asarray(d_history)
