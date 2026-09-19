"""
THE NEURAL NETWORK WITH PHYSICS INSIDE (PINN)
=============================================

WHAT A NEURAL NETWORK IS, IN PROGRAMMING TERMS
----------------------------------------------
A function. Nothing more.

An ordinary function you write yourself:  def f(x): return 3*x + 2
A neural network has the same shape, but with thousands of constants instead of
a 3 and a 2, and those constants you do NOT write: an algorithm searches for
them.

    theta, d = net(tau, context ; W)        W = the 13,058 constants

"Training" is the optimisation problem of finding the W that minimises a loss
function. It is gradient descent: work out which way each constant has to move
for the loss to go down, and take a step that way. Several thousand times.

Layers alternate with `tanh`. Without that nonlinearity, stacking ten linear
layers would give another linear function and the network could not represent a
curve at all.


WHAT "PHYSICS-INFORMED" ADDS
----------------------------
An ordinary network is trained like this:

    loss = (what it predicts - what was measured)^2

and it needs to know the correct answer at every point where it learns.

A PINN adds a second term:

    loss = (prediction - measurement)^2  +  (residual of the equations)^2

where now there are TWO residuals, one per equation:

    r_theta = dtheta/dtau - thermal_rate(theta, d, context)
    r_d     = dd/dtau     - wear_rate(d, theta, context)

If the network satisfied the system exactly, both would be zero everywhere. So
putting them in the loss is, literally, asking the network to obey the physics.

THE TRICK IS IN THE DERIVATIVES. They are not approximated by finite
differences: they are computed EXACTLY, with the same automatic differentiation
PyTorch already uses to train. The difference is that here the output is
differentiated with respect to the INPUT rather than with respect to the
weights.

And from that comes the property that makes all of this useful:

    *** evaluating a residual only needs a POINT of the domain.             ***
    *** It does NOT need to know the correct answer at that point.          ***

So the physics can be enforced where there is not a single data point: in
combinations of conditions that never occurred, and on laps beyond the end of
every stint in the dataset. Which are exactly the laps a strategist needs
somebody to predict. It matters twice over here, because theta is never
measured ANYWHERE -- the only thing holding the temperature curve in place is
the equation.


THE TERMS OF THE LOSS
---------------------
    physics_loss   residual of the WEAR equation, on unlabelled points
    thermal_loss   residual of the THERMAL equation, on the same points
    data_loss      the fit to the measured pace loss, only where data exists
    ic_loss        the initial conditions -- ONLY when they are imposed softly

Each has its own weight, and those weights are the first thing to reach for
when a run misbehaves. TUNING.md explains what each one does to the outcome.


THE INITIAL CONDITIONS, HARD OR SOFT
------------------------------------
Two states now start at zero: a tire leaving the pits is unworn (d=0) and sits
at the reference thermal state (theta=0).

SOFT (`ic="soft"`) puts that in the loss as one more term. It is the textbook
formulation, it is satisfied approximately, never exactly, and it competes for
gradient with everything else. With two states that competition roughly
doubles, and it is the most common reason a coupled PINN refuses to converge.

HARD (`ic="hard"`, the default) imposes them by TRANSFORMING the output, so
they hold exactly at every iteration and cost nothing:

    theta(tau) = tau * N0                    -> theta(0) = 0, exactly
    d(tau)     = 1 - exp(-tau * softplus(N1)) -> d(0) = 0, exactly
                                             -> and 0 <= d < 1, always

Because tau multiplies the network's output, at tau=0 the term vanishes no
matter what the network says. The second transform is worth reading twice: it
does not merely pin d(0), it makes the saturation bound structural. `softplus`
is non-negative, so the exponent is non-positive, so d can never leave [0,1).
Three failure modes deleted at the cost of one line, and nothing is lost --
every curve from 0 that stays under 1 can still be written this way.


THE INVERSE PROBLEM
-------------------
Ten physical constants are unknown and are estimated AT THE SAME TIME as the
network weights. The optimiser treats them like any other parameter.

They are stored in three different ways, and the choice encodes what we know:

  LOGARITHM (kw, m, Ea, Eq, Ev, zeta, h0, h1)
    The physical value is exp(raw), which is positive no matter what. This is
    not numerical caution: there is no such thing as a negative wear
    coefficient or a negative cooling rate, and asserting that through the
    parametrisation is stronger than trusting the optimiser to respect it.
    For h0 it is load-bearing in a second way: h0 <= 0 would make the thermal
    equation unstable and theta would run away to infinity.

  FREE (kappa)
    The one constant in the system whose sign is NOT fixed by physics. It says
    how much a harder compound resists wear, and there is analysis arguing that
    2026 inverted that ordering. Log-parametrising it would make the model
    incapable of expressing that. This way, the sign is decided by the data.

  BOUNDED BOX (gamma2)
    lo + (hi-lo)*sigmoid(raw), so it cannot leave GAMMA2_RANGE. gamma2 is the
    most weakly identified constant in the system: it only reaches the
    observable once d is close to 1, and teams pit before that. The box says
    something we actually know -- a destroyed tire costs a few seconds a lap,
    not millions. See physics.py, FIXED_PARAMS, for what happened without it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from physics import (
    CONTEXT_RANGES,
    GAMMA2_RANGE,
    GROUND_TRUTH,
    LAP_REF,
    LEARNABLE_PARAMS,
    N_CONTEXT,
    N_INPUTS,
    N_STATES,
    STRATEGY_HORIZON,
    Context,
    TireParams,
    pace_loss,
    thermal_rate,
    wear_rate,
)

# Constants that have to be positive: stored as log(value).
_POSITIVE = ("kw", "m", "Ea", "Eq", "Ev", "zeta", "h0", "h1")

# Constants confined to a box: stored as the inverse sigmoid of their position
# in the box. Only gamma2, for the reason in the module docstring.
_BOUNDED = {"gamma2": GAMMA2_RANGE}

# Starting values, deliberately away from the truth: the PINN has to recover
# GROUND_TRUTH from here. If it started at the answer, recovering it would
# prove nothing.
#
# These are also a TUNING KNOB. A coupled system has a much rougher loss
# surface than a single equation, and where you start decides which basin you
# land in. TUNING.md section 6 covers what to do when a constant never moves.
INITIAL_VALUES = {
    "kw": 0.30,
    "m": 1.00,
    "Ea": 0.50,
    "Eq": 0.20,
    "Ev": 0.20,
    "kappa": 0.30,
    "zeta": 0.50,
    "h0": 3.00,
    "h1": 1.50,
    "gamma2": 1.00,
}


# ---------------------------------------------------------------------------
# 1) THE NETWORK
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Multilayer perceptron: 6 inputs -> several hidden layers -> 2 outputs.

    Inputs (6):   tau, q_fric, load, speed, track_temp, compound
    Outputs (2):  the raw numbers behind (theta, d)

    By default 4 layers of 64 neurons = 13,058 constants to fit.

    ONE network with two outputs, not two networks. The two states are coupled
    -- theta drives the wear and the wear drives theta -- so the features that
    explain one are largely the features that explain the other, and sharing
    the hidden layers lets the model learn them once. Two separate networks
    would have to discover the same structure twice from the same data.

    Why a perceptron and not something fancier: the functions to be learned are
    smooth and have six inputs. There is no spatial structure to justify
    convolutions and no sequence to justify recurrence. What does matter is
    that the network be cheaply and stably DIFFERENTIABLE, because it has to be
    differentiated at every collocation point, twice, once per equation.

    Why tanh and not ReLU: the derivative of a ReLU network is piecewise
    constant, so the network would predict a stepped, discontinuous
    dtheta/dtau, impossible to match against a smooth right-hand side. tanh is
    infinitely differentiable.
    """

    def __init__(self, width: int = 64, layers: int = 4):
        """Build the stack: Linear, Tanh, Linear, Tanh, ..., Linear.

        The last layer has NO activation, so the outputs can take any value.
        Whatever shaping they need happens in the output transform, where it
        can be justified per state instead of applied blindly to both.
        """
        super().__init__()

        # [6, 64, 64, 64, 64, 2]
        dims = [N_INPUTS] + [width] * layers + [N_STATES]

        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                modules.append(nn.Tanh())

        self.layers = nn.Sequential(*modules)
        self._glorot_init()

    def _glorot_init(self) -> None:
        """Glorot (Xavier) initialisation, the one that suits tanh.

        It keeps the variance of the signal steady across layers. Without it,
        with four layers, tanh saturates from the very first iteration: every
        output pins to -1 or +1, the derivative goes to zero and the network
        stops learning.
        """
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tau: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Run one batch through the network, returning the raw (N, 2).

        tau and the context arrive separately rather than pre-joined so that
        tau can stay a leaf of the autograd graph and be differentiated
        against. They are concatenated here into the (N, 6) the layers expect.
        """
        return self.layers(torch.cat([tau, context], dim=1))


# ---------------------------------------------------------------------------
# 2) THE PINN
# ---------------------------------------------------------------------------

class PINN:
    """The network, plus the ten physical constants estimated alongside it."""

    def __init__(
        self,
        width: int = 64,
        layers: int = 4,
        gamma1: float = GROUND_TRUTH.gamma1,
        A_gen: float = GROUND_TRUTH.A_gen,
        ic: str = "hard",
        tau_max: float = STRATEGY_HORIZON / LAP_REF,
        seed: int = 0,
    ):
        """Create the network and the ten constants to be estimated.

        gamma1 and A_gen arrive as arguments and stay fixed: they are what
        anchor the scales of d and of theta respectively. See physics.py,
        FIXED_PARAMS, for why leaving either free breaks the inverse problem.

        `ic` selects how the initial conditions are imposed, "hard" or "soft";
        the module docstring compares them. tau_max says how far out the
        equations will be enforced, and defaults to the full strategy horizon
        rather than to the longest stint in the data.
        """
        if ic not in ("hard", "soft"):
            raise ValueError(f"ic must be 'hard' or 'soft', not {ic!r}")

        torch.manual_seed(seed)

        self.net = MLP(width=width, layers=layers)
        self.gamma1 = gamma1          # fixed: anchors the scale of d
        self.A_gen = A_gen            # fixed: anchors the scale of theta
        self.ic = ic
        self.tau_max = tau_max        # how far the equations are enforced
        self.history: list[dict[str, float]] = []

        # The ten constants to estimate, in their unconstrained "raw" form.
        self._raw: dict[str, nn.Parameter] = {
            name: nn.Parameter(torch.tensor(self._to_raw(name, INITIAL_VALUES[name])))
            for name in LEARNABLE_PARAMS
        }

    # -- the physical constants -------------------------------------------

    @staticmethod
    def _to_raw(name: str, value: float) -> float:
        """Physical value -> the unconstrained number the optimiser moves."""
        if name in _POSITIVE:
            return float(np.log(value))
        if name in _BOUNDED:
            low, high = _BOUNDED[name]
            fraction = (value - low) / (high - low)
            fraction = float(np.clip(fraction, 1e-4, 1 - 1e-4))
            return float(np.log(fraction / (1.0 - fraction)))      # logit
        return float(value)

    def _value(self, name: str) -> torch.Tensor:
        """The unconstrained number the optimiser moves -> physical value."""
        raw = self._raw[name]
        if name in _POSITIVE:
            return torch.exp(raw)
        if name in _BOUNDED:
            low, high = _BOUNDED[name]
            return low + (high - low) * torch.sigmoid(raw)
        return raw

    def params_tensor(self) -> TireParams:
        """The constants as tensors, so they enter the autograd graph."""
        values = {n: self._value(n) for n in LEARNABLE_PARAMS}
        return TireParams(gamma1=self.gamma1, A_gen=self.A_gen, **values)

    def learned_params(self) -> TireParams:
        """The same constants as plain numbers, for printing."""
        values = {n: float(self._value(n).detach()) for n in LEARNABLE_PARAMS}
        return TireParams(gamma1=self.gamma1, A_gen=self.A_gen, **values)

    def n_weights(self) -> int:
        """How many weights the network has, not counting the 10 constants."""
        return sum(p.numel() for p in self.net.parameters())

    # -- turning the network's raw output into physical states -------------

    def state(self, tau: torch.Tensor, context: torch.Tensor):
        """The two physical states (theta, d) at these points.

        This is where the initial conditions are enforced when `ic="hard"`, by
        transforming the network's output rather than by asking the loss
        nicely. See the module docstring for why each transform has the shape
        it has.
        """
        raw = self.net(tau, context)
        raw_theta = raw[:, 0:1]
        raw_d = raw[:, 1:2]

        if self.ic == "soft":
            return raw_theta, raw_d

        theta = tau * raw_theta
        # softplus >= 0  =>  the exponent is <= 0  =>  0 <= d < 1, structurally.
        d = 1.0 - torch.exp(-tau * nn.functional.softplus(raw_d))
        return theta, d

    # -- the residuals, which are the heart of the method -------------------

    def residuals(self, tau: torch.Tensor, context: torch.Tensor):
        """(r_theta, r_d): how badly each equation is violated at these points.

        Both are zero everywhere if and only if the network solves the system.
        """
        # 1) mark tau as something we want to differentiate with respect to
        tau = tau.requires_grad_(True)

        # 2) run it through the network and the output transform
        theta, d = self.state(tau, context)

        # 3) differentiate each OUTPUT with respect to the INPUT, exactly.
        #
        #    Two separate calls, on purpose. One call with grad_outputs=ones
        #    over both columns would return d(theta + d)/dtau -- the SUM of the
        #    derivatives -- which is not what either equation needs.
        dtheta_dtau = self._d_dtau(theta, tau)
        dd_dtau = self._d_dtau(d, tau)

        # 4) compare against what the physics says
        q_fric, load, speed, track_temp, compound = self._columns(context)
        p = self.params_tensor()

        thermal = thermal_rate(theta, d, q_fric, speed, p)
        wear = wear_rate(d, theta, q_fric, load, speed, track_temp, compound, p)

        return dtheta_dtau - thermal, dd_dtau - wear

    @staticmethod
    def _d_dtau(output: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Exact derivative of one output column with respect to tau.

        create_graph=True is essential: it lets the residual itself be
        differentiated again with respect to the weights, which is what the
        optimiser needs. Without it the physics terms would contribute no
        gradient at all and the equations would never be satisfied.
        """
        return torch.autograd.grad(
            outputs=output,
            inputs=tau,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
        )[0]

    @staticmethod
    def _columns(context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split the context matrix into its 5 named columns."""
        return tuple(context[:, i : i + 1] for i in range(N_CONTEXT))

    def collocation_points(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The points where the equations are enforced.

        Drawn uniformly:
          - tau out to the FULL 45-lap horizon, beyond the longest stint in the
            dataset;
          - each context variable within its physical range.

        They carry no labels whatsoever, and that is the entire reason these
        terms can cover ground the data never does.
        """
        tau = torch.rand(n, 1) * self.tau_max

        columns = []
        for name in CONTEXT_RANGES:
            low, high = CONTEXT_RANGES[name]
            columns.append(torch.rand(n, 1) * (high - low) + low)

        return tau, torch.cat(columns, dim=1)

    # -- the terms of the loss ---------------------------------------------
    #
    # One method each, so the training loop below reads like the formula in
    # this module's docstring instead of like twenty lines of tensor plumbing.

    def _physics_losses(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Terms 1 and 2. How badly each equation is violated, unlabelled.

        Both come from the SAME collocation points in one call: drawing two
        independent sets would double the cost for no benefit, and the two
        residuals are coupled anyway -- they have to be measured at the same
        place to be traded off against each other.
        """
        tau, context = self.collocation_points(n)
        r_theta, r_d = self.residuals(tau, context)
        return (r_d ** 2).mean(), (r_theta ** 2).mean()

    def _data_loss(self, tau_obs, context_obs, delta_obs) -> torch.Tensor:
        """Term 3. How far the predicted pace is from the measured pace.

        The network predicts theta and d, neither of which anybody measured.
        `pace_loss` turns d into seconds, which is what the data is in. Note
        that it uses the LEARNED gamma2, so the cliff term is being fitted here
        too -- it is the only place gamma2 ever reaches the loss.
        """
        _, d = self.state(tau_obs, context_obs)
        predicted_delta = pace_loss(d, self.params_tensor())
        return ((predicted_delta - delta_obs) ** 2).mean()

    def _initial_condition_loss(self, n: int = 64) -> torch.Tensor:
        """Term 4. How far (theta, d) at tau=0 are from zero.

        A new tire has worn nothing and sits at the reference thermal state.
        With `ic="hard"` this is satisfied by construction and the term is
        exactly zero, so it is skipped entirely rather than computed and
        wasted.

        The contexts are drawn at random so the condition is imposed across
        the whole range of conditions, not just at one of them.
        """
        _, context = self.collocation_points(n)
        tau_zero = torch.zeros(context.shape[0], 1)
        theta, d = self.state(tau_zero, context)
        return (theta ** 2).mean() + (d ** 2).mean()

    # -- training -----------------------------------------------------------

    def train(
        self,
        inputs: np.ndarray,
        delta: np.ndarray,
        iterations: int = 8000,
        n_collocation: int = 2000,
        lr: float = 3e-3,
        w_physics: float = 1.0,
        w_thermal: float = 1.0,
        w_data: float = 10.0,
        w_ic: float = 10.0,
        lbfgs_iterations: int = 0,
        record_every: int = 25,
        print_every: int = 500,
    ) -> None:
        """Adam first, then optionally L-BFGS.

        The weights of the terms are not importances: they are SCALES. The
        residuals are dimensionless and small; the data term is in seconds.
        Without the factors, the term that actually anchors the model to what
        was measured would sit below the numerical noise of the others. They
        are the main thing to turn when a run misbehaves -- see TUNING.md.

        `w_thermal` is separate from `w_physics` because the two residuals do
        not live on the same scale. dtheta/dtau is of order A_gen (a few units)
        while dd/dtau is of order kw (a fraction), so their squares differ by a
        couple of orders of magnitude and the thermal term will dominate unless
        it is scaled down.
        """
        # Split the input matrix into tau and context BEFORE converting to
        # tensors, so that tau is a leaf of the graph and can be differentiated.
        tau_obs = torch.tensor(inputs[:, 0:1], dtype=torch.float32)
        context_obs = torch.tensor(inputs[:, 1:], dtype=torch.float32)
        delta_obs = torch.tensor(delta.reshape(-1, 1), dtype=torch.float32)

        weights = {
            "physics": w_physics, "thermal": w_thermal,
            "data": w_data, "ic": w_ic,
        }
        observations = (tau_obs, context_obs, delta_obs)

        self._run_adam(observations, weights, iterations, n_collocation, lr,
                       record_every, print_every)

        if lbfgs_iterations > 0:
            self._run_lbfgs(observations, weights, lbfgs_iterations,
                            n_collocation, record_every)

    def _total_loss(self, observations, weights, n_collocation):
        """Assemble the weighted sum, and return the pieces for logging."""
        physics_loss, thermal_loss = self._physics_losses(n_collocation)
        data_loss = self._data_loss(*observations)
        ic_loss = (
            torch.zeros(()) if self.ic == "hard"
            else self._initial_condition_loss()
        )

        total = (
            weights["physics"] * physics_loss
            + weights["thermal"] * thermal_loss
            + weights["data"] * data_loss
            + weights["ic"] * ic_loss
        )
        return total, physics_loss, thermal_loss, data_loss, ic_loss

    def _run_adam(self, observations, weights, iterations, n_collocation, lr,
                  record_every, print_every) -> None:
        """Phase 1. Adam, which explores: cheap steps, fresh points each time.

        The collocation points are redrawn EVERY iteration. That is what stops
        the network from satisfying the equations at a fixed finite set of
        places and nowhere in between.
        """
        to_optimise = list(self.net.parameters()) + list(self._raw.values())
        optimiser = torch.optim.Adam(to_optimise, lr=lr)

        print(f"      phase 1: {iterations} Adam iterations at lr={lr}")
        for iteration in range(1, iterations + 1):
            optimiser.zero_grad()
            pieces = self._total_loss(observations, weights, n_collocation)
            pieces[0].backward()
            optimiser.step()

            if iteration % record_every == 0 or iteration == 1:
                self._record(iteration, *pieces)
                if iteration % print_every == 0 or iteration == 1:
                    print("  " + self._log_line(self.history[-1]))

    def _run_lbfgs(self, observations, weights, iterations, n_collocation,
                   record_every) -> None:
        """Phase 2. L-BFGS, which refines: expensive steps that use curvature.

        Why bother. The thermal constants (zeta, h0, h1) are the worst
        conditioned in the problem: they only reach the observable through two
        layers of composition -- they move theta, theta moves the wear rate,
        the wear rate moves d, and only then does anything happen in seconds.
        Adam, which scales each parameter by its own gradient history, tends to
        leave them sitting wherever they started. A quasi-Newton method uses
        the curvature of the loss and can move them.

        THE COLLOCATION POINTS ARE FROZEN HERE, drawn once before the loop.
        L-BFGS builds an internal model of the loss surface from successive
        evaluations, and resampling between them would make every evaluation a
        different function -- the approximation it builds would be of noise.
        Adam tolerates that; L-BFGS does not.
        """
        offset = self.history[-1]["iteration"] if self.history else 0
        print(f"      phase 2: up to {iterations} L-BFGS iterations")

        frozen_tau, frozen_context = self.collocation_points(n_collocation)
        to_optimise = list(self.net.parameters()) + list(self._raw.values())
        optimiser = torch.optim.LBFGS(
            to_optimise,
            max_iter=iterations,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        step = {"n": 0}

        def closure():
            """Re-evaluate the loss. L-BFGS calls this several times per step.

            Same terms as `_total_loss`, but over the FROZEN points, which is
            why it is written out here instead of reusing that method.
            """
            optimiser.zero_grad()
            r_theta, r_d = self.residuals(frozen_tau, frozen_context)
            physics_loss = (r_d ** 2).mean()
            thermal_loss = (r_theta ** 2).mean()
            data_loss = self._data_loss(*observations)
            ic_loss = (
                torch.zeros(()) if self.ic == "hard"
                else self._initial_condition_loss()
            )
            total = (
                weights["physics"] * physics_loss
                + weights["thermal"] * thermal_loss
                + weights["data"] * data_loss
                + weights["ic"] * ic_loss
            )
            total.backward()

            step["n"] += 1
            if step["n"] % record_every == 0:
                self._record(offset + step["n"], total, physics_loss,
                             thermal_loss, data_loss, ic_loss)
            return total

        optimiser.step(closure)
        if self.history:
            print("  " + self._log_line(self.history[-1]))

    def _record(self, iteration, loss, physics, thermal, data, ic) -> None:
        """Store this iteration's state so it can be plotted later.

        .item() is used rather than float(): the tensors are still attached to
        the autograd graph here, and float() on one of them raises a warning.
        """
        record = {
            "iteration": iteration,
            "total": loss.item(),
            "physics": physics.item(),
            "thermal": thermal.item(),
            "data": data.item(),
            "ic": ic.item(),
        }
        learned = self.learned_params()
        for name in LEARNABLE_PARAMS:
            record[name] = float(getattr(learned, name))
        self.history.append(record)

    @staticmethod
    def _log_line(r: dict[str, float]) -> str:
        """Format one progress line.

        It reads from the recorded dict rather than from the live tensors, so
        printing never touches the autograd graph. The four constants shown are
        the ones worth watching live: kw and Ea drive the wear, zeta and h0 are
        the pair most likely to sit still.
        """
        return (
            f"{r['iteration']:6d}  total {r['total']:9.5f}  "
            f"wear {r['physics']:8.5f}  heat {r['thermal']:8.5f}  "
            f"data {r['data']:8.5f}  |  kw {r['kw']:5.3f}  Ea {r['Ea']:5.3f}  "
            f"zeta {r['zeta']:5.3f}  h0 {r['h0']:5.3f}"
        )

    # -- prediction ---------------------------------------------------------

    def predict_state(self, context: Context, laps: np.ndarray):
        """Both latent states, lap by lap. Neither was ever measured.

        This is the window into what the model actually believes is happening
        inside the tire, and it is the first thing to look at when a run
        produces plausible seconds for implausible reasons. run.py draws it.
        """
        laps = np.asarray(laps, dtype=float).ravel()
        tau = torch.tensor((laps / LAP_REF).reshape(-1, 1), dtype=torch.float32)

        repeated_context = np.tile(context.vector().reshape(1, -1), (laps.size, 1))
        context_t = torch.tensor(repeated_context, dtype=torch.float32)

        with torch.no_grad():
            theta, d = self.state(tau, context_t)

        return theta.numpy().ravel(), d.numpy().ravel()

    def predict_stint(self, context: Context, laps: np.ndarray) -> np.ndarray:
        """Predicted pace loss, lap by lap.

        Same signature as the baseline's, so the evaluator measures both models
        with exactly the same yardstick.
        """
        _, d = self.predict_state(context, laps)
        return pace_loss(d, self.learned_params())

    def predict_wear(self, context: Context, laps: np.ndarray) -> np.ndarray:
        """Just the wear state d, for callers that do not want the pair."""
        return self.predict_state(context, laps)[1]
