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

    d = net(tau, context ; W)        W = the 12,993 constants

"Training" is the optimisation problem of finding the W that minimises a loss
function. It is gradient descent: work out which way each constant has to move
for the loss to go down, and take a step that way. Eight thousand times.

Layers alternate with `tanh`. Without that nonlinearity, stacking ten linear
layers would give another linear function and the network could not represent a
curve at all.


WHAT "PHYSICS-INFORMED" ADDS
----------------------------
An ordinary network is trained like this:

    loss = (what it predicts - what was measured)^2

and it needs to know the correct answer at every point where it learns.

A PINN adds a second term:

    loss = (prediction - measurement)^2  +  (residual of the equation)^2

where the residual is:

    r = dd/dtau - wear_rate(d, context)

If the network satisfied the equation exactly, r would be zero everywhere. So
putting r^2 in the loss is, literally, asking the network to obey the physics.

THE TRICK IS IN dd/dtau. That derivative is not approximated by finite
differences: it is computed EXACTLY, with the same automatic differentiation
PyTorch already uses to train. The difference is that here the output is
differentiated with respect to the INPUT rather than with respect to the
weights.

And from that comes the property that makes all of this useful:

    *** evaluating the residual only needs a POINT of the domain.           ***
    *** It does NOT need to know the correct answer at that point.          ***

So the physics can be enforced where there is not a single data point: in
combinations of conditions that never occurred, and on laps beyond the end of
every stint in the dataset. Which are exactly the laps a strategist needs
somebody to predict.


THE THREE TERMS OF THE LOSS
---------------------------
    physics_loss   the residual of the equation, on 2,000 unlabelled points
    data_loss      the fit to the measured pace loss, only where data exists
    ic_loss        the initial condition d(0) = 0

The third one being a loss term is a known weakness, kept deliberately because
it is the textbook formulation and because watching it fail is instructive: it
is satisfied approximately, never exactly, and it competes for gradient with
the other two. In `main` it is replaced by an output transform that makes it
exact and free. See ROADMAP.md, step 4.


THE INVERSE PROBLEM
-------------------
Six physical constants (kw, m, Ea, Eq, Ev, kappa) are unknown, and they are
estimated AT THE SAME TIME as the network weights. The optimiser treats them
like any other parameter.

The five that have to be positive are stored as their LOGARITHM. That way the
physical value is exp(raw), which is positive no matter what. This is not
numerical caution: there is no such thing as a negative wear coefficient, and
asserting that through the parametrisation is stronger than trusting the
optimiser to respect it.

kappa is the exception and its sign is left free, because it is the one
constant in the system whose sign is NOT fixed by physics: it says how much a
harder compound resists wear, and there is analysis arguing that 2026 inverted
that ordering. Log-parametrising it would make the model incapable of
expressing that. This way, the sign is decided by the data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from physics import (
    CONTEXT_RANGES,
    GROUND_TRUTH,
    LAP_REF,
    LEARNABLE_PARAMS,
    N_CONTEXT,
    N_INPUTS,
    STRATEGY_HORIZON,
    Context,
    TireParams,
    wear_rate,
)

# The five constants that have to be positive are stored as log(value).
_POSITIVE = ("kw", "m", "Ea", "Eq", "Ev")

# Starting values, deliberately wrong: the PINN has to recover the ones in
# GROUND_TRUTH from here. If it started at the answer, recovering it would
# prove nothing.
INITIAL_VALUES = {
    "kw": 0.30,      # the true value is 0.55
    "m": 1.00,       # the true value is 1.50
    "Ea": 0.50,      # the true value is 0.95
    "Eq": 0.20,      # the true value is 0.40
    "Ev": 0.20,      # the true value is 0.35
    "kappa": 0.30,   # the true value is 0.85
}


# ---------------------------------------------------------------------------
# 1) THE NETWORK
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Multilayer perceptron: 6 inputs -> several hidden layers -> 1 output.

    Inputs (6):  tau, q_fric, load, speed, track_temp, compound
    Output (1):  d, the fraction of tread consumed

    By default 4 layers of 64 neurons = 12,993 constants to fit.

    Why a perceptron and not something fancier: the function to be learned is
    smooth and has six inputs. There is no spatial structure to justify
    convolutions and no sequence to justify recurrence. What does matter is
    that the network be cheaply and stably DIFFERENTIABLE, because it has to be
    differentiated at every collocation point.

    Why tanh and not ReLU: the derivative of a ReLU network is piecewise
    constant, so the network would predict a stepped, discontinuous dd/dtau,
    impossible to match against a smooth right-hand side. tanh is infinitely
    differentiable.
    """

    def __init__(self, width: int = 64, layers: int = 4):
        """Build the stack: Linear, Tanh, Linear, Tanh, ..., Linear.

        The last layer has NO activation, so the output can take any value.
        A tanh there would clamp d to (-1, 1), which happens to be almost
        right and would hide errors for the wrong reason.
        """
        super().__init__()

        # [6, 64, 64, 64, 64, 1]
        dims = [N_INPUTS] + [width] * layers + [1]

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
        """Run one batch through the network.

        tau and the context arrive separately rather than pre-joined so that
        tau can stay a leaf of the autograd graph and be differentiated
        against. They are concatenated here into the (N, 6) the layers expect.
        """
        return self.layers(torch.cat([tau, context], dim=1))


# ---------------------------------------------------------------------------
# 2) THE PINN
# ---------------------------------------------------------------------------

class PINN:
    """The network, plus the six physical constants estimated alongside it."""

    def __init__(
        self,
        width: int = 64,
        layers: int = 4,
        gamma1: float = GROUND_TRUTH.gamma1,
        tau_max: float = STRATEGY_HORIZON / LAP_REF,
        seed: int = 0,
    ):
        """Create the network and the six constants to be estimated.

        gamma1 arrives as an argument and stays fixed: it is what anchors the
        scale of d. tau_max says how far out the equation will be enforced,
        and defaults to the full strategy horizon rather than to the longest
        stint in the data.
        """
        torch.manual_seed(seed)

        self.net = MLP(width=width, layers=layers)
        self.gamma1 = gamma1          # fixed: anchors the scale of d (physics.py)
        self.tau_max = tau_max        # how far the equation is enforced
        self.history: list[dict[str, float]] = []

        # The six constants to estimate, in their unconstrained "raw" form.
        self._raw: dict[str, nn.Parameter] = {}
        for name in LEARNABLE_PARAMS:
            initial = INITIAL_VALUES[name]
            raw = np.log(initial) if name in _POSITIVE else initial
            self._raw[name] = nn.Parameter(torch.tensor(float(raw)))

    # -- the physical constants -------------------------------------------

    def _value(self, name: str) -> torch.Tensor:
        """From the raw value the network optimises to the physical value."""
        raw = self._raw[name]
        return torch.exp(raw) if name in _POSITIVE else raw

    def params_tensor(self) -> TireParams:
        """The constants as tensors, so they enter the autograd graph."""
        values = {n: self._value(n) for n in LEARNABLE_PARAMS}
        return TireParams(gamma1=self.gamma1, **values)

    def learned_params(self) -> TireParams:
        """The same constants as plain numbers, for printing."""
        values = {n: float(self._value(n).detach()) for n in LEARNABLE_PARAMS}
        return TireParams(gamma1=self.gamma1, **values)

    def n_weights(self) -> int:
        """How many weights the network has, not counting the 6 constants."""
        return sum(p.numel() for p in self.net.parameters())

    # -- the residual, which is the heart of the method --------------------

    def residual(self, tau: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """r = dd/dtau - wear_rate(d, context).

        It is zero everywhere if and only if the network satisfies the equation.
        """
        # 1) mark tau as something we want to differentiate with respect to
        tau = tau.requires_grad_(True)

        # 2) run it through the network
        d = self.net(tau, context)

        # 3) differentiate the OUTPUT with respect to the INPUT, exactly
        #
        #    create_graph=True is essential: it lets the residual itself be
        #    differentiated again with respect to the weights, which is what
        #    the optimiser needs. Without it the physics term would contribute
        #    no gradient at all and the equation would never be satisfied.
        dd_dtau = torch.autograd.grad(
            outputs=d,
            inputs=tau,
            grad_outputs=torch.ones_like(d),
            create_graph=True,
        )[0]

        # 4) compare against what the physics says
        q_fric, load, speed, track_temp, compound = self._columns(context)
        physical_rate = wear_rate(
            d, q_fric, load, speed, track_temp, compound, self.params_tensor()
        )
        return dd_dtau - physical_rate

    @staticmethod
    def _columns(context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split the context matrix into its 5 named columns."""
        return tuple(context[:, i : i + 1] for i in range(N_CONTEXT))

    def collocation_points(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The points where the equation is enforced.

        Drawn uniformly:
          - tau out to the FULL 45-lap horizon, beyond the longest stint in the
            dataset;
          - each context variable within its physical range.

        They carry no labels whatsoever, and that is the entire reason this
        term can cover ground the data never does.
        """
        tau = torch.rand(n, 1) * self.tau_max

        columns = []
        for name in CONTEXT_RANGES:
            low, high = CONTEXT_RANGES[name]
            columns.append(torch.rand(n, 1) * (high - low) + low)

        return tau, torch.cat(columns, dim=1)

    # -- the three terms of the loss ---------------------------------------
    #
    # One method each, so the training loop below reads like the formula in
    # this module's docstring instead of like twelve lines of tensor plumbing.

    def _physics_loss(self, n: int) -> torch.Tensor:
        """Term 1. How badly the equation is violated, at unlabelled points."""
        tau, context = self.collocation_points(n)
        return (self.residual(tau, context) ** 2).mean()

    def _data_loss(self, tau_obs, context_obs, delta_obs) -> torch.Tensor:
        """Term 2. How far the predicted pace is from the measured pace.

        The network predicts d, which nobody measured. `gamma1 * d` turns it
        into seconds, which is what the data is in.
        """
        predicted_d = self.net(tau_obs, context_obs)
        predicted_delta = self.gamma1 * predicted_d
        return ((predicted_delta - delta_obs) ** 2).mean()

    def _initial_condition_loss(self, n: int = 64) -> torch.Tensor:
        """Term 3. How far d(0) is from zero: a new tire has worn nothing.

        The contexts are drawn at random so the condition is imposed across
        the whole range of conditions, not just at one of them.
        """
        _, context = self.collocation_points(n)
        tau_zero = torch.zeros(context.shape[0], 1)
        return (self.net(tau_zero, context) ** 2).mean()

    # -- training -----------------------------------------------------------

    def train(
        self,
        inputs: np.ndarray,
        delta: np.ndarray,
        iterations: int = 8000,
        n_collocation: int = 2000,
        lr: float = 3e-3,
        w_physics: float = 1.0,
        w_data: float = 10.0,
        w_ic: float = 10.0,
        record_every: int = 25,
        print_every: int = 500,
    ) -> None:
        """Gradient descent with Adam.

        The weights of the three terms are not importances: they are scales.
        The residual of the equation is dimensionless and small; the data one
        is in seconds. Without the factor, the term that actually anchors the
        model to what was measured would sit below the numerical noise of the
        other.
        """
        # Split the input matrix into tau and context BEFORE converting to
        # tensors, so that tau is a leaf of the graph and can be differentiated.
        tau_obs = torch.tensor(inputs[:, 0:1], dtype=torch.float32)
        context_obs = torch.tensor(inputs[:, 1:], dtype=torch.float32)
        delta_obs = torch.tensor(delta.reshape(-1, 1), dtype=torch.float32)

        # The optimiser moves the network weights and the 6 constants together.
        to_optimise = list(self.net.parameters()) + list(self._raw.values())
        optimiser = torch.optim.Adam(to_optimise, lr=lr)

        for iteration in range(1, iterations + 1):
            optimiser.zero_grad()

            physics_loss = self._physics_loss(n_collocation)
            data_loss = self._data_loss(tau_obs, context_obs, delta_obs)
            ic_loss = self._initial_condition_loss()

            loss = (
                w_physics * physics_loss
                + w_data * data_loss
                + w_ic * ic_loss
            )

            loss.backward()
            optimiser.step()

            if iteration % record_every == 0 or iteration == 1:
                self._record(iteration, loss, physics_loss, data_loss, ic_loss)
                if iteration % print_every == 0 or iteration == 1:
                    print("  " + self._log_line(self.history[-1]))

    def _record(self, iteration, loss, physics, data, ic) -> None:
        """Store this iteration's state so it can be plotted later.

        .item() is used rather than float(): the tensors are still attached to
        the autograd graph here, and float() on one of them raises a warning.
        """
        record = {
            "iteration": iteration,
            "total": loss.item(),
            "physics": physics.item(),
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
        printing never touches the autograd graph.
        """
        return (
            f"{r['iteration']:6d}  total {r['total']:9.5f}  "
            f"physics {r['physics']:8.5f}  data {r['data']:8.5f}  "
            f"ic {r['ic']:8.5f}  kw {r['kw']:5.3f}  kappa {r['kappa']:5.3f}"
        )

    # -- prediction ---------------------------------------------------------

    def predict_stint(self, context: Context, laps: np.ndarray) -> np.ndarray:
        """Predicted pace loss, lap by lap.

        Same signature as the baseline's, so the evaluator measures both models
        with exactly the same yardstick.
        """
        laps = np.asarray(laps, dtype=float).ravel()
        tau = torch.tensor((laps / LAP_REF).reshape(-1, 1), dtype=torch.float32)

        repeated_context = np.tile(context.vector().reshape(1, -1), (laps.size, 1))
        context_t = torch.tensor(repeated_context, dtype=torch.float32)

        with torch.no_grad():
            d = self.net(tau, context_t)

        return (self.gamma1 * d).numpy().ravel()

    def predict_wear(self, context: Context, laps: np.ndarray) -> np.ndarray:
        """The latent state d. It was never measured; it comes from the equation."""
        return self.predict_stint(context, laps) / self.gamma1
