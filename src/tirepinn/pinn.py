"""Parametric tire-degradation PINN, built on DeepXDE.

Design
------
A textbook PINN solves *one* trajectory: the network takes t and returns the
state. That would mean retraining for every stint, which is useless for live
inference. Here the network is a **parametric solution operator**:

    N(tau, q, lam, v, T_trk, c)  ->  (theta, d)

that is, it learns in one go the entire family of solutions of the ODE across
the full range of race conditions. Predicting a new stint is a single forward
pass, with no retraining and no integration.

Loss function
-------------
Five terms, three of physics and two of data:

  L1  residual of (E1), the thermal balance
  L2  residual of (E2), the wear law
  L3  penalty on the bound d <= d_max
  L4  fit to the measured pace loss (the only real observable)
  L5  weak temperature supervision (optional; see README)

L1-L3 are enforced across the whole condition hypercube, including where there
is no data: that is the edge over an LSTM, which outside its training
distribution has nothing tying it to thermodynamics.

Initial conditions
------------------
They are imposed *hard*, through an output transform rather than as extra loss
terms:

    theta(tau) = theta_0 + tau * N_0(x)      =>  theta(0) = theta_0 exactly
    d(tau)     = tau * softplus(N_1(x))      =>  d(0) = 0 exactly and d >= 0

This removes two loss terms and with them the weight-balancing problem, which is
the main source of convergence failures in PINNs.

Inverse problem
---------------
The physical coefficients (zeta, h0, h1, kw, m, Ea, kappa, gamma1, gamma2) are
unknown: they are estimated together with the network weights as `dde.Variable`.
The ODE ones are parametrised in log space, so they are positive by
construction, which is what their physical meaning requires.

The observable ones (gamma1, gamma2) additionally carry an upper bound, via a
sigmoid. This is not numerical caution but a necessity: (d, gamma1, gamma2) have
an exact scale degeneracy, since d -> e*d with gamma1 -> gamma1/e and
gamma2 -> gamma2/e^p leaves delta unchanged. On synthetic data the thermal proxy
and the stints that saturate at d = 1 break it; with real data neither exists
and the fit slides along that direction until it overflows. Bounding gamma
asserts something we do know: a destroyed tire costs a few seconds a lap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# The backend must be set before importing deepxde.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import torch
import torch.nn.functional as F

from .config import INPUT_DIM, LEARNABLE_PARAMS, OUTPUT_DIM, Config
from .dataset import StintDataset, input_bounds, stint_inputs
from .physics import (
    TireParams,
    pace_loss,
    theta_rhs,
    wear_lap,
    wear_rate,
)

# Re-exported for convenience: the canonical order of the physical parameters.
LEARNABLE = LEARNABLE_PARAMS


def _to_raw(name: str, value: float, cfg) -> float:
    """From a physical value to the unconstrained ("raw") variable the optimiser sees.

    Two regimes, chosen per parameter by whether it declares `<name>_bounds`:

    - No bounds: log parametrisation, so the value is positive by construction.
      That is what most ODE coefficients require -- there is no such thing as a
      negative cooling coefficient or a negative wear rate.
    - With bounds: sigmoid into the declared range. Used by gamma1/gamma2,
      because they are the degenerate direction of the problem, and by kappa,
      whose sign is genuinely free (see `PhysicsConfig.kappa_bounds`).
    """
    bounds = getattr(cfg, f"{name}_bounds", None)
    if bounds is None:
        return float(np.log(value))
    lo, hi = bounds
    z = float(np.clip((value - lo) / (hi - lo), 1e-4, 1 - 1e-4))
    return float(np.log(z / (1.0 - z)))


def _from_raw(name: str, raw, cfg):
    """Inverse of `_to_raw`: raw variable back to physical units.

    Works on plain floats (reporting) and on tensors (inside the autograd graph).
    """
    bounds = getattr(cfg, f"{name}_bounds", None)
    is_tensor = torch.is_tensor(raw)
    if bounds is None:
        return torch.exp(raw) if is_tensor else float(np.exp(raw))
    lo, hi = bounds
    if is_tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)
    return float(lo + (hi - lo) / (1.0 + np.exp(-raw)))


def _split_context(x):
    """Context columns of the network input, as (q, lam, v, track_temp, compound)."""
    return tuple(x[:, i : i + 1] for i in range(1, INPUT_DIM))


def _split_state(y):
    """Network output columns, as (theta, d)."""
    return y[:, 0:1], y[:, 1:2]


class TirePINN:
    """Parametric PINN: training, inference and physical-parameter estimation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: dde.Model | None = None
        self.net: dde.nn.NN | None = None
        self.loss_history = None
        self.var_history: list[tuple[int, dict[str, float]]] = []
        self._raw_vars: dict[str, torch.Tensor] = {}  # free parameters, unconstrained
        self._fixed: dict[str, float] = {}  # parameters held at their initial value
        self._bounds: tuple[list[float], list[float]] | None = None
        self._has_theta_obs = False

    # ------------------------------------------------------------------
    # Learnable physical parameters
    # ------------------------------------------------------------------
    def _param_inits(self) -> dict[str, float]:
        """Starting value of every physical parameter, free or fixed."""
        phys = self.cfg.physics
        inits = {name: float(getattr(phys, f"{name}_init")) for name in LEARNABLE}
        if phys.train_A_gen:
            inits["A_gen"] = phys.A_gen
        return inits

    def _init_variables(self) -> None:
        """Create the trainable variables; every other parameter stays fixed."""
        phys = self.cfg.physics
        free = set(self.cfg.pinn.free_params) | ({"A_gen"} if phys.train_A_gen else set())
        inits = self._param_inits()

        self._raw_vars = {
            name: dde.Variable(_to_raw(name, value, phys))
            for name, value in inits.items()
            if name in free
        }
        self._fixed = {name: value for name, value in inits.items() if name not in free}

    @property
    def trainable_variables(self) -> list[torch.Tensor]:
        return list(self._raw_vars.values())

    def _dtype(self):
        """Numeric type the network expects.

        It must follow `pinn.float64`: if double precision is enabled and float32
        were still sent here, prediction would fail on incompatible types.
        """
        return np.float64 if self.cfg.pinn.float64 else np.float32

    def _params_tensor(self) -> TireParams:
        """Parameters as torch tensors, so the residuals stay differentiable."""
        return self._assemble_params(
            {name: _from_raw(name, raw, self.cfg.physics) for name, raw in self._raw_vars.items()}
        )

    def learned_params(self) -> TireParams:
        """Estimated parameters, as plain floats."""
        return self._assemble_params(
            {
                name: _from_raw(name, raw.item(), self.cfg.physics)
                for name, raw in self._raw_vars.items()
            }
        )

    def _assemble_params(self, free_values: dict) -> TireParams:
        """Complete parameter set: the free values given, plus every fixed one."""
        phys = self.cfg.physics
        values = {**self._fixed, **free_values}
        return TireParams(
            A_gen=values.get("A_gen", phys.A_gen),
            **{name: values[name] for name in LEARNABLE},
            p=phys.cliff_exponent,
        )

    # ------------------------------------------------------------------
    # Residuals and observables
    # ------------------------------------------------------------------
    def _pde(self, x, y):
        """System residuals, evaluated at the collocation points."""
        phys = self.cfg.physics
        p = self._params_tensor()

        q, lam, v, track_temp, compound = _split_context(x)
        theta, d = _split_state(y)

        # Derivatives of each output (i) with respect to tau (input column j=0).
        dtheta_dtau = dde.grad.jacobian(y, x, i=0, j=0)
        dd_dtau = dde.grad.jacobian(y, x, i=1, j=0)

        r_theta = dtheta_dtau - theta_rhs(theta, d, q, v, p)
        r_wear = dd_dtau - wear_rate(theta, d, lam, track_temp, compound, p, phys.exp_clamp)
        r_bound = F.relu(d - phys.d_max)
        return [r_theta, r_wear, r_bound]

    def _obs_delta(self, x, y, _):
        """Observation operator: from the latent state d to the pace loss."""
        _, d = _split_state(y)
        return pace_loss(d, self._params_tensor())

    @staticmethod
    def _obs_theta(x, y, _):
        """Observation operator for the temperature proxy: theta itself."""
        theta, _ = _split_state(y)
        return theta

    def _output_transform(self, x, y):
        """Enforces theta(0) = theta_0, d(0) = 0 and d >= 0 exactly."""
        tau = x[:, 0:1]
        raw_theta, raw_d = _split_state(y)
        theta = self.cfg.physics.theta_init + tau * raw_theta
        d = tau * F.softplus(raw_d)
        return torch.cat([theta, d], dim=1)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _collocation_anchors(self, data: StintDataset) -> np.ndarray:
        """Dense collocation points in tau over every observed context.

        Uniform sampling of the hypercube covers context combinations that never
        occur in a race. These anchors additionally concentrate the residual on
        genuinely realisable trajectories, and they run out to the full decision
        horizon: the ODE is enforced on the laps the stint never reached, which
        are exactly the ones that need predicting.
        """
        phys = self.cfg.physics
        n_tau = self.cfg.pinn.tau_collocation
        tau = np.linspace(0.0, phys.strategy_horizon / phys.lap_ref, n_tau).reshape(-1, 1)
        return np.vstack(
            [np.hstack([tau, np.tile(s.context.reshape(1, -1), (n_tau, 1))]) for s in data.stints]
        )

    def build(self, data: StintDataset) -> None:
        """Assemble geometry, observation conditions, network and model."""
        cfg, phys = self.cfg, self.cfg.physics
        dde.config.set_random_seed(cfg.pinn.seed)
        if cfg.pinn.float64:
            dde.config.set_default_float("float64")

        self._init_variables()

        lows, highs = input_bounds(data, phys)
        self._bounds = (lows, highs)
        x_obs = data.inputs(phys)

        pde_data = dde.data.PDE(
            dde.geometry.Hypercube(lows, highs),
            self._pde,
            self._observation_conditions(data, x_obs),
            num_domain=cfg.pinn.num_domain,
            num_boundary=0,
            anchors=np.vstack([self._collocation_anchors(data), x_obs]),
        )
        self.net = self._make_network(cfg.pinn.hidden, cfg.pinn.activation)
        self.model = dde.Model(pde_data, self.net)

    def _observation_conditions(self, data: StintDataset, x_obs: np.ndarray) -> list:
        """Data terms of the loss: measured pace loss, plus the optional thermal proxy."""
        conditions = [dde.icbc.PointSetOperatorBC(x_obs, data.delta(), self._obs_delta)]

        theta_obs = None
        if self.cfg.pinn.use_theta_proxy:
            theta_obs = data.theta_observations(self.cfg.physics)
        self._has_theta_obs = theta_obs is not None
        if theta_obs is not None:
            x_theta, y_theta = theta_obs
            conditions.append(dde.icbc.PointSetOperatorBC(x_theta, y_theta, self._obs_theta))
        return conditions

    def _make_network(self, hidden, activation: str) -> dde.nn.FNN:
        """Fully connected network with the hard initial conditions attached."""
        net = dde.nn.FNN([INPUT_DIM, *hidden, OUTPUT_DIM], activation, self.cfg.pinn.initializer)
        net.apply_output_transform(self._output_transform)
        return net

    def _loss_weights(self) -> list[float]:
        """One weight per loss term, in the order DeepXDE reports them (L1..L5)."""
        c = self.cfg.pinn
        weights = [c.w_pde_theta, c.w_pde_wear, c.w_bound, c.w_data_delta]
        if self._has_theta_obs:
            weights.append(c.w_data_theta)
        return weights

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, out_dir: str | Path | None = None) -> None:
        """Adam to explore, L-BFGS to refine. The standard regime for PINNs."""
        if self.model is None:
            raise RuntimeError("Call build() before train()")
        cfg = self.cfg.pinn
        out_dir = Path(out_dir or self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        weights = self._loss_weights()
        var_file = str(out_dir / "variables.dat")
        var_cb = dde.callbacks.VariableValue(
            self.trainable_variables,
            period=cfg.display_every,
            filename=var_file,
            precision=6,
        )

        # Phase 1: Adam explores.
        self.model.compile(
            "adam",
            lr=cfg.lr,
            loss_weights=weights,
            external_trainable_variables=self.trainable_variables,
        )
        self.model.train(
            iterations=cfg.adam_iters,
            display_every=cfg.display_every,
            callbacks=[var_cb],
        )

        # Phase 2: L-BFGS refines. The thermal coefficients only converge here.
        if cfg.lbfgs_iters > 0:
            dde.optimizers.config.set_LBFGS_options(maxiter=cfg.lbfgs_iters)
            self.model.compile(
                "L-BFGS",
                loss_weights=weights,
                external_trainable_variables=self.trainable_variables,
            )
            self.model.train(display_every=cfg.display_every, callbacks=[var_cb])

        self.loss_history = self.model.losshistory
        self._read_variable_history(var_file)

    def _read_variable_history(self, path: str) -> None:
        """Recover the physical-parameter trace, to plot its convergence.

        DeepXDE writes lines of the form `<iteration> [v1, v2, ...]`, where the
        values are the unconstrained variables the network optimises; here they
        are mapped back into physical units. An unreadable file yields an empty
        history rather than an error: the trace is a diagnostic, not a result.
        """
        names = list(self._raw_vars)
        history: list[tuple[int, dict[str, float]]] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    entry = self._parse_variable_line(line, names)
                    if entry is not None:
                        history.append(entry)
        except (OSError, ValueError):
            history = []
        self.var_history = history

    def _parse_variable_line(
        self, line: str, names: list[str]
    ) -> tuple[int, dict[str, float]] | None:
        """One `<iteration> [v1, v2, ...]` line, in physical units. None if it has no values."""
        step_txt, _, values_txt = line.strip().partition(" ")
        values_txt = values_txt.strip().strip("[]")
        if not values_txt:
            return None
        raw_values = [float(v) for v in values_txt.split(",")]
        physical = {
            name: _from_raw(name, raw, self.cfg.physics)
            for name, raw in zip(names, raw_values, strict=False)
        }
        return int(step_txt), physical

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Latent state (theta, d) for an (n, 6) input matrix.

        A freshly trained model goes through DeepXDE; one loaded from disk only
        has the network, and takes the direct torch path. Both give the same
        result, so the rest of the code does not distinguish between them.
        """
        x = np.asarray(x, dtype=self._dtype()).reshape(-1, INPUT_DIM)
        if self.model is not None:
            y = self.model.predict(x)
        elif self.net is not None:
            y = self.forward_numpy(x)
        else:
            raise RuntimeError("The model is neither built nor loaded")
        return y[:, 0], y[:, 1]

    def predict_curve(self, context: np.ndarray, laps: np.ndarray) -> dict[str, np.ndarray]:
        """Full stint prediction: temperature, wear and pace."""
        laps = np.asarray(laps, dtype=float).ravel()
        theta, d = self.predict(stint_inputs(laps, context, self.cfg.physics))
        delta = pace_loss(d, self.learned_params())
        return {"laps": laps, "theta": theta, "d": d, "delta": delta}

    def predict_stint(self, context: np.ndarray, laps: np.ndarray) -> np.ndarray:
        """Predicted pace loss. Same signature as the baselines, so that
        `evaluate.py` measures all three models with the same yardstick."""
        return self.predict_curve(context, laps)["delta"]

    def strategy(
        self, context: np.ndarray, horizon: int | None = None, current_lap: float = 0.0
    ) -> dict:
        """Output for the pit wall: wear-limit lap and remaining useful life.

        Keyed to the latent state `d` crossing `d_crit`, not to the slope of the
        predicted pace curve: the state is available directly here, so there is
        no reason to re-infer it from a differentiated curve.
        """
        phys = self.cfg.physics
        horizon = horizon or phys.strategy_horizon
        pred = self.predict_curve(context, np.arange(1, horizon + 1))
        cliff = wear_lap(pred["laps"], pred["d"], phys)
        rul = None if cliff is None else max(cliff - current_lap, 0.0)
        return {
            "cliff_lap": cliff,
            "rul_laps": rul,
            "horizon": horizon,
            "curve": pred,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), out_dir / "pinn_weights.pt")
        payload = {
            "params": self.learned_params().as_dict(),
            "bounds": self._bounds,
            "hidden": list(self.cfg.pinn.hidden),
            "activation": self.cfg.pinn.activation,
            "theta_init": self.cfg.physics.theta_init,
            "lap_ref": self.cfg.physics.lap_ref,
        }
        with open(out_dir / "pinn_params.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, out_dir: str | Path, cfg: Config) -> TirePINN:
        """Rebuild the network for inference (no training data required)."""
        out_dir = Path(out_dir)
        with open(out_dir / "pinn_params.json", encoding="utf-8") as fh:
            payload = json.load(fh)

        obj = cls(cfg)
        obj._init_variables()
        with torch.no_grad():
            for name, value in payload["params"].items():
                if name in obj._raw_vars:
                    obj._raw_vars[name].fill_(_to_raw(name, value, cfg.physics))

        obj.net = obj._make_network(payload["hidden"], payload["activation"])
        obj.net.load_state_dict(torch.load(out_dir / "pinn_weights.pt", map_location="cpu"))
        obj.net.eval()
        obj._bounds = payload.get("bounds")
        return obj

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """Pure forward pass, without the DeepXDE machinery.

        This is the path an inference service would take: just the network.
        """
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(x, dtype=self._dtype()).reshape(-1, INPUT_DIM))
            return self.net(t).cpu().numpy()
