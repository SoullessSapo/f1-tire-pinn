"""PINN parametrica de degradacion de neumaticos, construida sobre DeepXDE.

Diseno
------
Una PINN clasica resuelve *una* trayectoria: la red toma t y devuelve el estado.
Eso obligaria a reentrenar por cada stint, lo que es inservible para inferencia
en vivo. Aqui la red es un **operador solucion parametrico**:

    N(tau, q, lam, v, T_trk, c)  ->  (theta, d)

es decir, aprende de una sola vez la familia completa de soluciones de la EDO
para todo el rango de condiciones de carrera. Predecir un stint nuevo es un
unico paso forward, sin reentrenar y sin integrar nada.

Funcion de perdida
------------------
Cinco terminos, tres de fisica y dos de datos:

  L1  residuo de (E1), balance termico
  L2  residuo de (E2), ley de desgaste
  L3  penalizacion de la cota d <= d_max
  L4  ajuste a la perdida de ritmo medida (el unico observable real)
  L5  supervision debil de temperatura (opcional; ver README)

L1-L3 se imponen sobre todo el hipercubo de condiciones, tambien donde no hay
datos: ahi esta la ventaja sobre una LSTM, que fuera de su distribucion de
entrenamiento no tiene ninguna restriccion que la ate a la termodinamica.

Condiciones iniciales
---------------------
Se imponen de forma *dura* mediante una transformacion de salida, no como un
termino mas de la perdida:

    theta(tau) = theta_0 + tau * N_0(x)      =>  theta(0) = theta_0 exacto
    d(tau)     = tau * softplus(N_1(x))      =>  d(0) = 0 exacto y d >= 0

Esto elimina dos terminos de la perdida y con ellos el problema de balancear
sus pesos, que es la principal fuente de fallo de convergencia en PINNs.

Problema inverso
----------------
Los coeficientes fisicos (zeta, h0, h1, kw, m, Ea, kappa, gamma1, gamma2) no se
conocen: se estiman junto con los pesos de la red como `dde.Variable`. Los de la
EDO se parametrizan en logaritmo, de modo que son positivos por construccion,
que es lo que exige su significado fisico.

Los del observable (gamma1, gamma2) llevan ademas cota superior, via sigmoide.
No es una precaucion numerica sino una necesidad: (d, gamma1, gamma2) tienen una
degeneracion exacta de escala, ya que d -> e*d con gamma1 -> gamma1/e y
gamma2 -> gamma2/e^p deja delta sin cambio. Con datos sinteticos la rompen el
proxy termico y los stints que saturan en d = 1; con datos reales no hay ninguna
de las dos y el ajuste se va por esa direccion hasta desbordar. Acotar gamma es
afirmar algo que si sabemos: un neumatico destruido cuesta unos pocos segundos
por vuelta.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# El backend debe fijarse antes de importar deepxde.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import torch
import torch.nn.functional as F

from .config import INPUT_DIM, LEARNABLE_PARAMS, OUTPUT_DIM, Config
from .dataset import StintDataset, input_bounds
from .physics import (
    TireParams,
    cliff_lap,
    pace_loss,
    theta_rhs,
    wear_rate,
)

# Re-exportado por conveniencia: el orden canonico de los parametros fisicos.
LEARNABLE = LEARNABLE_PARAMS


def _to_raw(name: str, value: float, cfg) -> float:
    """Del valor fisico a la variable sin restricciones que optimiza la red.

    Los parametros de la EDO solo necesitan ser positivos, asi que se
    parametrizan en logaritmo. Los del observable (gamma1, gamma2) ademas
    llevan cota superior, porque son la direccion degenerada del problema: se
    mapean con una sigmoide dentro de su rango fisico.
    """
    bounds = getattr(cfg, f"{name}_bounds", None)
    if bounds is None:
        return float(np.log(value))
    lo, hi = bounds
    z = float(np.clip((value - lo) / (hi - lo), 1e-4, 1 - 1e-4))
    return float(np.log(z / (1.0 - z)))


def _from_raw(name: str, raw, cfg):
    """Inversa de `_to_raw`. Funciona con escalares de numpy y con tensores."""
    bounds = getattr(cfg, f"{name}_bounds", None)
    is_tensor = torch.is_tensor(raw)
    if bounds is None:
        return torch.exp(raw) if is_tensor else float(np.exp(raw))
    lo, hi = bounds
    if is_tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)
    return float(lo + (hi - lo) / (1.0 + np.exp(-raw)))


class TirePINN:
    """PINN parametrica: entrenamiento, inferencia y estimacion de parametros."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: dde.Model | None = None
        self.net: dde.nn.NN | None = None
        self.loss_history = None
        self.var_history: list[tuple[int, dict[str, float]]] = []
        self._log_vars: dict[str, torch.Tensor] = {}
        self._fixed: dict[str, float] = {}
        self._bounds: tuple[list[float], list[float]] | None = None
        self._has_theta_obs = False

    # ------------------------------------------------------------------
    # Parametros fisicos aprendibles
    # ------------------------------------------------------------------
    def _param_inits(self) -> dict[str, float]:
        """Valor de partida de cada parametro fisico, libre o fijo."""
        phys = self.cfg.physics
        inits = {name: float(getattr(phys, f"{name}_init")) for name in LEARNABLE}
        if phys.train_A_gen:
            inits["A_gen"] = phys.A_gen
        return inits

    def _init_variables(self) -> None:
        """Crea las variables entrenables; el resto de parametros quedan fijos."""
        phys = self.cfg.physics
        free = set(self.cfg.pinn.free_params) | ({"A_gen"} if phys.train_A_gen else set())
        inits = self._param_inits()

        self._log_vars = {
            k: dde.Variable(_to_raw(k, v, phys)) for k, v in inits.items() if k in free
        }
        self._fixed = {k: v for k, v in inits.items() if k not in free}

    @property
    def trainable_variables(self) -> list[torch.Tensor]:
        return list(self._log_vars.values())

    def _dtype(self):
        """Tipo numerico que espera la red.

        Debe seguir a `pinn.float64`: si se activa la doble precision y aqui se
        siguiera enviando float32, la prediccion fallaria por tipos incompatibles.
        """
        return np.float64 if self.cfg.pinn.float64 else np.float32

    def _params_tensor(self) -> TireParams:
        """Parametros como tensores de torch, para el grafo de autograd."""
        phys = self.cfg.physics
        get = lambda k: (  # noqa: E731
            _from_raw(k, self._log_vars[k], phys) if k in self._log_vars else self._fixed[k]
        )
        a_gen = get("A_gen") if "A_gen" in self._log_vars else phys.A_gen
        return TireParams(
            A_gen=a_gen,
            zeta=get("zeta"),
            h0=get("h0"),
            h1=get("h1"),
            kw=get("kw"),
            m=get("m"),
            Ea=get("Ea"),
            kappa=get("kappa"),
            gamma1=get("gamma1"),
            gamma2=get("gamma2"),
            p=phys.cliff_exponent,
        )

    def learned_params(self) -> TireParams:
        """Parametros estimados, como escalares de numpy."""
        phys = self.cfg.physics
        vals = dict(self._fixed)
        vals.update({k: _from_raw(k, v.item(), phys) for k, v in self._log_vars.items()})
        return TireParams(
            A_gen=vals.get("A_gen", phys.A_gen),
            zeta=vals["zeta"],
            h0=vals["h0"],
            h1=vals["h1"],
            kw=vals["kw"],
            m=vals["m"],
            Ea=vals["Ea"],
            kappa=vals["kappa"],
            gamma1=vals["gamma1"],
            gamma2=vals["gamma2"],
            p=phys.cliff_exponent,
        )

    # ------------------------------------------------------------------
    # Residuos y observables
    # ------------------------------------------------------------------
    def _pde(self, x, y):
        """Residuos del sistema, evaluados en los puntos de colocacion."""
        phys = self.cfg.physics
        p = self._params_tensor()

        q, lam, v, trk, comp = (x[:, i : i + 1] for i in range(1, INPUT_DIM))
        theta, d = y[:, 0:1], y[:, 1:2]

        # Derivadas respecto a tau (dimension 0 de la entrada).
        dtheta_dtau = dde.grad.jacobian(y, x, i=0, j=0)
        dd_dtau = dde.grad.jacobian(y, x, i=1, j=0)

        r_theta = dtheta_dtau - theta_rhs(theta, d, q, v, p)
        r_wear = dd_dtau - wear_rate(theta, d, lam, trk, comp, p, phys.exp_clamp)
        r_bound = F.relu(d - phys.d_max)
        return [r_theta, r_wear, r_bound]

    def _obs_delta(self, x, y, _):
        """Operador de observacion: del estado latente d a la perdida de ritmo."""
        return pace_loss(y[:, 1:2], self._params_tensor())

    @staticmethod
    def _obs_theta(x, y, _):
        return y[:, 0:1]

    def _output_transform(self, x, y):
        """Impone theta(0) = theta_0, d(0) = 0 y d >= 0 de forma exacta."""
        tau = x[:, 0:1]
        theta = self.cfg.physics.theta_init + tau * y[:, 0:1]
        d = tau * F.softplus(y[:, 1:2])
        return torch.cat([theta, d], dim=1)

    # ------------------------------------------------------------------
    # Construccion
    # ------------------------------------------------------------------
    def _collocation_anchors(self, data: StintDataset) -> np.ndarray:
        """Puntos de colocacion densos en tau sobre cada contexto observado.

        El muestreo uniforme del hipercubo cubre combinaciones de contexto que
        no ocurren en carrera. Estos anclajes concentran ademas el residuo
        sobre trayectorias realmente realizables, y llegan hasta el horizonte
        de decision completo: la EDO se impone tambien en las vueltas que el
        stint nunca alcanzo, que son justamente las que hay que predecir.
        """
        phys = self.cfg.physics
        n_tau = self.cfg.pinn.tau_collocation
        tau_max = phys.strategy_horizon / phys.lap_ref
        blocks = []
        for stint in data.stints:
            tau = np.linspace(0.0, tau_max, n_tau).reshape(-1, 1)
            ctx = np.tile(stint.context.reshape(1, -1), (n_tau, 1))
            blocks.append(np.hstack([tau, ctx]))
        return np.vstack(blocks)

    def build(self, data: StintDataset) -> None:
        """Arma geometria, condiciones de observacion, red y modelo."""
        cfg, phys = self.cfg, self.cfg.physics
        dde.config.set_random_seed(cfg.pinn.seed)
        if cfg.pinn.float64:
            dde.config.set_default_float("float64")

        self._init_variables()

        lows, highs = input_bounds(data, phys)
        self._bounds = (lows, highs)
        geom = dde.geometry.Hypercube(lows, highs)

        # --- termino de datos: perdida de ritmo medida ---
        x_obs = data.inputs(phys)
        y_obs = data.delta()
        bcs = [dde.icbc.PointSetOperatorBC(x_obs, y_obs, self._obs_delta)]

        # --- termino de datos opcional: proxy de temperatura ---
        theta_obs = data.theta_observations(phys) if cfg.pinn.use_theta_proxy else None
        self._has_theta_obs = theta_obs is not None
        if theta_obs is not None:
            bcs.append(dde.icbc.PointSetOperatorBC(theta_obs[0], theta_obs[1], self._obs_theta))

        anchors = np.vstack([self._collocation_anchors(data), x_obs])
        pde_data = dde.data.PDE(
            geom,
            self._pde,
            bcs,
            num_domain=cfg.pinn.num_domain,
            num_boundary=0,
            anchors=anchors,
        )

        layers = [INPUT_DIM, *cfg.pinn.hidden, OUTPUT_DIM]
        self.net = dde.nn.FNN(layers, cfg.pinn.activation, cfg.pinn.initializer)
        self.net.apply_output_transform(self._output_transform)
        self.model = dde.Model(pde_data, self.net)

    def _loss_weights(self) -> list[float]:
        c = self.cfg.pinn
        w = [c.w_pde_theta, c.w_pde_wear, c.w_bound, c.w_data_delta]
        if self._has_theta_obs:
            w.append(c.w_data_theta)
        return w

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------
    def train(self, out_dir: str | Path | None = None) -> None:
        """Adam para explorar, L-BFGS para afinar. Es el regimen estandar en PINNs."""
        if self.model is None:
            raise RuntimeError("Llama a build() antes de train()")
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
        """Recupera la traza de los parametros fisicos para graficar su convergencia.

        DeepXDE escribe lineas con el formato `<iteracion> [v1, v2, ...]`, donde
        los valores son las variables sin restricciones que la red optimiza;
        aqui se transforman de vuelta a unidades fisicas.
        """
        phys = self.cfg.physics
        names = list(self._log_vars)
        history: list[tuple[int, dict[str, float]]] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    step_txt, _, vals_txt = line.strip().partition(" ")
                    vals_txt = vals_txt.strip().strip("[]")
                    if not vals_txt:
                        continue
                    vals = [float(v) for v in vals_txt.split(",")]
                    history.append(
                        (
                            int(step_txt),
                            {n: _from_raw(n, v, phys) for n, v in zip(names, vals, strict=False)},
                        )
                    )
        except (OSError, ValueError):
            history = []
        self.var_history = history

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estado latente (theta, d) para una matriz de entradas (n, 6).

        Un modelo recien entrenado pasa por DeepXDE; uno cargado desde disco
        solo tiene la red, y va por el camino directo de torch. Las dos rutas
        dan el mismo resultado, asi que el resto del codigo no las distingue.
        """
        x = np.asarray(x, dtype=self._dtype()).reshape(-1, INPUT_DIM)
        if self.model is not None:
            y = self.model.predict(x)
        elif self.net is not None:
            y = self.forward_numpy(x)
        else:
            raise RuntimeError("El modelo no esta construido ni cargado")
        return y[:, 0], y[:, 1]

    def predict_curve(self, context: np.ndarray, laps: np.ndarray) -> dict[str, np.ndarray]:
        """Prediccion completa de un stint: temperatura, desgaste y ritmo."""
        phys = self.cfg.physics
        laps = np.asarray(laps, dtype=float).ravel()
        tau = (laps / phys.lap_ref).reshape(-1, 1)
        ctx = np.tile(np.asarray(context, dtype=float).reshape(1, -1), (tau.shape[0], 1))
        theta, d = self.predict(np.hstack([tau, ctx]))
        delta = pace_loss(d, self.learned_params())
        return {"laps": laps, "theta": theta, "d": d, "delta": delta}

    def predict_stint(self, context: np.ndarray, laps: np.ndarray) -> np.ndarray:
        """Perdida de ritmo predicha. Misma firma que los baselines, para que
        `evaluate.py` mida los tres modelos con la misma vara."""
        return self.predict_curve(context, laps)["delta"]

    def strategy(
        self, context: np.ndarray, horizon: int | None = None, current_lap: float = 0.0
    ) -> dict:
        """Salida util para el muro de boxes: vuelta del cliff y vida util remanente."""
        phys = self.cfg.physics
        horizon = horizon or phys.strategy_horizon
        pred = self.predict_curve(context, np.arange(1, horizon + 1))
        cliff = cliff_lap(pred["laps"], pred["delta"], phys)
        rul = None if cliff is None else max(cliff - current_lap, 0.0)
        return {
            "cliff_lap": cliff,
            "rul_laps": rul,
            "horizon": horizon,
            "curve": pred,
        }

    # ------------------------------------------------------------------
    # Persistencia
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
        """Reconstruye la red para inferencia (no requiere los datos de entrenamiento)."""
        out_dir = Path(out_dir)
        with open(out_dir / "pinn_params.json", encoding="utf-8") as fh:
            payload = json.load(fh)

        obj = cls(cfg)
        obj._init_variables()
        for name, value in payload["params"].items():
            if name in obj._log_vars:
                with torch.no_grad():
                    obj._log_vars[name].fill_(_to_raw(name, value, cfg.physics))

        layers = [INPUT_DIM, *payload["hidden"], OUTPUT_DIM]
        obj.net = dde.nn.FNN(layers, payload["activation"], cfg.pinn.initializer)
        obj.net.apply_output_transform(obj._output_transform)
        obj.net.load_state_dict(torch.load(out_dir / "pinn_weights.pt", map_location="cpu"))
        obj.net.eval()
        obj._bounds = payload.get("bounds")
        return obj

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """Paso forward puro, sin la maquinaria de DeepXDE.

        Es el camino que usaria un servicio de inferencia: solo la red.
        """
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(x, dtype=self._dtype()).reshape(-1, INPUT_DIM))
            return self.net(t).cpu().numpy()
