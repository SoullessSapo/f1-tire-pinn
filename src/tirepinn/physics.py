"""Modelo fisico del neumatico de F1.

Sistema de dos EDOs adimensionales acopladas que describen la vida de un stint:

    (E1)  dtheta/dtau = A_gen * q * (1 + zeta * d)  -  (h0 + h1 * v) * theta
    (E2)  dd/dtau     = kw * lam^m * exp(Ea * (theta + T_trk) - kappa * c) * (1 - d)

con

    tau   = vuelta_del_stint / L_ref             tiempo adimensional
    theta = (T_sup - T_pista) / dT_ref           exceso termico de la banda
    d     = fraccion de banda de rodadura consumida (0 = nuevo, 1 = agotado)
    q     = energia friccional especifica por vuelta (telemetria)
    lam   = carga mecanica media en g (telemetria)
    v     = velocidad media (enfriamiento convectivo forzado)
    T_trk = temperatura de pista normalizada
    c     = indice de dureza del compuesto (0 blando .. 1 duro)

(E1) es un balance termico de capacitancia concentrada: generacion friccional
menos decaimiento exponencial superficial (enfriamiento base + conveccion
forzada). (E2) combina la ley de Archard (desgaste proporcional a la carga)
con una activacion termica tipo Arrhenius linealizada alrededor de la ventana
de trabajo: el desgaste crece exponencialmente con la temperatura.

El factor (1 - d) de (E2) es un termino de saturacion: acota estructuralmente
d en [0, 1] porque no se puede consumir mas banda de rodadura de la que hay.
Es una hipotesis de modelado explicita, no una regularizacion numerica, y
tiene la ventaja de que la propia EDO impone la cota fisica sobre la red.

El factor (1 + zeta * d) de (E1) es el acoplamiento que produce el cliff: a
medida que la banda se adelgaza, la misma energia friccional se deposita sobre
menos masa de caucho, la temperatura superficial sube, y por Arrhenius el
desgaste se acelera. Es una realimentacion positiva, de modo que el cliff
emerge de la dinamica acoplada en lugar de imponerse a mano. Esta es
justamente la clase de restriccion termodinamica que una LSTM no conoce.

El observable medible desde la telemetria no es d sino la perdida de ritmo:

    delta(tau) = gamma1 * d  +  gamma2 * d^p

El primer termino es la degradacion lineal; el segundo, con p grande, es
despreciable hasta que d se acerca a 1 y entonces explota: es el cliff.

Las funciones aceptan tanto arrays de numpy como tensores de torch, de modo que
el mismo codigo define la verdad de referencia (integracion numerica) y el
residuo de la PINN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import PhysicsConfig

try:  # torch solo hace falta al evaluar el residuo dentro de DeepXDE
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
    """Parametros fisicos del modelo, estimados por la PINN como problema inverso."""

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
        """Valores iniciales declarados en la configuracion."""
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


# Verdad de referencia usada por el generador sintetico. Difiere a proposito de
# los valores iniciales de PhysicsConfig: la PINN debe recuperarla desde los
# datos, y esa recuperacion es la validacion del problema inverso.
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
# Terminos del sistema
# --------------------------------------------------------------------------
def theta_rhs(theta, d, q, v, p: TireParams):
    """Lado derecho de (E1): generacion friccional menos decaimiento termico.

    El factor (1 + zeta * d) concentra la energia friccional sobre una banda de
    rodadura cada vez mas delgada: es el motor del cliff.
    """
    return p.A_gen * q * (1.0 + p.zeta * d) - (p.h0 + p.h1 * v) * theta


def wear_rate(theta, d, lam, track_temp, compound, p: TireParams, exp_clamp: float = 6.0):
    """Lado derecho de (E2): tasa de desgaste Archard por Arrhenius, con saturacion.

    Es no negativa mientras d <= 1, asi que la monotonia del desgaste
    (dd/dtau >= 0) y la cota d <= 1 quedan implicadas por el propio residuo de
    la EDO, sin necesidad de restricciones adicionales sobre la red.
    """
    expo = _clamp_max(p.Ea * (theta + track_temp) - p.kappa * compound, exp_clamp)
    return p.kw * _pow(lam, p.m) * _exp(expo) * _relu(1.0 - d)


def pace_loss(d, p: TireParams):
    """Observable: perdida de ritmo en segundos frente al mejor ritmo del stint.

    Con d acotada en [0, 1] por (E2), la perdida maxima es gamma1 + gamma2.
    """
    d = _clamp_max(_relu(d), 1.0)
    return p.gamma1 * d + p.gamma2 * _pow(d, p.p)


# --------------------------------------------------------------------------
# Integracion de referencia (numpy): genera la verdad de un stint
# --------------------------------------------------------------------------
def integrate_stint(
    n_laps: int,
    context: np.ndarray,
    p: TireParams,
    phys: PhysicsConfig,
    steps_per_lap: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integra (E1)-(E2) con Runge-Kutta 4 sobre un stint completo.

    Args:
        n_laps: numero de vueltas del stint.
        context: vector (q_fric, load, speed, track_temp, compound).
        p: parametros fisicos.
        phys: escalas de adimensionalizacion.
        steps_per_lap: subdivision temporal del integrador.

    Returns:
        laps: vueltas 1..n_laps.
        theta: exceso termico al final de cada vuelta.
        d: degradacion acumulada al final de cada vuelta.
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
# Post-proceso: cliff y vida util remanente
# --------------------------------------------------------------------------
def cliff_lap(
    laps: np.ndarray,
    delta: np.ndarray,
    phys: PhysicsConfig,
    smooth: bool = True,
) -> float | None:
    """Primera vuelta en la que la curva de ritmo entra en el cliff.

    Criterio operativo: la pendiente de la perdida de ritmo supera el umbral
    cliff_slope_s_per_lap. Devuelve None si el stint termina antes del cliff.

    Se suaviza con una media movil de tres vueltas antes de derivar. Sin eso, el
    ruido de cronometraje de una curva real (unas decimas de segundo) domina la
    derivada y dispara falsos cliffs. Es el mismo criterio para la verdad
    observada y para la prediccion de cualquier modelo, que es lo que hace
    comparable la metrica.
    """
    laps = np.asarray(laps, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if laps.size < 3:
        return None
    if smooth and delta.size >= 5:
        kernel = np.ones(3) / 3.0
        padded = np.pad(delta, 1, mode="edge")
        delta = np.convolve(padded, kernel, mode="valid")
    slope = np.gradient(delta, laps)
    hit = np.nonzero(slope >= phys.cliff_slope_s_per_lap)[0]
    if hit.size == 0:
        return None
    return float(laps[hit[0]])



def remaining_useful_life(current_lap: float, cliff: float | None, horizon: float) -> float:
    """RUL en vueltas. Si no se detecta cliff, se acota con el horizonte dado."""
    if cliff is None:
        return float(max(horizon - current_lap, 0.0))
    return float(max(cliff - current_lap, 0.0))
