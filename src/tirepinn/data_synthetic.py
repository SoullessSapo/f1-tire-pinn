"""Banco de pruebas sintetico con verdad fisica conocida.

Integra el sistema (E1)-(E2) con los parametros de `physics.GROUND_TRUTH` y le
anade ruido de medicion. Sirve para dos cosas:

1. Que el pipeline completo corra sin depender de la red ni de la API de FastF1.
2. Validar el problema inverso: la PINN parte de valores iniciales distintos y
   debe *recuperar* los parametros fisicos verdaderos solo a partir de la
   perdida de ritmo observada. Con datos reales esa verificacion es imposible
   porque no existe la verdad de referencia.

El generador imita la decision de un estratega: el stint se corta un par de
vueltas despues del cliff, porque ningun equipo rueda con el neumatico agotado.
"""

from __future__ import annotations

import numpy as np

from .config import COMPOUND_INDEX, ContextRanges, DataConfig, PhysicsConfig
from .dataset import Stint, StintDataset
from .physics import GROUND_TRUTH, TireParams, cliff_lap, integrate_stint, pace_loss

# Compuestos que se simulan y su rango tipico de exigencia relativa.
_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def _sample_context(rng: np.random.Generator, ranges: ContextRanges, compound: str) -> np.ndarray:
    """Muestrea un contexto plausible para un compuesto dado."""
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
    """Genera `cfg.n_stints` stints sinteticos."""
    ranges = ranges or ContextRanges()
    rng = np.random.default_rng(seed)
    stints: list[Stint] = []

    for i in range(cfg.n_stints):
        compound = _COMPOUNDS[i % len(_COMPOUNDS)]
        context = _sample_context(rng, ranges, compound)

        # Se integra hasta el maximo y luego se decide donde habria parado el equipo.
        laps_full, theta_full, d_full = integrate_stint(cfg.max_stint, context, params, phys)
        delta_full = pace_loss(d_full, params)
        cliff = cliff_lap(laps_full, delta_full, phys)

        if cliff is not None:
            # Se dejan algunas vueltas *despues* del cliff. Un equipo no para en
            # el instante exacto: pierde un par de vueltas decidiendo, esperando
            # hueco en boxes o cubriendo a un rival. Ademas son las unicas
            # vueltas que informan sobre el regimen d -> 1, del que dependen
            # gamma2 y la escala de kw (ver README, identificabilidad).
            n_laps = int(min(cfg.max_stint, cliff + rng.integers(2, 6)))
        else:
            n_laps = int(rng.integers(cfg.min_stint, cfg.max_stint + 1))
        n_laps = int(np.clip(n_laps, cfg.min_stint, cfg.max_stint))

        laps = laps_full[:n_laps]
        theta = theta_full[:n_laps]
        d = d_full[:n_laps]
        delta = delta_full[:n_laps]

        # Ruido de medicion: el cronometraje real tiene trafico, viento y errores
        # de pilotaje que no dependen del neumatico.
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
