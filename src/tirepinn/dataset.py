"""Estructuras de datos comunes al generador sintetico y al cargador de FastF1.

Un `Stint` es la unidad de aprendizaje: un juego de neumaticos desde que sale de
boxes hasta que entra. La PINN nunca ve vueltas sueltas sino stints completos,
porque la fisica que impone es una EDO en el tiempo dentro del stint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CONTEXT_NAMES, PhysicsConfig


@dataclass
class Stint:
    """Un stint: vueltas, perdida de ritmo observada y contexto constante."""

    stint_id: str
    driver: str
    compound: str
    laps: np.ndarray              # vuelta dentro del stint, 1..n
    delta: np.ndarray             # perdida de ritmo observada [s]
    context: np.ndarray           # (5,) q_fric, load, speed, track_temp, compound

    # Solo disponibles en el banco sintetico (verdad de referencia).
    theta_true: np.ndarray | None = None
    d_true: np.ndarray | None = None
    delta_true: np.ndarray | None = None   # curva de ritmo sin ruido de medicion
    # Solo disponible con datos reales: numero de vuelta de la carrera.
    race_laps: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.laps = np.asarray(self.laps, dtype=float).ravel()
        self.delta = np.asarray(self.delta, dtype=float).ravel()
        self.context = np.asarray(self.context, dtype=float).ravel()
        if self.laps.shape != self.delta.shape:
            raise ValueError(
                f"{self.stint_id}: laps {self.laps.shape} y delta {self.delta.shape} no coinciden"
            )
        if self.context.size != len(CONTEXT_NAMES):
            raise ValueError(
                f"{self.stint_id}: el contexto debe tener {len(CONTEXT_NAMES)} componentes"
            )

    @property
    def n_laps(self) -> int:
        return int(self.laps.size)

    def tau(self, phys: PhysicsConfig) -> np.ndarray:
        """Tiempo adimensional tau = vuelta / L_ref."""
        return self.laps / phys.lap_ref

    def inputs(self, phys: PhysicsConfig) -> np.ndarray:
        """Matriz (n_laps, 6) de entradas de la red: tau + contexto replicado."""
        tau = self.tau(phys).reshape(-1, 1)
        ctx = np.tile(self.context.reshape(1, -1), (tau.shape[0], 1))
        return np.hstack([tau, ctx])


@dataclass
class StintDataset:
    """Coleccion de stints con utilidades de armado de tensores y particion."""

    stints: list[Stint]
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.stints)

    @property
    def n_laps(self) -> int:
        return int(sum(s.n_laps for s in self.stints))

    def inputs(self, phys: PhysicsConfig) -> np.ndarray:
        """(N, 6) con todas las vueltas de todos los stints."""
        return np.vstack([s.inputs(phys) for s in self.stints])

    def delta(self) -> np.ndarray:
        """(N, 1) perdida de ritmo observada."""
        return np.concatenate([s.delta for s in self.stints]).reshape(-1, 1)

    def contexts(self) -> np.ndarray:
        """(n_stints, 5) contexto de cada stint."""
        return np.vstack([s.context.reshape(1, -1) for s in self.stints])

    def theta_observations(self, phys: PhysicsConfig) -> tuple[np.ndarray, np.ndarray] | None:
        """Supervision debil de temperatura, si la fuente la provee.

        Devuelve (X, theta) o None. Con datos reales esto normalmente es None:
        la temperatura interna del neumatico no es publica.
        """
        usable = [s for s in self.stints if s.theta_true is not None]
        if not usable:
            return None
        X = np.vstack([s.inputs(phys) for s in usable])
        theta = np.concatenate([s.theta_true for s in usable]).reshape(-1, 1)
        return X, theta

    def split(self, test_fraction: float, seed: int = 0) -> tuple[StintDataset, StintDataset]:
        """Particion por stint completo, nunca por vuelta.

        Partir por vuelta filtraria informacion del mismo stint entre train y
        test y sobreestimaria el desempenio de todos los modelos por igual.
        """
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.stints))
        n_test = max(1, round(test_fraction * len(self.stints)))
        test_idx = set(idx[:n_test].tolist())
        train = [s for i, s in enumerate(self.stints) if i not in test_idx]
        test = [s for i, s in enumerate(self.stints) if i in test_idx]
        return (
            StintDataset(train, self.source, dict(self.meta, split="train")),
            StintDataset(test, self.source, dict(self.meta, split="test")),
        )

    def describe(self) -> str:
        if not self.stints:
            return "Dataset vacio"
        lens = np.array([s.n_laps for s in self.stints])
        deltas = np.concatenate([s.delta for s in self.stints])
        comps: dict[str, int] = {}
        for s in self.stints:
            comps[s.compound] = comps.get(s.compound, 0) + 1
        comp_txt = ", ".join(f"{k}:{v}" for k, v in sorted(comps.items()))
        return (
            f"Fuente: {self.source} | stints: {len(self.stints)} | vueltas: {self.n_laps}\n"
            f"  Longitud de stint: min={lens.min()} med={np.median(lens):.0f} max={lens.max()}\n"
            f"  Perdida de ritmo:  min={deltas.min():.2f}s med={np.median(deltas):.2f}s max={deltas.max():.2f}s\n"
            f"  Compuestos: {comp_txt}"
        )


def input_bounds(dataset: StintDataset, phys: PhysicsConfig, pad: float = 0.05) -> tuple[list[float], list[float]]:
    """Hipercubo que envuelve los datos observados, con un margen relativo.

    Es el dominio donde se imponen los residuos de la EDO: la fisica regulariza
    a la red tambien en las combinaciones de contexto que no aparecen en los
    datos, que es precisamente la ventaja de una PINN sobre un ajuste puro.
    """
    X = dataset.inputs(phys)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    lo = lo - pad * span
    hi = hi + pad * span
    lo[0] = 0.0  # tau siempre arranca en el inicio del stint
    # tau se estira hasta el horizonte de decision completo, mas alla del stint
    # mas largo observado: ahi es donde la fisica debe sostener por si sola la
    # extrapolacion hacia el cliff, sin datos que la guien.
    hi[0] = max(hi[0] * 1.20, phys.strategy_horizon / phys.lap_ref)
    return lo.tolist(), hi.tolist()
