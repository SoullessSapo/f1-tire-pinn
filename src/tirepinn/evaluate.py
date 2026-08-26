"""Evaluacion cuantitativa: precision, acierto en el cliff y coherencia fisica.

Las tres dimensiones responden a preguntas distintas:

- **RMSE / MAE** sobre la perdida de ritmo: cuanto se equivoca el modelo en la
  vuelta que esta viendo. Es la metrica que pide el criterio de precision
  predictiva del proyecto.

- **Error de vuelta del cliff**: cuanto se equivoca en la unica prediccion que
  cambia una decision de estrategia. Un modelo puede tener buen RMSE global y
  aun asi fallar el cliff por cinco vueltas, que es lo unico que importa.

- **Violaciones de monotonia**: con que frecuencia el modelo predice que el
  neumatico recupera agarre. Es fisicamente imposible y ninguna metrica de
  error la penaliza, asi que se mide aparte. Es la comparacion que justifica
  meter la fisica dentro de la red.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PhysicsConfig
from .dataset import Stint, StintDataset
from .physics import cliff_lap


@dataclass
class Metrics:
    """Resultado de evaluar un modelo sobre un conjunto de stints."""

    name: str
    rmse: float
    mae: float
    max_error: float
    cliff_mae: float | None
    cliff_detected: int
    cliff_total: int
    violation_rate: float
    extrap_violation_rate: float
    n_laps: int
    per_stint: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> str:
        cliff = "n/d" if self.cliff_mae is None else f"{self.cliff_mae:.2f}"
        return (
            f"{self.name:22s} {self.rmse:7.3f} {self.mae:7.3f} {self.max_error:8.3f} "
            f"{cliff:>8s} {self.cliff_detected:3d}/{self.cliff_total:<3d} "
            f"{100 * self.violation_rate:7.1f}% {100 * self.extrap_violation_rate:9.1f}%"
        )


HEADER = (
    f"{'Modelo':22s} {'RMSE':>7s} {'MAE':>7s} {'MaxErr':>8s} "
    f"{'CliffMAE':>8s} {'Cliff':>7s} {'ViolIn':>8s} {'ViolExtrap':>10s}"
)


def _monotonicity_violation(delta: np.ndarray, tol: float = 1e-3) -> tuple[int, int]:
    """Cuenta pasos vuelta a vuelta en los que el ritmo *mejora* mas de `tol`.

    El neumatico solo se degrada: cualquier mejora sostenida es una prediccion
    termodinamicamente imposible.
    """
    if delta.size < 2:
        return 0, 0
    diffs = np.diff(np.asarray(delta, dtype=float))
    return int((diffs < -tol).sum()), int(diffs.size)


def _true_cliff(stint: Stint, phys: PhysicsConfig) -> float | None:
    """Vuelta del cliff de referencia.

    Siempre el mismo criterio que se aplica a las predicciones: pendiente de la
    curva de ritmo. Con datos sinteticos se usa la curva sin ruido, que existe;
    con datos reales, la medida, que es lo unico observable.
    """
    curve = stint.delta_true if stint.delta_true is not None else stint.delta
    return cliff_lap(stint.laps, curve, phys)


def evaluate(
    name: str,
    predict_stint,
    data: StintDataset,
    phys: PhysicsConfig,
    extrapolation_horizon: int | None = None,
) -> Metrics:
    """Mide un modelo cualquiera que exponga `predict_stint(context, laps)`."""
    extrapolation_horizon = extrapolation_horizon or phys.strategy_horizon
    errors: list[np.ndarray] = []
    cliff_errors: list[float] = []
    per_stint: dict[str, float] = {}
    viol, viol_total = 0, 0
    ex_viol, ex_total = 0, 0
    detected = 0
    total_with_cliff = 0

    for stint in data.stints:
        pred = np.asarray(predict_stint(stint.context, stint.laps), dtype=float).ravel()
        err = pred - stint.delta
        errors.append(err)
        per_stint[stint.stint_id] = float(np.sqrt(np.mean(err**2)))

        v, t = _monotonicity_violation(pred)
        viol += v
        viol_total += t

        # Extrapolacion: se pide al modelo el stint completo hasta el horizonte,
        # mas alla de lo que vio. Aqui es donde la fisica se nota.
        horizon = np.arange(1, extrapolation_horizon + 1)
        pred_long = np.asarray(predict_stint(stint.context, horizon), dtype=float).ravel()
        v, t = _monotonicity_violation(pred_long)
        ex_viol += v
        ex_total += t

        truth = _true_cliff(stint, phys)
        if truth is not None:
            total_with_cliff += 1
            got = cliff_lap(horizon, pred_long, phys)
            if got is not None:
                detected += 1
                cliff_errors.append(abs(got - truth))

    all_err = np.concatenate(errors)
    return Metrics(
        name=name,
        rmse=float(np.sqrt(np.mean(all_err**2))),
        mae=float(np.mean(np.abs(all_err))),
        max_error=float(np.max(np.abs(all_err))),
        cliff_mae=float(np.mean(cliff_errors)) if cliff_errors else None,
        cliff_detected=detected,
        cliff_total=total_with_cliff,
        violation_rate=viol / viol_total if viol_total else 0.0,
        extrap_violation_rate=ex_viol / ex_total if ex_total else 0.0,
        n_laps=int(all_err.size),
        per_stint=per_stint,
    )


def parameter_recovery(learned, truth, names: tuple[str, ...]) -> list[tuple[str, float, float, float]]:
    """Compara parametros estimados contra la verdad sintetica.

    Devuelve (nombre, estimado, verdadero, error relativo en %). Solo tiene
    sentido con el banco sintetico: con datos reales no existe la verdad.
    """
    rows = []
    for n in names:
        est = float(getattr(learned, n))
        ref = float(getattr(truth, n))
        rel = 100.0 * abs(est - ref) / abs(ref) if ref else float("nan")
        rows.append((n, est, ref, rel))
    return rows


def format_report(metrics: list[Metrics]) -> str:
    lines = [HEADER, "-" * len(HEADER)]
    lines.extend(m.as_row() for m in metrics)
    lines.append("")
    lines.append(
        "ViolIn / ViolExtrap = % de vueltas en las que el modelo predice que el "
        "neumatico recupera agarre,\ndentro del stint observado y extrapolando "
        "al horizonte completo. Lo fisicamente correcto es 0%."
    )
    return "\n".join(lines)
