"""Two metrics, because they answer different questions.

RMSE says how wrong a model is on the lap it is looking at. That is the obvious
one, and on its own it is not enough.

Monotonicity violations say how often a model predicts the tire REGAINING grip
between one lap and the next. That is physically impossible, no error metric
penalises it, and it is the failure mode that makes a prediction useless for
strategy: a model that says the tire gets better on lap 34 will not be trusted
on lap 20 either. The correct value is 0 %.

It is measured twice: inside the observed stint, and extrapolating to the full
45-lap decision horizon. The second one is where the difference shows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HORIZON = 45


@dataclass
class Metrics:
    name: str
    rmse: float
    mae: float
    violations_in: float
    violations_extrap: float

    def row(self) -> str:
        return (
            f"{self.name:18s} {self.rmse:8.3f} {self.mae:8.3f} "
            f"{100 * self.violations_in:9.1f}% {100 * self.violations_extrap:11.1f}%"
        )


HEADER = f"{'Modelo':18s} {'RMSE':>8s} {'MAE':>8s} {'ViolDentro':>10s} {'ViolExtrap':>12s}"


def _violations(delta: np.ndarray, tol: float = 1e-3) -> tuple[int, int]:
    """Count lap-to-lap steps where the predicted pace IMPROVES."""
    if delta.size < 2:
        return 0, 0
    diffs = np.diff(np.asarray(delta, dtype=float))
    return int((diffs < -tol).sum()), int(diffs.size)


def evaluate(name: str, predict_stint, stints) -> Metrics:
    """Measure any model exposing predict_stint(c, laps)."""
    errors, v_in, t_in, v_ex, t_ex = [], 0, 0, 0, 0

    for s in stints:
        pred = np.asarray(predict_stint(s.c, s.laps), dtype=float).ravel()
        errors.append(pred - s.delta)

        v, t = _violations(pred)
        v_in += v
        t_in += t

        # The model is asked for the full horizon, well past what it saw.
        pred_long = np.asarray(
            predict_stint(s.c, np.arange(1, HORIZON + 1)), dtype=float
        ).ravel()
        v, t = _violations(pred_long)
        v_ex += v
        t_ex += t

    err = np.concatenate(errors)
    return Metrics(
        name=name,
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        violations_in=v_in / t_in if t_in else 0.0,
        violations_extrap=v_ex / t_ex if t_ex else 0.0,
    )
