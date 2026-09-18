"""
HOW A MODEL IS JUDGED
=====================

Two metrics, because they answer different questions and one alone is not
enough.


1) RMSE (and MAE)
-----------------
How wrong the model is about the lap it is looking at right now.

    RMSE = sqrt( mean( (prediction - measurement)^2 ) )

Squaring makes large errors count more than small ones, and the square root
brings it back to seconds. MAE is the mean absolute error: easier to interpret,
less sensitive to one extreme case.

It is the obvious metric. On its own it is not sufficient.


2) MONOTONICITY VIOLATIONS
--------------------------
How often the model predicts that the tire REGAINS grip from one lap to the
next.

That is physically impossible: rubber does not come back. And the important
part is that NO error metric penalises it. A model can have a fine RMSE and
still say that on lap 34 the car will be faster than on lap 33, which makes the
prediction useless for a decision: if it says that on lap 34, you will not trust
what it says on lap 20 either.

The correct value is 0 %.

It is measured TWICE:
  - inside the observed stint
  - extrapolating to the full 45-lap horizon

The second is where the models separate, because that is where no data holds
the curve down and all that is left is what the model believes the world is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import STRATEGY_HORIZON


@dataclass
class Metrics:
    """The result of evaluating one model over a set of stints."""

    name: str
    rmse: float
    mae: float
    max_error: float
    violations_inside: float
    violations_extrapolating: float

    def row(self) -> str:
        """One aligned line of the results table."""
        return (
            f"{self.name:18s} {self.rmse:8.3f} {self.mae:8.3f} {self.max_error:9.3f} "
            f"{100 * self.violations_inside:9.1f}% {100 * self.violations_extrapolating:11.1f}%"
        )


HEADER = (
    f"{'Model':18s} {'RMSE':>8s} {'MAE':>8s} {'MaxErr':>9s} "
    f"{'ViolIn':>10s} {'ViolExtrap':>12s}"
)


def _count_violations(delta: np.ndarray, tolerance: float = 1e-3) -> tuple[int, int]:
    """Count lap-to-lap steps where the predicted pace IMPROVES.

    Returns (violations, total_steps). The tolerance avoids counting a
    one-millionth-of-a-second improvement, which would only be numerical noise.
    """
    if delta.size < 2:
        return 0, 0
    differences = np.diff(np.asarray(delta, dtype=float))
    return int((differences < -tolerance).sum()), int(differences.size)


def evaluate(name: str, predict_stint, stints) -> Metrics:
    """Measure any model exposing predict_stint(context, laps).

    That both models share this signature is not a style detail: it is what
    guarantees they are being asked exactly the same question.
    """
    errors = []
    viol_inside = steps_inside = 0
    viol_extrap = steps_extrap = 0

    horizon = np.arange(1, STRATEGY_HORIZON + 1)

    for stint in stints:
        # --- inside the observed stint ---
        prediction = np.asarray(
            predict_stint(stint.context, stint.laps), dtype=float
        ).ravel()
        errors.append(prediction - stint.delta)

        v, t = _count_violations(prediction)
        viol_inside += v
        steps_inside += t

        # --- extrapolating: the whole stint is asked for, out to the horizon ---
        long_prediction = np.asarray(
            predict_stint(stint.context, horizon), dtype=float
        ).ravel()

        v, t = _count_violations(long_prediction)
        viol_extrap += v
        steps_extrap += t

    error = np.concatenate(errors)

    return Metrics(
        name=name,
        rmse=float(np.sqrt(np.mean(error ** 2))),
        mae=float(np.mean(np.abs(error))),
        max_error=float(np.max(np.abs(error))),
        violations_inside=viol_inside / steps_inside if steps_inside else 0.0,
        violations_extrapolating=viol_extrap / steps_extrap if steps_extrap else 0.0,
    )


def parameter_recovery(learned, truth, names) -> list[tuple]:
    """Compare the estimated constants against the true ones.

    This only makes sense with synthetic data. With real data there is no
    ground truth to compare against, and that is exactly why a synthetic bench
    exists.

    Returns (name, estimated, true, relative error in %).
    """
    rows = []
    for name in names:
        estimated = float(getattr(learned, name))
        true = float(getattr(truth, name))
        error = 100.0 * abs(estimated - true) / abs(true) if true else float("nan")
        rows.append((name, estimated, true, error))
    return rows
