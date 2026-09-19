"""
HOW A MODEL IS JUDGED
=====================

Three metrics, because they answer different questions and no one of them is
enough on its own.


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

Note what the coupled model did NOT change here. dd/dtau = k*(1-d) with k > 0
stays non-negative even with the thermal feedback switched on, so the cliff is
a STEEPENING, not a reversal. A predicted cliff that dips is still a bug.


3) THE CLIFF LAP
----------------
WHERE the tire falls off, which is the entire question a strategist is asking.

A model can have an excellent RMSE and still put the cliff five laps late,
because the cliff occupies few laps and contributes little to a mean squared
error. That is exactly the failure that matters most, and the only way to see
it is to measure it directly.

    the first lap after which the pace loses more than 0.30 s/lap
    for 4 laps running

Both numbers were bought with a mistake. The original criterion was 0.15 s/lap
at a single point, and it fired on 100 % of curves that had no cliff at all as
soon as real timing noise was added: it was measuring noise. Requiring the
slope to be SUSTAINED is what makes the detector a detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import STRATEGY_HORIZON

# What counts as falling off the cliff. See section 3 of the module docstring
# for why a single-point criterion does not work.
CLIFF_THRESHOLD = 0.30      # seconds lost per lap
CLIFF_SUSTAINED = 4         # for this many consecutive laps


@dataclass
class Metrics:
    """The result of evaluating one model over a set of stints."""

    name: str
    rmse: float
    mae: float
    max_error: float
    violations_inside: float
    violations_extrapolating: float
    cliff_rate: float           # fraction of stints where a cliff is predicted
    cliff_error: float          # mean |predicted cliff lap - true cliff lap|

    def row(self) -> str:
        """One aligned line of the results table."""
        cliff_error = (
            "     -" if not np.isfinite(self.cliff_error)
            else f"{self.cliff_error:6.1f}"
        )
        return (
            f"{self.name:18s} {self.rmse:8.3f} {self.mae:8.3f} {self.max_error:9.3f} "
            f"{100 * self.violations_inside:9.1f}% {100 * self.violations_extrapolating:11.1f}% "
            f"{100 * self.cliff_rate:8.0f}% {cliff_error}"
        )


HEADER = (
    f"{'Model':18s} {'RMSE':>8s} {'MAE':>8s} {'MaxErr':>9s} "
    f"{'ViolIn':>10s} {'ViolExtrap':>12s} {'Cliff':>9s} {'CliffErr':>6s}"
)


def cliff_lap(
    delta: np.ndarray,
    threshold: float = CLIFF_THRESHOLD,
    sustained: int = CLIFF_SUSTAINED,
) -> int | None:
    """The lap the tire falls off, or None if it never does.

    `delta` is a pace-loss curve indexed from lap 1. The rule: find the first
    run of `sustained` consecutive laps each losing more than `threshold`
    seconds over the previous one, and report the lap the run starts on.

    Returns a lap NUMBER (1-based), which is why the index gets +2: np.diff
    shifts by one, and lap numbering starts at one rather than zero.
    """
    delta = np.asarray(delta, dtype=float).ravel()
    if delta.size < sustained + 1:
        return None

    per_lap = np.diff(delta)
    for i in range(per_lap.size - sustained + 1):
        if np.all(per_lap[i : i + sustained] > threshold):
            return i + 2
    return None


def _count_violations(delta: np.ndarray, tolerance: float = 1e-3) -> tuple[int, int]:
    """Count lap-to-lap steps where the predicted pace IMPROVES.

    Returns (violations, total_steps). The tolerance avoids counting a
    one-millionth-of-a-second improvement, which would only be numerical noise.
    """
    if delta.size < 2:
        return 0, 0
    differences = np.diff(np.asarray(delta, dtype=float))
    return int((differences < -tolerance).sum()), int(differences.size)


def evaluate(name: str, predict_stint, stints, true_stint=None) -> Metrics:
    """Measure any model exposing predict_stint(context, laps).

    That both models share this signature is not a style detail: it is what
    guarantees they are being asked exactly the same question.

    `true_stint` is optional and has the same signature. When it is given -- on
    the synthetic bench, where the answer is known -- the cliff lap of the
    prediction is compared against the cliff lap of the truth. With real data
    there is nothing to compare against, so that column comes out blank and
    only the cliff RATE is reported.
    """
    errors = []
    viol_inside = steps_inside = 0
    viol_extrap = steps_extrap = 0
    cliffs_found = 0
    cliff_errors: list[float] = []

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

        # --- the cliff, which only exists out at the horizon ---
        predicted_cliff = cliff_lap(long_prediction)
        cliffs_found += predicted_cliff is not None

        if true_stint is not None:
            true_cliff = cliff_lap(
                np.asarray(true_stint(stint.context, horizon), dtype=float).ravel()
            )
            # Only scored where the truth HAS a cliff and the model found one.
            # Counting a miss as an error of zero laps would flatter a model
            # that never predicts a cliff at all; the rate column is what
            # catches that.
            if true_cliff is not None and predicted_cliff is not None:
                cliff_errors.append(abs(predicted_cliff - true_cliff))

    error = np.concatenate(errors)

    return Metrics(
        name=name,
        rmse=float(np.sqrt(np.mean(error ** 2))),
        mae=float(np.mean(np.abs(error))),
        max_error=float(np.max(np.abs(error))),
        violations_inside=viol_inside / steps_inside if steps_inside else 0.0,
        violations_extrapolating=viol_extrap / steps_extrap if steps_extrap else 0.0,
        cliff_rate=cliffs_found / len(stints) if stints else 0.0,
        cliff_error=float(np.mean(cliff_errors)) if cliff_errors else float("nan"),
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
