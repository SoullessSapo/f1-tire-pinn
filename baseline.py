"""
THE MODEL THE PINN HAS TO BEAT
==============================

This is the model teams actually use: a degradation rate in seconds per lap,
fitted per compound and conditions. Here it is written as a least-squares
regression.

It is given a FAIR chance on purpose: besides the linear term in time it gets a
quadratic term and the interactions of time with all five context variables. A
straw man that loses for lack of flexibility proves nothing.

And even so it has two properties that are exactly what the comparison is about:

  - INSIDE the measured range it is hard to beat. Over 12-30 laps the true
    curve is gently bent, and a parabola fits a gentle bend very well.

  - OUTSIDE that range nothing holds it down. A parabola fitted to a saturating
    curve keeps curving, so extrapolated far enough it will end up predicting
    that the tire REGAINS grip. That is not a tuning problem: it is what that
    family of functions does.


LEAST SQUARES, IN ONE LINE
--------------------------
We look for the coefficient vector `w` that minimises ||X*w - y||^2, where each
row of X holds the features of one lap and each y is its measured pace loss.
The solution comes from solving (X'X)w = X'y.

The `ridge` term adds a tiny number to the diagonal of X'X. It is insurance: if
two features were nearly identical, X'X would be nearly singular and the system
would have no stable solution. It does not change the result, it prevents the
failure.
"""

from __future__ import annotations

import numpy as np

from physics import LAP_REF, Context


class LinearBaseline:
    """Least squares on features of (time, context)."""

    name = "Linear (classic)"

    def __init__(self, ridge: float = 1e-6):
        """Set up an unfitted model. `ridge` is the diagonal insurance term."""
        self.ridge = ridge
        self.coefficients: np.ndarray | None = None

    @staticmethod
    def _features(tau: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Build the design matrix X.

        Per lap, 13 numbers:
            1                     the intercept
            tau, tau^2            the shape of the curve in time
            the 5 context vars    the level each condition imposes
            tau * (the 5)         how each condition changes the SLOPE

        The interactions in that last group are what let it say "on a hot track
        it degrades faster", which is what makes the comparison honest.
        """
        tau = np.asarray(tau, dtype=float).reshape(-1, 1)
        context = np.asarray(context, dtype=float).reshape(tau.shape[0], -1)

        return np.hstack([
            np.ones_like(tau),      # 1
            tau,                    # 1
            tau ** 2,               # 1
            context,                # 5
            tau * context,          # 5
        ])                          # total: 13 columns

    def fit(self, inputs: np.ndarray, delta: np.ndarray) -> LinearBaseline:
        """`inputs` is the (N, 6) matrix returned by data.flatten()."""
        tau = inputs[:, 0]
        context = inputs[:, 1:]

        X = self._features(tau, context)
        y = np.asarray(delta, dtype=float).ravel()

        gram = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.coefficients = np.linalg.solve(gram, X.T @ y)
        return self

    def predict_stint(self, context: Context, laps: np.ndarray) -> np.ndarray:
        """Same signature as the PINN's, so the evaluator cannot tell them apart."""
        if self.coefficients is None:
            raise RuntimeError("Call fit() before predicting")

        laps = np.asarray(laps, dtype=float).ravel()
        tau = laps / LAP_REF
        repeated_context = np.tile(context.vector().reshape(1, -1), (laps.size, 1))

        return self._features(tau, repeated_context) @ self.coefficients
