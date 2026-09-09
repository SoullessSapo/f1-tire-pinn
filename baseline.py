"""The reference model the PINN has to beat.

This is the empirical model teams actually use: a degradation rate in seconds
per lap. Written here as least squares on features of (tau, compound), and
deliberately given a fair chance -- a quadratic term and a compound interaction,
so it is not a straw man that loses for lack of flexibility.

It has two properties worth naming, because they are what the comparison is
about:

  - Inside the measured range it is hard to beat. Over 12-30 laps the true curve
    is gently bent, and a parabola fits a gentle bend very well.
  - Outside it, nothing holds it. A parabola fitted to a saturating curve keeps
    curving, so extrapolated far enough it will predict the tire regaining grip.
    That is not a tuning problem, it is what the function class does.
"""

from __future__ import annotations

import numpy as np


class LinearDegBaseline:
    """Least squares on [1, tau, tau^2, c, tau*c]."""

    name = "Lineal clasico"

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.coef_: np.ndarray | None = None

    @staticmethod
    def _features(tau: np.ndarray, c: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float).reshape(-1, 1)
        c = np.asarray(c, dtype=float).reshape(-1, 1)
        return np.hstack([np.ones_like(tau), tau, tau**2, c, tau * c])

    def fit(self, tau: np.ndarray, c: np.ndarray, delta: np.ndarray) -> LinearDegBaseline:
        x = self._features(tau, c)
        y = np.asarray(delta, dtype=float).ravel()
        gram = x.T @ x + self.ridge * np.eye(x.shape[1])
        self.coef_ = np.linalg.solve(gram, x.T @ y)
        return self

    def predict_stint(self, c: float, laps: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Ajusta el modelo antes de predecir")
        from physics import LAP_REF

        laps = np.asarray(laps, dtype=float).ravel()
        tau = laps / LAP_REF
        return self._features(tau, np.full_like(tau, float(c))) @ self.coef_
