"""Reference models the PINN is measured against.

Two baselines, chosen because they represent the two extremes of the state of
the art described in the project:

- `LinearDegBaseline`: the empirical model teams use, a degradation rate in
  seconds per lap fitted per compound and conditions. Fast and interpretable,
  but it cannot represent the cliff.

- `LSTMBaseline`: the black-box recurrent network. It has ample capacity to
  capture the nonlinearity, but nothing forces it to respect thermodynamics; in
  particular it can predict that the tire *regains* grip, which is physically
  impossible.

Both expose the same interface as the PINN (`predict_stint`), so `evaluate.py`
measures them with exactly the same yardstick.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import PhysicsConfig
from .dataset import StintDataset


class LinearDegBaseline:
    """Ridge regression on polynomial features of lap and context.

    This is the formalisation of the classic linear degradation model,
    generously extended with a quadratic term and lap-context interactions so it
    is not a straw man.
    """

    name = "Linear (classic)"

    def __init__(self, phys: PhysicsConfig, ridge: float = 1e-3):
        self.phys = phys
        self.ridge = ridge
        self.coef_: np.ndarray | None = None

    @staticmethod
    def _features(tau: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Design matrix: constant, tau, tau^2, context and its interactions with tau."""
        tau = tau.reshape(-1, 1)
        ctx = np.tile(np.asarray(context).reshape(1, -1), (tau.shape[0], 1))
        return np.hstack([np.ones_like(tau), tau, tau**2, ctx, tau * ctx])

    def fit(self, data: StintDataset) -> LinearDegBaseline:
        rows, targets = [], []
        for stint in data.stints:
            rows.append(self._features(stint.tau(self.phys), stint.context))
            targets.append(stint.delta)
        x = np.vstack(rows)
        y = np.concatenate(targets)
        gram = x.T @ x + self.ridge * np.eye(x.shape[1])
        self.coef_ = np.linalg.solve(gram, x.T @ y)
        return self

    def predict_stint(self, context: np.ndarray, laps: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Fit the model before predicting")
        tau = np.asarray(laps, dtype=float).ravel() / self.phys.lap_ref
        return self._features(tau, context) @ self.coef_


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


class LSTMBaseline:
    """Recurrent network over the stint's lap sequence.

    Input per lap: (tau, context). Output: that lap's pace loss. It stands in
    for the pure deep-learning methods cited in the state of the art.
    """

    name = "LSTM (black box)"

    def __init__(
        self,
        phys: PhysicsConfig,
        hidden: int = 48,
        layers: int = 1,
        epochs: int = 600,
        lr: float = 5e-3,
        seed: int = 0,
    ):
        self.phys = phys
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.net: _LSTMNet | None = None
        self.history: list[float] = []

    def _sequences(self, data: StintDataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack the stints into a padded tensor plus a validity mask."""
        max_len = max(s.n_laps for s in data.stints)
        n_feat = 1 + data.stints[0].context.size
        x = np.zeros((len(data.stints), max_len, n_feat), dtype=np.float32)
        y = np.zeros((len(data.stints), max_len), dtype=np.float32)
        mask = np.zeros((len(data.stints), max_len), dtype=np.float32)
        for i, stint in enumerate(data.stints):
            n = stint.n_laps
            x[i, :n] = stint.inputs(self.phys).astype(np.float32)
            y[i, :n] = stint.delta.astype(np.float32)
            mask[i, :n] = 1.0
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)

    def fit(self, data: StintDataset) -> LSTMBaseline:
        torch.manual_seed(self.seed)
        x, y, mask = self._sequences(data)
        self.net = _LSTMNet(x.shape[-1], self.hidden, self.layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.net(x)
            # The loss only counts real laps, not the padding.
            loss = ((pred - y) ** 2 * mask).sum() / mask.sum()
            loss.backward()
            opt.step()
            self.history.append(float(loss.item()))
        self.net.eval()
        return self

    def predict_stint(self, context: np.ndarray, laps: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Fit the model before predicting")
        laps = np.asarray(laps, dtype=float).ravel()
        tau = (laps / self.phys.lap_ref).reshape(-1, 1)
        ctx = np.tile(np.asarray(context, dtype=float).reshape(1, -1), (tau.shape[0], 1))
        seq = np.hstack([tau, ctx]).astype(np.float32)[None, ...]
        with torch.no_grad():
            return self.net(torch.from_numpy(seq)).numpy().ravel()
