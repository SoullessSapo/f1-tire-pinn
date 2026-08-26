"""Modelos de referencia contra los que se mide la PINN.

Dos baselines, elegidos porque representan los dos extremos del estado del arte
descrito en el proyecto:

- `LinearDegBaseline`: el modelo empirico que usan los equipos, una tasa de
  degradacion en segundos por vuelta ajustada por compuesto y condiciones. Es
  rapido e interpretable pero no puede representar el cliff.

- `LSTMBaseline`: la red recurrente de caja negra. Tiene capacidad de sobra
  para capturar la no linealidad, pero nada la obliga a respetar la
  termodinamica; en particular puede predecir que el neumatico *recupera*
  agarre, que es fisicamente imposible.

Ambos exponen la misma interfaz que la PINN (`predict_stint`), de modo que
`evaluate.py` los mide exactamente con la misma vara.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import PhysicsConfig
from .dataset import StintDataset


class LinearDegBaseline:
    """Regresion ridge sobre caracteristicas polinomicas de vuelta y contexto.

    Es la formalizacion del modelo clasico de degradacion lineal, generosamente
    ampliado con un termino cuadratico e interacciones vuelta-contexto para no
    hacerlo de paja.
    """

    name = "Lineal (clasico)"

    def __init__(self, phys: PhysicsConfig, ridge: float = 1e-3):
        self.phys = phys
        self.ridge = ridge
        self.coef_: np.ndarray | None = None

    @staticmethod
    def _features(tau: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Diseno: constante, tau, tau^2, contexto e interacciones con tau."""
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
            raise RuntimeError("Ajusta el modelo antes de predecir")
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
    """Red recurrente sobre la secuencia de vueltas del stint.

    Entrada por vuelta: (tau, contexto). Salida: perdida de ritmo de esa vuelta.
    Es el representante de los metodos de aprendizaje profundo puro citados en
    el estado del arte.
    """

    name = "LSTM (caja negra)"

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
        """Empaqueta los stints en un tensor con relleno y mascara de validez."""
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
            # La perdida solo cuenta las vueltas reales, no el relleno.
            loss = ((pred - y) ** 2 * mask).sum() / mask.sum()
            loss.backward()
            opt.step()
            self.history.append(float(loss.item()))
        self.net.eval()
        return self

    def predict_stint(self, context: np.ndarray, laps: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Ajusta el modelo antes de predecir")
        laps = np.asarray(laps, dtype=float).ravel()
        tau = (laps / self.phys.lap_ref).reshape(-1, 1)
        ctx = np.tile(np.asarray(context, dtype=float).reshape(1, -1), (tau.shape[0], 1))
        seq = np.hstack([tau, ctx]).astype(np.float32)[None, ...]
        with torch.no_grad():
            return self.net(torch.from_numpy(seq)).numpy().ravel()
