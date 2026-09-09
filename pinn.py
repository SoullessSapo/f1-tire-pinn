"""The PINN, written directly in PyTorch.

No DeepXDE here, on purpose. A PINN is a small idea wrapped in a lot of
framework, and the idea is worth seeing without the wrapper at least once:

    1. a neural network is a parametrised function  d = N(tau, c ; W)
    2. autograd differentiates it with respect to its INPUT, exactly
    3. so the ODE residual  r = dd/dtau - rate(d, c)  can be evaluated anywhere
    4. adding r^2 to the loss makes the network obey the equation

Step 3 is the whole trick, and it is worth being precise about why it matters:
evaluating the residual needs a point in the domain, and nothing else. It does
NOT need to know the right answer there. So the physics term can be enforced at
points where no data exists -- including laps past the end of every stint in the
dataset, which are exactly the laps a strategist needs predicted.

The loss has three terms:

    L_phys   residual of the ODE, on 2 000 collocation points
    L_data   fit to the measured pace loss, on observed laps only
    L_ic     initial condition d(0) = 0

L_ic being a loss term is a known weakness, kept here because it is the textbook
formulation and because seeing it fail is instructive: it is satisfied
approximately, never exactly, and it competes with the other two terms for
gradient. ROADMAP.md step 4 replaces it with an output transform that makes it
exact and free.

One physical constant, kw, is estimated jointly with the network weights -- an
inverse problem in miniature. It is parametrised as log(kw) so it stays positive
by construction: there is no such thing as a negative wear rate, and asserting
that through the parametrisation is stronger than hoping the optimiser respects
it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from physics import GROUND_TRUTH, wear_rate


class MLP(nn.Module):
    """Multilayer perceptron: 2 inputs -> 2 hidden layers of 32 -> 1 output.

    This is the entire "neural network" part. Each layer is an affine map
    followed by tanh; without that nonlinearity the whole stack would collapse
    into a single linear map and could not represent a curve.

    1 185 parameters, which is tiny. It can afford to be: the physics carries
    most of the structure, so the network is not being asked to memorise
    anything.
    """

    def __init__(self, hidden: int = 32, layers: int = 2):
        super().__init__()
        dims = [2] + [hidden] * layers + [1]
        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, tau: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([tau, c], dim=1))


class TirePINN:
    """Wear PINN with one estimated physical constant."""

    def __init__(
        self,
        kappa: float = GROUND_TRUTH.kappa,
        gamma1: float = GROUND_TRUTH.gamma1,
        kw_init: float = 0.60,      # deliberately wrong: the truth is 1.50
        tau_max: float = 1.50,      # 45 laps, the strategy horizon
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.net = MLP()
        # Estimated jointly with the weights. Stored as log(kw) so kw > 0 always.
        self.log_kw = nn.Parameter(torch.tensor(float(np.log(kw_init))))
        self.kappa = kappa
        self.gamma1 = gamma1
        self.tau_max = tau_max
        self.history: list[dict[str, float]] = []

    @property
    def kw(self) -> torch.Tensor:
        return torch.exp(self.log_kw)

    @property
    def kw_value(self) -> float:
        """The estimate as a plain float, outside the autograd graph."""
        return float(torch.exp(self.log_kw).detach())

    # ------------------------------------------------------------------
    def predict_d(self, tau: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(tau, c)

    def residual(self, tau: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """r = dd/dtau - rate(d, c). Zero everywhere iff the ODE holds.

        `create_graph=True` is what allows the residual itself to be
        differentiated with respect to the weights, which is what the optimiser
        needs. Without it the physics term would contribute no gradient.
        """
        tau = tau.requires_grad_(True)
        d = self.predict_d(tau, c)
        dd_dtau = torch.autograd.grad(
            d, tau, grad_outputs=torch.ones_like(d), create_graph=True
        )[0]
        return dd_dtau - wear_rate(d, c, self.kw, self.kappa)

    def collocation(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Points where the physics is enforced.

        Uniform over tau up to the full strategy horizon -- beyond the longest
        stint in the data -- and over the compound axis. No labels needed, which
        is the entire reason this term can cover ground the data never does.
        """
        tau = torch.rand(n, 1) * self.tau_max
        c = torch.rand(n, 1)
        return tau, c

    # ------------------------------------------------------------------
    def train(
        self,
        tau_obs: np.ndarray,
        c_obs: np.ndarray,
        delta_obs: np.ndarray,
        iterations: int = 6000,
        n_collocation: int = 2000,
        lr: float = 5e-3,
        w_phys: float = 1.0,
        w_data: float = 10.0,
        w_ic: float = 10.0,
        record_every: int = 25,
        log_every: int = 500,
    ) -> None:
        """Adam only. See ROADMAP.md step 5 for why that stops being enough."""
        tau_t = torch.tensor(tau_obs, dtype=torch.float32)
        c_t = torch.tensor(c_obs, dtype=torch.float32)
        delta_t = torch.tensor(delta_obs, dtype=torch.float32)

        params = list(self.net.parameters()) + [self.log_kw]
        opt = torch.optim.Adam(params, lr=lr)

        for it in range(1, iterations + 1):
            opt.zero_grad()

            # --- physics: the ODE, on points with no data ---
            tau_c, c_c = self.collocation(n_collocation)
            loss_phys = (self.residual(tau_c, c_c) ** 2).mean()

            # --- data: the only thing actually measured ---
            delta_pred = self.gamma1 * self.predict_d(tau_t, c_t)
            loss_data = ((delta_pred - delta_t) ** 2).mean()

            # --- initial condition: a brand-new tire has consumed no tread ---
            c_ic = torch.rand(64, 1)
            loss_ic = (self.predict_d(torch.zeros_like(c_ic), c_ic) ** 2).mean()

            loss = w_phys * loss_phys + w_data * loss_data + w_ic * loss_ic
            loss.backward()
            opt.step()

            if it % record_every == 0 or it == 1:
                # .item() rather than float(): the tensors are still attached to
                # the autograd graph here, and float() warns about it.
                record = {
                    "iter": it,
                    "total": loss.item(),
                    "phys": loss_phys.item(),
                    "data": loss_data.item(),
                    "ic": loss_ic.item(),
                    "kw": self.kw_value,
                }
                self.history.append(record)
                if it % log_every == 0 or it == 1:
                    print(
                        f"  {it:6d}  total {record['total']:9.5f}  "
                        f"fisica {record['phys']:8.5f}  datos {record['data']:8.5f}  "
                        f"ci {record['ic']:8.5f}  kw {record['kw']:6.3f}"
                    )

    # ------------------------------------------------------------------
    def predict_stint(self, c: float, laps: np.ndarray) -> np.ndarray:
        """Predicted pace loss. Same signature as the baseline, so the evaluator
        measures both with the same yardstick."""
        from physics import LAP_REF

        laps = np.asarray(laps, dtype=float).ravel()
        tau = torch.tensor((laps / LAP_REF).reshape(-1, 1), dtype=torch.float32)
        c_t = torch.full_like(tau, float(c))
        with torch.no_grad():
            d = self.predict_d(tau, c_t)
        return (self.gamma1 * d).numpy().ravel()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters()) + 1
