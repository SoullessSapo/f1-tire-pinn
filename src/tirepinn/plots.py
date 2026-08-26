"""Experiment figures.

All are written as PNGs into the output directory and are meant to drop
straight into the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import PhysicsConfig
from .dataset import StintDataset
from .physics import cliff_lap, pace_loss

COLORS = {
    "PINN": "#d62728",
    "LSTM (black box)": "#1f77b4",
    "Linear (classic)": "#7f7f7f",
}
_DEFAULT_COLOR = "#2ca02c"


def _color(name: str) -> str:
    return COLORS.get(name, _DEFAULT_COLOR)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def plot_stint_grid(
    models: dict,
    data: StintDataset,
    phys: PhysicsConfig,
    path: str | Path,
    max_stints: int = 6,
) -> None:
    """Observed degradation curves against each model's prediction."""
    _style()
    stints = data.stints[:max_stints]
    ncol = 3
    nrow = int(np.ceil(len(stints) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False)

    for ax, stint in zip(axes.ravel(), stints, strict=False):
        ax.plot(stint.laps, stint.delta, "o", ms=3.5, color="black", label="Observed", zorder=5)
        for name, model in models.items():
            pred = np.asarray(model.predict_stint(stint.context, stint.laps)).ravel()
            ax.plot(stint.laps, pred, lw=1.8, color=_color(name), label=name)
        truth = cliff_lap(stint.laps, stint.delta, phys)
        if truth is not None:
            ax.axvline(truth, ls=":", lw=1.2, color="darkorange", label="Observed cliff")
        ax.set_title(f"{stint.stint_id} - {stint.compound}", fontsize=9)
        ax.set_xlabel("Stint lap")
        ax.set_ylabel("Pace loss [s]")

    for ax in axes.ravel()[len(stints) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle("Degradation per stint: prediction vs observation (test set)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def plot_latent_states(
    pinn, data: StintDataset, phys: PhysicsConfig, path: str | Path, n: int = 3
) -> None:
    """Latent states reconstructed by the PINN.

    `d` never appears in the loss function: the network only sees pace loss, and
    wear comes out of enforcing the ODE. `theta` does receive weak supervision
    from the thermal proxy when it is enabled (synthetic bench), at a far lower
    weight than the data term. The figure compares both against the synthetic
    ground truth.
    """
    _style()
    stints = [s for s in data.stints if s.d_true is not None][:n]
    if not stints:
        return
    fig, axes = plt.subplots(2, len(stints), figsize=(4.0 * len(stints), 5.6), squeeze=False)

    for j, stint in enumerate(stints):
        pred = pinn.predict_curve(stint.context, stint.laps)
        axes[0, j].plot(stint.laps, stint.theta_true, "o", ms=3, color="black", label="Ground truth")
        axes[0, j].plot(stint.laps, pred["theta"], lw=1.8, color="#d62728", label="PINN")
        axes[0, j].set_title(f"{stint.stint_id} - {stint.compound}", fontsize=9)
        axes[0, j].set_ylabel(r"$\theta$  (thermal excess)")

        axes[1, j].plot(stint.laps, stint.d_true, "o", ms=3, color="black", label="Ground truth")
        axes[1, j].plot(stint.laps, pred["d"], lw=1.8, color="#d62728", label="PINN")
        axes[1, j].axhline(phys.d_crit, ls=":", lw=1.2, color="darkorange", label=r"$d_{crit}$")
        axes[1, j].set_ylabel("d  (tread consumed)")
        axes[1, j].set_xlabel("Stint lap")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle(
        "Latent states: d is reconstructed from the ODE alone (never observed); "
        "theta carries weak supervision"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_extrapolation(
    models: dict,
    data: StintDataset,
    phys: PhysicsConfig,
    path: str | Path,
    horizon: int | None = None,
    n: int = 3,
) -> None:
    """Behaviour beyond the observed stint.

    This is the figure that separates the PINN from the black box: in the
    data-free zone the PINN keeps integrating the ODE, while the LSTM is left
    with no constraint at all and can return curves that bend downwards.
    """
    _style()
    horizon = horizon or phys.strategy_horizon
    stints = data.stints[:n]
    fig, axes = plt.subplots(1, len(stints), figsize=(4.3 * len(stints), 3.4), squeeze=False)
    laps_long = np.arange(1, horizon + 1)

    for ax, stint in zip(axes.ravel(), stints, strict=False):
        ax.axvspan(1, stint.laps.max(), color="0.92", zorder=0)
        ax.plot(stint.laps, stint.delta, "o", ms=3.5, color="black", label="Observed", zorder=5)
        for name, model in models.items():
            pred = np.asarray(model.predict_stint(stint.context, laps_long)).ravel()
            ax.plot(laps_long, pred, lw=1.8, color=_color(name), label=name)
        ax.axvline(stint.laps.max(), ls="--", lw=1.0, color="0.4")
        ax.set_title(f"{stint.stint_id} - {stint.compound}", fontsize=9)
        ax.set_xlabel("Stint lap")
        ax.set_ylabel("Pace loss [s]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels) + 1)
    fig.suptitle("Extrapolation beyond the data (grey band = observed laps)")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def plot_parameter_convergence(var_history, truth, path: str | Path) -> None:
    """Convergence of the estimated physical parameters towards the truth."""
    if not var_history:
        return
    _style()
    steps = [s for s, _ in var_history]
    names = list(var_history[0][1].keys())
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.4 * nrow), squeeze=False)

    for ax, name in zip(axes.ravel(), names, strict=False):
        values = [v[name] for _, v in var_history]
        ax.plot(steps, values, lw=1.8, color="#d62728", label="Estimated")
        if truth is not None and hasattr(truth, name):
            ax.axhline(
                float(getattr(truth, name)), ls="--", lw=1.2, color="black", label="Ground truth"
            )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Iteration")

    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Inverse problem: recovery of the physical parameters")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_loss_history(losshistory, path: str | Path, labels: list[str] | None = None) -> None:
    """Evolution of each term of the loss function."""
    if losshistory is None:
        return
    _style()
    steps = np.asarray(losshistory.steps)
    losses = np.asarray(losshistory.loss_train)
    labels = labels or [f"L{i + 1}" for i in range(losses.shape[1])]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i in range(losses.shape[1]):
        ax.semilogy(
            steps, losses[:, i], lw=1.5, label=labels[i] if i < len(labels) else f"L{i + 1}"
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Terms of the loss function")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cliff_map(
    pinn, phys: PhysicsConfig, path: str | Path, horizon: int | None = None, res: int = 24
) -> None:
    """Strategy map: expected cliff lap by compound and conditions.

    This is the end product from the pit wall's point of view, and it is only
    possible because the network is parametric: each cell is one forward pass.
    """
    _style()
    horizon = horizon or phys.strategy_horizon
    compounds = [("SOFT", 0.0), ("MEDIUM", 0.5), ("HARD", 1.0)]
    track = np.linspace(0.05, 0.95, res)
    load = np.linspace(0.6, 1.4, res)
    laps = np.arange(1, horizon + 1, dtype=float)
    tau = laps / phys.lap_ref
    params = pinn.learned_params()

    grid_load, grid_track = np.meshgrid(load, track, indexing="ij")
    n_cells = grid_load.size

    grids = []
    for _, comp_idx in compounds:
        contexts = np.stack(
            [
                np.ones(n_cells),
                grid_load.ravel(),
                np.ones(n_cells),
                grid_track.ravel(),
                np.full(n_cells, comp_idx),
            ],
            axis=1,
        )
        # Every cell in a single call: the network is parametric, so the whole
        # map costs one batched forward pass.
        x = np.empty((n_cells * laps.size, contexts.shape[1] + 1))
        x[:, 0] = np.tile(tau, n_cells)
        x[:, 1:] = np.repeat(contexts, laps.size, axis=0)
        _, d = pinn.predict(x)
        delta = pace_loss(d, params).reshape(n_cells, laps.size)

        cells = [cliff_lap(laps, curve, phys) for curve in delta]
        grid = np.array([np.nan if c is None else c for c in cells])
        grids.append(grid.reshape(res, res))

    finite = np.concatenate([g[np.isfinite(g)] for g in grids]) if grids else np.array([1.0])
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (1.0, horizon)

    # Risk scale: red = the cliff arrives early, green = it survives. Cells with
    # no cliff inside the horizon are painted explicit grey rather than white, so
    # they read as "survives" instead of as an empty panel.
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("0.85")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), squeeze=False)
    for ax, (label, _), grid in zip(axes.ravel(), compounds, grids, strict=False):
        im = ax.imshow(
            np.ma.masked_invalid(grid),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=(track[0], track[-1], load[0], load[-1]),
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Track temperature (norm.)")
        ax.set_ylabel("Mechanical load (norm.)")
        ax.set_facecolor("0.85")
        ax.grid(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Cliff lap")
    fig.suptitle(f"Decision map: expected cliff lap (grey = no cliff within {horizon} laps)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
