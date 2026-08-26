"""Figuras del experimento.

Todas se guardan como PNG en el directorio de salida y estan pensadas para
entrar directamente en el informe.
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
    "LSTM (caja negra)": "#1f77b4",
    "Lineal (clasico)": "#7f7f7f",
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
    """Curvas de degradacion observadas frente a las predichas por cada modelo."""
    _style()
    stints = data.stints[:max_stints]
    ncol = 3
    nrow = int(np.ceil(len(stints) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False)

    for ax, stint in zip(axes.ravel(), stints, strict=False):
        ax.plot(stint.laps, stint.delta, "o", ms=3.5, color="black", label="Observado", zorder=5)
        for name, model in models.items():
            pred = np.asarray(model.predict_stint(stint.context, stint.laps)).ravel()
            ax.plot(stint.laps, pred, lw=1.8, color=_color(name), label=name)
        truth = cliff_lap(stint.laps, stint.delta, phys)
        if truth is not None:
            ax.axvline(truth, ls=":", lw=1.2, color="darkorange", label="Cliff observado")
        ax.set_title(f"{stint.stint_id} - {stint.compound}", fontsize=9)
        ax.set_xlabel("Vuelta del stint")
        ax.set_ylabel("Perdida de ritmo [s]")

    for ax in axes.ravel()[len(stints) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle("Degradacion por stint: prediccion vs observacion (conjunto de prueba)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def plot_latent_states(pinn, data: StintDataset, phys: PhysicsConfig, path: str | Path, n: int = 3) -> None:
    """Estados latentes reconstruidos por la PINN.

    `d` no aparece nunca en la funcion de perdida: la red solo ve la perdida de
    ritmo, y el desgaste sale de imponer la EDO. `theta` si recibe supervision
    debil del proxy termico cuando esta activado (banco sintetico), con un peso
    muy inferior al del termino de datos. La figura compara ambos contra la
    verdad sintetica.
    """
    _style()
    stints = [s for s in data.stints if s.d_true is not None][:n]
    if not stints:
        return
    fig, axes = plt.subplots(2, len(stints), figsize=(4.0 * len(stints), 5.6), squeeze=False)

    for j, stint in enumerate(stints):
        pred = pinn.predict_curve(stint.context, stint.laps)
        axes[0, j].plot(stint.laps, stint.theta_true, "o", ms=3, color="black", label="Verdad")
        axes[0, j].plot(stint.laps, pred["theta"], lw=1.8, color="#d62728", label="PINN")
        axes[0, j].set_title(f"{stint.stint_id} - {stint.compound}", fontsize=9)
        axes[0, j].set_ylabel(r"$\theta$  (exceso termico)")

        axes[1, j].plot(stint.laps, stint.d_true, "o", ms=3, color="black", label="Verdad")
        axes[1, j].plot(stint.laps, pred["d"], lw=1.8, color="#d62728", label="PINN")
        axes[1, j].axhline(phys.d_crit, ls=":", lw=1.2, color="darkorange", label=r"$d_{crit}$")
        axes[1, j].set_ylabel("d  (banda consumida)")
        axes[1, j].set_xlabel("Vuelta del stint")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle(
        "Estados latentes: d se reconstruye solo con la EDO (nunca se observa); "
        "theta lleva supervision debil"
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
    """Comportamiento mas alla del stint observado.

    Es la figura que separa la PINN de la caja negra: en la zona sin datos la
    PINN sigue integrando la EDO, mientras que la LSTM queda sin ninguna
    restriccion y puede devolver curvas que bajan.
    """
    _style()
    horizon = horizon or phys.strategy_horizon
    stints = data.stints[:n]
    fig, axes = plt.subplots(1, len(stints), figsize=(4.3 * len(stints), 3.4), squeeze=False)
    laps_long = np.arange(1, horizon + 1)

    for ax, stint in zip(axes.ravel(), stints, strict=False):
        ax.axvspan(1, stint.laps.max(), color="0.92", zorder=0)
        ax.plot(stint.laps, stint.delta, "o", ms=3.5, color="black", label="Observado", zorder=5)
        for name, model in models.items():
            pred = np.asarray(model.predict_stint(stint.context, laps_long)).ravel()
            ax.plot(laps_long, pred, lw=1.8, color=_color(name), label=name)
        ax.axvline(stint.laps.max(), ls="--", lw=1.0, color="0.4")
        ax.set_title(f"{stint.stint_id} - {stint.compound}", fontsize=9)
        ax.set_xlabel("Vuelta del stint")
        ax.set_ylabel("Perdida de ritmo [s]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels) + 1)
    fig.suptitle("Extrapolacion fuera de los datos (zona gris = vueltas observadas)")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def plot_parameter_convergence(var_history, truth, path: str | Path) -> None:
    """Convergencia de los parametros fisicos estimados hacia la verdad."""
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
        ax.plot(steps, values, lw=1.8, color="#d62728", label="Estimado")
        if truth is not None and hasattr(truth, name):
            ax.axhline(float(getattr(truth, name)), ls="--", lw=1.2, color="black", label="Verdad")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Iteracion")

    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Problema inverso: recuperacion de los parametros fisicos")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_loss_history(losshistory, path: str | Path, labels: list[str] | None = None) -> None:
    """Evolucion de cada termino de la funcion de perdida."""
    if losshistory is None:
        return
    _style()
    steps = np.asarray(losshistory.steps)
    losses = np.asarray(losshistory.loss_train)
    labels = labels or [f"L{i + 1}" for i in range(losses.shape[1])]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i in range(losses.shape[1]):
        ax.semilogy(steps, losses[:, i], lw=1.5, label=labels[i] if i < len(labels) else f"L{i + 1}")
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Perdida (escala log)")
    ax.set_title("Terminos de la funcion de perdida")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cliff_map(
    pinn, phys: PhysicsConfig, path: str | Path, horizon: int | None = None, res: int = 24
) -> None:
    """Mapa estrategico: vuelta esperada del cliff segun compuesto y condiciones.

    Es el producto final desde el punto de vista del muro de boxes, y solo es
    posible porque la red es parametrica: cada celda es un paso forward.
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
        # Todas las celdas en una sola llamada: la red es paramétrica, asi que
        # el mapa completo cuesta un unico paso forward por lotes.
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

    # Escala de riesgo: rojo = el cliff llega pronto, verde = aguanta. Las
    # celdas sin cliff dentro del horizonte se pintan en gris explicito, no en
    # blanco, para que se lean como "aguanta" y no como panel vacio.
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
        ax.set_xlabel("Temperatura de pista (norm.)")
        ax.set_ylabel("Carga mecanica (norm.)")
        ax.set_facecolor("0.85")
        ax.grid(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Vuelta del cliff")
    fig.suptitle(
        f"Mapa de decision: vuelta esperada del cliff "
        f"(gris = sin cliff en {horizon} vueltas)"
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
