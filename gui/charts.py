"""Interactive Plotly figures for the GUI.

The PNGs in `plots.py` are for the report; these are for exploring. Every
figure is hoverable, zoomable and built from the same model interfaces the
evaluation uses (`predict_stint`, `predict_curve`), so what is shown here is what
was measured.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tirepinn.config import CONTEXT_NAMES, PhysicsConfig
from tirepinn.physics import wear_lap

# Categorical slots, in fixed order: one colour per model, always the same one.
MODEL_COLORS = {
    "PINN": "#2a78d6",
    "Linear (classic)": "#eb6834",
    "LSTM (black box)": "#1baf7a",
}
MODEL_DASH = {"PINN": "solid", "Linear (classic)": "dash", "LSTM (black box)": "dot"}
# Compounds keep the colours every F1 viewer already knows (hard is grey, not
# white, so it stays visible on a light background).
COMPOUND_COLORS = {
    "SOFT": "#e34948",
    "MEDIUM": "#eda100",
    "HARD": "#6b6a66",
    "INTERMEDIATE": "#008300",
    "WET": "#2a78d6",
}
OBSERVED = "#52514e"
TERM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#4a3aa7"]
SEQUENTIAL = [
    [0.0, "#104281"],
    [0.25, "#256abf"],
    [0.5, "#3987e5"],
    [0.75, "#86b6ef"],
    [1.0, "#cde2fb"],
]
CONTEXT_LABELS = {
    "q_fric": "Energía de fricción q",
    "load": "Carga mecánica λ",
    "speed": "Velocidad v",
    "track_temp": "Temp. de pista T_trk",
    "compound": "Compuesto c",
}


def _layout(fig: go.Figure, height: int = 420, title: str | None = None, **kwargs) -> go.Figure:
    """Shared look.

    The title is stored in `meta` rather than drawn by Plotly: the interface
    renders it as text above the chart, which keeps it clear of the legend.
    Figures with subplot titles get a vertical legend on the right, for the same
    reason.
    """
    has_subplot_titles = bool(fig.layout.annotations)
    legend = (
        dict(orientation="v", yanchor="top", y=1, x=1.02, xanchor="left")
        if has_subplot_titles
        else dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left")
    )
    fig.update_layout(
        height=height,
        meta={"title": title},
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=legend,
        **kwargs,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def title_of(fig: go.Figure) -> str | None:
    meta = fig.layout.meta
    return meta.get("title") if isinstance(meta, dict) else None


def _compound_color(name: str) -> str:
    return COMPOUND_COLORS.get(name.upper(), OBSERVED)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def dataset_curves(data, highlight: set[str] | None = None) -> go.Figure:
    """Every stint's observed pace loss, coloured by compound."""
    fig = go.Figure()
    shown = set()
    for s in data.stints:
        dim = highlight is not None and s.stint_id not in highlight
        fig.add_trace(
            go.Scatter(
                x=s.laps,
                y=s.delta,
                mode="lines+markers",
                line=dict(width=1.5, color=_compound_color(s.compound)),
                marker=dict(size=5),
                opacity=0.25 if dim else 0.85,
                name=s.compound,
                legendgroup=s.compound,
                showlegend=s.compound not in shown,
                customdata=[[s.stint_id, s.driver]] * s.n_laps,
                hovertemplate="%{customdata[0]} (%{customdata[1]})<br>vuelta %{x:.0f}"
                "<br>pérdida %{y:.2f} s<extra></extra>",
            )
        )
        shown.add(s.compound)
    fig.update_xaxes(title="Vuelta del stint")
    fig.update_yaxes(title="Pérdida de ritmo δ [s]")
    fig = _layout(fig, 460, title="Pérdida de ritmo observada en cada stint")
    fig.update_layout(hovermode="closest")
    return fig


def context_histograms(data) -> go.Figure:
    """Distribution of each context variable across the stints (small multiples)."""
    ctx = data.contexts()
    fig = make_subplots(
        rows=1, cols=len(CONTEXT_NAMES), subplot_titles=[CONTEXT_LABELS[n] for n in CONTEXT_NAMES]
    )
    for i, _ in enumerate(CONTEXT_NAMES):
        fig.add_trace(
            go.Histogram(
                x=ctx[:, i],
                nbinsx=15,
                marker=dict(color="#2a78d6", line=dict(width=1, color="white")),
                showlegend=False,
                hovertemplate="%{x}<br>%{y} stints<extra></extra>",
            ),
            row=1,
            col=i + 1,
        )
    fig.update_yaxes(title="Stints", row=1, col=1)
    fig = _layout(fig, 280, title="Distribución del contexto (normalizado)")
    fig.update_layout(hovermode="closest", bargap=0.05)
    return fig


def stint_lengths(data) -> go.Figure:
    fig = go.Figure()
    for compound in sorted({s.compound for s in data.stints}):
        fig.add_trace(
            go.Histogram(
                x=[s.n_laps for s in data.stints if s.compound == compound],
                name=compound,
                marker=dict(color=_compound_color(compound), line=dict(width=1, color="white")),
                xbins=dict(size=2),
            )
        )
    fig.update_xaxes(title="Vueltas por stint")
    fig.update_yaxes(title="Stints")
    fig = _layout(fig, 300, title="Longitud de los stints por compuesto", barmode="stack")
    fig.update_layout(hovermode="closest")
    return fig


def stint_detail(stint, phys: PhysicsConfig) -> go.Figure:
    """One stint: observed pace, plus the hidden true states on the synthetic bench."""
    has_truth = stint.d_true is not None
    rows = 3 if has_truth else 1
    titles = ["Pérdida de ritmo δ [s]"] + (
        ["Temperatura latente θ (verdad)", "Desgaste latente d (verdad)"] if has_truth else []
    )
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=titles,
                        vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(x=stint.laps, y=stint.delta, mode="markers", name="Observado",
                   marker=dict(size=8, color=OBSERVED)),
        row=1, col=1,
    )
    if stint.delta_true is not None:
        fig.add_trace(
            go.Scatter(x=stint.laps, y=stint.delta_true, mode="lines", name="Sin ruido",
                       line=dict(width=2, color="#2a78d6")),
            row=1, col=1,
        )
    if has_truth:
        fig.add_trace(go.Scatter(x=stint.laps, y=stint.theta_true, mode="lines", name="θ",
                                 line=dict(width=2, color="#eb6834")), row=2, col=1)
        fig.add_trace(go.Scatter(x=stint.laps, y=stint.d_true, mode="lines", name="d",
                                 line=dict(width=2, color="#1baf7a")), row=3, col=1)
        fig.add_hline(y=phys.d_crit, line=dict(dash="dash", width=1, color=OBSERVED),
                      annotation_text=f"d_crit = {phys.d_crit:g}", row=3, col=1)
    fig.update_xaxes(title="Vuelta del stint", row=rows, col=1)
    return _layout(fig, 250 + 170 * rows, title=f"Stint {stint.stint_id} · {stint.driver} · {stint.compound}")


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def loss_curves(steps, terms, labels) -> go.Figure:
    fig = go.Figure()
    arr = np.asarray(terms, dtype=float)
    if arr.size:
        fig.add_trace(go.Scatter(x=steps, y=arr.sum(axis=1), name="Total",
                                 line=dict(width=2.5, color=OBSERVED)))
        for i in range(arr.shape[1]):
            fig.add_trace(go.Scatter(x=steps, y=arr[:, i], name=labels[i],
                                     line=dict(width=2, color=TERM_COLORS[i % len(TERM_COLORS)])))
    fig.update_yaxes(type="log", title="Pérdida (escala log)", exponentformat="power")
    fig.update_xaxes(title="Iteración")
    return _layout(fig, 400, title="Términos de la función de pérdida")


def parameter_traces(trace, names, truth=None, initial=None) -> go.Figure:
    """Convergence of each free physical parameter (small multiples, own y scale)."""
    names = list(names)
    if not names:
        return _layout(go.Figure(), 200, title="Sin parámetros libres")
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names,
                        vertical_spacing=0.12, horizontal_spacing=0.07)
    steps = [s for s, _ in trace]
    for k, name in enumerate(names):
        r, c = k // cols + 1, k % cols + 1
        fig.add_trace(
            go.Scatter(x=steps, y=[p[name] for _, p in trace], name="Estimado",
                       line=dict(width=2, color="#2a78d6"), showlegend=k == 0,
                       legendgroup="est"),
            row=r, col=c,
        )
        if truth is not None and name in truth:
            fig.add_hline(y=truth[name], line=dict(dash="dash", width=1.5, color=OBSERVED),
                          row=r, col=c)
    title = "Problema inverso: parámetros físicos durante el entrenamiento"
    if truth is not None:
        title += " (línea discontinua = valor real)"
    fig.update_layout(showlegend=False)
    return _layout(fig, 230 * rows + 60, title=title)


def lstm_loss(history) -> go.Figure:
    fig = go.Figure(go.Scatter(y=history, mode="lines", name="MSE",
                               line=dict(width=2, color=MODEL_COLORS["LSTM (black box)"])))
    fig.update_yaxes(type="log", title="MSE", exponentformat="power")
    fig.update_xaxes(title="Época")
    return _layout(fig, 300, title="Pérdida de la LSTM (línea base)")


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
def metric_bars(metrics) -> go.Figure:
    fig = go.Figure()
    for label, attr in (("RMSE", "rmse"), ("MAE", "mae")):
        fig.add_trace(go.Bar(
            x=[m.name for m in metrics], y=[getattr(m, attr) for m in metrics], name=label,
            text=[f"{getattr(m, attr):.3f}" for m in metrics], textposition="outside",
            marker=dict(color="#2a78d6" if attr == "rmse" else "#86b6ef",
                        line=dict(width=2, color="rgba(0,0,0,0)")),
        ))
    fig.update_yaxes(title="Error [s]")
    fig = _layout(fig, 360, title="Error sobre stints no vistos (menor es mejor)", barmode="group")
    fig.update_layout(hovermode="closest", bargap=0.3, bargroupgap=0.05)
    return fig


def violation_bars(metrics) -> go.Figure:
    fig = go.Figure()
    for label, attr, color in (("Dentro de lo observado", "violation_rate", "#eb6834"),
                               ("Extrapolando", "extrap_violation_rate", "#f4a582")):
        fig.add_trace(go.Bar(
            x=[m.name for m in metrics], y=[100 * getattr(m, attr) for m in metrics], name=label,
            text=[f"{100 * getattr(m, attr):.1f}%" for m in metrics], textposition="outside",
            marker=dict(color=color),
        ))
    fig.update_yaxes(title="% de vueltas", rangemode="tozero")
    fig = _layout(fig, 360, title="Violaciones físicas: el neumático «recupera» agarre (ideal 0 %)",
                  barmode="group")
    fig.update_layout(hovermode="closest", bargap=0.3, bargroupgap=0.05)
    return fig


def per_stint_rmse(metrics) -> go.Figure:
    fig = go.Figure()
    for m in metrics:
        ids = list(m.per_stint)
        fig.add_trace(go.Bar(x=ids, y=[m.per_stint[i] for i in ids], name=m.name,
                             marker=dict(color=MODEL_COLORS.get(m.name, OBSERVED))))
    fig.update_yaxes(title="RMSE [s]")
    fig.update_xaxes(title="Stint de prueba", tickangle=-45)
    fig = _layout(fig, 380, title="RMSE por stint de prueba", barmode="group")
    fig.update_layout(hovermode="x unified", bargap=0.25)
    return fig


def stint_predictions(stint, models: dict, phys: PhysicsConfig, horizon: int) -> go.Figure:
    """Observed laps plus every model's prediction, extrapolated to the horizon."""
    laps = np.arange(1, horizon + 1, dtype=float)
    fig = go.Figure()
    fig.add_vrect(x0=0.5, x1=stint.laps.max() + 0.5, fillcolor="rgba(128,128,128,0.12)",
                  line_width=0, annotation_text="observado", annotation_position="top left")
    if stint.delta_true is not None:
        full_truth = stint.delta_true
        fig.add_trace(go.Scatter(x=stint.laps, y=full_truth, mode="lines", name="Verdad (sin ruido)",
                                 line=dict(width=1.5, color=OBSERVED, dash="dash")))
    fig.add_trace(go.Scatter(x=stint.laps, y=stint.delta, mode="markers", name="Observado",
                             marker=dict(size=8, color=OBSERVED, line=dict(width=2, color="white"))))
    for name, model in models.items():
        pred = np.asarray(model.predict_stint(stint.context, laps)).ravel()
        fig.add_trace(go.Scatter(x=laps, y=pred, mode="lines", name=name,
                                 line=dict(width=2.5, color=MODEL_COLORS.get(name, "#4a3aa7"),
                                           dash=MODEL_DASH.get(name, "solid"))))
    fig.update_xaxes(title="Vuelta del stint")
    fig.update_yaxes(title="Pérdida de ritmo δ [s]")
    return _layout(fig, 430, title=f"Predicción y extrapolación · {stint.stint_id} ({stint.compound})")


def latent_states(pinn, stint, phys: PhysicsConfig, horizon: int) -> go.Figure:
    """What the PINN believes is happening inside the tire."""
    laps = np.arange(1, horizon + 1, dtype=float)
    curve = pinn.predict_curve(stint.context, laps)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Temperatura latente θ", "Desgaste latente d"])
    fig.add_trace(go.Scatter(x=laps, y=curve["theta"], name="θ PINN",
                             line=dict(width=2.5, color="#eb6834")), row=1, col=1)
    fig.add_trace(go.Scatter(x=laps, y=curve["d"], name="d PINN",
                             line=dict(width=2.5, color="#1baf7a")), row=1, col=2)
    if stint.theta_true is not None:
        fig.add_trace(go.Scatter(x=stint.laps, y=stint.theta_true, name="θ real", mode="markers",
                                 marker=dict(size=7, color=OBSERVED)), row=1, col=1)
    if stint.d_true is not None:
        fig.add_trace(go.Scatter(x=stint.laps, y=stint.d_true, name="d real", mode="markers",
                                 marker=dict(size=7, color=OBSERVED, symbol="diamond")), row=1, col=2)
    fig.add_hline(y=phys.d_crit, line=dict(dash="dash", width=1, color=OBSERVED),
                  annotation_text="d_crit", row=1, col=2)
    fig.update_xaxes(title="Vuelta del stint")
    return _layout(fig, 360, title="Estados internos que la PINN infiere (nunca observados en datos reales)")


def parameter_recovery_bars(rows) -> go.Figure:
    names = [r[0] for r in rows]
    fig = go.Figure(go.Bar(
        x=names, y=[r[3] for r in rows], marker=dict(color="#2a78d6"),
        text=[f"{r[3]:.1f}%" for r in rows], textposition="outside",
        customdata=[[r[1], r[2]] for r in rows],
        hovertemplate="%{x}<br>estimado %{customdata[0]:.4f}<br>real %{customdata[1]:.4f}"
        "<br>error %{y:.1f}%<extra></extra>",
    ))
    fig.update_yaxes(title="Error relativo [%]", rangemode="tozero")
    fig = _layout(fig, 340, title="Recuperación de parámetros físicos (banco sintético)")
    fig.update_layout(hovermode="closest")
    return fig


# ----------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------
def strategy_curve(curve: dict, phys: PhysicsConfig, cliff: float | None, current_lap: float) -> go.Figure:
    laps = curve["laps"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=["Pérdida de ritmo δ [s]", "Desgaste d", "Temperatura θ"])
    fig.add_trace(go.Scatter(x=laps, y=curve["delta"], name="δ", line=dict(width=2.5, color="#2a78d6")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=laps, y=curve["d"], name="d", line=dict(width=2.5, color="#1baf7a")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=laps, y=curve["theta"], name="θ", line=dict(width=2.5, color="#eb6834")),
                  row=3, col=1)
    fig.add_hline(y=phys.d_crit, line=dict(dash="dash", width=1, color=OBSERVED),
                  annotation_text=f"d_crit = {phys.d_crit:g}", row=2, col=1)
    for r in (1, 2, 3):
        if cliff is not None:
            fig.add_vline(x=cliff, line=dict(width=2, color="#d03b3b"), row=r, col=1)
        if current_lap > 0:
            fig.add_vline(x=current_lap, line=dict(width=1.5, dash="dot", color=OBSERVED), row=r, col=1)
    fig.update_xaxes(title="Vuelta del stint", row=3, col=1)
    fig = _layout(fig, 640, title="Predicción de la PINN para estas condiciones")
    fig.update_layout(showlegend=False)
    return fig


def cliff_map(pinn, phys: PhysicsConfig, horizon: int, res: int = 24) -> go.Figure:
    """Lap at which d reaches d_crit, over (track temperature, load), per compound."""
    compounds = [("SOFT", 0.0), ("MEDIUM", 0.5), ("HARD", 1.0)]
    track = np.linspace(0.05, 0.95, res)
    load = np.linspace(0.6, 1.4, res)
    laps = np.arange(1, horizon + 1, dtype=float)
    tau = laps / phys.lap_ref
    gl, gt = np.meshgrid(load, track, indexing="ij")
    n = gl.size

    grids = []
    for _, c in compounds:
        ctx = np.stack([np.ones(n), gl.ravel(), np.ones(n), gt.ravel(), np.full(n, c)], axis=1)
        x = np.empty((n * laps.size, 6))
        x[:, 0] = np.tile(tau, n)
        x[:, 1:] = np.repeat(ctx, laps.size, axis=0)
        _, d = pinn.predict(x)
        hits = [wear_lap(laps, row, phys) for row in d.reshape(n, laps.size)]
        grids.append(np.array([np.nan if h is None else h for h in hits]).reshape(res, res))

    finite = np.concatenate([g[np.isfinite(g)] for g in grids])
    zmin, zmax = (float(finite.min()), float(finite.max())) if finite.size else (1, horizon)
    fig = make_subplots(rows=1, cols=3, subplot_titles=[c for c, _ in compounds],
                        horizontal_spacing=0.05)
    for i, grid in enumerate(grids):
        fig.add_trace(
            go.Heatmap(
                z=grid, x=track, y=load, zmin=zmin, zmax=zmax, colorscale=SEQUENTIAL,
                showscale=i == 2, colorbar=dict(title="Vuelta"),
                hovertemplate="T pista %{x:.2f}<br>carga %{y:.2f}<br>d_crit en vuelta %{z:.0f}<extra></extra>",
            ),
            row=1, col=i + 1,
        )
        fig.update_xaxes(title="Temp. de pista (norm.)", row=1, col=i + 1)
    fig.update_yaxes(title="Carga mecánica (norm.)", row=1, col=1)
    fig = _layout(fig, 380, title=f"Mapa de decisión: vuelta en la que d alcanza d_crit "
                                  f"(en blanco = no se alcanza en {horizon} vueltas)")
    fig.update_layout(hovermode="closest")
    return fig
