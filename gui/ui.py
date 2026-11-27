"""Building blocks shared by the views: cards, tiles, badges, header, status bar."""

from __future__ import annotations

import charts
import streamlit as st
from job import Job
from session import busy, harvest, ss

from tirepinn.config import Config
from tirepinn.physics import GROUND_TRUTH

TRUTH = GROUND_TRUTH.as_dict()

# Only what the theme in .streamlit/config.toml cannot express: the header
# banner and a little more presence for the tabs.
STYLE = """
<style>
/* Page: less empty space above the content. */
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; }

/* Top bar: title on the left, the pipeline's progress on the right. */
.topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 0.8rem 1.4rem;
  flex-wrap: wrap; padding: 0.85rem 1.2rem; margin-bottom: 0.5rem; border-radius: 0.8rem;
  background: linear-gradient(100deg, #15151e 0%, #22222e 68%, #3b1414 100%);
  border-left: 5px solid #e10600; color: #ffffff;
}
.topbar .name { font-size: 1.45rem; font-weight: 700; line-height: 1.15; color: #ffffff; }
.topbar .sub { font-size: 0.86rem; color: #b8b8c6; margin-top: 0.1rem; }
.stepper { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0; padding: 0; }
.stepper .s {
  display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.25rem 0.75rem 0.25rem 0.3rem;
  border-radius: 999px; border: 1px solid #3b3b4c; color: #b8b8c6; font-size: 0.85rem;
}
.stepper .s .n {
  width: 1.4rem; height: 1.4rem; border-radius: 50%; display: inline-flex; align-items: center;
  justify-content: center; font-size: 0.74rem; font-weight: 700; background: #2d2d3b; color: #d8d8e2;
}
.stepper .s.done { color: #ffffff; border-color: #2f6b45; }
.stepper .s.done .n { background: #1f9d55; color: #ffffff; }
.stepper .s.active { color: #ffffff; border-color: #e10600; background: rgba(225, 6, 0, 0.18); }
.stepper .s.active .n { background: #e10600; color: #ffffff; }
.stepper .s.extra { border-style: dashed; color: #ffffff; }

/* Tabs as pills. Older Streamlit draws tabs with baseweb, newer with react-aria
   (data-testid="stTab"): both are styled. */
.stTabs [role="tablist"] { gap: 0.4rem; flex-wrap: wrap; border-bottom: none; box-shadow: none; }
.stTabs [data-testid="stTab"], .stTabs [data-baseweb="tab"] {
  height: auto; padding: 0.38rem 0.95rem; border-radius: 999px;
  border: 1px solid rgba(128, 128, 140, 0.3); box-shadow: none;
}
.stTabs [data-testid="stTab"] p, .stTabs [data-baseweb="tab"] p { font-size: 0.98rem; font-weight: 600; }
.stTabs [data-testid="stTab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: #e10600; border-color: #e10600; }
.stTabs [data-testid="stTab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color: #ffffff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Metric tiles: small caps label, figures that line up. */
[data-testid="stMetricLabel"] p {
  text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.74rem; opacity: 0.75;
}
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

/* Sidebar steps. */
.brand { display: flex; align-items: center; gap: 0.6rem; margin: 0 0 0.2rem; }
.brand .bar { width: 6px; height: 2.1rem; border-radius: 3px; background: #e10600; }
.brand .name { font-size: 1.35rem; font-weight: 700; line-height: 1.1; }
.brand .sub { font-size: 0.82rem; opacity: 0.7; }
.step { display: flex; align-items: center; gap: 0.6rem; margin: 0.1rem 0 0.3rem; }
.step .n {
  width: 1.7rem; height: 1.7rem; flex: none; border-radius: 50%; background: #e10600; color: #ffffff;
  display: inline-flex; align-items: center; justify-content: center; font-weight: 700;
}
.step .t { font-size: 1.12rem; font-weight: 700; line-height: 1.1; }
.step .h { font-size: 0.8rem; opacity: 0.65; }

/* Empty states and tyre badges. */
.empty {
  text-align: center; padding: 2.6rem 1rem 2.2rem; margin: 0.4rem 0 1rem;
  border: 1px dashed rgba(128, 128, 140, 0.45); border-radius: 0.8rem;
  background: linear-gradient(180deg, rgba(225, 6, 0, 0.04), rgba(225, 6, 0, 0));
}
.empty .icon { font-size: 2.6rem; line-height: 1; margin-bottom: 0.6rem; }
.empty h3 { margin: 0 0 0.3rem; padding: 0; font-size: 1.25rem; }
.empty p { margin: 0 auto; max-width: 34rem; opacity: 0.75; }
.tyre {
  display: inline-block; padding: 0 0.6rem; border: 2px solid; border-radius: 999px;
  font-size: 0.85rem; font-weight: 700; vertical-align: middle;
}
</style>
"""


def fmt_seconds(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def plot(container, fig, key: str | None = None) -> None:
    """Draw a chart in its own card, titled in text above it (see charts._layout).

    Streamlit derives a chart's id from the figure itself, so the same figure
    drawn in two tabs (a run's curves in Entrenamiento and in Historial) needs
    a `key` to tell them apart.
    """
    card = container.container(border=True)
    title = charts.title_of(fig)
    if title:
        card.markdown(f"**{title}**")
    card.plotly_chart(fig, width="stretch", key=key)


def kpi(container, *args, **kwargs) -> None:
    """A metric in a card, so the numbers read as one row of tiles."""
    container.metric(*args, border=True, **kwargs)


def empty(icon: str, title: str, text: str) -> None:
    """Placeholder for a tab with nothing to show yet."""
    st.markdown(
        f'<div class="empty"><div class="icon">{icon}</div><h3>{title}</h3><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def compound_badge(compound: str) -> str:
    color = charts.COMPOUND_COLORS.get(compound.upper(), charts.OBSERVED)
    return (f'<span class="tyre" style="border-color:{color};color:{color}">'
            f'● {compound.title()}</span>')


def roles(cfg: Config) -> tuple[str, str]:
    """How to call the training and the test stints in this run."""
    if cfg.data.train_sessions:
        return "prácticas", "carrera"
    if cfg.data.test_gp:
        return "otras carreras", "carrera no vista"
    return "entrenamiento", "prueba"


def phase_counter(job: Job) -> str:
    s = job.settings
    if job.lbfgs_start is None:
        return f"Adam {job.adam_iteration:,} / {s.adam_iters:,}"
    return f"L-BFGS {job.lbfgs_iteration:,} / {s.lbfgs_iters:,} (Adam terminado)"


def header() -> None:
    """Top bar: the title, and where the pipeline stands (data, training, results)."""
    job: Job | None = ss.job
    loading = busy() and job.kind == "data"
    training = busy() and job.kind == "train"

    def state(done: bool, active: bool) -> str:
        return "active" if active else ("done" if done else "")

    steps = [
        ("Datos", state(ss.data is not None or ss.result is not None, loading)),
        ("Entrenamiento", state(ss.result is not None, training)),
        ("Resultados", state(ss.result is not None and not training, False)),
    ]
    html = "".join(f'<span class="s {cls}"><span class="n">{"✓" if cls == "done" else i}</span>'
                   f"{label}</span>" for i, (label, cls) in enumerate(steps, 1))
    if ss.loaded is not None:
        html += '<span class="s extra"><span class="n">💾</span>Modelo abierto</span>'
    st.html(
        '<div class="topbar"><div><div class="name">Degradación de neumáticos F1 · PINN</div>'
        '<div class="sub">Red neuronal informada por la física: ecuaciones térmica y de desgaste '
        f'más datos de vuelta</div></div><div class="stepper">{html}</div></div>'
    )


def status_bar() -> None:
    @st.fragment(run_every=1.0 if busy() else None)
    def _bar():
        job: Job | None = ss.job
        if job is None:
            return
        if job.running:
            progress = job.progress
            if job.kind == "train" and job.total_iterations and 0.08 <= progress < 0.80:
                progress = 0.08 + 0.72 * min(job.iteration / job.total_iterations, 1.0)
            c1, c2 = st.columns([6, 1])
            c1.progress(progress, text=f"⏳ **{job.stage}** · {fmt_seconds(job.elapsed)}"
                        + (f" · {phase_counter(job)}" if job.iteration else ""))
            if c2.button("⏹ Detener", width="stretch", disabled=job.kind != "train"):
                job.stop_requested.set()
                job.log("Parada solicitada: se termina la fase actual y se evalúa lo entrenado")
        else:
            if not job.harvested:
                harvest()
                st.rerun(scope="app")
            if job.error:
                st.error(f"La última tarea falló: {job.error}")
            else:
                st.success(f"✅ {job.stage} en {fmt_seconds(job.elapsed)}")

    _bar()
