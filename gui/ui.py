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
.hero {
  border-radius: 0.8rem;
  padding: 1.1rem 1.4rem 1rem;
  margin-bottom: 0.6rem;
  background: linear-gradient(110deg, #15151e 0%, #26263a 62%, #e10600 160%);
  color: #ffffff;
  border-left: 6px solid #e10600;
}
.hero h1 { color: #ffffff; font-size: 1.9rem; margin: 0; padding: 0; line-height: 1.2; }
.hero p { color: #c9c9d4; margin: 0.25rem 0 0.7rem; font-size: 0.98rem; }
.hero .chip {
  display: inline-block; margin: 0 0.4rem 0.3rem 0; padding: 0.15rem 0.7rem;
  border-radius: 999px; font-size: 0.85rem; border: 1px solid #4a4a5e; color: #c9c9d4;
}
.hero .chip.on { border-color: #e10600; background: rgba(225, 6, 0, 0.18); color: #ffffff; }
.stTabs [data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 600; }
.empty {
  text-align: center; padding: 2.6rem 1rem 2.2rem; margin: 0.4rem 0 1rem;
  border: 1px dashed rgba(128, 128, 140, 0.45); border-radius: 0.8rem;
}
.empty .icon { font-size: 2.6rem; line-height: 1; margin-bottom: 0.6rem; }
.empty h3 { margin: 0 0 0.3rem; padding: 0; font-size: 1.25rem; }
.empty p { margin: 0 auto; max-width: 34rem; opacity: 0.75; }
.tyre {
  display: inline-block; padding: 0 0.6rem; border: 2px solid; border-radius: 999px;
  font-size: 0.85rem; font-weight: 700; vertical-align: middle;
}
.brand { display: flex; align-items: center; gap: 0.6rem; margin: 0 0 0.2rem; }
.brand .bar { width: 6px; height: 2.1rem; border-radius: 3px; background: #e10600; }
.brand .name { font-size: 1.35rem; font-weight: 700; line-height: 1.1; }
.brand .sub { font-size: 0.82rem; opacity: 0.7; }
</style>
"""


def fmt_seconds(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def plot(container, fig) -> None:
    """Draw a chart in its own card, titled in text above it (see charts._layout)."""
    card = container.container(border=True)
    title = charts.title_of(fig)
    if title:
        card.markdown(f"**{title}**")
    card.plotly_chart(fig, width="stretch")


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
    return ("prácticas", "carrera") if cfg.data.train_sessions else ("entrenamiento", "prueba")


def phase_counter(job: Job) -> str:
    s = job.settings
    if job.lbfgs_start is None:
        return f"Adam {job.adam_iteration:,} / {s.adam_iters:,}"
    return f"L-BFGS {job.lbfgs_iteration:,} / {s.lbfgs_iters:,} (Adam terminado)"


def header() -> None:
    """Banner with the state of the pipeline: what is loaded, trained, open."""
    job: Job | None = ss.job
    steps = [
        ("📥 Datos cargados", ss.data is not None),
        ("🧠 Entrenando", busy() and job.kind == "train"),
        ("✅ Modelo entrenado", ss.result is not None),
        ("💾 Modelo abierto", ss.loaded is not None),
    ]
    chips = "".join(f'<span class="chip{" on" if on else ""}">{label}</span>' for label, on in steps)
    st.markdown(
        f"""<div class="hero"><h1>Degradación de neumáticos F1 · PINN</h1>
        <p>Red neuronal informada por la física: ecuaciones térmica y de desgaste + datos de vuelta</p>
        {chips}</div>""",
        unsafe_allow_html=True,
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
