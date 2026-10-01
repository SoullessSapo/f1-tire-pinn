"""🧠 Entrenamiento: start a run and watch Adam and L-BFGS converge live."""

from __future__ import annotations

from dataclasses import asdict

import charts
import numpy as np
import streamlit as st
from job import Job, Settings, loss_labels
from session import busy, ss, start
from ui import TRUTH, empty, fmt_seconds, kpi, plot


def render(settings: Settings) -> None:
    # Stopping is done from the status bar at the top, visible from every tab.
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    if c1.button("▶️ Entrenar", type="primary", disabled=busy(), width="stretch"):
        start("train", settings)
        st.rerun()
    c2.caption(
        f"PINN: Adam {settings.adam_iters:,} + L-BFGS {settings.lbfgs_iters:,} iteraciones · "
        f"red {settings.hidden_depth}x{settings.hidden_width} · "
        + ("con líneas base" if settings.train_baselines else "sin líneas base")
        + f" · resultados en `{settings.out_dir}`"
    )
    with st.expander("Configuración completa que se usará"):
        try:
            st.json(asdict(settings.to_config()), expanded=False)
        except SystemExit as exc:
            st.error(str(exc))

    @st.fragment(run_every=2.0 if busy() else None)
    def _live():
        job: Job | None = ss.job
        if job is None or job.kind != "train":
            empty("🧠", "Listo para entrenar",
                  "Pulsa <b>Entrenar</b>. Las curvas de pérdida de Adam y L-BFGS y los "
                  "parámetros físicos se dibujan en vivo mientras la red aprende.")
            return
        # The state itself is in the status bar at the top.
        k = st.columns(4)
        s = job.settings
        kpi(k[0], "Iteraciones Adam", f"{job.adam_iteration:,}", f"de {s.adam_iters:,}",
            delta_color="off", delta_arrow="off")
        kpi(k[1], "Iteraciones L-BFGS", f"{job.lbfgs_iteration:,}",
            f"de {s.lbfgs_iters:,}" if s.lbfgs_iters else "desactivado",
            delta_color="off", delta_arrow="off")
        total = sum(job.loss_terms[-1]) if job.loss_terms else None
        first = sum(job.loss_terms[0]) if job.loss_terms else None
        history = [float(np.log10(sum(t))) for t in job.loss_terms[: len(job.loss_steps)]]
        kpi(k[2], "Pérdida total", f"{total:.3e}" if total else "—",
            f"÷{first / total:,.0f} desde el inicio" if total and first else None,
            delta_color="normal", delta_arrow="down",
            chart_data=history if len(history) > 1 else None, chart_type="line")
        kpi(k[3], "Tiempo", fmt_seconds(job.elapsed))

        # The worker appends to these lists one after another; read a common prefix.
        n = min(len(job.loss_steps), len(job.loss_terms), len(job.param_trace), len(job.loss_phases))
        steps, terms = job.loss_steps[:n], job.loss_terms[:n]
        if terms:
            phases = job.loss_phases[:n]
            plot(st, charts.loss_curves(steps, terms, loss_labels(len(terms[0])), phases,
                                        lbfgs_planned=s.lbfgs_iters > 0))
            truth = TRUTH if s.source == "synthetic" else None
            plot(st, charts.parameter_traces(job.param_trace[:n], job.free_params,
                                             truth, phases))
        if job.lstm_loss:
            plot(st, charts.lstm_loss(list(job.lstm_loss)))
        with st.expander("Registro", expanded=job.error is not None):
            st.code("\n".join(job.lines[-300:]) or "…", language=None)

    _live()
