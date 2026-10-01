"""📊 Datos: what was loaded, stint by stint."""

from __future__ import annotations

import charts
import numpy as np
import pandas as pd
import streamlit as st
from job import Settings
from session import busy, ss, start
from ui import empty, kpi, plot, roles

from tirepinn.config import CONTEXT_NAMES, Config


def render(settings: Settings) -> None:
    data_entry = ss.data
    c1, c2 = st.columns([1, 3])
    label = "🔄 Recargar datos" if data_entry else "📥 Cargar datos"
    if c1.button(label, type="primary" if not data_entry else "secondary", disabled=busy()):
        ss.data = None  # force a fresh load
        start("data", settings)
        st.rerun()
    if settings.source == "fastf1":
        c2.caption("FastF1 descarga la telemetría la primera vez (minutos por carrera); "
                   "después se lee de la caché en `cache/`.")

    if data_entry is None:
        empty("📥", "Todavía no hay datos",
              "Pulsa <b>Cargar datos</b>, o directamente <b>Entrenar</b> en la pestaña de "
              "entrenamiento: carga los datos por ti.")
        return
    key, data = data_entry
    if key != settings.data_key():
        st.warning("La configuración de datos de la barra lateral cambió desde la última carga. "
                   "Lo que ves abajo son los datos anteriores.")

    phys = (ss.result.cfg if ss.result else Config()).physics
    trained_on_this = ss.result is not None and ss.result.data is data
    test_ids = {s.stint_id for s in ss.result.test.stints} if trained_on_this else None
    train_label, test_label = roles(ss.result.cfg) if trained_on_this else ("", "")

    deltas = np.concatenate([s.delta for s in data.stints])
    m = st.columns(5)
    kpi(m[0], "Stints", len(data))
    kpi(m[1], "Vueltas", f"{data.n_laps:,}")
    kpi(m[2], "Pilotos", len({s.driver for s in data.stints}))
    kpi(m[3], "Pérdida de ritmo mediana", f"{np.median(deltas):.2f} s")
    kpi(m[4], "Pérdida máxima", f"{deltas.max():.2f} s")
    st.caption(f"Fuente: {data.source}")

    highlight = None
    if test_ids and st.toggle(f"Resaltar los stints de {test_label} del último entrenamiento"):
        highlight = test_ids
    plot(st, charts.dataset_curves(data, highlight))

    c1, c2 = st.columns([1, 2])
    plot(c1, charts.stint_lengths(data))
    plot(c2, charts.context_histograms(data))

    st.subheader("Tabla de stints")
    rows = []
    for s in data.stints:
        row = {
            "Stint": s.stint_id, "Piloto": s.driver, "Compuesto": s.compound, "Vueltas": s.n_laps,
            "δ inicial [s]": float(s.delta[0]), "δ final [s]": float(s.delta[-1]),
            "δ máx [s]": float(s.delta.max()),
        }
        row["Curva δ"] = [float(v) for v in s.delta]
        row.update({n: float(v) for n, v in zip(CONTEXT_NAMES, s.context, strict=False)})
        if test_ids is not None:
            row["Conjunto"] = test_label if s.stint_id in test_ids else train_label
        rows.append(row)
    table = pd.DataFrame(rows)
    st.dataframe(
        table, width="stretch", hide_index=True, height=320,
        column_config={
            "Curva δ": st.column_config.LineChartColumn("Curva δ", width="small"),
            "δ máx [s]": st.column_config.ProgressColumn(
                "δ máx [s]", format="%.2f", min_value=0.0,
                max_value=float(max(deltas.max(), 1e-9)),
            ),
        },
    )

    laps = pd.DataFrame([
        {"stint": s.stint_id, "piloto": s.driver, "compuesto": s.compound, "vuelta": int(lap),
         "delta_s": float(d), **{n: float(v) for n, v in zip(CONTEXT_NAMES, s.context, strict=False)}}
        for s in data.stints for lap, d in zip(s.laps, s.delta, strict=False)
    ])
    st.download_button("⬇️ Descargar todas las vueltas (CSV)", laps.to_csv(index=False),
                       "vueltas.csv", "text/csv")

    st.subheader("Detalle de un stint")
    ids = [s.stint_id for s in data.stints]
    chosen = st.selectbox("Stint", ids, key="data_stint")
    stint = data.stints[ids.index(chosen)]
    plot(st, charts.stint_detail(stint, phys))
