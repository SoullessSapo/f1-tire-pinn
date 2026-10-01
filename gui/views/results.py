"""📈 Resultados: diagnosis, metrics, generalisation and learned parameters."""

from __future__ import annotations

import charts
import numpy as np
import pandas as pd
import streamlit as st
from analysis import diagnose
from session import ss
from ui import TRUTH, empty, fmt_seconds, kpi, plot, roles


def show_diagnosis(res) -> None:
    checks = diagnose(res)
    worst = "bad" if any(c[0] == "bad" for c in checks) else (
        "warn" if any(c[0] == "warn" for c in checks) else "ok")
    label = {"ok": "✅ Diagnóstico: el entrenamiento parece sano",
             "warn": "⚠️ Diagnóstico: revisa estos puntos",
             "bad": "🚨 Diagnóstico: el modelo no convergió bien"}[worst]
    with st.expander(label, expanded=worst != "ok"):
        for level, title, text in checks:
            box = {"ok": st.success, "warn": st.warning, "bad": st.error}[level]
            box(f"**{title}**  \n{text}")


def generalization(res) -> None:
    """Training error next to test error: did what was learned carry over?"""
    train_label, test_label = roles(res.cfg)
    c1, c2 = st.columns([3, 2])
    plot(c1, charts.generalization_bars(res.train_metrics, res.metrics, train_label, test_label))
    train = {m.name: m.rmse for m in res.train_metrics}
    with c2.container(border=True):
        st.markdown(f"**Lectura: {train_label} → {test_label}**")
        for m in res.metrics:
            seen, unseen = train[m.name], m.rmse
            ratio = unseen / max(seen, 1e-9)
            if seen > 0.6:
                verdict = ":red[no aprendió ni lo que vio]"
            elif ratio < 1.5:
                verdict = ":green[generaliza bien]"
            elif ratio < 3:
                verdict = ":orange[algo de sobreajuste]"
            else:
                verdict = ":red[no generaliza]"
            st.markdown(f"- **{m.name}**: {seen:.3f} s → {unseen:.3f} s (x{ratio:.1f}) · {verdict}")
        if res.cfg.data.train_sessions:
            st.caption("En carrera el modelo solo recibe el compuesto, la duración del stint y la "
                       "temperatura de pista; energía, carga y velocidad vienen de las prácticas.")


def render() -> None:
    res = ss.result
    if res is None:
        empty("📈", "Sin resultados todavía",
              "Aquí aparecen las métricas, el diagnóstico y las predicciones cuando termine "
              "un entrenamiento.")
        return
    train_label, test_label = roles(res.cfg)
    st.caption(f"Entrenado en {fmt_seconds(res.seconds)} · {len(res.train)} stints de "
               f"{train_label}, {len(res.test)} de {test_label} · guardado en `{res.out_dir}`")

    show_diagnosis(res)

    best = min(res.metrics, key=lambda m: m.rmse)
    cols = st.columns(len(res.metrics))
    for col, m in zip(cols, res.metrics, strict=False):
        if m is best:
            kpi(col, f"🏆 RMSE · {m.name}", f"{m.rmse:.3f} s", "el mejor", delta_arrow="off")
        else:
            kpi(col, f"RMSE · {m.name}", f"{m.rmse:.3f} s", f"+{m.rmse - best.rmse:.3f} s vs el mejor",
                       delta_color="inverse")

    table = pd.DataFrame([{
        "Modelo": m.name, "RMSE [s]": m.rmse, "MAE [s]": m.mae, "Error máx [s]": m.max_error,
        "Error del cliff [vueltas]": "n/d" if m.cliff_mae is None else f"{m.cliff_mae:.2f}",
        "Cliffs detectados": f"{m.cliff_detected}/{m.cliff_total}",
        "Violaciones dentro [%]": 100 * m.violation_rate,
        "Violaciones extrapolando [%]": 100 * m.extrap_violation_rate, "Vueltas evaluadas": m.n_laps,
    } for m in res.metrics])
    st.dataframe(table.style.format(precision=3, na_rep="n/d"), width="stretch", hide_index=True)
    generalization(res)

    c1, c2 = st.columns(2)
    plot(c1, charts.metric_bars(res.metrics))
    plot(c2, charts.violation_bars(res.metrics))
    plot(st, charts.per_stint_rmse(res.metrics, test_label))

    st.subheader("Parámetros físicos aprendidos")
    pinn = res.models["PINN"]
    learned = pinn.learned_params().as_dict()
    free = set(res.cfg.pinn.free_params)
    synthetic = res.cfg.data.source == "synthetic"
    ptable = pd.DataFrame([{
        "Parámetro": n, "Valor": v, "Estado": "estimado" if n in free else "fijo",
        **({"Valor real": TRUTH[n], "Error [%]": 100 * abs(v - TRUTH[n]) / abs(TRUTH[n]) if TRUTH[n] else None}
           if synthetic else {}),
    } for n, v in learned.items() if n != "p"])
    c1, c2 = st.columns([2, 3])
    c1.dataframe(ptable.style.format(precision=4, na_rep=""), width="stretch", hide_index=True)
    if res.recovery:
        plot(c2, charts.parameter_recovery_bars(res.recovery))
        c2.caption(f"Error medio: {np.mean([r[3] for r in res.recovery]):.1f} %")
    elif not synthetic:
        c2.info("Con datos reales no existe un valor verdadero con el que comparar; los parámetros "
                "no estimados se fijan en la calibración del banco sintético.")

    st.info("Para ver la predicción de cada stint, carrera por carrera, abre la pestaña "
            "**🔎 Carrera a carrera**.")

    st.subheader("Informe")
    report = (res.out_dir / "report.txt")
    text = report.read_text(encoding="utf-8") if report.exists() else res.report
    st.code(text, language=None)
    st.download_button("⬇️ Descargar informe", text, "report.txt")
