"""🗂️ Historial: every earlier training run, compared and reopened."""

from __future__ import annotations

from pathlib import Path

import charts
import history
import pandas as pd
import streamlit as st
from job import loss_labels
from session import ROOT, busy, ss
from ui import TRUTH, empty, fmt_seconds, kpi, plot

SHORT = {"PINN": "PINN", "Linear (classic)": "Lineal", "LSTM (black box)": "LSTM"}


def _is_open(folder: Path) -> bool:
    return ss.result is not None and Path(ss.result.out_dir).resolve() == Path(folder).resolve()


def _label(run: dict) -> str:
    pinn = run["rmse"].get("PINN")
    rmse = f" · PINN {pinn:.3f} s" if pinn is not None else ""
    return f"#{run['number']} · {run['saved_at']} · {run['data']}{rmse}"


def _table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        if r["mode"] == "practicas":
            mode = "Prácticas → carrera"
        elif r["mode"] == "carrera":
            mode = f"Carrera fuera ({r.get('test_gp')})"
        else:
            mode = f"Stints al azar ({100 * r['test_fraction']:.0f} % prueba)"
        rows.append({
            "#": r["number"],
            "": "● abierto" if _is_open(r["folder"]) else "",
            "Fecha": r["saved_at"],
            "Datos": r["data"],
            "Evaluación": mode,
            "Iteraciones": f"{r['adam_iters']:,} + {r['lbfgs_iters']:,}",
            "Red": f"{r['network']} {r['activation']}",
            "Stints": f"{r['n_train']} / {r['n_test']}",
            **{f"RMSE {short} [s]": r["rmse"].get(name) for name, short in SHORT.items()},
            "Duración": fmt_seconds(r["seconds"]),
            "Estado": "detenido" if r["stopped"] else "completo",
        })
    return pd.DataFrame(rows)


def render() -> None:
    runs = history.list_runs(ROOT / "outputs")
    if not runs:
        empty("🗂️", "Todavía no hay entrenamientos guardados",
              "Cada entrenamiento que lances desde <b>🧠 Entrenamiento</b> se guarda aquí con "
              "sus métricas, sus curvas y sus predicciones, para volver a abrirlo cuando quieras.")
        st.caption("Los entrenamientos hechos antes de esta versión no guardaban sus resultados; "
                   "sus modelos siguen disponibles en **🗄️ Archivo** → pestaña **Modelos guardados**.")
        return

    st.write(f"**{len(runs)} entrenamientos guardados.** Elige uno para ver cómo entrenó y "
             "ábrelo para recorrer sus resultados en **📈 Resultados** y **🔎 Carrera a carrera**.")
    st.dataframe(
        _table(runs).style.format(precision=3, na_rep="—"), width="stretch", hide_index=True,
        column_config={"Stints": st.column_config.TextColumn(
            "Stints", help="Entrenamiento / prueba")},
    )
    if len(runs) > 1:
        plot(st, charts.history_rmse(runs), key="history-rmse")

    st.divider()
    run = runs[[_label(r) for r in runs].index(
        st.selectbox("Entrenamiento", [_label(r) for r in runs]))]
    folder = run["folder"]

    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.caption(f"Carpeta `{folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder}` · "
               f"semilla {run['seed']} · parámetros estimados: {', '.join(run['free_params'])}")
    if _is_open(folder):
        c2.success("Abierto", icon="✅")
    elif c2.button("📂 Abrir resultados", type="primary", width="stretch", disabled=busy(),
                   help="Lo carga en 📈 Resultados, 🔎 Carrera a carrera y 🏁 Estrategia, "
                        "en lugar del último entrenamiento."):
        with st.spinner("Cargando el modelo y sus datos…"):
            ss.result = history.load_run(folder)
        st.toast("Entrenamiento abierto: míralo en 📈 Resultados o 🔎 Carrera a carrera")
        st.rerun()

    best = min(run["rmse"].values()) if run["rmse"] else None
    cols = st.columns(max(len(run["rmse"]), 1) + 1)
    for col, (name, rmse) in zip(cols, run["rmse"].items(), strict=False):
        seen = run["train_rmse"].get(name)
        kpi(col, f"{'🏆 ' if rmse == best else ''}RMSE · {SHORT.get(name, name)}", f"{rmse:.3f} s",
            f"{seen:.3f} s en lo visto" if seen is not None else None,
            delta_color="off", delta_arrow="off")
    kpi(cols[-1], "Duración", fmt_seconds(run["seconds"]),
        "detenido a mano" if run["stopped"] else None, delta_color="off", delta_arrow="off")

    traces = history.load_traces(folder)
    if not traces:
        return
    st.subheader("Cómo entrenó")
    if traces["loss_terms"]:
        terms = traces["loss_terms"]
        plot(st, charts.loss_curves(traces["loss_steps"], terms, loss_labels(len(terms[0])),
                                    traces["loss_phases"], lbfgs_planned=traces["lbfgs_planned"]),
             key=f"history-loss-{folder.name}")
        truth = TRUTH if run["source"] == "synthetic" else None
        plot(st, charts.parameter_traces(traces["param_trace"], traces["free_params"], truth,
                                         traces["loss_phases"]),
             key=f"history-params-{folder.name}")
    if traces["lstm_loss"]:
        plot(st, charts.lstm_loss(traces["lstm_loss"]), key=f"history-lstm-{folder.name}")
    with st.expander("Registro"):
        st.code("\n".join(traces["log"][-300:]) or "…", language=None)
    with st.expander("Ajustes de la barra lateral"):
        st.json(traces["settings"], expanded=False)
