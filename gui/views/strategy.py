"""🏁 Estrategia: what-if curves for any conditions and compound."""

from __future__ import annotations

import time

import charts
import numpy as np
import pandas as pd
import streamlit as st
from session import ss
from ui import empty, kpi, plot

from tirepinn.config import COMPOUND_INDEX


def _strategy_model():
    choices = {}
    if ss.result is not None:
        choices["Último entrenamiento"] = (ss.result.models["PINN"], ss.result.cfg)
    if ss.loaded is not None:
        choices[f"Modelo cargado ({ss.loaded['dir'].name})"] = (ss.loaded["pinn"], ss.loaded["cfg"])
    return choices


def render() -> None:
    choices = _strategy_model()
    if not choices:
        empty("🏁", "Simulador de estrategia",
              "Entrena un modelo o abre uno en <b>Modelos guardados</b> para usarlo aquí.")
        return
    name = st.radio("Modelo", list(choices), horizontal=True)
    pinn, cfg = choices[name]
    phys = cfg.physics

    c = st.columns(4)
    compound = c[0].selectbox("Compuesto", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], index=1)
    track = c[1].slider("Temp. de pista (0 fría - 1 caliente)", 0.0, 1.0, 0.5, 0.05)
    load = c[2].slider("Carga mecánica", 0.5, 1.5, 1.0, 0.05)
    q = c[3].slider("Energía de fricción", 0.4, 1.6, 1.0, 0.05)
    c = st.columns(4)
    speed = c[0].slider("Velocidad media", 0.6, 1.4, 1.0, 0.05)
    horizon = c[1].slider("Horizonte [vueltas]", 10, 80, phys.strategy_horizon)
    current = c[2].slider("Vueltas ya hechas con este juego", 0, 60, 0)

    context = np.array([q, load, speed, track, COMPOUND_INDEX[compound]])
    t0 = time.perf_counter()
    out = pinn.strategy(context, horizon=horizon, current_lap=float(current))
    latency = (time.perf_counter() - t0) * 1000
    curve = out["curve"]

    k = st.columns(4)
    cliff = out["cliff_lap"]
    kpi(k[0], "Vuelta en la que d ≥ d_crit", "no llega" if cliff is None else f"{cliff:.0f}")
    kpi(k[1], "Vida útil restante", f"> {horizon - current} vueltas" if cliff is None
                else f"{out['rul_laps']:.0f} vueltas")
    kpi(k[2], f"Pérdida en la vuelta {horizon}", f"{curve['delta'][-1]:.2f} s")
    kpi(k[3], "Latencia de la predicción", f"{latency:.1f} ms")
    plot(st, charts.strategy_curve(curve, phys, cliff, current))

    st.subheader("Comparación de compuestos en estas condiciones")
    rows = []
    for comp in ("SOFT", "MEDIUM", "HARD"):
        ctx = context.copy()
        ctx[4] = COMPOUND_INDEX[comp]
        o = pinn.strategy(ctx, horizon=horizon)
        cv = o["curve"]
        rows.append({
            "Compuesto": comp,
            "Vuelta d ≥ d_crit": "no llega" if o["cliff_lap"] is None else f"{o['cliff_lap']:.0f}",
            "δ a 10 vueltas [s]": float(cv["delta"][min(9, horizon - 1)]),
            "δ a 20 vueltas [s]": float(cv["delta"][min(19, horizon - 1)]),
            f"δ a {horizon} vueltas [s]": float(cv["delta"][-1]),
            "Tiempo perdido acumulado [s]": float(cv["delta"].sum()),
        })
    st.dataframe(pd.DataFrame(rows).style.format(precision=2, na_rep="no llega"),
                 width="stretch", hide_index=True)

    st.subheader("Mapa de decisión")
    if st.toggle("Calcular el mapa (una pasada de la red por celda)", value=True):
        plot(st, charts.cliff_map(pinn, phys, horizon))
