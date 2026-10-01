"""F1 Tire PINN: graphical interface.

Launch with

    python run_gui.py        (or: streamlit run gui/app.py)

Everything the command line does is here: load synthetic or real FastF1 data,
train the PINN and the two baselines while watching the loss and the physical
parameters converge, compare the models race by race, explore strategy
predictions, and reopen models trained earlier.

This file only assembles the page. The pieces live next to it:

    sidebar.py   the settings of a run
    session.py   session state and background jobs
    job.py       the training pipeline, run in a worker thread
    ui.py        shared widgets: cards, tiles, badges, header, status bar
    charts.py    the Plotly figures
    analysis.py  convergence checks and per-stint errors (no Streamlit)
    views/       one module per tab
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import session
import sidebar
import streamlit as st
import ui
from views import data, races, results, saved, strategy, training

TABS = (
    ("📊 Datos", data.render, True),
    ("🧠 Entrenamiento", training.render, True),
    ("📈 Resultados", results.render, False),
    ("🔎 Carrera a carrera", races.render, False),
    ("🏁 Estrategia", strategy.render, False),
    ("🖼️ Figuras", saved.render_figures, False),
    ("💾 Modelos guardados", saved.render_models, False),
)


def main() -> None:
    st.set_page_config(page_title="F1 Tire PINN", page_icon="🏎️", layout="wide")
    session.init()
    settings = sidebar.render()
    session.harvest()
    st.html(ui.STYLE)
    ui.header()
    ui.status_bar()
    for tab, (_, render, needs_settings) in zip(st.tabs([t[0] for t in TABS]), TABS, strict=True):
        with tab:
            if needs_settings:
                render(settings)
            else:
                render()


main()
