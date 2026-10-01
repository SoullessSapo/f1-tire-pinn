"""🔎 Carrera a carrera: every stint of every race, one at a time."""

from __future__ import annotations

import charts
import pandas as pd
import streamlit as st
from analysis import race_names, race_of, stint_errors
from session import ss
from ui import compound_badge, empty, kpi, plot, roles


def render() -> None:
    res = ss.result
    if res is None:
        empty("🔎", "Entrena un modelo para recorrer sus carreras",
              "Aquí verás, carrera por carrera y stint por stint, cómo predice cada modelo.")
        return
    phys = res.cfg.physics
    fallback = res.cfg.data.gp if res.cfg.data.source == "fastf1" else "Banco sintético"
    names = race_names(res.cfg)
    test_ids = {s.stint_id for s in res.test.stints}
    train_label, test_label = roles(res.cfg)

    by_race: dict[str, list] = {}
    for s in res.data.stints:
        by_race.setdefault(race_of(s, names, fallback), []).append(s)

    c1, c2 = st.columns([3, 2])
    race = c1.selectbox(
        "Carrera", list(by_race),
        format_func=lambda r: f"{r} · {len(by_race[r])} stints "
                              f"({sum(s.stint_id in test_ids for s in by_race[r])} de {test_label})",
    )
    options = ["Todos", f"Solo {test_label}", f"Solo {train_label}"]
    # Practice -> race is about the race: open on it.
    default = options[1] if res.cfg.data.train_sessions else "Todos"
    which = c2.segmented_control("Stints", options, default=default, key="race_which") or default
    stints = [
        s for s in by_race[race]
        if which == "Todos" or (s.stint_id in test_ids) == (which == options[1])
    ]
    if not stints:
        st.info("Esta carrera no tiene stints de ese tipo.")
        return
    st.caption(f"Los stints de **{test_label}** no se usaron para entrenar: ahí se mide de verdad "
               f"el modelo. En los de **{train_label}** la red ya vio esas vueltas, así que un buen "
               "ajuste ahí no demuestra nada; un mal ajuste sí indica que no convergió.")

    # One stint at a time. The selectbox's own state is the single source of
    # truth; the buttons move it in a callback, which runs before it is drawn.
    ids = [s.stint_id for s in stints]
    key = f"race_stint_{race}_{which}"
    if ss.get(key) not in ids:
        ss[key] = ids[0]
    i = ids.index(ss[key])

    def move(step: int) -> None:
        ss[key] = ids[min(max(ids.index(ss[key]) + step, 0), len(ids) - 1)]

    b1, b2, b3 = st.columns([1, 6, 1], vertical_alignment="bottom")
    b1.button("◀ Anterior", width="stretch", disabled=i == 0, on_click=move, args=(-1,),
              key=f"{key}_prev")
    b2.selectbox("Stint", ids, key=key, label_visibility="collapsed")
    b3.button("Siguiente ▶", width="stretch", disabled=i == len(ids) - 1, on_click=move,
              args=(1,), key=f"{key}_next")
    stint = stints[i]

    is_test = stint.stint_id in test_ids
    st.markdown(
        f"#### {stint.stint_id} &nbsp; {compound_badge(stint.compound)} &nbsp; "
        + (f":blue-badge[🧪 {test_label} · nunca visto]" if is_test
           else f":gray-badge[📚 {train_label}]")
        + f" &nbsp; :gray[{stint.driver} · {stint.n_laps} vueltas · {i + 1} de {len(ids)}]",
        unsafe_allow_html=True,
    )
    errors = stint_errors(stint, res.models)
    best = min(errors, key=errors.get)
    k = st.columns(len(errors))
    for col, (name, err) in zip(k, errors.items(), strict=False):
        kpi(col, f"{'🏆 ' if name == best else ''}RMSE · {name}", f"{err:.3f} s")
    horizon = max(phys.strategy_horizon, int(stint.laps.max()) + 5)
    plot(st, charts.stint_predictions(stint, res.models, phys, horizon))
    plot(st, charts.latent_states(res.models["PINN"], stint, phys, horizon))

    st.subheader(f"Vista general · {race}")
    plot(st, charts.race_grid(stints, res.models, test_ids, (train_label, test_label)))
    table = pd.DataFrame([
        {"Stint": s.stint_id, "Conjunto": test_label if s.stint_id in test_ids else train_label,
         "Compuesto": s.compound, "Vueltas": s.n_laps,
         **{f"RMSE {n} [s]": v for n, v in stint_errors(s, res.models).items()}}
        for s in stints
    ])
    st.dataframe(table.style.format(precision=3), width="stretch", hide_index=True)
