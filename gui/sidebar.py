"""The sidebar: every setting of a run, collected into a `Settings`."""

from __future__ import annotations

import streamlit as st
from job import Settings
from session import ss

from tirepinn.config import CONTEXT_NAMES, LEARNABLE_PARAMS, REAL_DATA_FREE_PARAMS

PRESETS = {
    "Rápido (prueba, 1-3 min)": dict(adam=1200, lbfgs=300, stints=16, lstm=200),
    "Estándar (igual que el CLI)": dict(adam=15000, lbfgs=3000, stints=64, lstm=800),
}
CIRCUITS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami", "Imola", "Monaco",
    "Canada", "Spain", "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands",
    "Monza", "Azerbaijan", "Singapore", "Austin", "Mexico", "Brazil", "Las Vegas", "Qatar",
    "Abu Dhabi",
]


def _apply_preset() -> None:
    preset = PRESETS.get(ss.preset)
    if preset:
        ss.adam, ss.lbfgs, ss.stints, ss.lstm = (
            preset["adam"], preset["lbfgs"], preset["stints"], preset["lstm"]
        )


def render() -> Settings:
    st.sidebar.markdown(
        '<div class="brand"><div class="bar"></div><div><div class="name">🏎️ F1 Tire PINN</div>'
        '<div class="sub">física + datos</div></div></div>',
        unsafe_allow_html=True,
    )

    sb = st.sidebar.container(border=True)
    sb.subheader("1 · Datos")
    source = sb.radio(
        "Fuente", ["synthetic", "fastf1"], horizontal=True,
        format_func=lambda s: "Sintético (con verdad)" if s == "synthetic" else "F1 real (FastF1)",
    )
    s = Settings(source=source)
    if source == "synthetic":
        ss.setdefault("stints", 64)
        s.n_stints = sb.number_input("Número de stints", 8, 400, key="stints", step=8)
        s.noise_delta_s = sb.number_input("Ruido en el tiempo de vuelta [s]", 0.0, 1.0, 0.06, 0.01)
    else:
        s.year = int(sb.number_input("Temporada", 2018, 2030, 2023))
        s.gps = tuple(sb.multiselect(
            "Carreras", CIRCUITS, default=["Monza"], accept_new_options=True,
            help="Con varias carreras el contexto varía de verdad; con una sola, el modelo "
                 "apenas ve algo más que el efecto del compuesto.",
        ))
        s.split_mode = sb.radio(
            "Cómo evaluar", ["random", "practice"],
            format_func=lambda m: {"random": "Stints al azar",
                                   "practice": "Prácticas → carrera"}[m],
            help="**Stints al azar**: se aparta una fracción de stints de las sesiones elegidas.\n\n"
                 "**Prácticas → carrera**: se entrena solo con las prácticas y se predice la "
                 "carrera del mismo fin de semana, dándole solo el compuesto, la duración de cada "
                 "stint y la temperatura de pista. La telemetría de carrera no se usa.",
        )
        if s.split_mode == "practice":
            s.practice_sessions = tuple(sb.multiselect(
                "Prácticas para entrenar", ["FP1", "FP2", "FP3"], default=["FP1", "FP2", "FP3"],
                help="Por defecto se descargan todas las prácticas del fin de semana. FP2 suele "
                     "tener las tandas largas con más combustible, las más parecidas a la "
                     "carrera. En fines de semana sprint solo hay FP1: las que falten se saltan.",
            )) or ("FP1", "FP2", "FP3")
            s.practice_fresh_only = not sb.toggle(
                "Usar juegos ya usados en prácticas", value=True,
                help="Los equipos reparten cada juego en varias tandas: en prácticas casi todas "
                     "son con neumáticos usados. Si se desactiva, solo quedan las primeras "
                     "tandas de cada juego y puede no quedar ninguna. El precio: esas tandas "
                     "empiezan con algo de desgaste, y el modelo supone d = 0 al inicio.",
            )
        else:
            s.session = sb.selectbox("Sesión", ["R", "S", "FP1", "FP2", "FP3", "Q"])
        drivers = sb.text_input("Pilotos (vacío = todos)", placeholder="VER HAM LEC")
        s.drivers = tuple(d.upper() for d in drivers.split())
        if 0 < len(s.drivers) < 3:
            sb.warning(
                "Con menos de 3 pilotos no se puede medir el efecto del combustible y de la "
                "pista en cada carrera: se usa una corrección fija. Para entrenar conviene "
                "dejarlo vacío (todos)."
            )
        s.aggregate_context = tuple(sb.multiselect(
            "Colapsar a la mediana de la carrera", CONTEXT_NAMES, default=["q_fric", "load"],
            help="Estos proxies varían mucho dentro de un mismo circuito sin predecir nada: "
                 "es ruido de medida. Ver dataset.aggregate_context_by_race.",
        ))
        s.min_stint_laps = int(sb.number_input(
            "Vueltas mínimas por stint", 4, 30, 6 if s.split_mode == "practice" else 8,
            help="Las tandas de práctica son cortas: con 8 vueltas mínimas se pierden muchas.",
        ))
        s.free_params = tuple(sb.multiselect(
            "Parámetros físicos a estimar", LEARNABLE_PARAMS, default=list(REAL_DATA_FREE_PARAMS),
            help="El resto se fija en los valores calibrados en el banco sintético.",
        ))
    if not s.practice_to_race:
        s.test_fraction = sb.slider("Fracción de stints de prueba", 0.1, 0.5, 0.25, 0.05)
    s.seed = int(sb.number_input("Semilla", 0, 10_000, 42))

    sb = st.sidebar.container(border=True)
    sb.subheader("2 · Entrenamiento")
    ss.setdefault("adam", 15000)
    ss.setdefault("lbfgs", 3000)
    ss.setdefault("lstm", 800)
    sb.selectbox("Preajuste", [*PRESETS, "Personalizado"], index=1, key="preset",
                 on_change=_apply_preset)
    s.adam_iters = int(sb.number_input("Iteraciones Adam", 0, 200_000, key="adam", step=500))
    s.lbfgs_iters = int(sb.number_input("Iteraciones L-BFGS", 0, 50_000, key="lbfgs", step=250))
    s.train_baselines = sb.checkbox("Entrenar también las líneas base (lineal y LSTM)", True)
    if s.train_baselines:
        s.lstm_epochs = int(sb.number_input("Épocas LSTM", 10, 10_000, key="lstm", step=50))

    with sb.expander("Avanzado: red y pérdida"):
        s.lr = float(st.number_input("Tasa de aprendizaje", 1e-5, 1e-1, 1e-3, format="%.0e"))
        s.hidden_width = int(st.number_input("Neuronas por capa", 8, 512, 64, 8))
        s.hidden_depth = int(st.number_input("Capas ocultas", 1, 10, 4))
        s.activation = st.selectbox(
            "Función de activación", ["tanh", "sin", "swish", "gelu"],
            help="Debe ser suave: el residuo físico deriva la red. Por eso no se ofrece ReLU, "
                 "cuya segunda derivada es cero. La forma de la curva de desgaste la fijan las "
                 "ecuaciones, no la activación.",
        )
        s.num_domain = int(st.number_input("Puntos de colocación", 200, 50_000, 4000, 200))
        s.display_every = int(st.number_input("Refrescar pérdida cada N iteraciones", 10, 5000, 250, 10))
        default_w = 20.0 if source == "synthetic" else 5.0
        s.w_data_delta = float(st.number_input(
            "Peso del término de datos (ritmo)", 0.1, 500.0, default_w, 0.5,
            help="20 en sintético, 5 con datos reales: el ruido de una vuelta real es ~10x mayor.",
        ))

    sb = st.sidebar.container(border=True)
    sb.subheader("3 · Salida")
    s.out_dir = sb.text_input("Carpeta de resultados", "outputs/gui")
    return s
