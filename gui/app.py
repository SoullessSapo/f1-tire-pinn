"""F1 Tire PINN: graphical interface.

Launch with

    streamlit run gui/app.py        (or: python run_gui.py)

Everything the command line does is here: load synthetic or real FastF1 data,
train the PINN and the two baselines while watching the loss and the physical
parameters converge, compare the models, explore strategy predictions, and
reopen models trained earlier.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts
import numpy as np
import pandas as pd
import streamlit as st
from job import Job, Settings, loss_labels

from tirepinn.config import (
    COMPOUND_INDEX,
    CONTEXT_NAMES,
    LEARNABLE_PARAMS,
    REAL_DATA_FREE_PARAMS,
    Config,
)
from tirepinn.physics import GROUND_TRUTH
from tirepinn.pinn import TirePINN

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="F1 Tire PINN", page_icon="🏎️", layout="wide")
ss = st.session_state
for key, default in (("job", None), ("data", None), ("result", None), ("loaded", None)):
    ss.setdefault(key, default)

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
</style>
"""


# ======================================================================
# Sidebar: configuration
# ======================================================================
def _apply_preset() -> None:
    preset = PRESETS.get(ss.preset)
    if preset:
        ss.adam, ss.lbfgs, ss.stints, ss.lstm = (
            preset["adam"], preset["lbfgs"], preset["stints"], preset["lstm"]
        )


def sidebar() -> Settings:
    sb = st.sidebar
    sb.title("🏎️ F1 Tire PINN")
    sb.caption("Degradación de neumáticos con una red neuronal informada por la física")

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
        s.session = sb.selectbox("Sesión", ["R", "S", "FP1", "FP2", "FP3", "Q"])
        drivers = sb.text_input("Pilotos (vacío = todos)", placeholder="VER HAM LEC")
        s.drivers = tuple(d.upper() for d in drivers.split())
        s.aggregate_context = tuple(sb.multiselect(
            "Colapsar a la mediana de la carrera", CONTEXT_NAMES, default=["q_fric", "load"],
            help="Estos proxies varían mucho dentro de un mismo circuito sin predecir nada: "
                 "es ruido de medida. Ver dataset.aggregate_context_by_race.",
        ))
        s.free_params = tuple(sb.multiselect(
            "Parámetros físicos a estimar", LEARNABLE_PARAMS, default=list(REAL_DATA_FREE_PARAMS),
            help="El resto se fija en los valores calibrados en el banco sintético.",
        ))
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


# ======================================================================
# Job bookkeeping
# ======================================================================
def start(kind: str, settings: Settings) -> None:
    job = Job(kind, settings, cached_data=ss.data)
    ss.job = job
    job.start()


def harvest() -> None:
    """Move the output of a finished job into the session."""
    job: Job | None = ss.job
    if job is None or job.running or getattr(job, "_harvested", False):
        return
    job._harvested = True
    if job.data is not None:
        ss.data = (job.settings.data_key(), job.data)
    if job.result is not None:
        ss.result = job.result


def busy() -> bool:
    return ss.job is not None and ss.job.running


def fmt_seconds(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


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
            if not getattr(job, "_harvested", False):
                harvest()
                st.rerun(scope="app")
            if job.error:
                st.error(f"La última tarea falló: {job.error}")
            else:
                st.success(f"✅ {job.stage} en {fmt_seconds(job.elapsed)}")

    _bar()


# ======================================================================
# Tabs
# ======================================================================
def plot(container, fig) -> None:
    """Draw a chart in its own card, titled in text above it (see charts._layout)."""
    card = container.container(border=True)
    title = charts.title_of(fig)
    if title:
        card.markdown(f"**{title}**")
    card.plotly_chart(fig, width="stretch")


def phase_counter(job: Job) -> str:
    s = job.settings
    if job.lbfgs_start is None:
        return f"Adam {job.adam_iteration:,} / {s.adam_iters:,}"
    return f"L-BFGS {job.lbfgs_iteration:,} / {s.lbfgs_iters:,} (Adam terminado)"


def kpi(container, *args, **kwargs) -> None:
    """A metric in a card, so the numbers read as one row of tiles."""
    container.metric(*args, border=True, **kwargs)


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


def tab_data(settings: Settings) -> None:
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
        st.info("Todavía no hay datos cargados. Pulsa **Cargar datos**, o directamente "
                "**Entrenar** en la pestaña de entrenamiento (carga los datos por ti).")
        return
    key, data = data_entry
    if key != settings.data_key():
        st.warning("La configuración de datos de la barra lateral cambió desde la última carga. "
                   "Lo que ves abajo son los datos anteriores.")

    phys = (ss.result.cfg if ss.result else Config()).physics
    test_ids = {s.stint_id for s in ss.result.test.stints} if ss.result and ss.result.data is data else None

    deltas = np.concatenate([s.delta for s in data.stints])
    m = st.columns(5)
    kpi(m[0], "Stints", len(data))
    kpi(m[1], "Vueltas", f"{data.n_laps:,}")
    kpi(m[2], "Pilotos", len({s.driver for s in data.stints}))
    kpi(m[3], "Pérdida de ritmo mediana", f"{np.median(deltas):.2f} s")
    kpi(m[4], "Pérdida máxima", f"{deltas.max():.2f} s")
    st.caption(f"Fuente: {data.source}")

    highlight = None
    if test_ids and st.toggle("Resaltar los stints de prueba del último entrenamiento"):
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
        row.update({n: float(v) for n, v in zip(CONTEXT_NAMES, s.context, strict=False)})
        if test_ids is not None:
            row["Conjunto"] = "prueba" if s.stint_id in test_ids else "entrenamiento"
        rows.append(row)
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True, height=320)

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


def tab_train(settings: Settings) -> None:
    c1, c2, c3 = st.columns([1.3, 1, 3])
    if c1.button("▶️ Entrenar", type="primary", disabled=busy(), width="stretch"):
        start("train", settings)
        st.rerun()
    if busy() and ss.job.kind == "train" and c2.button("⏹ Detener", width="stretch", key="stop2"):
        ss.job.stop_requested.set()
    c3.caption(
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
            st.info("Pulsa **Entrenar** para empezar. Las curvas de pérdida y los parámetros "
                    "físicos se dibujan en vivo mientras la red aprende.")
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
        kpi(k[2], "Pérdida total", f"{total:.3e}" if total else "—",
                    f"÷{first / total:,.0f} desde el inicio" if total and first else None,
                    delta_color="normal", delta_arrow="down")
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


def tab_results() -> None:
    res = ss.result
    if res is None:
        st.info("Aquí aparecen las métricas y las predicciones cuando termine un entrenamiento.")
        return
    phys = res.cfg.physics
    st.caption(f"Entrenado en {fmt_seconds(res.seconds)} · {len(res.train)} stints de "
               f"entrenamiento, {len(res.test)} de prueba · guardado en `{res.out_dir}`")

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

    c1, c2 = st.columns(2)
    plot(c1, charts.metric_bars(res.metrics))
    plot(c2, charts.violation_bars(res.metrics))
    plot(st, charts.per_stint_rmse(res.metrics))

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

    st.subheader("Predicción stint a stint (conjunto de prueba)")
    ids = [s.stint_id for s in res.test.stints]
    c1, c2 = st.columns([3, 1])
    chosen = c1.selectbox("Stint de prueba", ids, key="res_stint")
    horizon = c2.slider("Horizonte [vueltas]", 10, 80, phys.strategy_horizon, key="res_h")
    stint = res.test.stints[ids.index(chosen)]
    plot(st, charts.stint_predictions(stint, res.models, phys, horizon))
    plot(st, charts.latent_states(pinn, stint, phys, horizon))

    st.subheader("Informe")
    report = (res.out_dir / "report.txt")
    text = report.read_text(encoding="utf-8") if report.exists() else res.report
    st.code(text, language=None)
    st.download_button("⬇️ Descargar informe", text, "report.txt")


def _available_models() -> list[Path]:
    base = ROOT / "outputs"
    return sorted({p.parent for p in base.rglob("pinn_weights.pt")}) if base.exists() else []


def tab_models() -> None:
    st.write("Abre un modelo entrenado antes (por la interfaz o por `run_train.py`) para usarlo "
             "en **Estrategia** sin volver a entrenar.")
    options = _available_models()
    if not options:
        st.info("No hay modelos en `outputs/` todavía.")
        return
    rel = [str(p.relative_to(ROOT)) for p in options]
    c1, c2 = st.columns([3, 1])
    chosen = c1.selectbox("Modelo", rel)
    if c2.button("📂 Cargar modelo", width="stretch"):
        folder = ROOT / chosen
        cfg_path = folder / "config.json"
        cfg = Config.from_json(str(cfg_path)) if cfg_path.exists() else Config()
        ss.loaded = {"pinn": TirePINN.load(folder, cfg), "cfg": cfg, "dir": folder}
        st.toast(f"Modelo cargado: {chosen}")

    loaded = ss.loaded
    if loaded is None:
        return
    folder, cfg, pinn = loaded["dir"], loaded["cfg"], loaded["pinn"]
    st.success(f"Modelo activo: `{folder.relative_to(ROOT)}` · fuente {cfg.data.source}")
    c1, c2 = st.columns(2)
    params = pinn.learned_params().as_dict()
    c1.dataframe(pd.DataFrame(
        [{"Parámetro": n, "Valor": v, "Estado": "estimado" if n in cfg.pinn.free_params else "fijo"}
         for n, v in params.items() if n != "p"]
    ).style.format(precision=4), width="stretch", hide_index=True)
    report = folder / "report.txt"
    if report.exists():
        c2.code(report.read_text(encoding="utf-8"), language=None)
    with st.expander("config.json"):
        st.json(asdict(cfg), expanded=False)


def _strategy_model():
    choices = {}
    if ss.result is not None:
        choices["Último entrenamiento"] = (ss.result.models["PINN"], ss.result.cfg)
    if ss.loaded is not None:
        choices[f"Modelo cargado ({ss.loaded['dir'].name})"] = (ss.loaded["pinn"], ss.loaded["cfg"])
    return choices


def tab_strategy() -> None:
    choices = _strategy_model()
    if not choices:
        st.info("Entrena un modelo o carga uno en **Modelos guardados** para usar el simulador.")
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


def tab_figures() -> None:
    dirs = []
    if ss.result is not None:
        dirs.append(Path(ss.result.out_dir).resolve())
    if ss.loaded is not None:
        dirs.append(Path(ss.loaded["dir"]).resolve())
    dirs += [p for p in _available_models() if p.resolve() not in dirs]
    if not dirs:
        st.info("Todavía no hay figuras. Aparecen al terminar un entrenamiento.")
        return
    labels = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in dirs]
    folder = dirs[labels.index(st.selectbox("Carpeta", labels))]
    captions = {
        "01_stints.png": "Predicciones de cada modelo sobre stints de prueba",
        "02_extrapolation.png": "Extrapolación más allá de lo observado",
        "03_latent_states.png": "Estados latentes θ y d",
        "04_parameters.png": "Convergencia de los parámetros físicos",
        "05_loss.png": "Términos de la función de pérdida",
        "06_cliff_map.png": "Mapa de decisión por compuesto",
    }
    images = sorted(folder.glob("*.png"))
    if not images:
        st.info("Esta carpeta no tiene figuras.")
    for img in images:
        st.image(str(img), caption=captions.get(img.name, img.name), width="stretch")
        st.download_button("⬇️ " + img.name, img.read_bytes(), img.name, "image/png",
                           key=f"dl-{folder}-{img.name}")
    for extra in ("report.txt", "pinn_params.json"):
        path = folder / extra
        if path.exists():
            with st.expander(extra):
                text = path.read_text(encoding="utf-8")
                st.code(json.dumps(json.loads(text), indent=2) if extra.endswith(".json") else text,
                        language=None)


# ======================================================================
def main() -> None:
    settings = sidebar()
    harvest()
    st.html(STYLE)
    header()
    status_bar()
    tabs = st.tabs(["📊 Datos", "🧠 Entrenamiento", "📈 Resultados", "🏁 Estrategia",
                    "🖼️ Figuras", "💾 Modelos guardados"])
    with tabs[0]:
        tab_data(settings)
    with tabs[1]:
        tab_train(settings)
    with tabs[2]:
        tab_results()
    with tabs[3]:
        tab_strategy()
    with tabs[4]:
        tab_figures()
    with tabs[5]:
        tab_models()


main()
