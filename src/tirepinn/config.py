"""Configuracion central del proyecto.

Todos los hiperparametros (fisicos, de red y de datos) viven aqui para que los
experimentos sean reproducibles y el reporte pueda citar valores concretos.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

# Orden canonico del vector de entrada de la red: (tau, contexto...)
CONTEXT_NAMES = ("q_fric", "load", "speed", "track_temp", "compound")
INPUT_NAMES = ("tau", *CONTEXT_NAMES)
INPUT_DIM = len(INPUT_NAMES)
OUTPUT_NAMES = ("theta", "d")
OUTPUT_DIM = len(OUTPUT_NAMES)

# Parametros fisicos que la red puede estimar como problema inverso.
LEARNABLE_PARAMS = ("zeta", "h0", "h1", "kw", "m", "Ea", "kappa", "gamma1", "gamma2")

# Subconjunto que se estima cuando se entrena con telemetria real. El resto se
# fija a los valores calibrados en el banco fisico.
#
# La razon es de identificabilidad, no de comodidad. Una carrera aporta unas
# decenas de stints con un suelo de ruido de ~0.5 s por vuelta y poca variacion
# de condiciones, que no alcanza para separar nueve coeficientes: al intentarlo,
# los exponentes termicos colapsan a cero.
#
# Y hay algo mas de fondo. La escala absoluta de d NO es identificable desde
# datos de carrera, porque el unico observable es delta y nada ancla d salvo la
# saturacion en d = 1, a la que solo se llega destruyendo el neumatico. Los
# equipos paran mucho antes. Dejar gamma libre hace que el ajuste empuje d hacia
# cero y gamma hacia arriba hasta chocar con la cota: se comprobo con Monza +
# Hungria, donde gamma1 y gamma2 terminaron pegadas a sus dos topes.
#
# Por eso gamma se toma tambien de la calibracion: "un segundo de perdida de
# ritmo equivale a este desgaste" es una afirmacion de calibracion, no algo que
# los tiempos por vuelta puedan responder. Quedan libres las dos cantidades que
# si varian de un circuito o un compuesto a otro.
REAL_DATA_FREE_PARAMS = ("kw", "kappa")

# Indice de dureza del compuesto: 0 = mas blando, 1 = mas duro.
COMPOUND_INDEX = {
    "HYPERSOFT": 0.0,
    "ULTRASOFT": 0.0,
    "SUPERSOFT": 0.0,
    "SOFT": 0.0,
    "MEDIUM": 0.5,
    "HARD": 1.0,
    "INTERMEDIATE": 0.75,
    "WET": 1.0,
}


@dataclass(frozen=True)
class ContextRanges:
    """Rangos fisicos admisibles de cada variable de contexto (ya normalizada).

    Definen el hipercubo sobre el que se imponen los residuos de la EDO, es
    decir, la region donde la fisica regulariza a la red aunque no haya datos.
    """

    tau: tuple[float, float] = (0.0, 1.40)
    q_fric: tuple[float, float] = (0.40, 1.60)
    load: tuple[float, float] = (0.50, 1.50)
    speed: tuple[float, float] = (0.60, 1.40)
    track_temp: tuple[float, float] = (0.00, 1.00)
    compound: tuple[float, float] = (0.00, 1.00)

    def lows(self) -> list[float]:
        return [getattr(self, n)[0] for n in INPUT_NAMES]

    def highs(self) -> list[float]:
        return [getattr(self, n)[1] for n in INPUT_NAMES]


@dataclass
class PhysicsConfig:
    """Escalas de adimensionalizacion y parametros del modelo termo-mecanico."""

    # --- escalas ---
    lap_ref: float = 30.0          # L_ref: vueltas por unidad de tau
    dT_ref: float = 40.0           # Delta T_ref [K] para adimensionalizar theta
    theta_init: float = 0.5        # theta(tau=0): salida de boxes con mantas
    d_max: float = 1.10            # cota superior admisible de la degradacion

    # --- parametro que ancla la escala de temperatura (ver README: identificabilidad) ---
    A_gen: float = 8.0
    train_A_gen: bool = False

    # --- valores iniciales de los parametros aprendibles (problema inverso) ---
    zeta_init: float = 0.50        # acoplamiento desgaste -> temperatura (cliff)
    h0_init: float = 4.0           # enfriamiento base
    h1_init: float = 2.0           # enfriamiento forzado por velocidad
    kw_init: float = 0.80          # coeficiente de desgaste (Archard)
    m_init: float = 1.00           # exponente de carga mecanica
    Ea_init: float = 0.60          # activacion termica (Arrhenius linealizado)
    kappa_init: float = 0.50       # resistencia del compuesto al desgaste
    gamma1_init: float = 1.00      # perdida de ritmo lineal [s]
    gamma2_init: float = 1.50      # perdida de ritmo del cliff [s]

    # Cotas fisicas duras sobre los parametros del observable. Son necesarias,
    # no cosmeticas: (d, gamma1, gamma2) tienen una degeneracion exacta de
    # escala, porque d -> e*d con gamma1 -> gamma1/e y gamma2 -> gamma2/e^p
    # deja delta identica. Con datos sinteticos la degeneracion la rompen el
    # proxy termico y los stints que saturan en d = 1; con datos reales no hay
    # ninguna de las dos cosas y el optimizador se va por esa direccion hasta
    # desbordar. Lo que la cierra es una afirmacion fisica sencilla: un
    # neumatico destruido cuesta unos segundos por vuelta, no millones.
    gamma1_bounds: tuple[float, float] = (0.2, 4.0)   # [s]
    gamma2_bounds: tuple[float, float] = (0.2, 6.0)   # [s]

    # --- observable de ritmo ---
    cliff_exponent: float = 8.0    # p en delta = g1*d + g2*d^p

    # --- definicion operativa del cliff ---
    cliff_slope_s_per_lap: float = 0.15
    d_crit: float = 0.85
    # Horizonte de decision estrategica. Define hasta donde se impone la EDO:
    # el dominio fisico lo fija la pregunta que se le va a hacer al modelo, no
    # la longitud de los stints que se alcanzaron a observar.
    strategy_horizon: int = 45
    exp_clamp: float = 6.0         # cota del exponente de Arrhenius (estabilidad)


@dataclass
class PINNConfig:
    """Arquitectura y regimen de entrenamiento de la PINN (DeepXDE)."""

    hidden: Sequence[int] = (64, 64, 64, 64)
    activation: str = "tanh"
    initializer: str = "Glorot normal"

    lr: float = 1e-3
    adam_iters: int = 15000
    lbfgs_iters: int = 3000
    display_every: int = 1000

    num_domain: int = 4000         # colocacion uniforme en el hipercubo
    tau_collocation: int = 40      # puntos en tau por contexto observado

    # pesos de las componentes de la funcion de perdida
    w_pde_theta: float = 1.0
    w_pde_wear: float = 1.0
    w_bound: float = 10.0
    w_data_delta: float = 20.0
    w_data_theta: float = 1.0

    use_theta_proxy: bool = True   # supervision debil sobre la temperatura
    # Parametros fisicos que se estiman. Los que no esten aqui se mantienen
    # fijos en el valor de `PhysicsConfig`.
    free_params: tuple[str, ...] = LEARNABLE_PARAMS
    seed: int = 42
    float64: bool = False


@dataclass
class DataConfig:
    """Origen y preprocesamiento de los stints."""

    source: str = "synthetic"      # {"synthetic", "fastf1"}

    # --- generador sintetico (banco de pruebas con verdad fisica conocida) ---
    n_stints: int = 40
    min_stint: int = 12
    max_stint: int = 34
    noise_delta_s: float = 0.06
    noise_theta: float = 0.05

    # --- FastF1 ---
    year: int = 2023
    gp: str = "Monza"
    session: str = "R"
    drivers: tuple[str, ...] = ()
    cache_dir: str = "cache"
    fuel_effect_s_per_lap: float = 0.055   # correccion de combustible [s/vuelta]

    # Referencias fijas para adimensionalizar los proxys de telemetria.
    # Deliberadamente NO son la mediana de la sesion: normalizar cada carrera
    # contra si misma dejaria a Monza y a Hungria ambas en 1.0 y borraria
    # justamente la variacion entre circuitos que el modelo necesita aprender.
    # Calibradas sobre carreras de 2023 (Monza: 1877 / 3.39 / 66.4;
    # Hungria: 1968 / 4.35 / 51.6).
    q_fric_ref: float = 1900.0     # potencia friccional especifica [W/kg]
    load_ref: float = 3.8          # carga mecanica media [g]
    speed_ref: float = 58.0        # velocidad media [m/s]
    min_stint_laps: int = 8
    ref_window: int = 3            # vueltas para el ritmo de referencia del stint
    max_delta_s: float = 6.0       # descarta vueltas con trafico/incidentes
    only_fresh_tyres: bool = True  # d(0)=0 solo vale para un juego nuevo

    test_fraction: float = 0.25


@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    pinn: PINNConfig = field(default_factory=PINNConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ranges: ContextRanges = field(default_factory=ContextRanges)
    out_dir: str = "outputs"

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)
