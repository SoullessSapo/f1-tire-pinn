"""
DESCARGAR LA BASE DE DATOS DESDE LA API DE FORMULA 1
====================================================

Este script baja telemetria real con FastF1, la limpia, calcula las variables
que necesita el modelo y lo deja todo en un CSV que puedes abrir en Excel.

    python descargar_datos.py --anio 2023 --carreras Monza --salida datos/monza.csv

    python descargar_datos.py --anio 2023 \\
        --carreras Monza Hungary Spa Silverstone \\
        --salida datos/2023.csv

    # Prueba rapida: sin telemetria (unos segundos en vez de varios minutos)
    python descargar_datos.py --anio 2023 --carreras Monza --sin-telemetria \\
        --salida datos/prueba.csv

Ejecuta `python descargar_datos.py --ayuda-variables` para ver de donde sale
cada columna del CSV.


EL PROBLEMA DE FONDO
--------------------
Nada de lo que el modelo necesita es observable.

La temperatura interna del neumatico, la carga vertical sobre cada rueda y el
estado de la banda de rodadura son datos propios de cada equipo y no se
publican. Lo unico publico es la telemetria de a bordo (velocidad, acelerador,
freno, marcha, posicion GPS) y los tiempos por vuelta.

Asi que las variables del modelo se RECONSTRUYEN a partir de lo que si hay:

    q_friccion   energia de friccion por vuelta = integral de |a| * v
    carga        carga mecanica media, en g, combinando lateral y longitudinal
    velocidad    velocidad media de la vuelta
    temp_pista   del canal meteorologico de la sesion
    compuesto    del propio dato de vuelta

La aceleracion lateral NO viene en la telemetria: se reconstruye derivando dos
veces la trayectoria GPS. Eso amplifica el ruido de muestreo, asi que antes hay
que suavizar.

Y el observable, la perdida de ritmo, necesita una correccion obligatoria que
se explica en el paso 4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# `np.trapezoid` es el nombre moderno; en numpy 1.x se llamaba `np.trapz`.
_integrar = getattr(np, "trapezoid", None) or np.trapz

GRAVEDAD = 9.80665


# ---------------------------------------------------------------------------
# 0) REFERENCIAS PARA NORMALIZAR
# ---------------------------------------------------------------------------
#
# Las tres variables de telemetria se dividen por una referencia fija para que
# queden alrededor de 1.
#
# MUY IMPORTANTE: son constantes, NO la mediana de la propia sesion.
#
# Normalizar cada carrera contra su propia mediana dejaria a Monza en 1.0 y a
# Hungria tambien en 1.0 -- dos circuitos radicalmente distintos -- y borraria
# justo la variacion entre circuitos que el modelo tiene que aprender. Esto fue
# un error real del proyecto y costo un rediseno; ver ROADMAP.md.
#
# Los valores salen de medir carreras de 2023 (Monza: 1877 / 3.39 / 66.4;
# Hungria: 1968 / 4.35 / 51.6).
REF_Q_FRICCION = 1900.0   # potencia friccional especifica [W/kg]
REF_CARGA = 3.8           # carga mecanica media [g]
REF_VELOCIDAD = 58.0      # velocidad media [m/s]


def describir_variables() -> str:
    return """
COLUMNAS DEL CSV
================

Identificacion
  carrera .......... nombre del Gran Premio
  piloto ........... codigo de tres letras
  stint_id ......... identificador unico del juego de neumaticos
  vuelta_carrera ... numero de vuelta de la carrera
  vuelta_stint ..... numero de vuelta DEL STINT (1 = el pico de rendimiento)

Lo que se mide
  tiempo_vuelta .... tiempo de vuelta en segundos, tal cual
  tiempo_corregido . tiempo de vuelta menos el efecto de combustible y pista
  delta ............ perdida de ritmo respecto a la mejor vuelta del stint [s]
                     <<< ES EL OBSERVABLE: lo unico que el modelo ajusta >>>

Las variables del modelo (constantes dentro del stint: la mediana del stint)
  q_friccion ....... energia de friccion por vuelta / 1900 W/kg
  carga ............ carga mecanica media / 3.8 g
  velocidad ........ velocidad media / 58 m/s
  temp_pista ....... temperatura de pista, reescalada a 0..1
  compuesto ........ 0 = blando, 0.5 = medio, 1 = duro

Para inspeccionar a mano
  compuesto_nombre . SOFT / MEDIUM / HARD
  edad_neumatico ... vueltas que lleva ese juego, segun la FIA
"""


# ---------------------------------------------------------------------------
# 1) SUAVIZADO
# ---------------------------------------------------------------------------

def _ventana_de_suavizado(tiempo: np.ndarray, segundos: float = 1.0) -> int:
    """Cuantas muestras son `segundos` de senal.

    La ventana se fija en SEGUNDOS y no en numero de muestras, y el motivo es
    concreto: FastF1 fusiona la telemetria del coche (~10 Hz) con la posicion
    GPS (~4 Hz) interpolando, asi que la frecuencia efectiva cambia de una
    vuelta a otra. Una ventana fija en muestras aplicaria un filtro fisico
    distinto en cada vuelta.
    """
    dt = float(np.median(np.diff(tiempo)))
    if not np.isfinite(dt) or dt <= 0:
        return 11
    # El filtro exige una ventana impar; `| 1` fuerza el impar mas cercano.
    return max(5, round(segundos / dt) | 1)


def _suavizar(valores: np.ndarray, ventana: int) -> np.ndarray:
    """Filtro de Savitzky-Golay: ajusta una parabola local y evalua el centro.

    Se prefiere a una media movil porque conserva los picos. Una media movil
    aplana las frenadas, que es justo la parte con mas informacion.
    """
    n = valores.size
    if n < 5:
        return valores

    ventana = min(ventana if ventana % 2 else ventana + 1, n if n % 2 else n - 1)
    if ventana < 5:
        return valores

    try:
        from scipy.signal import savgol_filter
        return savgol_filter(valores, ventana, 2)
    except Exception:
        # Si scipy no esta, una media movil es peor pero sirve.
        nucleo = np.ones(ventana) / ventana
        return np.convolve(valores, nucleo, mode="same")


# ---------------------------------------------------------------------------
# 2) LAS VARIABLES DINAMICAS DE UNA VUELTA
# ---------------------------------------------------------------------------

def variables_de_una_vuelta(telemetria) -> dict[str, float] | None:
    """Saca q_friccion, carga y velocidad de la telemetria de UNA vuelta.

    Devuelve None si la vuelta tiene demasiado pocas muestras para derivar.
    """
    if telemetria is None or len(telemetria) < 20:
        return None

    tiempo = telemetria["Time"].dt.total_seconds().to_numpy(dtype=float)
    velocidad = telemetria["Speed"].to_numpy(dtype=float) / 3.6   # km/h -> m/s

    # El tiempo tiene que ser estrictamente creciente para poder derivar.
    if not np.all(np.diff(tiempo) > 0):
        conservar = np.concatenate([[True], np.diff(tiempo) > 0])
        tiempo, velocidad = tiempo[conservar], velocidad[conservar]
        telemetria = telemetria.loc[conservar]
        if tiempo.size < 20:
            return None

    ventana = _ventana_de_suavizado(tiempo)
    velocidad = _suavizar(velocidad, ventana)

    # --- aceleracion longitudinal: derivar la velocidad ---
    a_longitudinal = np.gradient(velocidad, tiempo)

    # --- aceleracion lateral: derivar DOS VECES la trayectoria GPS ---
    if {"X", "Y"}.issubset(telemetria.columns):
        # FastF1 da X, Y en decimetros.
        x = _suavizar(telemetria["X"].to_numpy(dtype=float) / 10.0, ventana)
        y = _suavizar(telemetria["Y"].to_numpy(dtype=float) / 10.0, ventana)

        dx, dy = np.gradient(x, tiempo), np.gradient(y, tiempo)      # velocidad
        ddx, ddy = np.gradient(dx, tiempo), np.gradient(dy, tiempo)  # aceleracion

        # La componente de la aceleracion perpendicular a la velocidad es
        # |v x a| / |v|. Eso es exactamente la aceleracion lateral.
        rapidez = np.sqrt(dx ** 2 + dy ** 2)
        a_lateral = np.where(
            rapidez > 1.0,
            np.abs(dx * ddy - dy * ddx) / np.maximum(rapidez, 1e-6),
            0.0,
        )
        # 6 g es el techo real de un F1. Por encima es artefacto numerico.
        a_lateral = np.clip(_suavizar(a_lateral, ventana), 0.0, 6.0 * GRAVEDAD)
    else:
        a_lateral = np.zeros_like(velocidad)

    duracion = float(tiempo[-1] - tiempo[0])
    if duracion <= 0:
        return None

    a_total = np.sqrt(a_lateral ** 2 + a_longitudinal ** 2)

    return {
        # Potencia friccional especifica: |a| * v, promediada sobre la vuelta.
        "q_friccion_bruto": float(_integrar(a_total * velocidad, tiempo) / duracion),
        "carga_bruta": float(np.mean(a_total) / GRAVEDAD),
        "velocidad_bruta": float(np.mean(velocidad)),
    }


# ---------------------------------------------------------------------------
# 3) FILTROS DE CALIDAD
# ---------------------------------------------------------------------------

def vuelta_utilizable(vuelta, solo_goma_nueva: bool) -> bool:
    """Decide si una vuelta sirve para medir degradacion.

    Cada filtro quita vueltas cuyo tiempo cambia por motivos que NO son el
    neumatico. Si se dejaran, el modelo aprenderia que la degradacion depende
    del trafico.
    """
    import pandas as pd

    if pd.isna(vuelta.get("LapTime")):
        return False                              # sin tiempo
    if not bool(vuelta.get("IsAccurate", False)):
        return False                              # la propia FIA la marca dudosa
    if str(vuelta.get("TrackStatus", "")).strip() != "1":
        return False                              # bandera amarilla o safety car
    if pd.notna(vuelta.get("PitInTime")) or pd.notna(vuelta.get("PitOutTime")):
        return False                              # vuelta de entrada o salida de boxes
    if bool(vuelta.get("Deleted", False)):
        return False                              # anulada por limites de pista
    if solo_goma_nueva and not bool(vuelta.get("FreshTyre", True)):
        return False                              # d(0)=0 solo vale con goma nueva

    # Nota: se usa bool(...) y no `is False`. Segun la version de pandas el
    # valor llega como bool de Python o como numpy.bool_, y con `is` el filtro
    # fallaria en silencio en el segundo caso.
    return True


# ---------------------------------------------------------------------------
# 4) LA CORRECCION OBLIGATORIA: COMBUSTIBLE Y EVOLUCION DE PISTA
# ---------------------------------------------------------------------------

def estimar_efecto_vuelta_de_carrera(filas: list[dict]) -> float:
    """Cuanto baja el tiempo por vuelta a lo largo de la carrera, sin el neumatico.

    Dos cosas aceleran al coche segun avanza la carrera y no son la goma:
    quema unos 100 kg de combustible, y el asfalto se va engomando y da mas
    agarre. Sin corregirlo, el coche yendo cada vez mas rapido parece LO
    CONTRARIO de la degradacion, y la tapa entera.

    Las dos son funciones suaves y decrecientes de la vuelta de carrera, asi
    que NO son separables entre si. Lo que si se puede estimar es su suma, y es
    lo que devuelve esta funcion: un numero en segundos por vuelta, negativo.

    Como se identifica: los coches llevan neumaticos de edades distintas en la
    misma vuelta de carrera, porque paran en momentos diferentes. Si todos
    pararan a la vez, "vuelta de carrera" y "edad del neumatico" serian la
    misma variable y nada podria separarlas.

    El ajuste es:
        tiempo ~ piloto + pendiente*vuelta_carrera + degradacion(edad, compuesto)

    Los terminos de degradacion estan SOLO para que la pendiente no se los
    coma. Es un patron clasico de regresion: se incluye lo que no interesa para
    que no contamine lo que si.
    """
    if len(filas) < 50:
        return 0.0     # con tan pocas vueltas el ajuste no se sostiene

    pilotos = sorted({f["piloto"] for f in filas})
    compuestos = sorted({f["compuesto_nombre"] for f in filas})

    columnas = []
    for fila in filas:
        # Una columna por piloto (1 si es el suyo, 0 si no): absorbe que unos
        # coches sean mas rapidos que otros.
        fila_piloto = [1.0 if fila["piloto"] == p else 0.0 for p in pilotos]

        # La variable que nos interesa.
        vuelta = [float(fila["vuelta_carrera"])]

        # Una columna por compuesto con la edad del neumatico dentro.
        fila_degradacion = [
            float(fila["edad_neumatico"]) if fila["compuesto_nombre"] == c else 0.0
            for c in compuestos
        ]

        columnas.append(fila_piloto + vuelta + fila_degradacion)

    X = np.array(columnas, dtype=float)
    y = np.array([f["tiempo_vuelta"] for f in filas], dtype=float)

    validas = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[validas], y[validas]

    # Con menos de 5 vueltas por incognita el ajuste no es fiable.
    if X.shape[0] < 5 * X.shape[1]:
        return 0.0

    coeficientes = np.linalg.lstsq(
        X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ y, rcond=None
    )[0]

    # La pendiente es el coeficiente que sigue a las columnas de piloto.
    return float(coeficientes[len(pilotos)])


# ---------------------------------------------------------------------------
# 5) MONTAR LOS STINTS
# ---------------------------------------------------------------------------

def montar_stints(filas: list[dict], args) -> list[dict]:
    """Agrupa las vueltas en stints y calcula la perdida de ritmo de cada una.

    Aqui esta la decision mas sutil del fichero: DONDE EMPIEZA LA DEGRADACION.

    No en la primera vuelta. Un juego nuevo sale frio y se hace MAS RAPIDO
    durante dos o tres vueltas antes de empezar a caer: es la fase de
    calentamiento. El modelo es monotono por construccion y no puede
    representar eso, asi que si se dejara dentro, cada stint arrancaria con un
    conflicto irresoluble entre lo observado y lo predecible, y la red lo
    compensaria distorsionando las constantes fisicas.

    Asi que d = 0 se define en el PICO de rendimiento y las vueltas de
    calentamiento se descartan. En el proyecto principal esta correccion llevo
    el RMSE de 2.82 s a 0.567 s y las violaciones de monotonia del 16.4 % al 0 %.
    """
    from physics import INDICE_COMPUESTO

    # Agrupar por (piloto, numero de stint).
    grupos: dict[tuple, list[dict]] = {}
    for fila in filas:
        grupos.setdefault((fila["piloto"], fila["stint"]), []).append(fila)

    salida: list[dict] = []

    for (piloto, n_stint), vueltas in sorted(grupos.items()):
        vueltas.sort(key=lambda f: f["vuelta_carrera"])

        if len(vueltas) < args.min_vueltas:
            continue

        tiempos = np.array([f["tiempo_corregido"] for f in vueltas])

        # --- el pico: la vuelta mas rapida de las primeras ---
        ventana = min(args.vueltas_referencia + 1, len(tiempos))
        indice_pico = int(np.argmin(tiempos[:ventana]))

        vueltas = vueltas[indice_pico:]
        tiempos = tiempos[indice_pico:]

        # La perdida de ritmo se mide contra el pico, que pasa a ser la vuelta 1.
        delta = tiempos - tiempos[0]

        # Fuera las vueltas perdidas por trafico o por un error del piloto.
        conservar = delta <= args.max_delta
        if conservar.sum() < args.min_vueltas:
            continue

        vueltas = [v for v, ok in zip(vueltas, conservar) if ok]
        delta = delta[conservar]

        # --- el contexto: la mediana SOLO de las vueltas que se quedan ---
        # No del grupo entero: las de calentamiento ya se han descartado como
        # observacion y no deben influir en la mediana que representa al stint.
        contexto = {
            "q_friccion": float(np.median([v["q_friccion"] for v in vueltas])),
            "carga": float(np.median([v["carga"] for v in vueltas])),
            "velocidad": float(np.median([v["velocidad"] for v in vueltas])),
            "temp_pista": float(np.median([v["temp_pista"] for v in vueltas])),
            "compuesto": INDICE_COMPUESTO.get(vueltas[0]["compuesto_nombre"], 0.5),
        }

        carrera = vueltas[0]["carrera"]
        stint_id = f"{carrera[:3].upper()}-{piloto}-S{n_stint}"

        for i, (vuelta, d) in enumerate(zip(vueltas, delta), start=1):
            salida.append({
                "carrera": carrera,
                "piloto": piloto,
                "stint_id": stint_id,
                "vuelta_carrera": vuelta["vuelta_carrera"],
                "vuelta_stint": i,
                "tiempo_vuelta": round(vuelta["tiempo_vuelta"], 3),
                "tiempo_corregido": round(vuelta["tiempo_corregido"], 3),
                "delta": round(float(d), 3),
                "compuesto_nombre": vuelta["compuesto_nombre"],
                "edad_neumatico": vuelta["edad_neumatico"],
                **{k: round(v, 4) for k, v in contexto.items()},
            })

    return salida


# ---------------------------------------------------------------------------
# 6) UNA CARRERA, DE PRINCIPIO A FIN
# ---------------------------------------------------------------------------

def procesar_carrera(nombre_carrera: str, args) -> list[dict]:
    """Descarga una carrera y devuelve sus filas listas para el CSV."""
    import fastf1
    import pandas as pd

    print(f"\n  {nombre_carrera}")
    print(f"    descargando sesion {args.sesion} de {args.anio} ...")

    fastf1.Cache.enable_cache(args.cache)
    sesion = fastf1.get_session(args.anio, nombre_carrera, args.sesion)
    sesion.load(
        laps=True,
        telemetry=not args.sin_telemetria,
        weather=True,
        messages=False,
    )

    vueltas_sesion = sesion.laps
    if vueltas_sesion is None or len(vueltas_sesion) == 0:
        raise RuntimeError("la sesion no trae vueltas")

    # --- temperatura de pista a lo largo de la sesion ---
    meteo = getattr(sesion, "weather_data", None)
    if meteo is not None and len(meteo) and "TrackTemp" in meteo:
        meteo_t = meteo["Time"].dt.total_seconds().to_numpy(dtype=float)
        meteo_temp = meteo["TrackTemp"].to_numpy(dtype=float)
    else:
        meteo_t = meteo_temp = None

    pilotos = list(args.pilotos) if args.pilotos else sorted(
        vueltas_sesion["Driver"].dropna().unique()
    )
    if args.max_pilotos:
        pilotos = pilotos[: args.max_pilotos]

    print(f"    {len(pilotos)} pilotos, procesando vueltas ...")

    filas: list[dict] = []
    descartadas = 0

    for piloto in pilotos:
        vueltas_piloto = vueltas_sesion[vueltas_sesion["Driver"] == piloto]

        for _, vuelta in vueltas_piloto.iterrows():
            if not vuelta_utilizable(vuelta, args.solo_goma_nueva):
                descartadas += 1
                continue

            # --- las tres variables de telemetria ---
            if args.sin_telemetria:
                # Sin telemetria no se pueden calcular: se dejan en 1.0 (el
                # valor de referencia). El modelo seguira viendo variar la
                # temperatura y el compuesto, pero no la carga ni la energia.
                dinamica = {
                    "q_friccion_bruto": REF_Q_FRICCION,
                    "carga_bruta": REF_CARGA,
                    "velocidad_bruta": REF_VELOCIDAD,
                }
            else:
                try:
                    dinamica = variables_de_una_vuelta(vuelta.get_telemetry())
                except Exception:
                    dinamica = None
                if dinamica is None:
                    descartadas += 1
                    continue

            # --- temperatura de pista en el momento de esa vuelta ---
            inicio = vuelta.get("LapStartTime")
            inicio_s = inicio.total_seconds() if pd.notna(inicio) else np.nan
            if meteo_t is not None and np.isfinite(inicio_s):
                temp_pista_c = float(np.interp(inicio_s, meteo_t, meteo_temp))
            else:
                temp_pista_c = 35.0     # valor tipico si no hay dato

            edad = vuelta.get("TyreLife")

            filas.append({
                "carrera": nombre_carrera,
                "piloto": str(piloto),
                "stint": int(vuelta.get("Stint", 1)),
                "vuelta_carrera": int(vuelta["LapNumber"]),
                "tiempo_vuelta": float(vuelta["LapTime"].total_seconds()),
                "compuesto_nombre": str(vuelta.get("Compound", "MEDIUM")).upper(),
                "edad_neumatico": float(edad) if pd.notna(edad) else float("nan"),
                # normalizadas contra las referencias FIJAS
                "q_friccion": dinamica["q_friccion_bruto"] / REF_Q_FRICCION,
                "carga": dinamica["carga_bruta"] / REF_CARGA,
                "velocidad": dinamica["velocidad_bruta"] / REF_VELOCIDAD,
                # la temperatura se reescala de grados a 0..1
                "temp_pista": float(np.clip((temp_pista_c - 20.0) / 40.0, 0.0, 1.0)),
            })

    if not filas:
        raise RuntimeError("ninguna vuelta paso los filtros de calidad")

    print(f"    {len(filas)} vueltas validas ({descartadas} descartadas)")

    # --- la correccion de combustible + evolucion de pista ---
    if args.efecto_vuelta == "auto":
        pendiente = estimar_efecto_vuelta_de_carrera(filas)
        print(f"    efecto de vuelta de carrera estimado: {pendiente:+.4f} s/vuelta")
    else:
        pendiente = float(args.efecto_vuelta)
        print(f"    efecto de vuelta de carrera fijado a: {pendiente:+.4f} s/vuelta")

    for fila in filas:
        fila["tiempo_corregido"] = fila["tiempo_vuelta"] - pendiente * fila["vuelta_carrera"]

    # Si falta la edad del neumatico, se usa la posicion dentro del stint.
    for fila in filas:
        if not np.isfinite(fila["edad_neumatico"]):
            fila["edad_neumatico"] = float(fila["vuelta_carrera"])

    return montar_stints(filas, args)


# ---------------------------------------------------------------------------
# 7) LINEA DE COMANDOS
# ---------------------------------------------------------------------------

def parsear_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Descarga telemetria de F1 y la convierte en un CSV para entrenar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ejemplo:\n"
               "  python descargar_datos.py --anio 2023 --carreras Monza Hungary "
               "--salida datos/2023.csv",
    )

    p.add_argument("--ayuda-variables", action="store_true",
                   help="explica cada columna del CSV y termina")

    g = p.add_argument_group("que descargar")
    g.add_argument("--anio", type=int, default=2023, help="temporada (por defecto 2023)")
    g.add_argument("--carreras", nargs="+", default=["Monza"],
                   help="uno o mas Grandes Premios: Monza Hungary Spa ...")
    g.add_argument("--sesion", default="R",
                   help="R=carrera, Q=clasificacion, FP1/FP2/FP3=libres")
    g.add_argument("--pilotos", nargs="*", default=[],
                   help="codigos de tres letras, p.ej. VER HAM LEC. Vacio = todos")
    g.add_argument("--max-pilotos", type=int, default=0,
                   help="limitar a los N primeros pilotos (util para probar)")

    g = p.add_argument_group("calidad de los datos")
    g.add_argument("--min-vueltas", type=int, default=8,
                   help="descarta stints con menos vueltas validas (por defecto 8)")
    g.add_argument("--max-delta", type=float, default=6.0,
                   help="descarta vueltas que pierden mas de N segundos (trafico)")
    g.add_argument("--vueltas-referencia", type=int, default=3,
                   help="cuantas vueltas se miran para encontrar el pico de rendimiento")
    g.add_argument("--solo-goma-nueva", action="store_true", default=True,
                   help="usar solo juegos nuevos (d=0 solo vale con goma nueva)")
    g.add_argument("--incluir-goma-usada", dest="solo_goma_nueva",
                   action="store_false", help="incluir tambien juegos ya rodados")
    g.add_argument("--efecto-vuelta", default="auto",
                   help="correccion de combustible+pista en s/vuelta. "
                        "'auto' la estima por carrera; o da un numero, p.ej. -0.055")

    g = p.add_argument_group("velocidad y salida")
    g.add_argument("--sin-telemetria", action="store_true",
                   help="no descargar telemetria: mucho mas rapido, pero deja "
                        "q_friccion, carga y velocidad fijas en 1.0")
    g.add_argument("--cache", default="cache",
                   help="carpeta de cache de FastF1 (se llena sola, ~200 MB)")
    g.add_argument("--salida", default="datos/carreras.csv",
                   help="fichero CSV de salida")

    return p.parse_args()


def main() -> int:
    args = parsear_argumentos()

    if args.ayuda_variables:
        print(describir_variables())
        return 0

    try:
        import fastf1  # noqa: F401
    except ImportError:
        print("Falta FastF1. Instalalo con:\n  pip install -r requirements.txt")
        return 1

    print("=" * 70)
    print(f"Descargando {args.sesion} de {args.anio}: {', '.join(args.carreras)}")
    print("=" * 70)
    if args.sin_telemetria:
        print("\n  MODO RAPIDO: sin telemetria. q_friccion, carga y velocidad")
        print("  quedaran fijas en 1.0 y el modelo no podra aprender su efecto.")

    todas: list[dict] = []
    fallos: list[str] = []

    for carrera in args.carreras:
        try:
            filas = procesar_carrera(carrera, args)
        except Exception as error:
            fallos.append(f"{carrera}: {error}")
            print(f"    FALLO: {error}")
            continue

        n_stints = len({f["stint_id"] for f in filas})
        print(f"    -> {n_stints} stints, {len(filas)} vueltas")
        todas.extend(filas)

    if not todas:
        print("\nNinguna carrera produjo datos validos:")
        for fallo in fallos:
            print(f"  {fallo}")
        return 1

    # --- escribir el CSV ---
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    import csv
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(todas[0].keys()))
        escritor.writeheader()
        escritor.writerows(todas)

    n_stints = len({f["stint_id"] for f in todas})
    deltas = np.array([f["delta"] for f in todas])

    print("\n" + "=" * 70)
    print(f"Guardado en {salida.resolve()}")
    print(f"  {n_stints} stints | {len(todas)} vueltas | {len(args.carreras)} carreras")
    print(f"  Perdida de ritmo: mediana {np.median(deltas):.2f}s  max {deltas.max():.2f}s")
    if fallos:
        print(f"  {len(fallos)} carreras fallaron: {'; '.join(fallos)}")
    print("\nAhora puedes entrenar con:")
    print(f"  python run.py --fuente csv --csv {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
