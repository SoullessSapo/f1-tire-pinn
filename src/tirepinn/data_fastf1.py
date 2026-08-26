"""Carga de telemetria real de F1 y construccion de las variables del modelo.

El problema central de los datos reales es que **nada de lo que el modelo
necesita es directamente observable**: la temperatura interna del neumatico, la
carga vertical y el estado de la banda de rodadura son datos propietarios de
cada equipo. Lo unico publico es la telemetria de a bordo (velocidad,
acelerador, freno, marcha, posicion GPS) y los tiempos por vuelta.

Este modulo cierra esa brecha con variables proxy derivadas de la telemetria:

    q_fric  energia friccional especifica por vuelta, integrando el producto de
            la aceleracion total por la velocidad. Es el termino de generacion
            de calor de (E1).
    load    carga mecanica media en g, combinando aceleracion lateral y
            longitudinal. Es el termino de Archard de (E2).
    speed   velocidad media, que gobierna el enfriamiento convectivo.

La aceleracion lateral no viene en la telemetria: se reconstruye derivando dos
veces la trayectoria GPS, con suavizado previo porque la doble derivada
numerica amplifica el ruido de muestreo.

El observable de degradacion es la perdida de ritmo corregida por combustible:
un coche se aligera unos 100 kg a lo largo de la carrera y eso solo vale mas de
un segundo por vuelta, asi que sin corregirlo la degradacion queda enmascarada.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import COMPOUND_INDEX, DataConfig, PhysicsConfig
from .dataset import Stint, StintDataset

_G = 9.80665


def _require_fastf1():
    try:
        import fastf1
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastF1 no esta instalado. Instala las dependencias con "
            "`pip install -r requirements.txt` o usa --source synthetic."
        ) from exc
    return fastf1


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Suavizado Savitzky-Golay, con respaldo de media movil si scipy falla."""
    n = values.size
    if n < 5:
        return values
    window = min(window if window % 2 else window + 1, n if n % 2 else n - 1)
    if window < 5:
        return values
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(values, window, 2)
    except Exception:  # pragma: no cover
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="same")


def _smoothing_window(t: np.ndarray, seconds: float = 1.0) -> int:
    """Ventana de suavizado en muestras, fijada por tiempo y no por conteo.

    FastF1 fusiona la telemetria del coche (~10 Hz) con la posicion GPS (~4 Hz)
    interpolando, asi que la frecuencia efectiva cambia entre vueltas y sesiones.
    Una ventana fija en muestras suavizaria distinto en cada caso; una ventana
    fija en segundos aplica el mismo filtro fisico siempre. Importa porque la
    aceleracion lateral sale de una doble derivada, y los tramos interpolados
    linealmente producen segundas derivadas artificialmente grandes.
    """
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return 11
    return max(5, round(seconds / dt) | 1)


def _lap_dynamics(tel: pd.DataFrame) -> dict[str, float] | None:
    """Extrae las variables dinamicas de una vuelta a partir de su telemetria.

    Devuelve None si la vuelta no tiene muestras suficientes para derivar.
    """
    if tel is None or len(tel) < 20:
        return None

    t = tel["Time"].dt.total_seconds().to_numpy(dtype=float)
    speed = tel["Speed"].to_numpy(dtype=float) / 3.6  # km/h -> m/s
    if not np.all(np.diff(t) > 0):
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, speed, tel = t[keep], speed[keep], tel.loc[keep]
        if t.size < 20:
            return None

    window = _smoothing_window(t)
    speed = _smooth(speed, window)
    a_long = np.gradient(speed, t)

    # Aceleracion lateral desde la trayectoria GPS. FastF1 entrega X, Y en
    # decimetros; el modulo del producto cruzado velocidad-aceleracion dividido
    # por la rapidez da directamente la componente normal de la aceleracion.
    if {"X", "Y"}.issubset(tel.columns):
        x = _smooth(tel["X"].to_numpy(dtype=float) / 10.0, window)
        y = _smooth(tel["Y"].to_numpy(dtype=float) / 10.0, window)
        dx, dy = np.gradient(x, t), np.gradient(y, t)
        ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
        planar = np.sqrt(dx**2 + dy**2)
        a_lat = np.where(planar > 1.0, np.abs(dx * ddy - dy * ddx) / np.maximum(planar, 1e-6), 0.0)
        a_lat = np.clip(_smooth(a_lat, window), 0.0, 6.0 * _G)  # 6 g es el techo de un F1
    else:  # pragma: no cover - sesiones sin datos de posicion
        a_lat = np.zeros_like(speed)

    duration = float(t[-1] - t[0])
    if duration <= 0:
        return None

    a_total = np.sqrt(a_lat**2 + a_long**2)
    # Potencia friccional especifica: |a| * v, promediada en el tiempo de vuelta.
    q_fric = float(np.trapezoid(a_total * speed, t) / duration)
    return {
        "q_fric_raw": q_fric,
        "load_raw": float(np.mean(a_total) / _G),
        "speed_raw": float(np.mean(speed)),
    }


def _track_temp_series(session) -> tuple[np.ndarray, np.ndarray] | None:
    weather = getattr(session, "weather_data", None)
    if weather is None or len(weather) == 0 or "TrackTemp" not in weather:
        return None
    t = weather["Time"].dt.total_seconds().to_numpy(dtype=float)
    temp = weather["TrackTemp"].to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(temp)
    if ok.sum() < 2:
        return None
    return t[ok], temp[ok]


def _is_green(track_status) -> bool:
    """Solo se conservan vueltas en bandera verde.

    Un safety car o una bandera amarilla cambian el tiempo por vuelta varios
    segundos por razones que no tienen nada que ver con el neumatico.
    """
    if track_status is None or (isinstance(track_status, float) and np.isnan(track_status)):
        return False
    return str(track_status).strip() == "1"


def load_session(cfg: DataConfig):
    """Descarga (o lee de cache) la sesion pedida."""
    fastf1 = _require_fastf1()
    fastf1.Cache.enable_cache(cfg.cache_dir)
    session = fastf1.get_session(cfg.year, cfg.gp, cfg.session)
    session.load(laps=True, telemetry=True, weather=True, messages=False)
    return session


def build_dataset(cfg: DataConfig, phys: PhysicsConfig, session=None) -> StintDataset:
    """Construye el conjunto de stints a partir de una sesion de FastF1."""
    session = session or load_session(cfg)
    laps = session.laps
    if laps is None or len(laps) == 0:
        raise RuntimeError("La sesion no contiene vueltas cargadas")

    total_laps = int(getattr(session, "total_laps", 0) or laps["LapNumber"].max())
    weather = _track_temp_series(session)
    drivers = list(cfg.drivers) if cfg.drivers else sorted(laps["Driver"].dropna().unique())

    # --- Paso 1: caracteristicas crudas por vuelta ---
    records: list[dict] = []
    for driver in drivers:
        driver_laps = laps[laps["Driver"] == driver]
        for _, lap in driver_laps.iterrows():
            if pd.isna(lap.get("LapTime")) or not bool(lap.get("IsAccurate", False)):
                continue
            if not _is_green(lap.get("TrackStatus")):
                continue
            if pd.notna(lap.get("PitInTime")) or pd.notna(lap.get("PitOutTime")):
                continue
            if bool(lap.get("Deleted", False)):
                continue
            # `bool(...)` en lugar de `is False`: segun la version de pandas el
            # valor puede llegar como bool de Python o como numpy.bool_, y con
            # `is` el filtro fallaria en silencio para el segundo caso.
            if cfg.only_fresh_tyres and not bool(lap.get("FreshTyre", True)):
                continue

            try:
                # `iterrows` sobre un `Laps` devuelve objetos `Lap`, que ya
                # saben recuperar su propia telemetria fusionada (coche + GPS).
                tel = lap.get_telemetry()
            except Exception:
                continue
            dyn = _lap_dynamics(tel)
            if dyn is None:
                continue

            lap_start = lap.get("LapStartTime")
            start_s = lap_start.total_seconds() if pd.notna(lap_start) else np.nan
            if weather is not None and np.isfinite(start_s):
                track_temp = float(np.interp(start_s, weather[0], weather[1]))
            else:
                track_temp = 35.0

            tyre_life = lap.get("TyreLife")
            records.append(
                {
                    "driver": driver,
                    "stint": int(lap.get("Stint", 1)),
                    "lap_number": int(lap["LapNumber"]),
                    "tyre_life": float(tyre_life) if pd.notna(tyre_life) else np.nan,
                    "compound": str(lap.get("Compound", "MEDIUM")).upper(),
                    "lap_time": float(lap["LapTime"].total_seconds()),
                    "track_temp": track_temp,
                    **dyn,
                }
            )

    if not records:
        raise RuntimeError(
            "Ningun stint sobrevivio a los filtros de calidad. Prueba con otra "
            "sesion, mas pilotos, o relaja `only_fresh_tyres`."
        )

    df = pd.DataFrame.from_records(records)

    # --- Paso 2: adimensionalizacion contra referencias fijas ---
    # Las referencias son constantes de `DataConfig`, no estadisticos de la
    # sesion. Es lo que permite entrenar sobre varias carreras a la vez: un
    # circuito de alta carga y otro de baja velocidad quedan en puntos
    # distintos del espacio de contexto, en lugar de colapsar los dos a 1.0.
    df["q_fric"] = df["q_fric_raw"] / cfg.q_fric_ref
    df["load"] = df["load_raw"] / cfg.load_ref
    df["speed"] = df["speed_raw"] / cfg.speed_ref
    df["track_temp_norm"] = np.clip((df["track_temp"] - 20.0) / 40.0, 0.0, 1.0)
    df["compound_idx"] = df["compound"].map(lambda c: COMPOUND_INDEX.get(c, 0.5))

    # --- Paso 3: correccion de combustible ---
    # El coche pierde ~1.7 kg por vuelta y cada 10 kg valen ~0.3 s. Sin corregir,
    # la mejora por aligeramiento se confunde con lo contrario de la degradacion.
    df["lap_time_corr"] = df["lap_time"] - cfg.fuel_effect_s_per_lap * (total_laps - df["lap_number"])

    # --- Paso 4: armado de stints ---
    stints: list[Stint] = []
    for (driver, stint_no), group in df.groupby(["driver", "stint"], sort=True):
        group = group.sort_values("lap_number")
        if len(group) < cfg.min_stint_laps:
            continue

        # Edad del neumatico: TyreLife respeta los juegos ya usados; si falta,
        # se cae a la posicion dentro del stint.
        life = group["tyre_life"].to_numpy(dtype=float)
        if not np.all(np.isfinite(life)):
            life = np.arange(1, len(group) + 1, dtype=float)

        # Origen de la degradacion: el PICO de rendimiento del neumatico, no su
        # primera vuelta. Un juego nuevo sale frio y se hace mas rapido durante
        # dos o tres vueltas antes de empezar a caer. El modelo es monotono por
        # construccion, asi que no puede representar esa fase de calentamiento:
        # se descarta, y d = 0 se define en el pico. Sin este anclaje cada stint
        # arranca con medio segundo de desfase contra la prediccion, y ese
        # conflicto sistematico degenera la estimacion de los parametros.
        times = group["lap_time_corr"].to_numpy()
        ref_idx = int(np.argmin(times[: cfg.ref_window + 1]))

        times = times[ref_idx:]
        life = life[ref_idx:]
        delta = times - times[0]
        age = life - life[0] + 1.0  # el pico pasa a ser la vuelta 1 del stint

        keep = delta <= cfg.max_delta_s  # descarta trafico y errores de pilotaje
        if keep.sum() < cfg.min_stint_laps:
            continue

        # El contexto se resume sobre las vueltas que de verdad entran al ajuste,
        # no sobre el grupo completo: las de calentamiento ya se descartaron y no
        # deben influir en la mediana que representa al stint.
        used = group.iloc[ref_idx:][keep]
        context = np.array(
            [
                float(used["q_fric"].median()),
                float(used["load"].median()),
                float(used["speed"].median()),
                float(used["track_temp_norm"].median()),
                float(used["compound_idx"].iloc[0]),
            ]
        )
        stints.append(
            Stint(
                stint_id=f"{driver}-S{stint_no}",
                driver=str(driver),
                compound=str(used["compound"].iloc[0]),
                laps=age[keep],
                delta=delta[keep],
                context=context,
                race_laps=group["lap_number"].to_numpy()[ref_idx:][keep],
            )
        )

    if not stints:
        raise RuntimeError(
            f"No hay stints con al menos {cfg.min_stint_laps} vueltas validas. "
            "Baja `min_stint_laps` o elige una carrera con menos neutralizaciones."
        )

    return StintDataset(
        stints=stints,
        source=f"fastf1:{cfg.year}-{cfg.gp}-{cfg.session}",
        meta={
            "total_laps": total_laps,
            "drivers": drivers,
            "fuel_effect_s_per_lap": cfg.fuel_effect_s_per_lap,
            "laps_after_filters": len(df),
        },
    )


def build_multi_dataset(cfg: DataConfig, phys: PhysicsConfig, gps: Sequence[str]) -> StintDataset:
    """Combina varias carreras en un solo conjunto.

    Es la forma recomendada de entrenar con datos reales. En una sola carrera
    las variables de contexto casi no varian (mismo circuito, mismo clima), asi
    que la red no puede aprender como responde la degradacion a las
    condiciones: solo ve el efecto del compuesto y del tiempo. Con varias
    carreras el espacio de contexto se puebla de verdad.
    """
    all_stints: list[Stint] = []
    sources, failures = [], []

    for gp in gps:
        race_cfg = replace(cfg, gp=gp)
        try:
            part = build_dataset(race_cfg, phys)
        except Exception as exc:
            failures.append(f"{gp}: {exc}")
            continue
        for stint in part.stints:
            stint.stint_id = f"{gp[:3].upper()}-{stint.stint_id}"
        all_stints.extend(part.stints)
        sources.append(part.source)

    if not all_stints:
        raise RuntimeError("Ninguna carrera produjo stints validos:\n  " + "\n  ".join(failures))

    return StintDataset(
        stints=all_stints,
        source=" + ".join(sources),
        meta={"races": list(gps), "failures": failures},
    )
