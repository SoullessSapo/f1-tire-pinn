"""Real F1 data through the FastF1 API: sessions, per-lap features and stints.

The central problem with real data is that **nothing the model needs is
directly observable**: the tire's internal temperature, the vertical load and
the state of the tread are proprietary to each team. All that is public is
onboard telemetry (speed, throttle, brake, gear, GPS position), lap timing and
the session's weather feed.

This module bridges that gap with proxy variables derived from telemetry:

    q_fric  specific frictional energy per lap, integrating total acceleration
            times speed. This is the heat-generation term of (E1).
    load    mean mechanical load in g, combining lateral and longitudinal
            acceleration. This is the Archard term of (E2).
    speed   mean speed, which governs convective cooling.

Lateral acceleration is not in the telemetry: it is reconstructed by
differentiating the GPS trajectory twice, with prior smoothing because a
numerical second derivative amplifies sampling noise.

Everything else is read from the API rather than assumed:

    compound, tire age, stint      lap timing
    track temperature              weather feed, at the moment each lap started
    air temperature, humidity,     weather feed; kept with every lap and stint,
    pressure, wind, rain           rain also filters out wet laps
    deleted lap times              race-control messages
    race distance                  session
    races of a season              event schedule

A lap the API cannot describe -- an unknown compound, no weather to place it in
-- is dropped rather than filled in with a made-up value.

The degradation observable is pace loss corrected for everything that changes
lap time with race lap but is not the tire: fuel burn plus track evolution. That
combined effect is estimated per race rather than assumed, because it varies
from -0.026 to -0.097 s/lap across circuits and a wrong constant biases each
circuit differently. See `_estimate_race_lap_effect`.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMPOUND_INDEX, DataConfig, PhysicsConfig
from .dataset import Stint, StintDataset

# --- Telemetry units and limits ---------------------------------------------
G = 9.80665                  # standard gravity [m/s^2]
KMH_PER_MS = 3.6             # FastF1 speed is in km/h
DM_PER_M = 10.0              # FastF1 X/Y positions are in decimetres
MAX_ACCEL = 6.0 * G          # an F1 car's ceiling; anything above is GPS noise
MIN_SAMPLES_PER_LAP = 20     # fewer and a second derivative means nothing
MIN_PLANAR_SPEED = 1.0       # [m/s] below this the GPS heading is undefined
SMOOTHING_SECONDS = 1.0      # width of the telemetry smoothing filter

# Weather channels of the F1 live-timing feed, and their names in this project.
WEATHER_CHANNELS = {
    "TrackTemp": "track_temp_c",
    "AirTemp": "air_temp_c",
    "Humidity": "humidity_pct",
    "Pressure": "pressure_mbar",
    "WindSpeed": "wind_speed_ms",
    "WindDirection": "wind_direction_deg",
    "Rainfall": "rainfall",
}

# Bump whenever the way stints are built changes, so datasets cached on disk by
# an older version are rebuilt instead of silently reused.
_DATASET_FORMAT = 2


# =============================================================================
# The API
# =============================================================================
def _require_fastf1():
    try:
        import fastf1
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastF1 is not installed. Install the dependencies with "
            "`pip install -r requirements.txt` or use --source synthetic."
        ) from exc
    return fastf1


def _enable_cache(cfg: DataConfig):
    fastf1 = _require_fastf1()
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)  # FastF1 will not create it
    fastf1.Cache.enable_cache(cfg.cache_dir)
    return fastf1


def load_session(cfg: DataConfig, gp: str | int):
    """Download (or read from the cache) one session of `gp`.

    Race-control messages are loaded along with laps, telemetry and weather
    because FastF1 fills each lap's `Deleted` flag from them: without them every
    lap reads as not deleted.
    """
    fastf1 = _enable_cache(cfg)
    session = fastf1.get_session(cfg.year, gp, cfg.session)
    session.load(laps=True, telemetry=True, weather=True, messages=True)
    return session


def season_races(cfg: DataConfig) -> list[str]:
    """Every event of `cfg.year` whose `cfg.session` has already been held.

    Read from the API's event schedule, in calendar order, as event names
    ("Belgian Grand Prix") because those are unique within a season, unlike
    locations: 2020 raced twice at both Spielberg and Silverstone.
    """
    fastf1 = _enable_cache(cfg)
    schedule = fastf1.get_event_schedule(cfg.year, include_testing=False)
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    races = []
    for _, event in schedule.iterrows():
        try:
            held_at = event.get_session_date(cfg.session, utc=True)
        except ValueError:  # this event has no such session (e.g. no sprint)
            continue
        if pd.notna(held_at) and held_at < now_utc:
            races.append(str(event["EventName"]))
    return races


def _race_name(session, fallback: str) -> str:
    """The race's name as the API gives it ("Italian Grand Prix")."""
    event = getattr(session, "event", None)
    name = event.get("EventName") if event is not None else None
    return str(name) if name else fallback


def _race_distance(session) -> int:
    """Scheduled race distance in laps, from the API (the laps actually run if it has none)."""
    try:
        scheduled = session.total_laps
    except Exception:  # FastF1 raises when the lap count failed to load
        scheduled = None
    return int(scheduled) if scheduled else int(session.laps["LapNumber"].max())


# =============================================================================
# Weather
# =============================================================================
@dataclass
class WeatherFeed:
    """The session's weather channels, sampled about once a minute.

    Laps take 70-110 s, so each spans one or two samples. Continuous channels are
    interpolated at the moment a lap starts; wind direction is an angle, so it
    takes the sample in force instead of being interpolated; and a lap counts as
    wet if any sample from the one in force at its start to the last one before
    its end reports rain.
    """

    time_s: np.ndarray
    channels: dict[str, np.ndarray]

    @classmethod
    def from_session(cls, session) -> WeatherFeed:
        try:
            weather = session.weather_data
        except Exception:  # FastF1 raises when the feed failed to load
            weather = None
        if weather is None or len(weather) == 0 or "TrackTemp" not in weather:
            # Track temperature is a model input and there is nothing
            # truthful to put in its place.
            raise RuntimeError("The API has no weather data for this session")

        time_s = weather["Time"].dt.total_seconds().to_numpy(dtype=float)
        track_temp = weather["TrackTemp"].to_numpy(dtype=float)
        ok = np.isfinite(time_s) & np.isfinite(track_temp)
        if ok.sum() < 2:
            raise RuntimeError("The API has no usable track temperature for this session")

        channels = {
            name: (
                pd.to_numeric(weather[api_name], errors="coerce").to_numpy(dtype=float)[ok]
                if api_name in weather
                else np.full(ok.sum(), np.nan)
            )
            for api_name, name in WEATHER_CHANNELS.items()
        }
        return cls(time_s=time_s[ok], channels=channels)

    def during(self, start_s: float, end_s: float) -> dict[str, float]:
        """Conditions for a lap run between two session times [s]."""
        # Index of the sample in force when the lap starts, and of the last one before it ends.
        first = max(int(np.searchsorted(self.time_s, start_s, side="right")) - 1, 0)
        last = max(int(np.searchsorted(self.time_s, end_s, side="right")) - 1, first)

        conditions = {}
        for name, values in self.channels.items():
            if name == "rainfall":
                conditions[name] = float(np.any(values[first : last + 1] > 0.5))
            elif name == "wind_direction_deg":
                conditions[name] = float(values[first])
            else:
                conditions[name] = float(np.interp(start_s, self.time_s, values))
        return conditions


def _summarise_weather(laps: pd.DataFrame) -> dict[str, float]:
    """Conditions over a set of laps.

    Medians, except rain (the fraction of wet laps) and wind direction (a
    circular mean: the average of 350 and 10 degrees is 0, not 180).
    """
    summary = {}
    for name in WEATHER_CHANNELS.values():
        values = laps[name].to_numpy(dtype=float)
        if name == "rainfall":
            summary[name] = float(np.mean(values))
        elif name == "wind_direction_deg":
            rad = np.deg2rad(values)
            summary[name] = float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)
        else:
            summary[name] = float(np.median(values))
    return summary


def normalise_track_temp(temp_c, cfg: DataConfig, phys: PhysicsConfig):
    """Track temperature on the model's scale: (T - T_ref) / dT_ref, clipped to [0, 1].

    It shares dT_ref with theta, so that theta + T_trk in (E2) is one temperature.
    """
    return np.clip((temp_c - cfg.track_temp_ref_c) / phys.dT_ref, 0.0, 1.0)


# =============================================================================
# Telemetry: the dynamic proxies of one lap
# =============================================================================
def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing, falling back to a moving average if scipy fails."""
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


def _smoothing_window(t: np.ndarray, seconds: float = SMOOTHING_SECONDS) -> int:
    """Smoothing window in samples, set by time rather than by count.

    FastF1 merges car telemetry (~10 Hz) with GPS position (~4 Hz) by
    interpolating, so the effective sampling rate varies between laps and
    sessions. A window fixed in samples would smooth differently in each case; a
    window fixed in seconds always applies the same physical filter. This
    matters because lateral acceleration comes from a second derivative, and
    linearly interpolated stretches produce artificially large ones.
    """
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return 11
    return max(5, round(seconds / dt) | 1)


def _lap_dynamics(tel: pd.DataFrame) -> dict[str, float] | None:
    """Extract the dynamic variables of one lap from its telemetry.

    Returns None if the lap has too few samples to differentiate.
    """
    if tel is None or len(tel) < MIN_SAMPLES_PER_LAP:
        return None

    t = tel["Time"].dt.total_seconds().to_numpy(dtype=float)
    speed = tel["Speed"].to_numpy(dtype=float) / KMH_PER_MS
    if not np.all(np.diff(t) > 0):  # drop repeated timestamps
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, speed, tel = t[keep], speed[keep], tel.loc[keep]
        if t.size < MIN_SAMPLES_PER_LAP:
            return None

    window = _smoothing_window(t)
    speed = _smooth(speed, window)
    a_long = np.gradient(speed, t)

    # Lateral acceleration from the GPS trajectory: the magnitude of the
    # velocity-acceleration cross product divided by the speed is exactly the
    # normal component of acceleration.
    if {"X", "Y"}.issubset(tel.columns):
        x = _smooth(tel["X"].to_numpy(dtype=float) / DM_PER_M, window)
        y = _smooth(tel["Y"].to_numpy(dtype=float) / DM_PER_M, window)
        dx, dy = np.gradient(x, t), np.gradient(y, t)
        ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
        planar = np.sqrt(dx**2 + dy**2)
        a_lat = np.where(
            planar > MIN_PLANAR_SPEED, np.abs(dx * ddy - dy * ddx) / np.maximum(planar, 1e-6), 0.0
        )
        a_lat = np.clip(_smooth(a_lat, window), 0.0, MAX_ACCEL)
    else:  # pragma: no cover - sessions without position data
        a_lat = np.zeros_like(speed)

    duration = float(t[-1] - t[0])
    if duration <= 0:
        return None

    a_total = np.sqrt(a_lat**2 + a_long**2)
    # Specific frictional power: |a| * v, averaged over the lap time.
    q_fric = float(np.trapezoid(a_total * speed, t) / duration)
    return {
        "q_fric_raw": q_fric,
        "load_raw": float(np.mean(a_total) / G),
        "speed_raw": float(np.mean(speed)),
    }


# =============================================================================
# Step 1: one record per usable lap
# =============================================================================
def _flag(value, default: bool = False) -> bool:
    """A boolean lap flag from the API.

    `bool(...)` rather than `is True`: depending on the pandas version a flag
    arrives as a Python bool or as numpy.bool_, and `is` would silently fail on
    the second. Missing values (None/NaN) take `default`.
    """
    return default if value is None or pd.isna(value) else bool(value)


def _seconds(value) -> float:
    """A FastF1 session timestamp (Timedelta) in seconds; NaN if missing."""
    return value.total_seconds() if pd.notna(value) else np.nan


def _is_green(track_status) -> bool:
    """Only green-flag laps are kept.

    A safety car or a yellow flag changes the lap time by several seconds for
    reasons that have nothing to do with the tire.
    """
    if track_status is None or (isinstance(track_status, float) and np.isnan(track_status)):
        return False
    return str(track_status).strip() == "1"


def _passes_timing_filters(lap, cfg: DataConfig) -> bool:
    """Keep only laps whose time says something about the tire, and that the API fully describes."""
    if pd.isna(lap.get("LapTime")) or not _flag(lap.get("IsAccurate")):
        return False  # no reliable lap time
    if not _is_green(lap.get("TrackStatus")):
        return False  # safety car, VSC or yellow flag
    if pd.notna(lap.get("PitInTime")) or pd.notna(lap.get("PitOutTime")):
        return False  # in- or out-lap
    if _flag(lap.get("Deleted")):
        return False  # time deleted by race control
    if cfg.only_fresh_tyres and not _flag(lap.get("FreshTyre"), default=True):
        return False  # d(0) = 0 only holds for a brand-new set
    # Last, the API must say which tire this was.
    return str(lap.get("Compound", "")).upper() in COMPOUND_INDEX and pd.notna(lap.get("Stint"))


def _lap_record(lap, weather: WeatherFeed, cfg: DataConfig) -> dict | None:
    """Everything the pipeline needs from one lap, or None if the lap is not usable."""
    if not _passes_timing_filters(lap, cfg):
        return None

    lap_time_s = lap["LapTime"].total_seconds()
    start_s = _seconds(lap.get("LapStartTime"))
    if not np.isfinite(start_s):
        return None  # cannot be placed in the weather feed
    conditions = weather.during(start_s, start_s + lap_time_s)
    if cfg.skip_wet_laps and conditions["rainfall"]:
        return None  # a wet track, slow for reasons other than wear

    try:
        # `iterrows` over a `Laps` object yields `Lap` objects, which already
        # know how to fetch their own merged telemetry (car + GPS).
        telemetry = lap.get_telemetry()
    except Exception:
        return None
    dynamics = _lap_dynamics(telemetry)
    if dynamics is None:
        return None

    tyre_life = lap.get("TyreLife")
    return {
        "stint": int(lap["Stint"]),
        "lap_number": int(lap["LapNumber"]),
        "tyre_life": float(tyre_life) if pd.notna(tyre_life) else np.nan,
        "compound": str(lap["Compound"]).upper(),
        "lap_time": float(lap_time_s),
        **conditions,
        **dynamics,
    }


def _lap_records(
    session, cfg: DataConfig, drivers: Sequence[str], last_lap: int | None = None
) -> pd.DataFrame:
    """One row per usable lap of `drivers`, in driver then lap order.

    `last_lap` keeps only the laps run up to that race lap, for predictions that
    must not see the future.
    """
    weather = WeatherFeed.from_session(session)
    records: list[dict] = []
    for driver in drivers:
        for _, lap in session.laps[session.laps["Driver"] == driver].iterrows():
            if last_lap is not None and lap["LapNumber"] > last_lap:
                continue
            record = _lap_record(lap, weather, cfg)
            if record is not None:
                records.append({"driver": driver, **record})
    return pd.DataFrame.from_records(records)


# =============================================================================
# Step 2: the model's context, dimensionless
# =============================================================================
def _add_context_columns(laps: pd.DataFrame, cfg: DataConfig, phys: PhysicsConfig) -> None:
    """Make the proxies dimensionless against fixed references.

    The references are constants from `DataConfig`, not session statistics.
    That is what allows training across several races: a high-load circuit and
    a low-speed one land at different points of the context space, instead of
    both collapsing to 1.0.
    """
    laps["q_fric"] = laps["q_fric_raw"] / cfg.q_fric_ref
    laps["load"] = laps["load_raw"] / cfg.load_ref
    laps["speed"] = laps["speed_raw"] / cfg.speed_ref
    laps["track_temp_norm"] = normalise_track_temp(laps["track_temp_c"], cfg, phys)
    laps["compound_idx"] = laps["compound"].map(COMPOUND_INDEX)


# =============================================================================
# Step 3: race-lap correction (fuel burn + track evolution)
# =============================================================================
def _estimate_race_lap_effect(df: pd.DataFrame, n_knots: int = 4) -> np.ndarray:
    """Estimate how lap time changes with race lap, for reasons other than the tire.

    Two effects make a car faster as a race progresses: it burns off ~100 kg of
    fuel, and the circuit rubbers in. **They are not separable from each other** --
    both are smooth monotone functions of race lap -- so estimating them
    individually would invent a decomposition the data cannot support. What is
    estimable, and what this returns, is their sum.

    Estimating it beats assuming it. A fixed s/lap fuel figure cannot know that a
    7 km lap burns more fuel per lap than a 4 km one: measured across 2026, the
    combined effect ranges from -0.026 s/lap at Miami to -0.097 at Spa, against a
    typical assumed -0.055. Over a 20-lap stint that is a bias between -0.57 s
    and +0.83 s, comparable to the entire degradation signal, and with different
    signs at different circuits -- so it does not cancel, it distorts precisely
    the circuit-to-degradation relationship the model is trying to learn.

    Identification comes from cars carrying tires of different ages at the same
    race lap, because they pit at different times. Measured on 2026 races, tire
    age has a spread of 2-7 laps at a given race lap and correlates only 0.22-0.76
    with it; if everyone pitted together the two effects would be one variable and
    nothing could separate them.

    The fit is `lap_time ~ driver + f(race_lap) + degradation(age, compound)`,
    with `f` a piecewise-linear spline so its shape is measured rather than
    assumed, and the degradation terms present only to stop `f` absorbing them.

    Returns f evaluated at laps 0..max, normalised to f(0) = 0.
    """
    lap_max = int(df["lap_number"].max())
    race_lap = df["lap_number"].to_numpy(dtype=float)
    knots = np.linspace(1, lap_max, n_knots + 2)[1:-1]

    # Columns of the regression: one intercept per driver, the spline f in race
    # lap, and a linear degradation rate per compound.
    driver_terms = pd.get_dummies(df["driver"], prefix="D").to_numpy(dtype=float)
    spline_terms = [race_lap] + [np.maximum(race_lap - k, 0.0) for k in knots]
    compound_dummies = pd.get_dummies(df["compound"], prefix="C").to_numpy(dtype=float)
    tyre_age = df["tyre_life"].to_numpy(dtype=float)
    wear_terms = [compound_dummies[:, j] * tyre_age for j in range(compound_dummies.shape[1])]

    x = np.column_stack([driver_terms, *spline_terms, *wear_terms])
    y = df["lap_time"].to_numpy(dtype=float)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.shape[0] < 5 * x.shape[1]:  # too few laps to identify this many terms
        return np.zeros(lap_max + 2)

    coef = np.linalg.lstsq(x.T @ x + 1e-6 * np.eye(x.shape[1]), x.T @ y, rcond=None)[0]
    first_spline = driver_terms.shape[1]
    spline_coef = coef[first_spline : first_spline + len(spline_terms)]

    grid = np.arange(0, lap_max + 2, dtype=float)
    f = spline_coef[0] * grid + sum(
        c * np.maximum(grid - k, 0.0) for c, k in zip(spline_coef[1:], knots, strict=True)
    )
    return f - f[0]


def _correct_for_race_lap(laps: pd.DataFrame, cfg: DataConfig, total_laps: int) -> float:
    """Add `lap_time_corr`: the lap time without what the race lap itself explains.

    Uncorrected, the car speeding up as it lightens and as the track rubbers in
    looks like the opposite of degradation, and masks it entirely. The combined
    effect is measured per race (`_estimate_race_lap_effect`) unless disabled, in
    which case a fixed fuel figure is assumed.

    Returns the correction's mean slope in s/lap, for the record.
    """
    if cfg.estimate_race_lap_effect:
        effect = _estimate_race_lap_effect(laps)
        idx = np.clip(laps["lap_number"].to_numpy(dtype=int), 0, len(effect) - 1)
        laps["lap_time_corr"] = laps["lap_time"] - effect[idx]
        return float((effect[-1] - effect[0]) / (len(effect) - 1))

    laps["lap_time_corr"] = laps["lap_time"] - cfg.fuel_effect_s_per_lap * (
        total_laps - laps["lap_number"]
    )
    return -cfg.fuel_effect_s_per_lap


# =============================================================================
# Step 4: stints
# =============================================================================
def _stint_context(laps: pd.DataFrame) -> np.ndarray:
    """(5,) context of a stint: the median of each proxy over its laps, and its compound."""
    return np.array(
        [
            float(laps["q_fric"].median()),
            float(laps["load"].median()),
            float(laps["speed"].median()),
            float(laps["track_temp_norm"].median()),
            float(laps["compound_idx"].iloc[0]),
        ]
    )


def _build_stint(laps: pd.DataFrame, cfg: DataConfig, race: str) -> Stint | None:
    """One stint from its laps (already sorted), or None if too few are usable."""
    if len(laps) < cfg.min_stint_laps:
        return None

    # Tire age: TyreLife accounts for already-used sets; if missing, fall back
    # to the position within the stint.
    life = laps["tyre_life"].to_numpy(dtype=float)
    if not np.all(np.isfinite(life)):
        life = np.arange(1, len(laps) + 1, dtype=float)

    # Origin of degradation: the tire's performance PEAK, not its first lap.
    # A new set comes out cold and gets faster for two or three laps before
    # it starts falling away. The model is monotone by construction and so
    # cannot represent that warm-up phase: it is discarded, and d = 0 is
    # defined at the peak. Without this anchoring every stint starts half a
    # second offset from the prediction, and that systematic conflict
    # degenerates the parameter estimates.
    times = laps["lap_time_corr"].to_numpy()
    peak = int(np.argmin(times[: cfg.ref_window + 1]))
    times, life = times[peak:], life[peak:]
    delta = times - times[0]
    age = life - life[0] + 1.0  # the peak becomes lap 1 of the stint

    keep = delta <= cfg.max_delta_s  # discards traffic and driver errors
    if keep.sum() < cfg.min_stint_laps:
        return None

    # The context is summarised over the laps that actually enter the fit, not
    # over the whole stint: the warm-up laps were already discarded and must
    # not influence the median that represents the stint.
    used = laps.iloc[peak:][keep]
    driver, stint_no = used["driver"].iloc[0], used["stint"].iloc[0]
    return Stint(
        stint_id=f"{race}-{driver}-S{stint_no}",
        driver=str(driver),
        compound=str(used["compound"].iloc[0]),
        laps=age[keep],
        delta=delta[keep],
        context=_stint_context(used),
        race_laps=laps["lap_number"].to_numpy()[peak:][keep],
        race=race,
        weather=_summarise_weather(used),
    )


def _assemble_stints(laps: pd.DataFrame, cfg: DataConfig, race: str) -> list[Stint]:
    """Group the laps by driver and stint, and build each stint."""
    stints = []
    for _, group in laps.groupby(["driver", "stint"], sort=True):
        stint = _build_stint(group.sort_values("lap_number"), cfg, race)
        if stint is not None:
            stints.append(stint)
    return stints


# =============================================================================
# Datasets
# =============================================================================
def build_dataset(
    cfg: DataConfig, phys: PhysicsConfig, gp: str | int, session=None
) -> StintDataset:
    """Build the stints of one race from its FastF1 session (loaded from the API if not given)."""
    if session is None:
        session = load_session(cfg, gp)
    if session.laps is None or len(session.laps) == 0:
        raise RuntimeError("The session contains no loaded laps")

    race = _race_name(session, fallback=str(gp))
    total_laps = _race_distance(session)
    drivers = list(cfg.drivers) or sorted(session.laps["Driver"].dropna().unique())

    laps = _lap_records(session, cfg, drivers)
    if laps.empty:
        raise RuntimeError(
            "No lap survived the quality filters. Try another session, more "
            "drivers, or relax `only_fresh_tyres`."
        )
    _add_context_columns(laps, cfg, phys)
    race_lap_effect = _correct_for_race_lap(laps, cfg, total_laps)

    stints = _assemble_stints(laps, cfg, race)
    if not stints:
        raise RuntimeError(
            f"No stint has at least {cfg.min_stint_laps} valid laps. "
            "Lower `min_stint_laps` or pick a race with fewer neutralisations."
        )

    return StintDataset(
        stints=stints,
        source=f"fastf1:{cfg.year}-{race}-{cfg.session}",
        meta={
            "race": race,
            "total_laps": total_laps,
            "drivers": drivers,
            "race_lap_effect_s_per_lap": race_lap_effect,
            "laps_after_filters": len(laps),
            "weather": _summarise_weather(laps),
        },
    )


def _dataset_cache_path(cfg: DataConfig, phys: PhysicsConfig, gps: Sequence[str]) -> Path:
    """Cache file for one (year, session, races) combination.

    Keyed by a hash of everything that changes the result, so a config change
    silently invalidates the cache instead of returning a stale dataset.

    Caching happens **per race**, not only for the combined set. Parsing one
    season takes the better part of an hour, and caching only the final result
    means any interruption throws all of it away. With per-race entries a run
    can be stopped and resumed, and adding a race to the list only costs that
    race.
    """
    key = repr(
        (
            _DATASET_FORMAT,
            cfg.year,
            cfg.session,
            tuple(gps),
            tuple(cfg.drivers),
            cfg.fuel_effect_s_per_lap,
            cfg.estimate_race_lap_effect,
            cfg.q_fric_ref,
            cfg.load_ref,
            cfg.speed_ref,
            cfg.track_temp_ref_c,
            phys.dT_ref,
            cfg.min_stint_laps,
            cfg.ref_window,
            cfg.max_delta_s,
            cfg.only_fresh_tyres,
            cfg.skip_wet_laps,
        )
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(cfg.cache_dir) / "datasets" / f"{cfg.year}-{cfg.session}-{len(gps)}races-{digest}.pkl"


def _read_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _write_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def build_multi_dataset(
    cfg: DataConfig, phys: PhysicsConfig, gps: Sequence[str], use_cache: bool = True
) -> StintDataset:
    """Combine several races into a single dataset.

    This is the recommended way to train on real data. Within a single race the
    context variables barely vary (same circuit, same weather), so the network
    cannot learn how degradation responds to conditions: it only sees the effect
    of compound and time. With several races the context space is genuinely
    populated.

    Parsing telemetry is by far the slowest step -- minutes per race, since every
    lap is fetched and differentiated individually. The assembled dataset is
    cached to disk so that retraining, or changing a network hyperparameter, does
    not pay that cost again.
    """
    cache_path = _dataset_cache_path(cfg, phys, gps)
    if use_cache and cache_path.exists():
        data = _read_pickle(cache_path)
        print(f"  [cache] {len(data)} stints read from {cache_path.name}")
        return data

    stints: list[Stint] = []
    sources, race_meta, failures = [], [], []
    for i, gp in enumerate(gps, 1):
        progress = f"  [{i}/{len(gps)}] {gp}"
        race_cache = _dataset_cache_path(cfg, phys, [gp])
        if use_cache and race_cache.exists():
            part = _read_pickle(race_cache)
            print(f"{progress}: {len(part)} stints (cached)")
        else:
            try:
                part = build_dataset(cfg, phys, gp)
            except Exception as exc:
                failures.append(f"{gp}: {exc}")
                print(f"{progress}: FAILED ({exc})")
                continue
            if use_cache:
                _write_pickle(race_cache, part)
            print(f"{progress}: {len(part)} stints")
        stints.extend(part.stints)
        sources.append(part.source)
        race_meta.append(part.meta)

    if not stints:
        raise RuntimeError("No race produced valid stints:\n  " + "\n  ".join(failures))

    data = StintDataset(
        stints=stints,
        source=" + ".join(sources),
        meta={"races": list(gps), "per_race": race_meta, "failures": failures},
    )
    if use_cache:
        _write_pickle(cache_path, data)
    return data


# =============================================================================
# Live inference: one driver at one lap
# =============================================================================
@dataclass
class RaceSituation:
    """Where one driver stands at one lap of a race, as the API reports it."""

    race: str
    driver: str
    lap_number: int
    total_laps: int
    compound: str
    tyre_age: float           # laps already run on this set (the API's TyreLife)
    fresh_tyre: bool          # whether the set was new when fitted
    weather: dict[str, float]  # conditions on this lap, keyed as WEATHER_CHANNELS
    context: np.ndarray       # (5,) model context, normalised as in training
    clean_laps_so_far: int    # laps the telemetry proxies were measured on...
    clean_laps_this_stint: int  # ...and how many of them on the current set

    @property
    def laps_remaining(self) -> int:
        return max(self.total_laps - self.lap_number, 0)


def race_situation(
    session,
    driver: str,
    cfg: DataConfig,
    phys: PhysicsConfig,
    lap_number: int | None = None,
) -> RaceSituation:
    """Read `driver`'s situation at `lap_number` (default: their last completed lap).

    Compound, tire age and weather are the API's values for that lap. The
    telemetry proxies are medians over the driver's clean laps up to it, and
    never later, exactly as a live prediction would have them. Proxies the model
    saw as per-race medians in training (`cfg.aggregate_context`) use every such
    lap of the race; the others use only the current stint, which is how each
    stint's context was built in training.
    """
    driver_laps = session.laps.pick_drivers(driver)
    completed = driver_laps[driver_laps["LapTime"].notna()]
    if completed.empty:
        raise ValueError(f"{driver} has no completed lap in this session")
    driver = str(completed["Driver"].iloc[0])  # canonical abbreviation, even if a number was given

    if lap_number is None:
        lap_number = int(completed["LapNumber"].max())
    current = driver_laps[driver_laps["LapNumber"] == lap_number]
    if current.empty:
        first, last = int(driver_laps["LapNumber"].min()), int(driver_laps["LapNumber"].max())
        raise ValueError(f"{driver} did not run lap {lap_number} (ran laps {first}-{last})")
    lap = current.iloc[0]

    compound = str(lap.get("Compound", "")).upper()
    if compound not in COMPOUND_INDEX:
        raise ValueError(f"The API does not say which compound {driver} was on at lap {lap_number}")
    if pd.isna(lap.get("TyreLife")):
        raise ValueError(f"The API has no tire age for {driver} at lap {lap_number}")

    start_s, end_s = _seconds(lap.get("LapStartTime")), _seconds(lap.get("Time"))
    if not np.isfinite(start_s):
        start_s = end_s
    if not np.isfinite(end_s):
        end_s = start_s
    if not np.isfinite(start_s):
        raise ValueError(f"The API has no timing for {driver}'s lap {lap_number}")
    weather = WeatherFeed.from_session(session).during(start_s, end_s)

    history = _lap_records(session, cfg, [driver], last_lap=lap_number)
    if history.empty:
        raise ValueError(f"{driver} has no clean lap up to lap {lap_number} to measure the circuit on")
    _add_context_columns(history, cfg, phys)
    stint_no = lap.get("Stint")
    this_stint = history[history["stint"] == int(stint_no)] if pd.notna(stint_no) else history.iloc[:0]

    def proxy(name: str) -> float:
        laps = history if name in cfg.aggregate_context or this_stint.empty else this_stint
        return float(laps[name].median())

    context = np.array(
        [
            proxy("q_fric"),
            proxy("load"),
            proxy("speed"),
            float(normalise_track_temp(weather["track_temp_c"], cfg, phys)),
            COMPOUND_INDEX[compound],
        ]
    )
    return RaceSituation(
        race=_race_name(session, fallback="?"),
        driver=driver,
        lap_number=int(lap_number),
        total_laps=_race_distance(session),
        compound=compound,
        tyre_age=float(lap["TyreLife"]),
        fresh_tyre=_flag(lap.get("FreshTyre"), default=True),
        weather=weather,
        context=context,
        clean_laps_so_far=len(history),
        clean_laps_this_stint=len(this_stint),
    )
