"""Loading real F1 telemetry and building the model's variables.

The central problem with real data is that **nothing the model needs is
directly observable**: the tire's internal temperature, the vertical load and
the state of the tread are proprietary to each team. All that is public is
onboard telemetry (speed, throttle, brake, gear, GPS position) and lap times.

This module bridges that gap with proxy variables derived from telemetry:

    q_fric  specific frictional energy per lap, integrating total acceleration
            times speed. This is the heat-generation term of (E1).
    load    mean mechanical load in g, combining lateral and longitudinal
            acceleration. This is the Archard term of (E2).
    speed   mean speed, which governs convective cooling.

Lateral acceleration is not in the telemetry: it is reconstructed by
differentiating the GPS trajectory twice, with prior smoothing because a
numerical second derivative amplifies sampling noise.

The degradation observable is fuel-corrected pace loss: a car sheds around
100 kg over a race and that alone is worth more than a second per lap, so
without correcting for it degradation is completely masked.
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
            "FastF1 is not installed. Install the dependencies with "
            "`pip install -r requirements.txt` or use --source synthetic."
        ) from exc
    return fastf1


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


def _smoothing_window(t: np.ndarray, seconds: float = 1.0) -> int:
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

    # Lateral acceleration from the GPS trajectory. FastF1 gives X, Y in
    # decimetres; the magnitude of the velocity-acceleration cross product
    # divided by the speed is exactly the normal component of acceleration.
    if {"X", "Y"}.issubset(tel.columns):
        x = _smooth(tel["X"].to_numpy(dtype=float) / 10.0, window)
        y = _smooth(tel["Y"].to_numpy(dtype=float) / 10.0, window)
        dx, dy = np.gradient(x, t), np.gradient(y, t)
        ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
        planar = np.sqrt(dx**2 + dy**2)
        a_lat = np.where(planar > 1.0, np.abs(dx * ddy - dy * ddx) / np.maximum(planar, 1e-6), 0.0)
        a_lat = np.clip(_smooth(a_lat, window), 0.0, 6.0 * _G)  # 6 g is an F1 car's ceiling
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
    """Only green-flag laps are kept.

    A safety car or a yellow flag changes the lap time by several seconds for
    reasons that have nothing to do with the tire.
    """
    if track_status is None or (isinstance(track_status, float) and np.isnan(track_status)):
        return False
    return str(track_status).strip() == "1"


def load_session(cfg: DataConfig):
    """Download (or read from cache) the requested session."""
    fastf1 = _require_fastf1()
    fastf1.Cache.enable_cache(cfg.cache_dir)
    session = fastf1.get_session(cfg.year, cfg.gp, cfg.session)
    session.load(laps=True, telemetry=True, weather=True, messages=False)
    return session


def build_dataset(cfg: DataConfig, phys: PhysicsConfig, session=None) -> StintDataset:
    """Build the stint set from a FastF1 session."""
    session = session or load_session(cfg)
    laps = session.laps
    if laps is None or len(laps) == 0:
        raise RuntimeError("The session contains no loaded laps")

    total_laps = int(getattr(session, "total_laps", 0) or laps["LapNumber"].max())
    weather = _track_temp_series(session)
    drivers = list(cfg.drivers) if cfg.drivers else sorted(laps["Driver"].dropna().unique())

    # --- Step 1: raw per-lap features ---
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
            # `bool(...)` rather than `is False`: depending on the pandas version
            # the value can arrive as a Python bool or as numpy.bool_, and with
            # `is` the filter would silently fail in the second case.
            if cfg.only_fresh_tyres and not bool(lap.get("FreshTyre", True)):
                continue

            try:
                # `iterrows` over a `Laps` object yields `Lap` objects, which
                # already know how to fetch their own merged telemetry (car + GPS).
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
            "No stint survived the quality filters. Try another session, more "
            "drivers, or relax `only_fresh_tyres`."
        )

    df = pd.DataFrame.from_records(records)

    # --- Step 2: non-dimensionalisation against fixed references ---
    # The references are constants from `DataConfig`, not session statistics.
    # That is what allows training across several races: a high-load circuit and
    # a low-speed one land at different points of the context space, instead of
    # both collapsing to 1.0.
    df["q_fric"] = df["q_fric_raw"] / cfg.q_fric_ref
    df["load"] = df["load_raw"] / cfg.load_ref
    df["speed"] = df["speed_raw"] / cfg.speed_ref
    df["track_temp_norm"] = np.clip((df["track_temp"] - 20.0) / 40.0, 0.0, 1.0)
    df["compound_idx"] = df["compound"].map(lambda c: COMPOUND_INDEX.get(c, 0.5))

    # --- Step 3: fuel correction ---
    # The car loses ~1.7 kg per lap and every 10 kg is worth ~0.3 s. Uncorrected,
    # the gain from getting lighter looks like the opposite of degradation.
    df["lap_time_corr"] = df["lap_time"] - cfg.fuel_effect_s_per_lap * (
        total_laps - df["lap_number"]
    )

    # --- Step 4: assembling stints ---
    stints: list[Stint] = []
    for (driver, stint_no), group in df.groupby(["driver", "stint"], sort=True):
        group = group.sort_values("lap_number")
        if len(group) < cfg.min_stint_laps:
            continue

        # Tire age: TyreLife accounts for already-used sets; if missing, fall
        # back to the position within the stint.
        life = group["tyre_life"].to_numpy(dtype=float)
        if not np.all(np.isfinite(life)):
            life = np.arange(1, len(group) + 1, dtype=float)

        # Origin of degradation: the tire's performance PEAK, not its first lap.
        # A new set comes out cold and gets faster for two or three laps before
        # it starts falling away. The model is monotone by construction and so
        # cannot represent that warm-up phase: it is discarded, and d = 0 is
        # defined at the peak. Without this anchoring every stint starts half a
        # second offset from the prediction, and that systematic conflict
        # degenerates the parameter estimates.
        times = group["lap_time_corr"].to_numpy()
        ref_idx = int(np.argmin(times[: cfg.ref_window + 1]))

        times = times[ref_idx:]
        life = life[ref_idx:]
        delta = times - times[0]
        age = life - life[0] + 1.0  # the peak becomes lap 1 of the stint

        keep = delta <= cfg.max_delta_s  # discards traffic and driver errors
        if keep.sum() < cfg.min_stint_laps:
            continue

        # The context is summarised over the laps that actually enter the fit,
        # not over the whole group: the warm-up laps were already discarded and
        # must not influence the median that represents the stint.
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
            f"No stint has at least {cfg.min_stint_laps} valid laps. "
            "Lower `min_stint_laps` or pick a race with fewer neutralisations."
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
    """Combine several races into a single dataset.

    This is the recommended way to train on real data. Within a single race the
    context variables barely vary (same circuit, same weather), so the network
    cannot learn how degradation responds to conditions: it only sees the effect
    of compound and time. With several races the context space is genuinely
    populated.
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
        raise RuntimeError("No race produced valid stints:\n  " + "\n  ".join(failures))

    return StintDataset(
        stints=all_stints,
        source=" + ".join(sources),
        meta={"races": list(gps), "failures": failures},
    )
