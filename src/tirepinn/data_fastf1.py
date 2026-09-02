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
from dataclasses import replace
from pathlib import Path

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
    lap = df["lap_number"].to_numpy(dtype=float)
    knots = np.linspace(1, lap_max, n_knots + 2)[1:-1]

    drivers = pd.get_dummies(df["driver"], prefix="D").to_numpy(dtype=float)
    evo = [lap] + [np.maximum(lap - k, 0.0) for k in knots]
    compounds = pd.get_dummies(df["compound"], prefix="C").to_numpy(dtype=float)
    deg = [compounds[:, j] * df["tyre_life"].to_numpy(dtype=float) for j in range(compounds.shape[1])]

    x = np.column_stack([drivers, *evo, *deg])
    y = df["lap_time"].to_numpy(dtype=float)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.shape[0] < 5 * x.shape[1]:  # too few laps to identify this many terms
        return np.zeros(lap_max + 2)

    coef = np.linalg.lstsq(x.T @ x + 1e-6 * np.eye(x.shape[1]), x.T @ y, rcond=None)[0]
    ev = coef[drivers.shape[1] : drivers.shape[1] + len(evo)]

    grid = np.arange(0, lap_max + 2, dtype=float)
    f = ev[0] * grid + sum(e * np.maximum(grid - k, 0.0) for e, k in zip(ev[1:], knots, strict=False))
    return f - f[0]


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

    # --- Step 3: race-lap correction (fuel burn + track evolution) ---
    # Uncorrected, the car speeding up as it lightens and as the track rubbers in
    # looks like the opposite of degradation, and masks it entirely.
    # `estimate_race_lap_effect` measures the combined effect per race instead of
    # assuming a fixed s/lap figure; see that function for why the two cannot be
    # separated and why assuming one number biases circuits differently.
    if cfg.estimate_race_lap_effect:
        effect = _estimate_race_lap_effect(df)
        idx = np.clip(df["lap_number"].to_numpy(dtype=int), 0, len(effect) - 1)
        df["lap_time_corr"] = df["lap_time"] - effect[idx]
    else:
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


def _dataset_cache_path(cfg: DataConfig, gps: Sequence[str]) -> Path:
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
            cfg.year,
            cfg.session,
            tuple(gps),
            tuple(cfg.drivers),
            cfg.fuel_effect_s_per_lap,
            cfg.estimate_race_lap_effect,
            cfg.q_fric_ref,
            cfg.load_ref,
            cfg.speed_ref,
            cfg.min_stint_laps,
            cfg.ref_window,
            cfg.max_delta_s,
            cfg.only_fresh_tyres,
        )
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(cfg.cache_dir) / "datasets" / f"{cfg.year}-{cfg.session}-{len(gps)}races-{digest}.pkl"


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
    cache_path = _dataset_cache_path(cfg, gps)
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
        print(f"  [cache] {len(data)} stints read from {cache_path.name}")
        return data

    all_stints: list[Stint] = []
    sources, failures = [], []

    for i, gp in enumerate(gps, 1):
        race_cfg = replace(cfg, gp=gp)
        race_cache = _dataset_cache_path(race_cfg, [gp])
        if use_cache and race_cache.exists():
            with open(race_cache, "rb") as fh:
                part = pickle.load(fh)
            print(f"  [{i}/{len(gps)}] {gp}: {len(part.stints)} stints (cached)")
        else:
            try:
                part = build_dataset(race_cfg, phys)
            except Exception as exc:
                failures.append(f"{gp}: {exc}")
                print(f"  [{i}/{len(gps)}] {gp}: FAILED ({exc})")
                continue
            if use_cache:
                race_cache.parent.mkdir(parents=True, exist_ok=True)
                with open(race_cache, "wb") as fh:
                    pickle.dump(part, fh)
            print(f"  [{i}/{len(gps)}] {gp}: {len(part.stints)} stints")
        for stint in part.stints:
            stint.stint_id = f"{gp[:3].upper()}-{stint.stint_id}"
        all_stints.extend(part.stints)
        sources.append(part.source)

    if not all_stints:
        raise RuntimeError("No race produced valid stints:\n  " + "\n  ".join(failures))

    data = StintDataset(
        stints=all_stints,
        source=" + ".join(sources),
        meta={"races": list(gps), "failures": failures},
    )
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(data, fh)
    return data
