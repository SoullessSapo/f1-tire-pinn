"""
DOWNLOAD THE DATASET FROM THE FORMULA 1 API
===========================================

This script pulls real telemetry with FastF1, cleans it, computes the variables
the model needs and leaves everything in a CSV you can open in a spreadsheet.

    python download_data.py --year 2023 --races Monza --out data/monza.csv

    python download_data.py --year 2023 \\
        --races Monza Hungary Spa Silverstone \\
        --out data/2023.csv

    # Quick check: no telemetry (seconds instead of several minutes)
    python download_data.py --year 2023 --races Monza --no-telemetry \\
        --out data/quick.csv

Run `python download_data.py --explain-columns` to see where every column of
the CSV comes from.


THE UNDERLYING PROBLEM
----------------------
Nothing the model needs is observable.

The tire's internal temperature, the vertical load on each wheel and the state
of the tread belong to each team and are not published. All that is public is
onboard telemetry (speed, throttle, brake, gear, GPS position) and lap times.

So the model's variables are RECONSTRUCTED from what does exist:

    q_fric       frictional energy per lap = integral of |a| * v
    load         mean mechanical load, in g, combining lateral and longitudinal
    speed        mean speed of the lap
    track_temp   from the session's weather channel
    compound     from the lap record itself

Lateral acceleration is NOT in the telemetry: it is reconstructed by
differentiating the GPS trajectory twice. That amplifies sampling noise, so it
has to be smoothed first.

And the observable, pace loss, needs a mandatory correction explained in step 4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# `np.trapezoid` is the modern name; in numpy 1.x it was called `np.trapz`.
_integrate = getattr(np, "trapezoid", None) or np.trapz

GRAVITY = 9.80665


# ---------------------------------------------------------------------------
# 0) REFERENCES FOR NORMALISATION
# ---------------------------------------------------------------------------
#
# The three telemetry variables are divided by a fixed reference so they land
# around 1.
#
# VERY IMPORTANT: these are constants, NOT the session's own median.
#
# Normalising each race against its own median would put Monza at 1.0 and
# Hungary at 1.0 too -- two radically different circuits -- and would erase
# exactly the between-circuit variation the model has to learn. This was a real
# mistake in the project and cost a redesign; see ROADMAP.md.
#
# The values come from measuring 2023 races (Monza: 1877 / 3.39 / 66.4;
# Hungary: 1968 / 4.35 / 51.6).
REF_Q_FRIC = 1900.0    # specific frictional power [W/kg]
REF_LOAD = 3.8         # mean mechanical load [g]
REF_SPEED = 58.0       # mean speed [m/s]


def describe_columns() -> str:
    """The text printed by --explain-columns: what every CSV column means."""
    return """
CSV COLUMNS
===========

Identification
  race ............. name of the Grand Prix
  driver ........... three-letter code
  stint_id ......... unique identifier of the set of tires
  race_lap ......... lap number within the race
  stint_lap ........ lap number WITHIN THE STINT (1 = the performance peak)

What is measured
  lap_time ......... lap time in seconds, as-is
  corrected_time ... lap time minus the fuel and track-evolution effect
  delta ............ pace loss relative to the stint's best lap [s]
                     <<< THE OBSERVABLE: the only thing the model fits >>>

The model's variables (constant within the stint: the stint median)
  q_fric ........... frictional energy per lap / 1900 W/kg
  load ............. mean mechanical load / 3.8 g
  speed ............ mean speed / 58 m/s
  track_temp ....... track temperature, rescaled to 0..1
  compound ......... 0 = soft, 0.5 = medium, 1 = hard

For inspecting by hand
  compound_name .... SOFT / MEDIUM / HARD
  tyre_life ........ laps this set has run, according to the FIA
"""


# ---------------------------------------------------------------------------
# 1) SMOOTHING
# ---------------------------------------------------------------------------

def _smoothing_window(time: np.ndarray, seconds: float = 1.0) -> int:
    """How many samples make up `seconds` of signal.

    The window is fixed in SECONDS rather than in number of samples, for a
    concrete reason: FastF1 merges car telemetry (~10 Hz) with GPS position
    (~4 Hz) by interpolating, so the effective rate changes from one lap to the
    next. A window fixed in samples would apply a different physical filter on
    every lap.
    """
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        return 11
    # The filter needs an odd window; `| 1` forces the nearest odd number.
    return max(5, round(seconds / dt) | 1)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay filter: fits a local parabola and evaluates its centre.

    It is preferred over a moving average because it preserves peaks. A moving
    average flattens the braking zones, which are the part carrying the most
    information.
    """
    n = values.size
    if n < 5:
        return values

    window = min(window if window % 2 else window + 1, n if n % 2 else n - 1)
    if window < 5:
        return values

    try:
        from scipy.signal import savgol_filter
        return savgol_filter(values, window, 2)
    except Exception:
        # If scipy is missing, a moving average is worse but it works.
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="same")


# ---------------------------------------------------------------------------
# 2) THE DYNAMIC VARIABLES OF ONE LAP
# ---------------------------------------------------------------------------


def _clean_time_and_speed(telemetry):
    """Pull out time and speed, dropping samples that break differentiation.

    Returns (telemetry, time, speed_in_m_s) or None if too little is left.
    Differentiating needs the time axis to be strictly increasing, and merged
    telemetry occasionally repeats a timestamp.
    """
    time = telemetry["Time"].dt.total_seconds().to_numpy(dtype=float)
    speed = telemetry["Speed"].to_numpy(dtype=float) / 3.6   # km/h -> m/s

    if np.all(np.diff(time) > 0):
        return telemetry, time, speed

    keep = np.concatenate([[True], np.diff(time) > 0])
    if keep.sum() < 20:
        return None
    return telemetry.loc[keep], time[keep], speed[keep]


def _lateral_acceleration(telemetry, time: np.ndarray, window: int) -> np.ndarray:
    """Lateral acceleration, reconstructed from the GPS trajectory.

    It is not in the telemetry, so it has to be derived. Differentiating the
    position twice gives the full acceleration vector; the part perpendicular
    to the velocity is |v x a| / |v|, and that is the lateral component.

    Returns zeros if the session carries no position channel.
    """
    if not {"X", "Y"}.issubset(telemetry.columns):
        return np.zeros(time.size)

    # FastF1 gives X, Y in decimetres.
    x = _smooth(telemetry["X"].to_numpy(dtype=float) / 10.0, window)
    y = _smooth(telemetry["Y"].to_numpy(dtype=float) / 10.0, window)

    dx, dy = np.gradient(x, time), np.gradient(y, time)      # velocity
    ddx, ddy = np.gradient(dx, time), np.gradient(dy, time)  # acceleration

    planar_speed = np.sqrt(dx ** 2 + dy ** 2)
    a_lat = np.where(
        planar_speed > 1.0,
        np.abs(dx * ddy - dy * ddx) / np.maximum(planar_speed, 1e-6),
        0.0,
    )

    # 6 g is an F1 car's real ceiling. Anything above is a numerical artefact
    # of differentiating an interpolated GPS trace twice.
    return np.clip(_smooth(a_lat, window), 0.0, 6.0 * GRAVITY)


def lap_dynamics(telemetry) -> dict[str, float] | None:
    """Extract q_fric, load and speed from ONE lap's telemetry.

    Four steps: clean the signal, smooth it, differentiate it both ways, then
    average over the lap. Returns None if the lap cannot be used.
    """
    if telemetry is None or len(telemetry) < 20:
        return None

    cleaned = _clean_time_and_speed(telemetry)
    if cleaned is None:
        return None
    telemetry, time, speed = cleaned

    window = _smoothing_window(time)
    speed = _smooth(speed, window)

    a_long = np.gradient(speed, time)               # from the speed channel
    a_lat = _lateral_acceleration(telemetry, time, window)   # from the GPS
    a_total = np.sqrt(a_lat ** 2 + a_long ** 2)

    duration = float(time[-1] - time[0])
    if duration <= 0:
        return None

    return {
        # Specific frictional power: |a| * v, averaged over the lap.
        "q_fric_raw": float(_integrate(a_total * speed, time) / duration),
        "load_raw": float(np.mean(a_total) / GRAVITY),
        "speed_raw": float(np.mean(speed)),
    }


# ---------------------------------------------------------------------------
# 3) QUALITY FILTERS
# ---------------------------------------------------------------------------

def lap_is_usable(lap, only_fresh_tyres: bool) -> bool:
    """Decide whether a lap can be used to measure degradation.

    Each filter removes laps whose time changes for reasons that are NOT the
    tire. Leaving them in would teach the model that degradation depends on
    traffic.
    """
    import pandas as pd

    if pd.isna(lap.get("LapTime")):
        return False                              # no time recorded
    if not bool(lap.get("IsAccurate", False)):
        return False                              # the FIA itself flags it doubtful
    if str(lap.get("TrackStatus", "")).strip() != "1":
        return False                              # yellow flag or safety car
    if pd.notna(lap.get("PitInTime")) or pd.notna(lap.get("PitOutTime")):
        return False                              # in-lap or out-lap
    if bool(lap.get("Deleted", False)):
        return False                              # deleted for track limits
    if only_fresh_tyres and not bool(lap.get("FreshTyre", True)):
        return False                              # d(0)=0 only holds on new rubber

    # Note: bool(...) is used rather than `is False`. Depending on the pandas
    # version the value arrives as a Python bool or as numpy.bool_, and with
    # `is` the filter would silently fail in the second case.
    return True


# ---------------------------------------------------------------------------
# 4) THE MANDATORY CORRECTION: FUEL AND TRACK EVOLUTION
# ---------------------------------------------------------------------------

def estimate_race_lap_effect(rows: list[dict]) -> float:
    """How much lap time drops over the race for reasons other than the tire.

    Two things speed the car up as the race goes on that are not the rubber: it
    burns about 100 kg of fuel, and the asphalt rubbers in and gives more grip.
    Uncorrected, the car getting steadily faster looks like the OPPOSITE of
    degradation, and masks it entirely.

    Both are smooth, decreasing functions of race lap, so they are NOT separable
    from each other. What can be estimated is their sum, and that is what this
    function returns: a number in seconds per lap, negative.

    How it is identified: cars carry tires of different ages at the same race
    lap, because they pit at different times. If everyone pitted together,
    "race lap" and "tire age" would be the same variable and nothing could
    separate them.

    The fit is:
        lap_time ~ driver + slope*race_lap + degradation(age, compound)

    The degradation terms are there ONLY to stop the slope absorbing them. It is
    a classic regression pattern: include what you do not care about so it does
    not contaminate what you do.
    """
    if len(rows) < 50:
        return 0.0     # with this few laps the fit does not hold up

    drivers = sorted({r["driver"] for r in rows})
    compounds = sorted({r["compound_name"] for r in rows})

    columns = []
    for row in rows:
        # One column per driver (1 if it is theirs, 0 otherwise): absorbs the
        # fact that some cars are simply faster than others.
        driver_row = [1.0 if row["driver"] == d else 0.0 for d in drivers]

        # The variable we actually care about.
        lap = [float(row["race_lap"])]

        # One column per compound, holding the tire age.
        degradation_row = [
            float(row["tyre_life"]) if row["compound_name"] == c else 0.0
            for c in compounds
        ]

        columns.append(driver_row + lap + degradation_row)

    X = np.array(columns, dtype=float)
    y = np.array([r["lap_time"] for r in rows], dtype=float)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]

    # With fewer than 5 laps per unknown the fit is not trustworthy.
    if X.shape[0] < 5 * X.shape[1]:
        return 0.0

    coefficients = np.linalg.lstsq(
        X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ y, rcond=None
    )[0]

    # The slope is the coefficient right after the driver columns.
    return float(coefficients[len(drivers)])


# ---------------------------------------------------------------------------
# 5) ASSEMBLING THE STINTS
# ---------------------------------------------------------------------------


def _find_peak(times: np.ndarray, search_laps: int) -> int:
    """Index of the fastest lap among the first few. This is where d = 0.

    NOT the first lap. A new set comes out cold and gets FASTER for two or
    three laps before it starts falling away: that is the warm-up phase. The
    model is monotonic by construction and cannot represent it, so leaving it
    in would give every stint an irreconcilable conflict between what was
    observed and what can be predicted, and the network would compensate by
    distorting the physical constants.

    In the main project this correction took RMSE from 2.82 s to 0.567 s and
    monotonicity violations from 16.4 % to 0 %.
    """
    window = min(search_laps + 1, len(times))
    return int(np.argmin(times[:window]))


def _stint_context(laps: list[dict]) -> dict[str, float]:
    """The five model variables for a stint: the median over its laps.

    Only over the laps that survived the filters. The warm-up laps were already
    discarded as observations and must not sneak back in through the median
    that represents the stint.
    """
    from physics import COMPOUND_INDEX

    return {
        "q_fric": float(np.median([lap["q_fric"] for lap in laps])),
        "load": float(np.median([lap["load"] for lap in laps])),
        "speed": float(np.median([lap["speed"] for lap in laps])),
        "track_temp": float(np.median([lap["track_temp"] for lap in laps])),
        "compound": COMPOUND_INDEX.get(laps[0]["compound_name"], 0.5),
    }


def _one_stint(laps: list[dict], args) -> list[dict] | None:
    """Turn one group of laps into CSV rows, or None if it does not qualify."""
    if len(laps) < args.min_laps:
        return None

    times = np.array([lap["corrected_time"] for lap in laps])

    # Everything before the performance peak is warm-up: drop it.
    peak = _find_peak(times, args.reference_laps)
    laps, times = laps[peak:], times[peak:]

    # Pace loss is measured against the peak, which becomes lap 1.
    delta = times - times[0]

    # Drop laps lost to traffic or to a driver mistake.
    keep = delta <= args.max_delta
    if keep.sum() < args.min_laps:
        return None

    laps = [lap for lap, ok in zip(laps, keep) if ok]
    delta = delta[keep]

    context = _stint_context(laps)
    race = laps[0]["race"]
    stint_id = f"{race[:3].upper()}-{laps[0]['driver']}-S{laps[0]['stint']}"

    return [
        {
            "race": race,
            "driver": lap["driver"],
            "stint_id": stint_id,
            "race_lap": lap["race_lap"],
            "stint_lap": i,
            "lap_time": round(lap["lap_time"], 3),
            "corrected_time": round(lap["corrected_time"], 3),
            "delta": round(float(d), 3),
            "compound_name": lap["compound_name"],
            "tyre_life": lap["tyre_life"],
            **{k: round(v, 4) for k, v in context.items()},
        }
        for i, (lap, d) in enumerate(zip(laps, delta), start=1)
    ]


def assemble_stints(rows: list[dict], args) -> list[dict]:
    """Group laps into stints and compute each one's pace loss.

    The interesting decision -- where degradation begins -- lives in
    `_find_peak`. This function only does the grouping.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["driver"], row["stint"]), []).append(row)

    output: list[dict] = []
    for _, laps in sorted(groups.items()):
        laps.sort(key=lambda r: r["race_lap"])
        stint_rows = _one_stint(laps, args)
        if stint_rows:
            output.extend(stint_rows)

    return output


# ---------------------------------------------------------------------------
# 6) ONE RACE, END TO END
# ---------------------------------------------------------------------------


def _load_session(race_name: str, args):
    """Download (or read from cache) one session."""
    import fastf1

    print(f"\n  {race_name}")
    print(f"    downloading {args.session} session of {args.year} ...")

    fastf1.Cache.enable_cache(args.cache)
    session = fastf1.get_session(args.year, race_name, args.session)
    session.load(
        laps=True,
        telemetry=not args.no_telemetry,
        weather=True,
        messages=False,
    )
    return session


def _track_temp_series(session):
    """The session's track-temperature trace, or None if it was not recorded."""
    weather = getattr(session, "weather_data", None)
    if weather is None or len(weather) == 0 or "TrackTemp" not in weather:
        return None
    return (
        weather["Time"].dt.total_seconds().to_numpy(dtype=float),
        weather["TrackTemp"].to_numpy(dtype=float),
    )


def _track_temp_at(lap, weather) -> float:
    """Track temperature when this lap started, interpolated from the trace."""
    import pandas as pd

    start = lap.get("LapStartTime")
    start_s = start.total_seconds() if pd.notna(start) else np.nan

    if weather is None or not np.isfinite(start_s):
        return 35.0     # a typical value when no data exists
    return float(np.interp(start_s, weather[0], weather[1]))


def _lap_row(lap, race_name: str, driver: str, dynamics: dict,
             track_temp_c: float) -> dict:
    """One raw lap, as the dict every later step expects.

    This is where the three telemetry variables get normalised against the
    FIXED references, and where the track temperature goes from degrees to 0..1.
    """
    import pandas as pd

    life = lap.get("TyreLife")

    return {
        "race": race_name,
        "driver": driver,
        "stint": int(lap.get("Stint", 1)),
        "race_lap": int(lap["LapNumber"]),
        "lap_time": float(lap["LapTime"].total_seconds()),
        "compound_name": str(lap.get("Compound", "MEDIUM")).upper(),
        "tyre_life": float(life) if pd.notna(life) else float("nan"),
        "q_fric": dynamics["q_fric_raw"] / REF_Q_FRIC,
        "load": dynamics["load_raw"] / REF_LOAD,
        "speed": dynamics["speed_raw"] / REF_SPEED,
        "track_temp": float(np.clip((track_temp_c - 20.0) / 40.0, 0.0, 1.0)),
    }


def _collect_laps(session, race_name: str, args) -> list[dict]:
    """Walk every driver's laps, keep the usable ones, build their rows."""
    session_laps = session.laps
    if session_laps is None or len(session_laps) == 0:
        raise RuntimeError("the session contains no laps")

    weather = _track_temp_series(session)

    drivers = list(args.drivers) if args.drivers else sorted(
        session_laps["Driver"].dropna().unique()
    )
    if args.max_drivers:
        drivers = drivers[: args.max_drivers]

    print(f"    {len(drivers)} drivers, processing laps ...")

    rows: list[dict] = []
    dropped = 0

    for driver in drivers:
        for _, lap in session_laps[session_laps["Driver"] == driver].iterrows():
            if not lap_is_usable(lap, args.only_fresh_tyres):
                dropped += 1
                continue

            dynamics = _lap_variables(lap, args)
            if dynamics is None:
                dropped += 1
                continue

            rows.append(_lap_row(
                lap, race_name, str(driver), dynamics,
                _track_temp_at(lap, weather),
            ))

    if not rows:
        raise RuntimeError("no lap passed the quality filters")

    print(f"    {len(rows)} valid laps ({dropped} dropped)")
    return rows


def _lap_variables(lap, args) -> dict | None:
    """The three telemetry variables for one lap, honouring --no-telemetry."""
    if args.no_telemetry:
        # Without telemetry they cannot be computed, so they are left at the
        # reference value (which normalises to 1.0). The model will still see
        # temperature and compound vary, but not load or frictional energy.
        return {
            "q_fric_raw": REF_Q_FRIC,
            "load_raw": REF_LOAD,
            "speed_raw": REF_SPEED,
        }

    try:
        return lap_dynamics(lap.get_telemetry())
    except Exception:
        return None


def _apply_race_lap_correction(rows: list[dict], args) -> None:
    """Subtract the fuel + track-evolution effect. Modifies `rows` in place."""
    if args.race_lap_effect == "auto":
        slope = estimate_race_lap_effect(rows)
        print(f"    race-lap effect estimated at: {slope:+.4f} s/lap")
    else:
        slope = float(args.race_lap_effect)
        print(f"    race-lap effect fixed at: {slope:+.4f} s/lap")

    for row in rows:
        row["corrected_time"] = row["lap_time"] - slope * row["race_lap"]

        # If tire age is missing, fall back to the lap number.
        if not np.isfinite(row["tyre_life"]):
            row["tyre_life"] = float(row["race_lap"])


def process_race(race_name: str, args) -> list[dict]:
    """One race, end to end: download, filter, correct, assemble."""
    session = _load_session(race_name, args)
    rows = _collect_laps(session, race_name, args)
    _apply_race_lap_correction(rows, args)
    return assemble_stints(rows, args)


# ---------------------------------------------------------------------------
# 7) COMMAND LINE
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define and read the command-line options, grouped by what they affect."""
    p = argparse.ArgumentParser(
        description="Download F1 telemetry and turn it into a CSV for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
               "  python download_data.py --year 2023 --races Monza Hungary "
               "--out data/2023.csv",
    )

    p.add_argument("--explain-columns", action="store_true",
                   help="explain every column of the CSV and exit")

    g = p.add_argument_group("what to download")
    g.add_argument("--year", type=int, default=2023, help="season (default 2023)")
    g.add_argument("--races", nargs="+", default=["Monza"],
                   help="one or more Grands Prix: Monza Hungary Spa ...")
    g.add_argument("--session", default="R",
                   help="R=race, Q=qualifying, FP1/FP2/FP3=practice")
    g.add_argument("--drivers", nargs="*", default=[],
                   help="three-letter codes, e.g. VER HAM LEC. Empty = all")
    g.add_argument("--max-drivers", type=int, default=0,
                   help="limit to the first N drivers (handy for testing)")

    g = p.add_argument_group("data quality")
    g.add_argument("--min-laps", type=int, default=8,
                   help="discard stints with fewer valid laps (default 8)")
    g.add_argument("--max-delta", type=float, default=6.0,
                   help="discard laps losing more than N seconds (traffic)")
    g.add_argument("--reference-laps", type=int, default=3,
                   help="how many laps to search for the performance peak")
    g.add_argument("--only-fresh-tyres", action="store_true", default=True,
                   help="use only new sets (d=0 only holds on fresh rubber)")
    g.add_argument("--include-used-tyres", dest="only_fresh_tyres",
                   action="store_false", help="also include already-run sets")
    g.add_argument("--race-lap-effect", default="auto",
                   help="fuel+track correction in s/lap. 'auto' estimates it "
                        "per race; or give a number, e.g. -0.055")

    g = p.add_argument_group("speed and output")
    g.add_argument("--no-telemetry", action="store_true",
                   help="skip telemetry: far faster, but leaves q_fric, load "
                        "and speed pinned at 1.0")
    g.add_argument("--cache", default="cache",
                   help="FastF1 cache folder (fills itself, ~200 MB)")
    g.add_argument("--out", default="data/races.csv",
                   help="output CSV file")

    return p.parse_args()


def _write_csv(rows: list[dict], out: Path) -> None:
    """Write every row to the CSV, creating the folder if needed."""
    import csv

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict], out: Path, args, failures: list[str]) -> None:
    """What was downloaded, and the command to train on it."""
    n_stints = len({r["stint_id"] for r in rows})
    deltas = np.array([r["delta"] for r in rows])

    print("\n" + "=" * 70)
    print(f"Saved to {out.resolve()}")
    print(f"  {n_stints} stints | {len(rows)} laps | {len(args.races)} races")
    print(f"  Pace loss: median {np.median(deltas):.2f}s  max {deltas.max():.2f}s")
    if failures:
        print(f"  {len(failures)} races failed: {'; '.join(failures)}")
    print("\nNow you can train with:")
    print(f"  python run.py --source csv --csv {out}")


def _download_all(args) -> tuple[list[dict], list[str]]:
    """Process every requested race. Returns (rows, failures).

    One race failing does not stop the rest: its reason is collected and
    reported at the end, which matters when a season takes the better part of
    an hour to parse.
    """
    rows: list[dict] = []
    failures: list[str] = []

    for race in args.races:
        try:
            race_rows = process_race(race, args)
        except Exception as error:
            failures.append(f"{race}: {error}")
            print(f"    FAILED: {error}")
            continue

        n_stints = len({r["stint_id"] for r in race_rows})
        print(f"    -> {n_stints} stints, {len(race_rows)} laps")
        rows.extend(race_rows)

    return rows, failures


def main() -> int:
    """Download every requested race and write one CSV. Returns the exit code."""
    args = parse_args()

    if args.explain_columns:
        print(describe_columns())
        return 0

    try:
        import fastf1  # noqa: F401
    except ImportError:
        print("FastF1 is missing. Install it with:\n  pip install -r requirements.txt")
        return 1

    print("=" * 70)
    print(f"Downloading {args.session} of {args.year}: {', '.join(args.races)}")
    print("=" * 70)
    if args.no_telemetry:
        print("\n  FAST MODE: no telemetry. q_fric, load and speed will stay")
        print("  pinned at 1.0 and the model cannot learn their effect.")

    rows, failures = _download_all(args)

    if not rows:
        print("\nNo race produced valid data:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    out = Path(args.out)
    _write_csv(rows, out)
    _print_summary(rows, out, args, failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
