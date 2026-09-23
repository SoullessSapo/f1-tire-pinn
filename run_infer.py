"""Predict a driver's tire wear from the race situation, and measure latency.

Nothing is typed in by hand: for the requested lap, the FastF1 API provides the
compound, the tire's age, the track temperature and the rest of the weather,
the telemetry proxies of the laps run so far, and the laps left to the flag. The
model is loaded with the configuration it was trained with, so the context is
normalised exactly as in training.

The forward pass is then timed on its own. That is the same code path an
inference service would run, and it separates the model's cost from transport
cost in the end-to-end latency budget.

Example
-------
    python run_infer.py --model outputs/2026 --gp Spa --driver VER --lap 25
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from tirepinn import data_fastf1
from tirepinn.config import CONTEXT_NAMES, DataConfig
from tirepinn.data_fastf1 import RaceSituation
from tirepinn.dataset import input_matrix
from tirepinn.physics import cliff_lap
from tirepinn.pinn import TirePINN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="outputs", help="directory holding the trained model")
    p.add_argument("--gp", required=True, help="race, by any name FastF1 recognises (Monza, Spa, ...)")
    p.add_argument("--driver", required=True, help="three-letter code or car number (VER, 1, ...)")
    p.add_argument(
        "--lap", type=int, default=None, help="race lap to predict from (default: the driver's last)"
    )
    p.add_argument("--year", type=int, default=None, help="season (default: the model's training season)")
    p.add_argument("--session", default=None, help="session (default: the model's training session)")
    p.add_argument("--bench", type=int, default=500, help="repetitions for the latency measurement")
    return p.parse_args()


def prediction_horizon(situation: RaceSituation, pinn: TirePINN) -> tuple[int, bool]:
    """Stint laps to predict: to the flag on this set, but not past the trained domain.

    Returns the horizon and whether it was cut short of the flag.
    """
    age_at_flag = situation.tyre_age + situation.laps_remaining
    horizon = min(age_at_flag, pinn.max_trained_lap)
    horizon = max(int(horizon), int(np.ceil(situation.tyre_age)), 1)
    return horizon, horizon < age_at_flag


def print_situation(situation: RaceSituation, data_cfg: DataConfig) -> None:
    w = situation.weather
    print("=" * 66)
    print(
        f"{situation.race} {data_cfg.year} ({data_cfg.session}) | {situation.driver}, "
        f"lap {situation.lap_number} of {situation.total_laps}"
    )
    print("=" * 66)
    fresh = "new set" if situation.fresh_tyre else "used set"
    print(f"Tire      {situation.compound}, {situation.tyre_age:.0f} laps old ({fresh})")
    print(
        f"Weather   track {w['track_temp_c']:.1f} C | air {w['air_temp_c']:.1f} C | "
        f"humidity {w['humidity_pct']:.0f} % | pressure {w['pressure_mbar']:.0f} mbar"
    )
    print(
        f"          wind {w['wind_speed_ms']:.1f} m/s from {w['wind_direction_deg']:.0f} deg | "
        f"{'RAIN' if w['rainfall'] else 'dry'}"
    )
    context = " | ".join(f"{n} {v:.2f}" for n, v in zip(CONTEXT_NAMES, situation.context, strict=True))
    print(f"Context   {context}")
    print(
        f"          telemetry proxies from {situation.clean_laps_so_far} clean laps so far, "
        f"{situation.clean_laps_this_stint} on this set"
    )
    if w["rainfall"]:
        print("Warning   the API reports rain on this lap; the model describes a dry tire")
    if not situation.fresh_tyre:
        print("Warning   this set was used before; the model was trained on new sets only")


def print_prediction(plan: dict, situation: RaceSituation, pinn: TirePINN, cut_short: bool) -> None:
    phys = pinn.cfg.physics
    curve = plan["curve"]
    now = max(round(situation.tyre_age), 1)
    rows = sorted(set(range(1, plan["horizon"] + 1, 5)) | {now, plan["horizon"]})

    print()
    print(f"{'Lap':>7s} {'theta':>8s} {'d':>7s} {'Pace loss':>11s}")
    for lap in rows:
        i = lap - 1
        marker = "  <- now" if lap == now else ""
        print(f"{lap:7d} {curve['theta'][i]:8.2f} {curve['d'][i]:7.3f} {curve['delta'][i]:10.2f}s{marker}")

    print()
    limit, rul = plan["wear_limit_lap"], plan["rul_laps"]
    remaining = situation.laps_remaining
    rule = f"Wear limit (d >= {phys.d_crit:g})"
    if limit is None:
        print(f"{rule} not reached within {plan['horizon']} laps of the stint.")
    elif rul == 0:
        print(f"{rule} passed on lap {limit:.0f} of the stint: the set is already beyond it.")
    else:
        print(f"{rule} on lap {limit:.0f} of the stint: {rul:.0f} laps of useful life left.")

    if remaining == 0:
        print("That was the last lap of the race.")
    elif cut_short:
        print(
            f"The flag is {remaining} laps away, beyond the {pinn.max_trained_lap:.0f}-lap "
            "stint the model was trained on: the prediction stops there."
        )
    elif limit is None or rul >= remaining:
        print(f"With {remaining} laps to the flag, the set makes it to the end.")
    elif rul > 0:
        print(
            f"With {remaining} laps to the flag, the set reaches its limit "
            f"{remaining - rul:.0f} laps before the end."
        )

    cliff = cliff_lap(curve["laps"], curve["delta"], phys)
    slope_rule = f">= {phys.cliff_slope_s_per_lap:g} s/lap over {phys.cliff_min_run} laps"
    if cliff is None:
        print(f"Pace cliff ({slope_rule}): none within {plan['horizon']} laps.")
    else:
        print(f"Pace cliff ({slope_rule}): from lap {cliff:.0f} of the stint.")


def benchmark(pinn: TirePINN, x: np.ndarray, repetitions: int) -> None:
    """Time the bare forward pass, the part of the latency that belongs to the model."""
    pinn.forward_numpy(x)  # warm-up
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        pinn.forward_numpy(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    print()
    print(
        f"Inference latency ({len(x)} laps per call, {repetitions} repetitions): "
        f"mean {times.mean():.3f} ms | p50 {np.percentile(times, 50):.3f} ms | "
        f"p95 {np.percentile(times, 95):.3f} ms"
    )


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model)
    if not (model_dir / "pinn_weights.pt").exists():
        print(f"No model found in {model_dir.resolve()}. Run run_train.py first")
        return 1

    pinn = TirePINN.load(model_dir)  # with the config.json it was trained with
    cfg = pinn.cfg
    data_cfg = replace(
        cfg.data, year=args.year or cfg.data.year, session=args.session or cfg.data.session
    )

    session = data_fastf1.load_session(data_cfg, args.gp)
    try:
        situation = data_fastf1.race_situation(session, args.driver, data_cfg, cfg.physics, args.lap)
    except ValueError as exc:
        print(exc)
        return 1

    horizon, cut_short = prediction_horizon(situation, pinn)
    plan = pinn.strategy(situation.context, horizon=horizon, current_lap=situation.tyre_age)

    print_situation(situation, data_cfg)
    print_prediction(plan, situation, pinn, cut_short)
    benchmark(pinn, input_matrix(plan["curve"]["laps"], situation.context, cfg.physics), args.bench)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
