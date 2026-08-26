"""Inference with a trained model, plus latency measurement.

Loads only the network weights (no data, no training machinery), asks for a
strategy prediction under given conditions, and measures how long a forward pass
takes. This is the same code path an inference service would run, and it serves
to separate the model's cost from transport cost in the end-to-end latency
budget.

Example
-------
    python run_infer.py --compound SOFT --track-temp 0.8 --load 1.2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from tirepinn.config import COMPOUND_INDEX, Config
from tirepinn.physics import cliff_lap, pace_loss
from tirepinn.pinn import TirePINN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="outputs", help="directory holding the trained model")
    p.add_argument("--compound", default="MEDIUM", choices=sorted(COMPOUND_INDEX))
    p.add_argument("--q-fric", type=float, default=1.0, help="relative frictional energy")
    p.add_argument("--load", type=float, default=1.0, help="relative mechanical load")
    p.add_argument("--speed", type=float, default=1.0, help="relative mean speed")
    p.add_argument("--track-temp", type=float, default=0.5, help="normalised track temperature 0-1")
    p.add_argument("--current-lap", type=float, default=0.0, help="laps already run on this set")
    p.add_argument("--horizon", type=int, default=45)
    p.add_argument("--bench", type=int, default=500, help="repetitions for the latency measurement")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model)
    if not (model_dir / "pinn_weights.pt").exists():
        print(f"No model found in {model_dir.resolve()}. Run run_train.py first")
        return 1

    pinn = TirePINN.load(model_dir, Config())
    context = np.array(
        [args.q_fric, args.load, args.speed, args.track_temp, COMPOUND_INDEX[args.compound]]
    )
    laps = np.arange(1, args.horizon + 1, dtype=float)
    x = np.hstack(
        [(laps / pinn.cfg.physics.lap_ref).reshape(-1, 1), np.tile(context, (laps.size, 1))]
    )

    y = pinn.forward_numpy(x)
    theta, d = y[:, 0], y[:, 1]
    delta = pace_loss(d, pinn.learned_params())
    cliff = cliff_lap(laps, delta, pinn.cfg.physics)

    print("=" * 62)
    print(f"Compound {args.compound} | track {args.track_temp:.2f} | load {args.load:.2f}")
    print("=" * 62)
    print(f"{'Lap':>7s} {'theta':>8s} {'d':>7s} {'Pace loss':>11s}")
    for i in range(0, args.horizon, 5):
        print(f"{laps[i]:7.0f} {theta[i]:8.2f} {d[i]:7.3f} {delta[i]:10.2f}s")

    print()
    if cliff is None:
        print(f"No cliff within {args.horizon} laps: the set survives the horizon.")
    else:
        rul = max(cliff - args.current_lap, 0.0)
        print(f"Cliff expected on lap {cliff:.0f} of the stint.")
        print(f"Remaining useful life (RUL): {rul:.0f} laps from lap {args.current_lap:.0f}.")

    # --- model latency ---
    pinn.forward_numpy(x)  # warm-up
    times = []
    for _ in range(args.bench):
        t0 = time.perf_counter()
        pinn.forward_numpy(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    print()
    print(
        f"Inference latency ({args.horizon} laps per call, {args.bench} repetitions): "
        f"mean {times.mean():.3f} ms | p50 {np.percentile(times, 50):.3f} ms | "
        f"p95 {np.percentile(times, 95):.3f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
