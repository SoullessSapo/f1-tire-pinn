"""Inferencia con el modelo ya entrenado, y medicion de latencia.

Carga solo los pesos de la red (sin datos ni maquinaria de entrenamiento), pide
una prediccion de estrategia para unas condiciones dadas y mide cuanto tarda un
paso forward. Es el mismo camino de codigo que ejecutaria un servicio de
inferencia, y sirve para separar el costo del modelo del costo del transporte
en el presupuesto de latencia extremo a extremo.

Ejemplo
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="outputs", help="directorio con el modelo entrenado")
    p.add_argument("--compound", default="MEDIUM", choices=sorted(COMPOUND_INDEX))
    p.add_argument("--q-fric", type=float, default=1.0, help="energia friccional relativa")
    p.add_argument("--load", type=float, default=1.0, help="carga mecanica relativa")
    p.add_argument("--speed", type=float, default=1.0, help="velocidad media relativa")
    p.add_argument("--track-temp", type=float, default=0.5, help="temperatura de pista normalizada 0-1")
    p.add_argument("--current-lap", type=float, default=0.0, help="vueltas ya rodadas con este juego")
    p.add_argument("--horizon", type=int, default=45)
    p.add_argument("--bench", type=int, default=500, help="repeticiones para medir latencia")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model)
    if not (model_dir / "pinn_weights.pt").exists():
        print(f"No hay modelo en {model_dir.resolve()}. Corre primero run_train.py")
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
    print(f"Compuesto {args.compound} | pista {args.track_temp:.2f} | carga {args.load:.2f}")
    print("=" * 62)
    print(f"{'Vuelta':>7s} {'theta':>8s} {'d':>7s} {'Perdida':>9s}")
    for i in range(0, args.horizon, 5):
        print(f"{laps[i]:7.0f} {theta[i]:8.2f} {d[i]:7.3f} {delta[i]:8.2f}s")

    print()
    if cliff is None:
        print(f"Sin cliff dentro de {args.horizon} vueltas: el juego aguanta el horizonte.")
    else:
        rul = max(cliff - args.current_lap, 0.0)
        print(f"Cliff previsto en la vuelta {cliff:.0f} del stint.")
        print(f"Vida util remanente (RUL): {rul:.0f} vueltas desde la vuelta {args.current_lap:.0f}.")

    # --- latencia del modelo ---
    pinn.forward_numpy(x)  # calentamiento
    times = []
    for _ in range(args.bench):
        t0 = time.perf_counter()
        pinn.forward_numpy(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    print()
    print(
        f"Latencia de inferencia ({args.horizon} vueltas por llamada, {args.bench} repeticiones): "
        f"media {times.mean():.3f} ms | p50 {np.percentile(times, 50):.3f} ms | "
        f"p95 {np.percentile(times, 95):.3f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
