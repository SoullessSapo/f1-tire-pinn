"""Entrena la PINN de degradacion y la compara contra los baselines.

Ejemplos
--------
    python run_train.py --source synthetic --stints 60
    python run_train.py --source synthetic --quick
    python run_train.py --source fastf1 --year 2023 --gp Monza --session R
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from tirepinn import plots
from tirepinn.baselines import LinearDegBaseline, LSTMBaseline
from tirepinn.config import REAL_DATA_FREE_PARAMS, Config
from tirepinn.evaluate import evaluate, format_report, parameter_recovery
from tirepinn.physics import GROUND_TRUTH
from tirepinn.pinn import LEARNABLE, TirePINN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "fastf1"], default="synthetic")
    p.add_argument("--out", default="outputs", help="directorio de salida")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="corrida corta para validar el pipeline")

    g = p.add_argument_group("sintetico")
    g.add_argument("--stints", type=int, default=64)

    g = p.add_argument_group("fastf1")
    g.add_argument("--year", type=int, default=2023)
    g.add_argument(
        "--gp",
        nargs="+",
        default=["Monza"],
        help="una o varias carreras. Con varias el contexto varia de verdad; "
        "con una sola el modelo casi solo ve el efecto del compuesto.",
    )
    g.add_argument("--session", default="R")
    g.add_argument("--drivers", nargs="*", default=[], help="p. ej. VER HAM LEC")

    g = p.add_argument_group("entrenamiento")
    g.add_argument("--adam", type=int, default=15000)
    g.add_argument("--lbfgs", type=int, default=3000)
    g.add_argument("--lstm-epochs", type=int, default=800)
    g.add_argument("--no-baselines", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(out_dir=args.out)
    cfg.pinn.seed = args.seed
    cfg.pinn.adam_iters = 1200 if args.quick else args.adam
    cfg.pinn.lbfgs_iters = 300 if args.quick else args.lbfgs
    cfg.data.source = args.source
    cfg.data.n_stints = 16 if args.quick else args.stints
    cfg.data.year = args.year
    cfg.data.gp = args.gp[0]
    cfg.data.session = args.session
    cfg.data.drivers = tuple(args.drivers)
    # La temperatura no es observable en datos reales: solo el banco sintetico
    # puede darle a la red una referencia termica.
    cfg.pinn.use_theta_proxy = args.source == "synthetic"

    if args.source == "fastf1":
        # La ley termo-mecanica se calibra en el banco fisico, donde existe
        # verdad de referencia; sobre telemetria real solo se ajustan las
        # cantidades que cambian de un circuito o un lote de neumaticos a otro.
        # Ver config.REAL_DATA_FREE_PARAMS para el argumento completo.
        for name in LEARNABLE:
            if name not in REAL_DATA_FREE_PARAMS:
                setattr(cfg.physics, f"{name}_init", float(getattr(GROUND_TRUTH, name)))
        cfg.pinn.free_params = REAL_DATA_FREE_PARAMS

        # El peso del termino de datos debe escalar con la calidad del dato. El
        # cronometraje sintetico tiene sigma ~0.06 s, asi que su termino baja
        # hasta ~0.004 y deja de tirar del gradiente. Una vuelta real tiene
        # sigma ~0.5 s por trafico, viento y pilotaje, y su termino se estanca
        # en ~0.25: con el mismo peso dominaria para siempre y arrastraria a la
        # red fuera de la EDO. Se baja el peso para que la fisica siga contando
        # una vez alcanzado el suelo de ruido.
        cfg.pinn.w_data_delta = 5.0
    return cfg


def load_data(cfg: Config, args: argparse.Namespace):
    if cfg.data.source == "synthetic":
        from tirepinn import data_synthetic

        return data_synthetic.generate(cfg.data, cfg.physics, cfg.ranges, seed=args.seed)
    from tirepinn import data_fastf1

    if len(args.gp) == 1:
        return data_fastf1.build_dataset(cfg.data, cfg.physics)
    return data_fastf1.build_multi_dataset(cfg.data, cfg.physics, args.gp)


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PINN de degradacion de neumaticos de F1")
    print("=" * 78)

    data = load_data(cfg, args)
    print(data.describe())
    train, test = data.split(cfg.data.test_fraction, seed=args.seed)
    print(f"  Particion por stint: {len(train)} entrenamiento / {len(test)} prueba\n")

    # ---------------------------------------------------------------- PINN --
    print("[1/4] Entrenando la PINN ...")
    t0 = time.perf_counter()
    pinn = TirePINN(cfg)
    pinn.build(train)
    pinn.train(out)
    print(f"      listo en {time.perf_counter() - t0:.1f} s\n")

    models = {"PINN": pinn}

    # ----------------------------------------------------------- baselines --
    if not args.no_baselines:
        print("[2/4] Ajustando baselines ...")
        models["Lineal (clasico)"] = LinearDegBaseline(cfg.physics).fit(train)
        models["LSTM (caja negra)"] = LSTMBaseline(
            cfg.physics, epochs=200 if args.quick else args.lstm_epochs, seed=args.seed
        ).fit(train)
        print("      listo\n")
    else:
        print("[2/4] Baselines omitidos\n")

    # ---------------------------------------------------------- evaluacion --
    print("[3/4] Evaluando sobre stints no vistos ...\n")
    metrics = [evaluate(name, m.predict_stint, test, cfg.physics) for name, m in models.items()]
    report = format_report(metrics)
    print(report)

    recovery_txt = ""
    if cfg.data.source == "synthetic":
        rows = parameter_recovery(pinn.learned_params(), GROUND_TRUTH, cfg.pinn.free_params)
        lines = [
            "",
            "Recuperacion de parametros fisicos (problema inverso)",
            f"{'Parametro':10s} {'Estimado':>10s} {'Verdadero':>10s} {'Error rel.':>11s}",
            "-" * 45,
        ]
        lines += [f"{n:10s} {est:10.4f} {ref:10.4f} {rel:10.1f}%" for n, est, ref, rel in rows]
        lines.append(
            f"{'':10s} {'':>10s} {'media':>10s} "
            f"{np.mean([r[3] for r in rows]):10.1f}%"
        )
        recovery_txt = "\n".join(lines)
        print(recovery_txt)

    # -------------------------------------------------------------- salidas --
    print("\n[4/4] Generando figuras ...")
    loss_labels = ["EDO termica", "EDO desgaste", "Cota d<=dmax", "Datos: ritmo"]
    if cfg.pinn.use_theta_proxy:
        loss_labels.append("Datos: temperatura")

    plots.plot_stint_grid(models, test, cfg.physics, out / "01_stints.png")
    plots.plot_extrapolation(models, test, cfg.physics, out / "02_extrapolacion.png")
    plots.plot_latent_states(pinn, test, cfg.physics, out / "03_estados_latentes.png")
    plots.plot_parameter_convergence(
        pinn.var_history,
        GROUND_TRUTH if cfg.data.source == "synthetic" else None,
        out / "04_parametros.png",
    )
    plots.plot_loss_history(pinn.loss_history, out / "05_perdida.png", loss_labels)
    plots.plot_cliff_map(pinn, cfg.physics, out / "06_mapa_cliff.png")

    pinn.save(out)
    cfg.to_json(str(out / "config.json"))
    with open(out / "report.txt", "w", encoding="utf-8") as fh:
        fh.write(data.describe() + "\n\n" + report + "\n" + recovery_txt + "\n")

    print(f"      figuras, modelo y reporte en {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
