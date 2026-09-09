"""Train the v0 PINN, compare it against the linear baseline, draw the figures.

    python run.py                 # 24 stints, 6 000 iteraciones (~1 min en CPU)
    python run.py --quick         # version corta para comprobar que corre
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
from baseline import LinearDegBaseline
from evaluate import HEADER, evaluate
from physics import COMPOUND_INDEX, GROUND_TRUTH, LAP_REF, exact_solution
from pinn import TirePINN

COLORS = {"pinn": "#B93A24", "linear": "#7C8593", "exact": "#2C6C8C"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stints", type=int, default=24)
    p.add_argument("--iterations", type=int, default=6000)
    p.add_argument("--noise", type=float, default=0.05, help="ruido de cronometraje [s]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs")
    p.add_argument("--quick", action="store_true", help="1 000 iteraciones, 12 stints")
    return p.parse_args()


def plot_fit(models: dict, stints, out: Path) -> None:
    """One panel per compound: the data, the exact curve, and both models."""
    by_compound = {}
    for s in stints:
        by_compound.setdefault(s.compound, s)
    order = [c for c in ("SOFT", "MEDIUM", "HARD") if c in by_compound]

    fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 3.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, compound in zip(axes, order):
        s = by_compound[compound]
        horizon = np.arange(1, 46)

        ax.axvspan(1, s.laps[-1], color="#000000", alpha=0.05, lw=0)
        ax.axvline(s.laps[-1], color="#7C8593", ls="--", lw=1)
        ax.plot(s.laps, s.delta, "o", ms=3.5, color="#14181F", label="medido (con ruido)")
        ax.plot(
            horizon,
            GROUND_TRUTH.gamma1 * exact_solution(horizon / LAP_REF, s.c, GROUND_TRUTH),
            color=COLORS["exact"], lw=4, alpha=0.85, label="solucion exacta",
        )
        ax.plot(
            horizon, models["PINN"].predict_stint(s.c, horizon),
            color=COLORS["pinn"], lw=1.6, label="PINN (encima de la exacta)",
        )
        ax.plot(
            horizon, models["Lineal"].predict_stint(s.c, horizon),
            color=COLORS["linear"], lw=1.8, ls="-.", label="lineal",
        )
        ax.set_title(f"{compound}  ({s.stint_id})", fontsize=10)
        ax.set_xlabel("vuelta del stint")
        ax.grid(alpha=0.15)

    axes[0].set_ylabel("perdida de ritmo [s]")
    axes[0].set_ylim(-0.3, 1.7)
    axes[-1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "A la derecha de la linea discontinua, todos los modelos extrapolan",
        fontsize=9.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_training(model: TirePINN, out: Path) -> None:
    """Loss terms and the estimated physical constant, side by side."""
    hist = model.history
    it = [h["iter"] for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    for key, label, color in [
        ("phys", "fisica (residuo EDO)", COLORS["exact"]),
        ("data", "datos (ritmo medido)", COLORS["pinn"]),
        ("ic", "condicion inicial", COLORS["linear"]),
    ]:
        ax1.plot(it, [h[key] for h in hist], label=label, lw=1.6, color=color)
    ax1.set_yscale("log")
    ax1.set_xlabel("iteracion")
    ax1.set_ylabel("perdida")
    ax1.set_title("Los tres terminos de la perdida", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.15)

    ax2.plot(it, [h["kw"] for h in hist], lw=2, color=COLORS["pinn"], label="kw estimado")
    ax2.axhline(GROUND_TRUTH.kw, color="#14181F", ls="--", lw=1.2, label="valor real")
    ax2.set_xlabel("iteracion")
    ax2.set_ylabel("kw")
    ax2.set_title("Problema inverso: recuperar una constante fisica", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    n_stints = 12 if args.quick else args.stints
    iterations = 1000 if args.quick else args.iterations
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PINN de degradacion de neumaticos - v0 (implementacion inicial)")
    print("=" * 70)

    stints = data.generate(n_stints, noise_s=args.noise, seed=args.seed)
    train, test = data.split(stints, seed=args.seed)
    tau, c, delta = data.as_arrays(train)
    print(f"\n{len(stints)} stints sinteticos, {sum(s.laps.size for s in stints)} vueltas")
    print(f"Particion por stint: {len(train)} entrenamiento / {len(test)} test")
    print(f"Ruido de cronometraje: sigma = {args.noise} s\n")

    print(f"[1/3] Entrenando el PINN ({iterations} iteraciones de Adam) ...")
    t0 = time.perf_counter()
    model = TirePINN(seed=args.seed)
    print(f"      red: {model.n_parameters()} parametros, kw inicial {model.kw_value:.3f}")
    model.train(tau, c, delta, iterations=iterations)
    print(f"      listo en {time.perf_counter() - t0:.1f} s\n")

    print("[2/3] Ajustando el baseline lineal ...")
    linear = LinearDegBaseline().fit(tau, c, delta)
    print("      listo\n")

    print("[3/3] Evaluando sobre stints no vistos\n")
    models = {"PINN": model, "Lineal": linear}
    metrics = [
        evaluate("PINN", model.predict_stint, test),
        evaluate("Lineal clasico", linear.predict_stint, test),
    ]
    lines = [HEADER, "-" * len(HEADER)] + [m.row() for m in metrics]
    kw_est = model.kw_value
    lines += [
        "",
        "ViolDentro / ViolExtrap = % de vueltas donde el modelo predice que el",
        "neumatico RECUPERA agarre. Es fisicamente imposible; el valor correcto es 0 %.",
        "",
        "Problema inverso (una constante):",
        f"  kw estimado {kw_est:.4f}  |  real {GROUND_TRUTH.kw:.4f}  |  "
        f"error {100 * abs(kw_est - GROUND_TRUTH.kw) / GROUND_TRUTH.kw:.1f} %",
    ]
    report = "\n".join(lines)
    print(report)

    plot_fit(models, test, out / "01_ajuste.png")
    plot_training(model, out / "02_entrenamiento.png")
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nFiguras e informe en {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
