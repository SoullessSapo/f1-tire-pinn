"""
ENTRENAR Y COMPARAR
===================

    python run.py                                  # datos sinteticos
    python run.py --quick                          # version corta, para probar
    python run.py --fuente csv --csv datos/2023.csv  # datos reales descargados

    python run.py --neuronas 96 --capas 5          # red mas grande
    python run.py --stints 64 --iteraciones 12000  # entrenamiento mas largo

El programa hace cuatro cosas, en este orden:

  1. consigue los stints (simulados o del CSV que bajo descargar_datos.py)
  2. los parte en entrenamiento y prueba, POR STINT ENTERO
  3. entrena el PINN y ajusta el baseline lineal sobre los mismos datos
  4. mide a los dos sobre los stints que ninguno ha visto, y dibuja

Con datos sinteticos hace ademas una quinta cosa que con datos reales es
imposible: comprobar si el PINN ha recuperado las constantes fisicas
verdaderas. Esa es la prueba de que el metodo funciona.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

# Backend sin pantalla. Tiene que ir ANTES de importar pyplot, porque pyplot
# elige backend al importarse. Sin esto el script falla en un servidor.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import data
from baseline import BaselineLineal
from evaluate import CABECERA, evaluar, recuperacion_de_parametros
from physics import (
    HORIZONTE_VUELTAS,
    PARAMETROS_LIBRES,
    VALORES_REALES,
    VUELTAS_REF,
    solucion_exacta,
)
from pinn import PINN

COLORES = {
    "pinn": "#B93A24",
    "lineal": "#7C8593",
    "exacta": "#2C6C8C",
    "medido": "#14181F",
}


# ---------------------------------------------------------------------------
# LINEA DE COMANDOS
# ---------------------------------------------------------------------------

def parsear_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = p.add_argument_group("de donde salen los datos")
    g.add_argument("--fuente", choices=["sintetico", "csv"], default="sintetico")
    g.add_argument("--csv", default="datos/carreras.csv",
                   help="fichero que produjo descargar_datos.py")
    g.add_argument("--stints", type=int, default=48,
                   help="cuantos stints simular (solo con --fuente sintetico)")
    g.add_argument("--ruido", type=float, default=0.05,
                   help="ruido de cronometraje en segundos (solo sintetico)")
    g.add_argument("--min-vueltas", type=int, default=8,
                   help="descartar stints mas cortos que esto (solo con --fuente csv)")

    g = p.add_argument_group("la red")
    g.add_argument("--neuronas", type=int, default=64, help="neuronas por capa")
    g.add_argument("--capas", type=int, default=4, help="numero de capas ocultas")
    g.add_argument("--iteraciones", type=int, default=8000)
    g.add_argument("--colocacion", type=int, default=2000,
                   help="puntos donde se exige la ecuacion en cada iteracion")
    g.add_argument("--lr", type=float, default=3e-3, help="tasa de aprendizaje")

    g = p.add_argument_group("otros")
    g.add_argument("--semilla", type=int, default=0)
    g.add_argument("--salida", default="outputs")
    g.add_argument("--quick", action="store_true",
                   help="1500 iteraciones y 18 stints, para comprobar que corre")

    return p.parse_args()


# ---------------------------------------------------------------------------
# GRAFICAS
# ---------------------------------------------------------------------------

def grafica_ajuste(modelos: dict, stints, sintetico: bool, ruta: Path) -> None:
    """Un panel por compuesto: lo medido, lo exacto y lo que dice cada modelo.

    La zona sombreada es donde HAY datos. A su derecha todos los modelos estan
    extrapolando, y ahi es donde se ve la diferencia entre tener fisica dentro
    y no tenerla.
    """
    # Un stint representativo de cada compuesto.
    por_compuesto: dict[str, object] = {}
    for stint in stints:
        por_compuesto.setdefault(stint.compuesto, stint)
    orden = [c for c in ("SOFT", "MEDIUM", "HARD") if c in por_compuesto]
    if not orden:
        orden = list(por_compuesto)[:3]

    fig, ejes = plt.subplots(
        1, len(orden), figsize=(4.3 * len(orden), 3.7), sharey=True, squeeze=False
    )
    horizonte = np.arange(1, HORIZONTE_VUELTAS + 1)

    for eje, compuesto in zip(ejes[0], orden):
        stint = por_compuesto[compuesto]

        eje.axvspan(1, stint.vueltas[-1], color="#000000", alpha=0.05, lw=0)
        eje.axvline(stint.vueltas[-1], color="#7C8593", ls="--", lw=1)

        eje.plot(stint.vueltas, stint.delta, "o", ms=3.5,
                 color=COLORES["medido"], label="medido (con ruido)")

        if sintetico:
            d_exacto = solucion_exacta(horizonte / VUELTAS_REF, stint.contexto, VALORES_REALES)
            eje.plot(horizonte, VALORES_REALES.gamma1 * d_exacto,
                     color=COLORES["exacta"], lw=4, alpha=0.85, label="solucion exacta")

        eje.plot(horizonte, modelos["PINN"].predecir_stint(stint.contexto, horizonte),
                 color=COLORES["pinn"], lw=1.7, label="PINN")
        eje.plot(horizonte, modelos["Lineal"].predecir_stint(stint.contexto, horizonte),
                 color=COLORES["lineal"], lw=1.8, ls="-.", label="lineal")

        eje.set_title(f"{compuesto}  ({stint.stint_id})", fontsize=10)
        eje.set_xlabel("vuelta del stint")
        eje.grid(alpha=0.15)

    ejes[0][0].set_ylabel("perdida de ritmo [s]")
    ejes[0][-1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "A la derecha de la linea discontinua, todos los modelos extrapolan",
        fontsize=9.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    plt.close(fig)


def grafica_entrenamiento(modelo: PINN, ruta: Path) -> None:
    """Como bajan los tres terminos del coste."""
    historial = modelo.historial
    iteraciones = [h["iteracion"] for h in historial]

    fig, eje = plt.subplots(figsize=(6.4, 3.8))
    for clave, etiqueta, color in [
        ("fisica", "fisica (residuo de la ecuacion)", COLORES["exacta"]),
        ("datos", "datos (ritmo medido)", COLORES["pinn"]),
        ("ci", "condicion inicial d(0)=0", COLORES["lineal"]),
    ]:
        eje.plot(iteraciones, [h[clave] for h in historial], lw=1.6,
                 color=color, label=etiqueta)

    eje.set_yscale("log")
    eje.set_xlabel("iteracion")
    eje.set_ylabel("coste (escala logaritmica)")
    eje.set_title("Los tres terminos del coste", fontsize=10)
    eje.legend(fontsize=8)
    eje.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    plt.close(fig)


def grafica_parametros(modelo: PINN, sintetico: bool, ruta: Path) -> None:
    """Las seis constantes fisicas mientras se estiman.

    Con datos sinteticos se dibuja tambien el valor verdadero: si las curvas
    aterrizan en las lineas de puntos, el problema inverso ha funcionado.
    """
    historial = modelo.historial
    iteraciones = [h["iteracion"] for h in historial]

    fig, ejes = plt.subplots(2, 3, figsize=(11, 5.4), squeeze=False)

    for eje, nombre in zip(ejes.ravel(), PARAMETROS_LIBRES):
        eje.plot(iteraciones, [h[nombre] for h in historial],
                 lw=1.8, color=COLORES["pinn"], label="estimado")
        if sintetico:
            eje.axhline(float(getattr(VALORES_REALES, nombre)),
                        color=COLORES["medido"], ls="--", lw=1.2, label="valor real")
        eje.set_title(nombre, fontsize=10)
        eje.set_xlabel("iteracion")
        eje.grid(alpha=0.15)

    ejes[0][0].legend(fontsize=8)
    fig.suptitle(
        "Problema inverso: seis constantes fisicas estimadas junto a los pesos",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> int:
    args = parsear_argumentos()

    n_stints = 18 if args.quick else args.stints
    iteraciones = 1500 if args.quick else args.iteraciones

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    sintetico = args.fuente == "sintetico"

    print("=" * 74)
    print("PINN de degradacion de neumaticos - v0 ampliado")
    print("=" * 74)

    # --- 1) los datos -----------------------------------------------------
    if sintetico:
        stints = data.generar_sinteticos(
            n_stints=n_stints, ruido_s=args.ruido, semilla=args.semilla
        )
        print(f"\nFuente: banco sintetico (ruido sigma = {args.ruido} s)")
    else:
        try:
            stints = data.cargar_csv(args.csv, min_vueltas=args.min_vueltas)
        except (FileNotFoundError, ValueError) as error:
            # Un mensaje claro vale mas que una traza de error de veinte lineas.
            print(f"\nNo se pudieron cargar los datos:\n  {error}")
            return 1
        print(f"\nFuente: {args.csv}")

    print(data.resumen(stints))

    entrenamiento, prueba = data.partir(stints, semilla=args.semilla)
    print(f"  Particion por stint entero: {len(entrenamiento)} entrenar "
          f"/ {len(prueba)} probar")

    entradas, delta = data.aplanar(entrenamiento)

    # --- 2) el PINN -------------------------------------------------------
    print(f"\n[1/3] Entrenando el PINN ({iteraciones} iteraciones de Adam) ...")
    modelo = PINN(neuronas=args.neuronas, capas=args.capas, semilla=args.semilla)
    print(f"      red: {args.capas} capas de {args.neuronas} neuronas, "
          f"{modelo.n_parametros_red()} pesos")
    print(f"      mas {len(PARAMETROS_LIBRES)} constantes fisicas a estimar: "
          f"{', '.join(PARAMETROS_LIBRES)}")

    t0 = time.perf_counter()
    modelo.entrenar(
        entradas, delta,
        iteraciones=iteraciones,
        n_colocacion=args.colocacion,
        lr=args.lr,
    )
    print(f"      listo en {time.perf_counter() - t0:.1f} s")

    # --- 3) el rival ------------------------------------------------------
    print("\n[2/3] Ajustando el baseline lineal ...")
    lineal = BaselineLineal().ajustar(entradas, delta)
    print("      listo")

    # --- 4) medir ---------------------------------------------------------
    print("\n[3/3] Midiendo sobre stints que ninguno ha visto\n")
    modelos = {"PINN": modelo, "Lineal": lineal}
    metricas = [
        evaluar("PINN", modelo.predecir_stint, prueba),
        evaluar("Lineal clasico", lineal.predecir_stint, prueba),
    ]

    lineas = [CABECERA, "-" * len(CABECERA)] + [m.fila() for m in metricas]
    lineas += [
        "",
        "ViolDentro / ViolExtrap = % de vueltas donde el modelo predice que el",
        "neumatico RECUPERA agarre. Es fisicamente imposible: el valor correcto es 0 %.",
    ]

    if sintetico:
        estimados = modelo.parametros_estimados()
        filas = recuperacion_de_parametros(estimados, VALORES_REALES, PARAMETROS_LIBRES)
        lineas += [
            "",
            "Problema inverso: recuperacion de las constantes fisicas",
            f"{'Constante':12s} {'Estimado':>10s} {'Real':>10s} {'Error':>9s}",
            "-" * 45,
        ]
        lineas += [
            f"{n:12s} {est:10.4f} {real:10.4f} {err:8.1f}%"
            for n, est, real, err in filas
        ]
        lineas.append(
            f"{'':12s} {'':>10s} {'media':>10s} "
            f"{np.mean([f[3] for f in filas]):8.1f}%"
        )
    else:
        estimados = modelo.parametros_estimados()
        lineas += [
            "",
            "Constantes fisicas estimadas (con datos reales no hay verdad de",
            "referencia contra la que comparar):",
        ]
        lineas += [
            f"  {n:8s} {float(getattr(estimados, n)):8.4f}" for n in PARAMETROS_LIBRES
        ]

    informe = "\n".join(lineas)
    print(informe)

    # --- 5) dibujar y guardar --------------------------------------------
    grafica_ajuste(modelos, prueba, sintetico, salida / "01_ajuste.png")
    grafica_entrenamiento(modelo, salida / "02_entrenamiento.png")
    grafica_parametros(modelo, sintetico, salida / "03_parametros.png")
    (salida / "report.txt").write_text(
        data.resumen(stints) + "\n\n" + informe + "\n", encoding="utf-8"
    )

    print(f"\nFiguras e informe en {salida.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
