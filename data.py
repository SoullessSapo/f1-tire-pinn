"""
DE DONDE SALEN LOS STINTS
=========================

Un "stint" es un juego de neumaticos: desde que el coche sale de boxes hasta
que vuelve a entrar. Es la unidad de aprendizaje de todo el proyecto. La red
nunca ve vueltas sueltas, porque la fisica que se le impone es una ecuacion en
el tiempo DENTRO de un stint.

Este fichero sabe construir stints de dos sitios:

  generar_sinteticos()  simula stints resolviendo la ecuacion con constantes
                        conocidas y anadiendo ruido de cronometraje.

  cargar_csv()          lee el CSV que produce `descargar_datos.py` a partir de
                        la telemetria real de FastF1.

Los dos devuelven exactamente lo mismo: una lista de objetos `Stint`. A partir
de ahi, el resto del proyecto no sabe ni le importa de donde vinieron.


POR QUE EMPEZAR POR LOS SINTETICOS
----------------------------------
Porque de ellos se conoce la respuesta. Con datos sinteticos se sabe cuanto
valen de verdad las constantes fisicas y cuanto vale de verdad el desgaste d en
cada vuelta, asi que se puede comprobar si el modelo acierta. Con datos reales
eso es imposible: nadie publica el estado de la banda de rodadura.

Si el modelo no recupera las constantes en el banco sintetico, apuntarlo a
datos reales solo sirve para obtener numeros equivocados con mas trabajo.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physics import (
    INDICE_COMPUESTO,
    RANGOS_CONTEXTO,
    VALORES_REALES,
    VUELTAS_REF,
    Contexto,
    ParametrosFisicos,
    integrar_stint,
    perdida_ritmo,
)

_COMPUESTOS = ("SOFT", "MEDIUM", "HARD")


# ---------------------------------------------------------------------------
# 1) LA UNIDAD DE DATOS
# ---------------------------------------------------------------------------

@dataclass
class Stint:
    """Un juego de neumaticos, de boxes a boxes."""

    stint_id: str
    contexto: Contexto        # las 5 condiciones, constantes en todo el stint
    vueltas: np.ndarray       # vuelta dentro del stint: 1, 2, 3, ...
    delta: np.ndarray         # perdida de ritmo MEDIDA [s] (lleva ruido)

    # Solo existen en los stints sinteticos, donde se conoce la verdad.
    # Se usan para comprobar, jamas para entrenar.
    d_real: np.ndarray | None = None
    delta_sin_ruido: np.ndarray | None = None

    @property
    def n_vueltas(self) -> int:
        return int(self.vueltas.size)

    @property
    def tau(self) -> np.ndarray:
        """El tiempo adimensional de cada vuelta."""
        return self.vueltas / VUELTAS_REF

    @property
    def compuesto(self) -> str:
        return self.contexto.nombre_compuesto


# ---------------------------------------------------------------------------
# 2) STINTS SINTETICOS
# ---------------------------------------------------------------------------

def _sortear_contexto(rng: np.random.Generator, compuesto: str) -> Contexto:
    """Sortea unas condiciones plausibles para un compuesto dado."""
    valores = {}
    for nombre, (bajo, alto) in RANGOS_CONTEXTO.items():
        valores[nombre] = float(rng.uniform(bajo, alto))
    # El compuesto no se sortea: lo decide quien llama.
    valores["compuesto"] = INDICE_COMPUESTO[compuesto]
    return Contexto(**valores)


def generar_sinteticos(
    n_stints: int = 24,
    p: ParametrosFisicos = VALORES_REALES,
    ruido_s: float = 0.05,
    min_vueltas: int = 12,
    max_vueltas: int = 30,
    semilla: int = 0,
) -> list[Stint]:
    """Genera `n_stints` stints simulados, rotando entre los tres compuestos.

    El ruido importa y no es decorativo. Sin el, el ajuste es trivial y el
    termino de fisica de la perdida no tiene nada que hacer. Con el, se nota
    enseguida la diferencia entre un modelo que persigue el ruido y uno que
    esta sujeto por una ecuacion.
    """
    rng = np.random.default_rng(semilla)
    stints: list[Stint] = []

    for i in range(n_stints):
        compuesto = _COMPUESTOS[i % len(_COMPUESTOS)]
        contexto = _sortear_contexto(rng, compuesto)
        n_vueltas = int(rng.integers(min_vueltas, max_vueltas + 1))

        # 1) resolver la ecuacion -> el desgaste real vuelta a vuelta
        vueltas, d = integrar_stint(n_vueltas, contexto, p)

        # 2) traducirlo a lo unico observable: segundos perdidos
        delta_limpio = perdida_ritmo(d, p)

        # 3) ensuciarlo como lo ensucia la realidad: trafico, viento, el piloto
        delta_medido = delta_limpio + rng.normal(0.0, ruido_s, size=delta_limpio.shape)

        stints.append(
            Stint(
                stint_id=f"SIN{i:02d}",
                contexto=contexto,
                vueltas=vueltas,
                delta=delta_medido,
                d_real=d,
                delta_sin_ruido=delta_limpio,
            )
        )

    return stints


# ---------------------------------------------------------------------------
# 3) STINTS REALES, DESDE EL CSV DESCARGADO
# ---------------------------------------------------------------------------

# Columnas que `cargar_csv` necesita. `descargar_datos.py` escribe estas y
# algunas mas (tiempo de vuelta, vuelta de carrera...) para poder inspeccionar
# el fichero a mano en una hoja de calculo.
_COLUMNAS_REQUERIDAS = (
    "stint_id", "vuelta_stint", "delta",
    "q_friccion", "carga", "velocidad", "temp_pista", "compuesto",
)


def cargar_csv(ruta: str | Path, min_vueltas: int = 8) -> list[Stint]:
    """Lee el CSV de `descargar_datos.py` y lo convierte en stints.

    El CSV tiene una fila por vuelta. Aqui se agrupan por `stint_id` y se
    comprueba que el contexto sea de verdad constante dentro del stint, porque
    todo el modelo se apoya en esa suposicion.

    Los stints con menos de `min_vueltas` vueltas validas se descartan: con
    cuatro puntos no se distingue una curva de una recta.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Descarga los datos primero:\n"
            f"  python descargar_datos.py --anio 2023 --carreras Monza --salida {ruta}"
        )

    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))

    if not filas:
        raise ValueError(f"{ruta} esta vacio")

    faltan = [c for c in _COLUMNAS_REQUERIDAS if c not in filas[0]]
    if faltan:
        raise ValueError(f"A {ruta} le faltan columnas: {', '.join(faltan)}")

    # Agrupar las filas por stint, conservando el orden de aparicion.
    grupos: dict[str, list[dict]] = {}
    for fila in filas:
        grupos.setdefault(fila["stint_id"], []).append(fila)

    stints: list[Stint] = []
    descartados = 0

    for stint_id, filas_stint in grupos.items():
        filas_stint.sort(key=lambda f: float(f["vuelta_stint"]))

        if len(filas_stint) < min_vueltas:
            descartados += 1
            continue

        vueltas = np.array([float(f["vuelta_stint"]) for f in filas_stint])
        delta = np.array([float(f["delta"]) for f in filas_stint])

        # El contexto se toma de la primera fila. `descargar_datos.py` ya
        # escribe el mismo valor en todas las filas del stint (usa la mediana
        # del stint), asi que cualquiera sirve.
        primera = filas_stint[0]
        contexto = Contexto(
            q_friccion=float(primera["q_friccion"]),
            carga=float(primera["carga"]),
            velocidad=float(primera["velocidad"]),
            temp_pista=float(primera["temp_pista"]),
            compuesto=float(primera["compuesto"]),
        )

        stints.append(Stint(stint_id=stint_id, contexto=contexto,
                            vueltas=vueltas, delta=delta))

    if not stints:
        raise ValueError(
            f"Ningun stint de {ruta} llega a {min_vueltas} vueltas validas. "
            "Baja --min-vueltas o descarga mas carreras."
        )

    if descartados:
        print(f"  {descartados} stints descartados por tener menos de {min_vueltas} vueltas")

    return stints


# ---------------------------------------------------------------------------
# 4) PARTIR EN ENTRENAMIENTO Y PRUEBA
# ---------------------------------------------------------------------------

def partir(
    stints: list[Stint], fraccion_prueba: float = 0.25, semilla: int = 0
) -> tuple[list[Stint], list[Stint]]:
    """Partir por STINT ENTERO, nunca por vuelta.

    Si se partiera por vuelta, las vueltas 5 y 6 del mismo juego acabarian una
    en entrenamiento y otra en prueba. El modelo se estaria evaluando sobre una
    curva que ya ha visto a medias: eso es fuga de informacion, y ademas es la
    peor clase de fuga, porque mejora a TODOS los modelos por igual y por tanto
    no se nota comparandolos entre si.
    """
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(len(stints))
    n_prueba = max(1, round(fraccion_prueba * len(stints)))
    indices_prueba = set(orden[:n_prueba].tolist())

    entrenamiento = [s for i, s in enumerate(stints) if i not in indices_prueba]
    prueba = [s for i, s in enumerate(stints) if i in indices_prueba]
    return entrenamiento, prueba


def aplanar(stints: list[Stint]) -> tuple[np.ndarray, np.ndarray]:
    """Convierte una lista de stints en las dos matrices que entrena la red.

    Devuelve:
      entradas  (N, 6)  cada fila es [tau, y las 5 variables de contexto]
      delta     (N, 1)  la perdida de ritmo medida en esa vuelta

    El contexto es constante dentro del stint, asi que se REPITE en cada una de
    sus vueltas. Eso es lo que permite que la red aprenda a la vez el efecto del
    tiempo y el de las condiciones.
    """
    bloques_entrada = []
    bloques_delta = []

    for s in stints:
        tau = s.tau.reshape(-1, 1)
        contexto_repetido = np.tile(s.contexto.vector().reshape(1, -1), (tau.shape[0], 1))
        bloques_entrada.append(np.hstack([tau, contexto_repetido]))
        bloques_delta.append(s.delta.reshape(-1, 1))

    return np.vstack(bloques_entrada), np.vstack(bloques_delta)


def resumen(stints: list[Stint]) -> str:
    """Un parrafo describiendo el conjunto, para imprimir antes de entrenar."""
    if not stints:
        return "Conjunto vacio"

    longitudes = np.array([s.n_vueltas for s in stints])
    deltas = np.concatenate([s.delta for s in stints])

    cuenta: dict[str, int] = {}
    for s in stints:
        cuenta[s.compuesto] = cuenta.get(s.compuesto, 0) + 1
    por_compuesto = ", ".join(f"{k}:{v}" for k, v in sorted(cuenta.items()))

    return (
        f"{len(stints)} stints | {int(longitudes.sum())} vueltas\n"
        f"  Longitud del stint:  min={longitudes.min()} "
        f"mediana={np.median(longitudes):.0f} max={longitudes.max()}\n"
        f"  Perdida de ritmo:    min={deltas.min():.2f}s "
        f"mediana={np.median(deltas):.2f}s max={deltas.max():.2f}s\n"
        f"  Compuestos: {por_compuesto}"
    )
