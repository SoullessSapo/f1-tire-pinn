"""
COMO SE MIDE SI UN MODELO ES BUENO
==================================

Dos metricas, porque responden a preguntas distintas y una sola no basta.


1) RMSE (y MAE)
---------------
Cuanto se equivoca el modelo en la vuelta que esta mirando ahora mismo.

    RMSE = raiz( media( (prediccion - medida)^2 ) )

Se eleva al cuadrado para que los errores grandes pesen mas que los pequenos, y
luego se toma la raiz para volver a segundos. El MAE es la media del error en
valor absoluto: mas facil de interpretar, menos sensible a un caso extremo.

Es la metrica obvia. Por si sola no es suficiente.


2) VIOLACIONES DE MONOTONIA
---------------------------
Cuantas veces el modelo predice que el neumatico RECUPERA agarre de una vuelta
a la siguiente.

Eso es fisicamente imposible: la goma no vuelve. Y lo importante es que NINGUNA
metrica de error lo penaliza. Un modelo puede tener un RMSE estupendo y aun asi
decir que en la vuelta 34 el coche va a ir mas rapido que en la 33, lo cual
hace la prediccion inservible para tomar una decision: si dice eso en la 34,
tampoco te vas a fiar de lo que diga en la 20.

El valor correcto es 0 %.

Se mide DOS VECES:
  - dentro del stint observado
  - extrapolando al horizonte completo de 45 vueltas

La segunda es donde se separan los modelos, porque es donde ya no hay datos que
sujeten la curva y lo unico que queda es lo que el modelo cree que es el mundo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import HORIZONTE_VUELTAS


@dataclass
class Metricas:
    """El resultado de evaluar un modelo sobre un conjunto de stints."""

    nombre: str
    rmse: float
    mae: float
    error_max: float
    violaciones_dentro: float
    violaciones_extrapolando: float

    def fila(self) -> str:
        return (
            f"{self.nombre:18s} {self.rmse:8.3f} {self.mae:8.3f} {self.error_max:9.3f} "
            f"{100 * self.violaciones_dentro:10.1f}% {100 * self.violaciones_extrapolando:11.1f}%"
        )


CABECERA = (
    f"{'Modelo':18s} {'RMSE':>8s} {'MAE':>8s} {'ErrorMax':>9s} "
    f"{'ViolDentro':>11s} {'ViolExtrap':>12s}"
)


def _contar_violaciones(delta: np.ndarray, tolerancia: float = 1e-3) -> tuple[int, int]:
    """Cuenta los pasos de una vuelta a la siguiente en los que el ritmo MEJORA.

    Devuelve (violaciones, pasos_totales). La tolerancia evita contar como
    violacion una mejora de una millonesima de segundo, que solo seria ruido
    numerico.
    """
    if delta.size < 2:
        return 0, 0
    diferencias = np.diff(np.asarray(delta, dtype=float))
    return int((diferencias < -tolerancia).sum()), int(diferencias.size)


def evaluar(nombre: str, predecir_stint, stints) -> Metricas:
    """Mide cualquier modelo que exponga predecir_stint(contexto, vueltas).

    Que los dos modelos compartan esa firma no es un detalle de estilo: es lo
    que garantiza que se les esta preguntando exactamente lo mismo.
    """
    errores = []
    viol_dentro = pasos_dentro = 0
    viol_extrap = pasos_extrap = 0

    horizonte = np.arange(1, HORIZONTE_VUELTAS + 1)

    for stint in stints:
        # --- dentro del stint observado ---
        prediccion = np.asarray(
            predecir_stint(stint.contexto, stint.vueltas), dtype=float
        ).ravel()
        errores.append(prediccion - stint.delta)

        v, t = _contar_violaciones(prediccion)
        viol_dentro += v
        pasos_dentro += t

        # --- extrapolando: se le pide el stint entero hasta el horizonte ---
        prediccion_larga = np.asarray(
            predecir_stint(stint.contexto, horizonte), dtype=float
        ).ravel()

        v, t = _contar_violaciones(prediccion_larga)
        viol_extrap += v
        pasos_extrap += t

    error = np.concatenate(errores)

    return Metricas(
        nombre=nombre,
        rmse=float(np.sqrt(np.mean(error ** 2))),
        mae=float(np.mean(np.abs(error))),
        error_max=float(np.max(np.abs(error))),
        violaciones_dentro=viol_dentro / pasos_dentro if pasos_dentro else 0.0,
        violaciones_extrapolando=viol_extrap / pasos_extrap if pasos_extrap else 0.0,
    )


def recuperacion_de_parametros(estimados, reales, nombres) -> list[tuple]:
    """Compara las constantes estimadas con las verdaderas.

    Solo tiene sentido con datos sinteticos. Con datos reales no existe ninguna
    verdad de referencia contra la que comparar, y ese es precisamente el
    motivo de tener un banco sintetico.

    Devuelve (nombre, estimado, real, error relativo en %).
    """
    filas = []
    for nombre in nombres:
        estimado = float(getattr(estimados, nombre))
        real = float(getattr(reales, nombre))
        error = 100.0 * abs(estimado - real) / abs(real) if real else float("nan")
        filas.append((nombre, estimado, real, error))
    return filas
