"""
EL MODELO CONTRA EL QUE HAY QUE COMPETIR
========================================

Este es el modelo que los equipos usan de verdad: una tasa de degradacion en
segundos por vuelta, ajustada por compuesto y condiciones. Aqui esta escrito
como una regresion por minimos cuadrados.

Se le da una oportunidad JUSTA a proposito: ademas del termino lineal en el
tiempo lleva un termino cuadratico y las interacciones del tiempo con las cinco
variables de contexto. Un rival de paja que pierde por falta de flexibilidad no
demuestra nada.

Y aun asi tiene dos propiedades que son justo de lo que va la comparacion:

  - DENTRO del rango medido es dificil de batir. A lo largo de 12-30 vueltas la
    curva real esta suavemente doblada, y una parabola ajusta muy bien una
    curva suavemente doblada.

  - FUERA de ese rango no lo sujeta nada. Una parabola ajustada a una curva que
    satura sigue curvandose, asi que extrapolada lo suficiente acabara
    prediciendo que el neumatico RECUPERA agarre. Eso no es un problema de
    ajuste: es lo que hace esa familia de funciones.


MINIMOS CUADRADOS, EN UNA LINEA
-------------------------------
Se busca el vector de coeficientes `w` que minimiza ||X*w - y||^2, donde cada
fila de X son las caracteristicas de una vuelta y cada y es su perdida de ritmo
medida. La solucion se obtiene resolviendo (X'X)w = X'y.

El termino `ridge` suma un numero minusculo a la diagonal de X'X. Es un seguro:
si dos caracteristicas fueran casi identicas, X'X seria casi singular y el
sistema no tendria solucion estable. No cambia el resultado, evita el fallo.
"""

from __future__ import annotations

import numpy as np

from physics import VUELTAS_REF, Contexto


class BaselineLineal:
    """Minimos cuadrados sobre caracteristicas de (tiempo, contexto)."""

    nombre = "Lineal clasico"

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.coeficientes: np.ndarray | None = None

    @staticmethod
    def _caracteristicas(tau: np.ndarray, contexto: np.ndarray) -> np.ndarray:
        """Construye la matriz de diseno X.

        Por cada vuelta, 13 numeros:
            1                        el termino independiente
            tau, tau^2               la forma de la curva en el tiempo
            las 5 de contexto        el nivel que impone cada condicion
            tau * (las 5)            como cada condicion cambia la PENDIENTE

        Las interacciones del ultimo grupo son las que le permiten decir "en
        pista caliente se degrada mas deprisa", que es lo que hace que la
        comparacion sea honesta.
        """
        tau = np.asarray(tau, dtype=float).reshape(-1, 1)
        contexto = np.asarray(contexto, dtype=float).reshape(tau.shape[0], -1)

        return np.hstack([
            np.ones_like(tau),      # 1
            tau,                    # 1
            tau ** 2,               # 1
            contexto,               # 5
            tau * contexto,         # 5
        ])                          # total: 13 columnas

    def ajustar(self, entradas: np.ndarray, delta: np.ndarray) -> BaselineLineal:
        """`entradas` es la matriz (N, 6) que devuelve data.aplanar()."""
        tau = entradas[:, 0]
        contexto = entradas[:, 1:]

        X = self._caracteristicas(tau, contexto)
        y = np.asarray(delta, dtype=float).ravel()

        gram = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.coeficientes = np.linalg.solve(gram, X.T @ y)
        return self

    def predecir_stint(self, contexto: Contexto, vueltas: np.ndarray) -> np.ndarray:
        """Misma firma que la del PINN, para que el evaluador no distinga."""
        if self.coeficientes is None:
            raise RuntimeError("Llama a ajustar() antes de predecir")

        vueltas = np.asarray(vueltas, dtype=float).ravel()
        tau = vueltas / VUELTAS_REF
        contexto_repetido = np.tile(contexto.vector().reshape(1, -1), (vueltas.size, 1))

        return self._caracteristicas(tau, contexto_repetido) @ self.coeficientes
