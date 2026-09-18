"""
LA RED NEURONAL CON FISICA DENTRO (PINN)
========================================

QUE ES UNA RED NEURONAL, EN TERMINOS DE PROGRAMACION
----------------------------------------------------
Una funcion. Nada mas.

Una funcion normal la escribes tu:  def f(x): return 3*x + 2
Una red neuronal tiene la misma forma, pero con miles de constantes en vez de
un 3 y un 2, y esas constantes NO las escribes: las busca un algoritmo.

    d = red(tau, contexto ; W)        W = las 12 993 constantes

"Entrenar" es el problema de optimizacion de buscar la W que minimiza una
funcion de coste. Es descenso por gradiente: calcular en que direccion hay que
mover cada constante para que el coste baje, y dar un paso en esa direccion.
Ocho mil veces.

Las capas se alternan con `tanh`. Sin esa no linealidad, encadenar diez capas
lineales daria otra funcion lineal y la red no podria representar una curva.


QUE ANADE EL "PHYSICS-INFORMED"
-------------------------------
Una red normal se entrena asi:

    coste = (lo que predice - lo que se midio)^2

y necesita conocer la respuesta correcta en cada punto donde aprende.

Una PINN anade un segundo termino:

    coste = (prediccion - medida)^2  +  (residuo de la ecuacion)^2

donde el residuo es:

    r = dd/dtau - velocidad_desgaste(d, contexto)

Si la red cumpliera la ecuacion exactamente, r valdria cero en todas partes.
Asi que meter r^2 en el coste es, literalmente, pedirle a la red que obedezca
la fisica.

EL TRUCO ESTA EN dd/dtau. Esa derivada no se aproxima por diferencias finitas:
se calcula EXACTA, con la misma diferenciacion automatica que ya usa PyTorch
para entrenar. La diferencia es que aqui se deriva la salida respecto a la
ENTRADA en lugar de respecto a los pesos.

Y de ahi sale la propiedad que hace util todo esto:

    *** evaluar el residuo solo necesita un PUNTO del dominio.            ***
    *** NO necesita saber la respuesta correcta en ese punto.             ***

Por eso la fisica se puede exigir donde no hay ni un dato: en combinaciones de
condiciones que nunca se dieron, y en vueltas mas alla del final de todos los
stints del conjunto. Que son exactamente las vueltas que un estratega necesita
que alguien le prediga.


LOS TRES TERMINOS DEL COSTE
---------------------------
    coste_fisica   el residuo de la ecuacion, en 2 000 puntos sin etiqueta
    coste_datos    el ajuste a la perdida de ritmo medida, solo donde hay datos
    coste_ci       la condicion inicial d(0) = 0

El tercero siendo un termino del coste es una debilidad conocida, y se deja
asi a proposito porque es la formulacion de libro y porque verla fallar ensena:
se cumple de forma aproximada, nunca exacta, y compite por gradiente con los
otros dos. En `main` se sustituye por una transformacion de la salida que la
hace exacta y gratis. Ver ROADMAP.md, paso 4.


EL PROBLEMA INVERSO
-------------------
Seis constantes fisicas (kw, m, Ea, Eq, Ev, kappa) no se conocen, y se estiman
A LA VEZ que los pesos de la red. El optimizador las trata igual que a
cualquier otro parametro.

Las cinco que tienen que ser positivas se guardan como su LOGARITMO. Asi el
valor fisico es exp(crudo), que es positivo pase lo que pase. No es cautela
numerica: no existe un coeficiente de desgaste negativo, y afirmarlo a traves
de la parametrizacion es mas fuerte que confiar en que el optimizador lo
respete.

kappa es la excepcion y se deja con el signo libre, porque es la unica
constante del sistema cuyo signo NO esta fijado por la fisica: dice cuanto
resiste un compuesto mas duro, y hay analisis que sostienen que 2026 invirtio
esa jerarquia. Parametrizarla en logaritmo haria al modelo incapaz de
expresarlo. Asi, el signo lo decide el dato.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from physics import (
    HORIZONTE_VUELTAS,
    N_CONTEXTO,
    N_ENTRADAS_RED,
    PARAMETROS_LIBRES,
    RANGOS_CONTEXTO,
    VALORES_REALES,
    VUELTAS_REF,
    Contexto,
    ParametrosFisicos,
    velocidad_desgaste,
)

# Las cinco constantes que tienen que ser positivas se guardan como log(valor).
_POSITIVAS = ("kw", "m", "Ea", "Eq", "Ev")

# Valores de arranque, deliberadamente equivocados: el PINN tiene que recuperar
# los de VALORES_REALES partiendo de aqui. Si arrancara en la respuesta, la
# recuperacion no demostraria nada.
VALORES_INICIALES = {
    "kw": 0.30,      # el real es 0.55
    "m": 1.00,       # el real es 1.50
    "Ea": 0.50,      # el real es 0.95
    "Eq": 0.20,      # el real es 0.40
    "Ev": 0.20,      # el real es 0.35
    "kappa": 0.30,   # el real es 0.85
}


# ---------------------------------------------------------------------------
# 1) LA RED
# ---------------------------------------------------------------------------

class RedNeuronal(nn.Module):
    """Perceptron multicapa: 6 entradas -> varias capas ocultas -> 1 salida.

    Entradas (6):  tau, q_friccion, carga, velocidad, temp_pista, compuesto
    Salida   (1):  d, la fraccion de banda consumida

    Por defecto 4 capas de 64 neuronas = 12 993 constantes que ajustar.

    Por que un perceptron y no algo mas sofisticado: la funcion que hay que
    aprender es suave y tiene seis entradas. No hay estructura espacial que
    justifique convoluciones ni secuencia que justifique recurrencia. Lo que si
    importa es que la red sea DERIVABLE de forma barata y estable, porque en
    cada punto de colocacion hay que derivarla.

    Por que tanh y no ReLU: la derivada de una red ReLU es constante a trozos,
    asi que la red predeciria un dd/dtau escalonado y discontinuo, imposible de
    casar con un lado derecho suave. tanh es infinitamente derivable.
    """

    def __init__(self, neuronas: int = 64, capas: int = 4):
        super().__init__()

        # [6, 64, 64, 64, 64, 1]
        dimensiones = [N_ENTRADAS_RED] + [neuronas] * capas + [1]

        modulos: list[nn.Module] = []
        for i in range(len(dimensiones) - 1):
            modulos.append(nn.Linear(dimensiones[i], dimensiones[i + 1]))
            es_ultima = i == len(dimensiones) - 2
            if not es_ultima:
                modulos.append(nn.Tanh())

        self.capas = nn.Sequential(*modulos)
        self._inicializar_glorot()

    def _inicializar_glorot(self) -> None:
        """Inicializacion de Glorot (Xavier), la que le va bien a tanh.

        Mantiene la varianza de las senales al atravesar capas. Sin esto, con
        cuatro capas, tanh satura desde la primera iteracion: todas las salidas
        se pegan a -1 o +1, la derivada se hace cero y la red deja de aprender.
        """
        for modulo in self.capas:
            if isinstance(modulo, nn.Linear):
                nn.init.xavier_normal_(modulo.weight)
                nn.init.zeros_(modulo.bias)

    def forward(self, tau: torch.Tensor, contexto: torch.Tensor) -> torch.Tensor:
        return self.capas(torch.cat([tau, contexto], dim=1))


# ---------------------------------------------------------------------------
# 2) EL PINN
# ---------------------------------------------------------------------------

class PINN:
    """La red, mas las seis constantes fisicas que se estiman con ella."""

    def __init__(
        self,
        neuronas: int = 64,
        capas: int = 4,
        gamma1: float = VALORES_REALES.gamma1,
        tau_max: float = HORIZONTE_VUELTAS / VUELTAS_REF,
        semilla: int = 0,
    ):
        torch.manual_seed(semilla)

        self.red = RedNeuronal(neuronas=neuronas, capas=capas)
        self.gamma1 = gamma1          # fija: ancla la escala de d (ver physics.py)
        self.tau_max = tau_max        # hasta donde se exige la ecuacion
        self.historial: list[dict[str, float]] = []

        # Las seis constantes a estimar, en su forma "cruda" (sin restricciones).
        self._crudos: dict[str, nn.Parameter] = {}
        for nombre in PARAMETROS_LIBRES:
            inicial = VALORES_INICIALES[nombre]
            crudo = np.log(inicial) if nombre in _POSITIVAS else inicial
            self._crudos[nombre] = nn.Parameter(torch.tensor(float(crudo)))

    # -- las constantes fisicas -------------------------------------------

    def _valor(self, nombre: str) -> torch.Tensor:
        """Del valor crudo que optimiza la red al valor fisico."""
        crudo = self._crudos[nombre]
        return torch.exp(crudo) if nombre in _POSITIVAS else crudo

    def parametros_fisicos(self) -> ParametrosFisicos:
        """Las constantes como tensores, para que entren al grafo de autograd."""
        valores = {n: self._valor(n) for n in PARAMETROS_LIBRES}
        return ParametrosFisicos(gamma1=self.gamma1, **valores)

    def parametros_estimados(self) -> ParametrosFisicos:
        """Las mismas constantes como numeros normales, para imprimir."""
        valores = {n: float(self._valor(n).detach()) for n in PARAMETROS_LIBRES}
        return ParametrosFisicos(gamma1=self.gamma1, **valores)

    def n_parametros_red(self) -> int:
        return sum(p.numel() for p in self.red.parameters())

    # -- el residuo, que es el corazon del metodo --------------------------

    def residuo(self, tau: torch.Tensor, contexto: torch.Tensor) -> torch.Tensor:
        """r = dd/dtau - velocidad_desgaste(d, contexto).

        Vale cero en todas partes si y solo si la red cumple la ecuacion.
        """
        # 1) marcar tau como algo respecto a lo que queremos derivar
        tau = tau.requires_grad_(True)

        # 2) pasar por la red
        d = self.red(tau, contexto)

        # 3) derivar la SALIDA respecto a la ENTRADA, de forma exacta
        #
        #    create_graph=True es imprescindible: permite que el residuo se
        #    vuelva a derivar despues respecto a los pesos, que es lo que
        #    necesita el optimizador. Sin el, el termino de fisica no aportaria
        #    ningun gradiente y la ecuacion no se cumpliria nunca.
        dd_dtau = torch.autograd.grad(
            outputs=d,
            inputs=tau,
            grad_outputs=torch.ones_like(d),
            create_graph=True,
        )[0]

        # 4) comparar contra lo que dice la fisica
        q, carga, velocidad, temp_pista, compuesto = self._columnas(contexto)
        ritmo_fisico = velocidad_desgaste(
            d, q, carga, velocidad, temp_pista, compuesto, self.parametros_fisicos()
        )
        return dd_dtau - ritmo_fisico

    @staticmethod
    def _columnas(contexto: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Separa la matriz de contexto en sus 5 columnas con nombre."""
        return tuple(contexto[:, i : i + 1] for i in range(N_CONTEXTO))

    def puntos_colocacion(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Los puntos donde se exige que se cumpla la ecuacion.

        Se sortean uniformemente:
          - tau hasta el horizonte COMPLETO de 45 vueltas, mas alla del stint
            mas largo del conjunto;
          - cada variable de contexto dentro de su rango fisico.

        No llevan etiqueta ninguna, y esa es la razon entera de que este termino
        pueda cubrir terreno que los datos no cubren.
        """
        tau = torch.rand(n, 1) * self.tau_max

        columnas = []
        for nombre in RANGOS_CONTEXTO:
            bajo, alto = RANGOS_CONTEXTO[nombre]
            columnas.append(torch.rand(n, 1) * (alto - bajo) + bajo)

        return tau, torch.cat(columnas, dim=1)

    # -- entrenamiento ------------------------------------------------------

    def entrenar(
        self,
        entradas: np.ndarray,
        delta: np.ndarray,
        iteraciones: int = 8000,
        n_colocacion: int = 2000,
        lr: float = 3e-3,
        peso_fisica: float = 1.0,
        peso_datos: float = 10.0,
        peso_ci: float = 10.0,
        registrar_cada: int = 25,
        imprimir_cada: int = 500,
    ) -> None:
        """Descenso por gradiente con Adam.

        Los pesos de los tres terminos no son importancias: son escalas. El
        residuo de la ecuacion es adimensional y pequeno; el de datos esta en
        segundos. Sin el factor, el termino que ancla el modelo a lo medido de
        verdad quedaria por debajo del ruido numerico del otro.
        """
        # Separar la matriz de entrada en tau y contexto ANTES de convertirla a
        # tensores, para que tau sea una hoja del grafo y se pueda derivar.
        tau_obs = torch.tensor(entradas[:, 0:1], dtype=torch.float32)
        contexto_obs = torch.tensor(entradas[:, 1:], dtype=torch.float32)
        delta_obs = torch.tensor(delta.reshape(-1, 1), dtype=torch.float32)

        # El optimizador mueve a la vez los pesos de la red y las 6 constantes.
        a_optimizar = list(self.red.parameters()) + list(self._crudos.values())
        optimizador = torch.optim.Adam(a_optimizar, lr=lr)

        for iteracion in range(1, iteraciones + 1):
            optimizador.zero_grad()

            # --- 1) FISICA: la ecuacion, en puntos donde no hay datos ---
            tau_col, contexto_col = self.puntos_colocacion(n_colocacion)
            coste_fisica = (self.residuo(tau_col, contexto_col) ** 2).mean()

            # --- 2) DATOS: lo unico que se ha medido de verdad ---
            d_predicho = self.red(tau_obs, contexto_obs)
            delta_predicho = self.gamma1 * d_predicho
            coste_datos = ((delta_predicho - delta_obs) ** 2).mean()

            # --- 3) CONDICION INICIAL: un neumatico nuevo no ha gastado nada ---
            _, contexto_ci = self.puntos_colocacion(64)
            tau_cero = torch.zeros(contexto_ci.shape[0], 1)
            coste_ci = (self.red(tau_cero, contexto_ci) ** 2).mean()

            coste = (
                peso_fisica * coste_fisica
                + peso_datos * coste_datos
                + peso_ci * coste_ci
            )

            coste.backward()
            optimizador.step()

            if iteracion % registrar_cada == 0 or iteracion == 1:
                self._registrar(iteracion, coste, coste_fisica, coste_datos, coste_ci)
                if iteracion % imprimir_cada == 0 or iteracion == 1:
                    print("  " + self._linea_de_log(self.historial[-1]))

    def _registrar(self, iteracion, coste, fisica, datos, ci) -> None:
        """Guarda el estado de esta iteracion para poder dibujarlo despues.

        Se usa .item() y no float(): los tensores siguen enganchados al grafo de
        autograd aqui, y float() sobre uno de ellos avisa con un warning.
        """
        registro = {
            "iteracion": iteracion,
            "total": coste.item(),
            "fisica": fisica.item(),
            "datos": datos.item(),
            "ci": ci.item(),
        }
        estimados = self.parametros_estimados()
        for nombre in PARAMETROS_LIBRES:
            registro[nombre] = float(getattr(estimados, nombre))
        self.historial.append(registro)

    @staticmethod
    def _linea_de_log(r: dict[str, float]) -> str:
        return (
            f"{r['iteracion']:6d}  total {r['total']:9.5f}  "
            f"fisica {r['fisica']:8.5f}  datos {r['datos']:8.5f}  "
            f"ci {r['ci']:8.5f}  kw {r['kw']:5.3f}  kappa {r['kappa']:5.3f}"
        )

    # -- prediccion ---------------------------------------------------------

    def predecir_stint(self, contexto: Contexto, vueltas: np.ndarray) -> np.ndarray:
        """Perdida de ritmo predicha, vuelta a vuelta.

        Misma firma que la del baseline, para que el evaluador mida a los dos
        modelos con exactamente la misma vara.
        """
        vueltas = np.asarray(vueltas, dtype=float).ravel()
        tau = torch.tensor((vueltas / VUELTAS_REF).reshape(-1, 1), dtype=torch.float32)

        contexto_repetido = np.tile(contexto.vector().reshape(1, -1), (vueltas.size, 1))
        contexto_t = torch.tensor(contexto_repetido, dtype=torch.float32)

        with torch.no_grad():
            d = self.red(tau, contexto_t)

        return (self.gamma1 * d).numpy().ravel()

    def predecir_desgaste(self, contexto: Contexto, vueltas: np.ndarray) -> np.ndarray:
        """El estado latente d. Nunca se midio; sale de imponer la ecuacion."""
        return self.predecir_stint(contexto, vueltas) / self.gamma1
