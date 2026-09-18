"""
MODELO FISICO DEL DESGASTE DE UN NEUMATICO
==========================================

COMO LEER ESTE FICHERO
----------------------
Todo lo que hay aqui gira alrededor de UNA ecuacion diferencial.

Una ecuacion diferencial no dice cuanto vale algo. Dice a que VELOCIDAD cambia.
Es la diferencia entre "el deposito tiene 40 litros" y "el deposito pierde 2
litros por minuto". Con la segunda frase, mas el nivel de partida, puedes
reconstruir el nivel en cualquier instante futuro: eso es integrar.

Aqui lo que cambia es el desgaste del neumatico, y la ecuacion dice a que
velocidad se gasta segun las condiciones en las que esta rodando.


LAS VARIABLES, UNA POR UNA
--------------------------
Hay tres grupos y conviene no mezclarlos nunca:

1) EL TIEMPO
   tau .............. tiempo adimensional = vuelta_del_stint / 30
                      Un stint normal de 30 vueltas va de tau=0 a tau=1.
                      Se divide por 30 para que los numeros que entran a la red
                      esten cerca de 1. Una red entrenada con entradas de
                      escalas muy distintas converge mal.

2) EL ESTADO (lo que evoluciona)
   d ................ fraccion de banda de rodadura consumida.
                      0 = neumatico nuevo, 1 = neumatico agotado.
                      *** ES LATENTE: no se mide NUNCA, ni en el banco ni en
                      carrera. El modelo lo reconstruye. ***

3) EL CONTEXTO (5 variables, constantes dentro de un stint)
   q_friccion ....... energia de friccion por vuelta, normalizada (~1)
                      Cuanta energia mete el piso contra la goma.
   carga ............ carga mecanica media en g, normalizada (~1)
                      Cuanto peso aparente soporta el neumatico en curva.
   velocidad ........ velocidad media, normalizada (~1)
                      Mas velocidad = mas aire = mas refrigeracion.
   temp_pista ....... temperatura de pista, normalizada de 0 a 1
   compuesto ........ dureza: 0 = blando, 0.5 = medio, 1 = duro

Y una cosa mas, la unica que de verdad se mide:

   delta ............ perdida de ritmo en segundos respecto a la mejor vuelta
                      del stint. *** ES EL UNICO OBSERVABLE. ***


LA ECUACION
-----------
    dd/dtau = k(contexto) * (1 - d)

    con   k(contexto) = kw * (carga/carga_ref)^m
                           * exp( Ea*(temp_pista - temp_ref)
                                + Eq*(q_friccion - q_ref)
                                - Ev*(velocidad  - vel_ref)
                                - kappa*(compuesto - comp_ref) )

Leida en castellano: "el neumatico se gasta a una velocidad que depende de las
condiciones, y que se frena a medida que queda menos goma".

Cada pieza del exponente es una afirmacion fisica, y cada signo esta elegido:

    + Ea*temp_pista .... mas calor, mas desgaste (activacion termica)
    + Eq*q_friccion .... mas energia de friccion, mas desgaste
    - Ev*velocidad ..... mas velocidad, mas refrigeracion, MENOS desgaste
    - kappa*compuesto .. mas duro, MENOS desgaste
    carga^m ............ ley de Archard: el desgaste crece con la carga


POR QUE SE RESTA UNA REFERENCIA EN CADA TERMINO
-----------------------------------------------
Fijate en que cada variable aparece como (variable - su_referencia). Eso hace
que en condiciones de referencia el exponente valga CERO y por tanto k = kw.
Es decir: kw pasa a significar literalmente "la velocidad de desgaste en
condiciones normales", que es una frase que se puede discutir con un ingeniero.

Pero no es solo cosmetica. Sin restar la referencia, kw y los demas
coeficientes se pisan: subir kw y bajar Ev a la vez deja el desgaste medio
igual, asi que hay infinitas combinaciones casi equivalentes y el optimizador
no sabe cual elegir. Se midio en este mismo proyecto: sin centrar, kw salia con
un 10 % de error y Ev con un 33 %, PERO la combinacion log(kw) - Ev se recuperaba
con un error de 0.0098. O sea, el modelo sabia perfectamente cuanto se gasta el
neumatico; lo que no sabia era a cual de las dos constantes atribuirlo.

Centrar los regresores es el remedio clasico de ese problema en regresion, y
aqui hace exactamente lo mismo.

El factor (1 - d) es una saturacion. Acota d entre 0 y 1 por construccion,
porque no puedes consumir mas goma de la que hay, y mantiene la velocidad de
desgaste positiva. Es decir: el neumatico solo puede ir a peor, y eso sale de la
ECUACION, no de una restriccion pegada por fuera.


LO QUE ESTE MODELO NO TIENE
---------------------------
No tiene temperatura del neumatico como estado propio. La temperatura entra de
forma aproximada, a traves de temp_pista, q_friccion y velocidad, pero no
evoluciona por si sola.

Eso tiene una consecuencia importante: no hay realimentacion entre el desgaste
y el calor, y por tanto NO HAY CLIFF. La curva de ritmo solo puede doblarse
hacia un lado. Meter esa realimentacion es el salto a la rama `main`, donde hay
una segunda ecuacion diferencial para la temperatura. Ver ROADMAP.md.


LA PROPIEDAD QUE HACE UTIL ESTE MODELO
--------------------------------------
Como el contexto es constante dentro de un stint, k(contexto) es una CONSTANTE,
y entonces la ecuacion tiene SOLUCION EXACTA escrita a mano:

    d(tau) = 1 - exp(-k * tau)

Eso es justo lo que se necesita para empezar: se puede comprobar que la red
acierta comparandola con la respuesta exacta, antes de apuntar el metodo a un
sistema donde no existe ninguna respuesta exacta contra la que comparar.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

try:
    # torch solo hace falta cuando el PINN evalua el residuo durante el
    # entrenamiento. Para generar datos o integrar, numpy basta.
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------------
# 1) CONSTANTES DE ESCALA
# ---------------------------------------------------------------------------

# Cuantas vueltas son una unidad de tau. Un stint de 30 vueltas -> tau de 0 a 1.
VUELTAS_REF = 30.0

# Hasta donde miramos al extrapolar: el horizonte de una decision de estrategia.
HORIZONTE_VUELTAS = 45

# Traduccion del nombre del compuesto a un numero entre 0 (blando) y 1 (duro).
INDICE_COMPUESTO = {
    "SOFT": 0.0,
    "MEDIUM": 0.5,
    "HARD": 1.0,
    # Los que pueden aparecer en datos reales y no son secos:
    "INTERMEDIATE": 0.75,
    "WET": 1.0,
}


# ---------------------------------------------------------------------------
# 2) EL CONTEXTO DE UN STINT
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Contexto:
    """Las 5 condiciones en las que rueda un stint.

    Son constantes dentro del stint: un juego de neumaticos corre siempre el
    mismo circuito, con la misma temperatura aproximada y el mismo compuesto.

    Se usa una clase con nombres en vez de una lista de 5 numeros porque
    `contexto.temp_pista` se entiende al leerlo y `contexto[3]` no.
    """

    q_friccion: float = 1.0
    carga: float = 1.0
    velocidad: float = 1.0
    temp_pista: float = 0.5
    compuesto: float = 0.0

    def vector(self) -> np.ndarray:
        """Los 5 numeros en el orden canonico, para dárselos a la red."""
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=float)

    @classmethod
    def desde_vector(cls, v) -> Contexto:
        """Operacion inversa de `vector()`."""
        return cls(*(float(x) for x in np.asarray(v, dtype=float).ravel()))

    @property
    def nombre_compuesto(self) -> str:
        """El compuesto mas cercano, para etiquetar graficas."""
        return min(INDICE_COMPUESTO, key=lambda k: abs(INDICE_COMPUESTO[k] - self.compuesto))


# Nombres en el orden canonico. TODO el proyecto asume este orden: la red, el
# baseline, el generador y el lector de CSV. Es el contrato del proyecto.
CONTEXTO_NOMBRES = tuple(f.name for f in fields(Contexto))
N_CONTEXTO = len(CONTEXTO_NOMBRES)          # 5
N_ENTRADAS_RED = 1 + N_CONTEXTO             # 6 = tau + contexto

# Rango fisicamente razonable de cada variable de contexto.
# Sirve para dos cosas: generar stints sinteticos, y decidir DONDE se le exige
# a la red que cumpla la ecuacion (ver pinn.py, puntos de colocacion).
RANGOS_CONTEXTO = {
    "q_friccion": (0.40, 1.60),
    "carga": (0.50, 1.50),
    "velocidad": (0.60, 1.40),
    "temp_pista": (0.00, 1.00),
    "compuesto": (0.00, 1.00),
}

# Condiciones de referencia: el centro de cada rango. En este punto el
# exponente de la ecuacion vale cero, asi que kw es exactamente "la velocidad
# de desgaste en condiciones normales". Ver la explicacion de arriba sobre por
# que centrar importa.
REFERENCIA = Contexto(
    q_friccion=1.0,
    carga=1.0,
    velocidad=1.0,
    temp_pista=0.5,
    compuesto=0.5,
)


# ---------------------------------------------------------------------------
# 3) LAS CONSTANTES FISICAS DEL MODELO
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParametrosFisicos:
    """Las 7 constantes de la ecuacion.

    Seis de ellas las ESTIMA el PINN mientras entrena (problema inverso). La
    septima, gamma1, se mantiene fija, y hay un motivo importante para ello que
    se explica justo debajo de la clase.
    """

    kw: float = 0.55        # velocidad base de desgaste
    m: float = 1.50         # exponente de carga (ley de Archard)
    Ea: float = 0.95        # cuanto acelera el desgaste la temperatura de pista
    Eq: float = 0.40        # cuanto lo acelera la energia de friccion
    Ev: float = 0.35        # cuanto lo frena la refrigeracion por velocidad
    kappa: float = 0.85     # cuanto resiste al desgaste un compuesto mas duro
    gamma1: float = 1.35    # segundos perdidos por unidad de desgaste [s]


# Los valores "verdaderos" que usa el generador sintetico. El PINN arranca de
# valores deliberadamente equivocados y tiene que RECUPERAR estos a partir de
# tiempos por vuelta con ruido. Que lo consiga es la prueba de que el metodo
# funciona, y es la unica comprobacion que con datos reales es imposible hacer,
# porque alli no existe ninguna verdad de referencia.
VALORES_REALES = ParametrosFisicos()

# Las seis que el PINN estima. gamma1 NO esta en la lista, a proposito.
#
# Por que gamma1 se queda fija
# ----------------------------
# Lo unico que se mide es delta = gamma1 * d. Si se dejan libres a la vez d y
# gamma1, hay infinitas combinaciones que dan exactamente el mismo delta:
# multiplicar d por 2 y dividir gamma1 por 2 no cambia nada de lo observable.
# El optimizador no tiene forma de preferir una y se desliza por esa direccion
# hasta desbordar. Fijar gamma1 ancla la escala de d y cierra el problema.
#
# En la rama `main` esto mismo aparece a lo grande y fue el error mas grave del
# proyecto: el entrenamiento divergio hasta un RMSE de miles de millones de
# segundos mientras la perdida de entrenamiento se veia baja.
PARAMETROS_LIBRES = ("kw", "m", "Ea", "Eq", "Ev", "kappa")


# ---------------------------------------------------------------------------
# 4) LA ECUACION
# ---------------------------------------------------------------------------

def _es_tensor(x) -> bool:
    """True si x es un tensor de torch (y no un array de numpy)."""
    return torch is not None and torch.is_tensor(x)


def constante_desgaste(q_friccion, carga, velocidad, temp_pista, compuesto, p):
    """k(contexto): todo lo de la ecuacion que NO depende de d.

    Se separa en su propia funcion porque aparece en dos sitios que tienen que
    coincidir exactamente: la velocidad de desgaste y la solucion exacta.

    Funciona igual con numeros de numpy y con tensores de torch. Esa dualidad
    es deliberada: asi la MISMA funcion define la verdad de referencia (cuando
    se generan datos) y el residuo que la red minimiza (cuando se entrena). Si
    hubiera dos copias, tarde o temprano una se corregiria y la otra no.
    """
    if _es_tensor(q_friccion):
        exp, potencia = torch.exp, torch.pow
        carga_relativa = torch.clamp(carga / REFERENCIA.carga, min=1e-6)
    else:
        exp, potencia = np.exp, np.power
        carga_relativa = np.maximum(carga / REFERENCIA.carga, 1e-6)

    # La carga nunca es negativa fisicamente, pero durante el entrenamiento la
    # red puede recibir puntos raros y `potencia` con base negativa da NaN.

    # Cada variable entra como "cuanto se desvia de lo normal". Asi kw queda
    # con significado propio y no se pisa con los demas coeficientes.
    exponente = (
        p.Ea * (temp_pista - REFERENCIA.temp_pista)
        + p.Eq * (q_friccion - REFERENCIA.q_friccion)
        - p.Ev * (velocidad - REFERENCIA.velocidad)
        - p.kappa * (compuesto - REFERENCIA.compuesto)
    )
    return p.kw * potencia(carga_relativa, p.m) * exp(exponente)


def velocidad_desgaste(d, q_friccion, carga, velocidad, temp_pista, compuesto, p):
    """El lado derecho de la ecuacion: dd/dtau.

    Es no negativa mientras d <= 1, asi que el desgaste monotono (el neumatico
    solo va a peor) y la cota d <= 1 salen las dos de la propia ecuacion.
    """
    k = constante_desgaste(q_friccion, carga, velocidad, temp_pista, compuesto, p)
    return k * (1.0 - d)


def perdida_ritmo(d, p: ParametrosFisicos):
    """Convierte el estado latente d en lo unico observable: segundos.

    Esto es el "operador de observacion". La red predice d, que nadie ha medido
    nunca; esta funcion lo traduce a algo que si se puede comparar con los
    datos. Por eso d puede reconstruirse sin haber aparecido jamas en la
    funcion de coste.
    """
    return p.gamma1 * d


# ---------------------------------------------------------------------------
# 5) RESOLVER LA ECUACION (dos formas)
# ---------------------------------------------------------------------------

def solucion_exacta(tau, contexto: Contexto, p: ParametrosFisicos) -> np.ndarray:
    """La solucion escrita a mano, sin aproximar nada.

    Como el contexto es constante dentro del stint, k tambien lo es, y separar
    variables en dd/dtau = k*(1-d) con d(0)=0 da directamente:

        d(tau) = 1 - exp(-k * tau)

    Esta es la referencia contra la que se comprueba todo lo demas. Tener una
    respuesta exacta es la razon de que este modelo sea el punto de partida
    correcto: se valida el metodo donde se PUEDE validar.
    """
    k = constante_desgaste(
        contexto.q_friccion, contexto.carga, contexto.velocidad,
        contexto.temp_pista, contexto.compuesto, p,
    )
    return 1.0 - np.exp(-k * np.asarray(tau, dtype=float))


def integrar_stint(
    n_vueltas: int,
    contexto: Contexto,
    p: ParametrosFisicos,
    pasos_por_vuelta: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolver la ecuacion paso a paso, con Runge-Kutta de orden 4.

    Aqui esto es redundante: arriba esta la solucion exacta. Y ese es justo el
    motivo de tenerlo. Se puede comprobar que el integrador coincide con la
    respuesta exacta AHORA, que es barato comprobarlo. En `main` hay dos
    ecuaciones acopladas, la solucion exacta desaparece, y lo unico que queda
    es este integrador: mas vale que este validado.

    Runge-Kutta 4 en vez de Euler porque Euler acumula un sesgo sistematico, y
    el problema inverso interpretaria ese sesgo como si fuera fisica.

    Devuelve (vueltas, d) con las vueltas numeradas 1..n_vueltas.
    """
    dt = 1.0 / (VUELTAS_REF * pasos_por_vuelta)

    def ritmo(d_actual: float) -> float:
        return velocidad_desgaste(
            d_actual, contexto.q_friccion, contexto.carga, contexto.velocidad,
            contexto.temp_pista, contexto.compuesto, p,
        )

    d = 0.0            # neumatico nuevo
    historial = []

    for _ in range(n_vueltas):
        for _ in range(pasos_por_vuelta):
            # Runge-Kutta 4: en vez de fiarse de la pendiente en un solo punto,
            # promedia cuatro pendientes tomadas a lo largo del paso.
            k1 = ritmo(d)
            k2 = ritmo(d + 0.5 * dt * k1)
            k3 = ritmo(d + 0.5 * dt * k2)
            k4 = ritmo(d + dt * k3)
            d = d + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        historial.append(d)

    vueltas = np.arange(1, n_vueltas + 1, dtype=float)
    return vueltas, np.asarray(historial)
