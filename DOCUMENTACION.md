# F1 Tire PINN — Guía completa del proyecto

**Autor:** Esteban Valencia
**Repositorio:** https://github.com/SoullessSapo/F1-CNN-IA
**Alcance:** la red neuronal y el uso de DeepXDE. La capa cloud de AWS **no** está incluida.

> *El código y el README están en inglés; este documento está en español a propósito.*

---

## Cómo leer este documento

Está escrito para alguien que sabe programar pero no necesariamente viene de
modelado matemático ni de machine learning. Por eso las partes 1 a 5 construyen
los conceptos desde cero, con analogías de ingeniería de software, antes de
entrar a las decisiones concretas.

| Parte | Qué explica |
|---|---|
| **1** | Aclaración importante: esto **no** es un LLM |
| **2** | El problema, en términos de ingeniería |
| **3** | Qué es realmente la red neuronal aquí, y qué significa "entrenar" |
| **4** | Qué es una ecuación diferencial y por qué modelar así |
| **5** | Qué significa "physics-informed" — la idea central |
| **6** | De dónde salieron las ecuaciones (la física real) |
| **7** | La arquitectura concreta y qué hace DeepXDE por nosotros |
| **8** | El problema inverso: estimar constantes físicas desde datos |
| **9** | Los datos: de dónde salen y qué hubo que inventar |
| **10** | Los cuatro problemas serios que aparecieron |
| **11** | Resultados |
| **12** | Limitaciones y siguientes pasos |
| **13** | Verificación contra datos publicados independientes |

---

# 1. Primero, una aclaración: esto no es un LLM

Conviene despejarlo porque cambia toda la intuición.

Un **LLM** (GPT, Claude) es un transformer con miles de millones de parámetros,
entrenado sobre terabytes de texto, que predice el siguiente token. Cuesta
millones de dólares entrenarlo y necesita centros de datos llenos de GPUs.

Lo que hay aquí es una **red densa pequeña**: 6 entradas → 4 capas de 64
neuronas → 2 salidas. Son unos **13 000 parámetros**, cuatro órdenes de magnitud
menos que el modelo más pequeño que llamarías LLM. Entrena en 30 minutos en la
CPU de un portátil, sin GPU.

Tampoco es una **CNN** (red convolucional). Las CNN sirven para datos con
estructura espacial —imágenes, señales— donde tiene sentido deslizar un filtro.
Aquí las entradas son seis números sueltos sin ninguna relación espacial entre
sí, así que una convolución no aportaría nada.

La categoría correcta es **PINN**: *Physics-Informed Neural Network*. Lo
distintivo no es la arquitectura de la red, que es de lo más simple que existe.
Lo distintivo es **cómo se entrena**, y eso es lo que explica la parte 5.

---

# 2. El problema, en términos de ingeniería

Un neumático de F1 se degrada durante un stint. Al principio la pérdida de ritmo
es gradual —unas décimas por vuelta— y en algún momento se cae por un
precipicio: el **cliff**. El agarre colapsa y el coche pierde varios segundos por
vuelta de golpe.

Predecir la vuelta exacta del cliff es **la** decisión estratégica de una
carrera. Equivocarse por una vuelta cuesta posiciones que no se recuperan.

Formulado como problema de ingeniería:

- **Entrada:** telemetría de a bordo y cronometraje, en vivo.
- **Salida:** ¿en qué vuelta llega el cliff con este juego, en estas condiciones?
- **Restricción dura:** la respuesta tiene que llegar en menos de 500 ms.
- **Restricción incómoda:** lo que quieres predecir (el desgaste) **no se puede
  medir**. Solo ves tiempos por vuelta.

Ese último punto es el que hace el problema interesante y el que motiva todo lo
demás.

## 2.1 Por qué no bastan los enfoques obvios

El PDF del proyecto ya identificaba tres familias y sus límites:

**Modelos empíricos lineales** (lo que usan muchos equipos): ajustas una recta de
"segundos perdidos por vuelta" por compuesto. Es rápido e interpretable, pero una
recta **no puede representar un precipicio** — es la definición de ser lineal.

**Deep learning puro** (LSTM, GRU): tiene capacidad de sobra para capturar la no
linealidad. El problema es que **nada la obliga a respetar la física**. Puede
predecir tranquilamente que un neumático usado recupera agarre. Y lo hace: en mis
mediciones, un 11.5 % de las vueltas al extrapolar.

**Simulación por elementos finitos**: físicamente exacta pero tarda horas. Sirve
para diseñar el neumático, no para decidir en carrera.

La PINN intenta quedarse con lo bueno de las dos últimas: la flexibilidad de una
red neuronal y las garantías de la física.

---

# 3. Qué es la red neuronal aquí, en términos de programación

## 3.1 Una red es una función parametrizada

Olvida por un momento la analogía con neuronas. Para lo que nos ocupa, una red
neuronal es simplemente **una función matemática con muchos parámetros
ajustables**:

```python
def red(entrada, parametros):    # parametros = ~13.000 números
    ...
    return salida
```

La estructura interna aquí es la más simple posible: multiplicar por una matriz,
aplicar una función no lineal (`tanh`), repetir cuatro veces. La no linealidad es
lo que le permite representar curvas y no solo rectas.

Lo único que importa conceptualmente es esto: **cambiando esos 13 000 parámetros,
esa función puede tomar casi cualquier forma**. Es un molde universal.

## 3.2 "Entrenar" es un problema de optimización

Entrenar no tiene nada de mágico. Es esto:

1. Defines una función de coste que mide **cuán mal** lo está haciendo la red.
2. Calculas la derivada del coste respecto a cada uno de los 13 000 parámetros.
3. Mueves cada parámetro un poquito en la dirección que reduce el coste.
4. Repites 18 000 veces.

Es descenso de gradiente. Lo mismo que harías para minimizar cualquier función,
solo que en un espacio de 13 000 dimensiones.

**La analogía que probablemente te sirva más:** entrenar es como un *linter con
autofix*. Defines un conjunto de reglas (la función de coste), y el optimizador
va reescribiendo los parámetros hasta que las cumple lo mejor posible. **Todo el
diseño del proyecto consiste en elegir bien las reglas.**

## 3.3 Diferenciación automática: la pieza que lo hace posible

Para el paso 2 hay que derivar el coste respecto a 13 000 parámetros. A mano es
inviable.

PyTorch lo hace solo, con una técnica llamada **diferenciación automática**: cada
operación que ejecutas queda registrada en un grafo, y al final se recorre hacia
atrás aplicando la regla de la cadena. El resultado es **exacto** — no es una
aproximación numérica del tipo `(f(x+h) − f(x)) / h`.

Esto es central para el proyecto, y no solo por los parámetros. También necesito
derivar **la salida de la red respecto a su propia entrada** (`dθ/dτ`, la
velocidad a la que cambia la temperatura). Sin diferenciación automática exacta,
las PINN sencillamente no funcionarían.

---

# 4. Qué es una ecuación diferencial y por qué modelar con ellas

## 4.1 La idea

Una ecuación diferencial no describe **cuánto vale** algo, sino **a qué velocidad
cambia**. En vez de decir "la temperatura en la vuelta 12 es 95 °C", dice "la
temperatura sube a razón de tanto por vuelta, y ese ritmo depende de la
temperatura actual".

En pseudocódigo, una ecuación diferencial es esencialmente el cuerpo de un bucle
de simulación:

```python
estado = estado_inicial
for paso in range(n):
    ritmo_de_cambio = f(estado, entradas)   # <-- esto es la ecuación diferencial
    estado = estado + ritmo_de_cambio * dt
```

## 4.2 Por qué esto encaja tan bien aquí

Porque la degradación es **acumulativa e histórica**. El desgaste de la vuelta 20
depende de todo lo que pasó en las 19 anteriores. Una ecuación diferencial
expresa exactamente eso: el estado presente es la integral de todo el historial.

Y hay una ventaja adicional que resultó decisiva. Puedo escribir reglas sobre
`dd/dτ` (la *velocidad* de desgaste) que garantizan propiedades sobre `d` (el
desgaste acumulado). Concretamente: **si la velocidad de desgaste nunca es
negativa, el desgaste nunca puede disminuir**. Ésa es la garantía física que la
LSTM no puede dar, y se obtiene gratis por la estructura del modelo.

## 4.3 El sistema completo

El modelo son dos ecuaciones acopladas —"acopladas" significa que cada una
depende de la otra, así que hay que resolverlas a la vez:

```
(E1)  dθ/dτ = A_gen · q · (1 + ζ·d)  −  (h₀ + h₁·v) · θ
(E2)  dd/dτ = k_w · λ^m · exp(E_a·(θ + T_trk) − κ·c) · (1 − d)
```

Con dos variables de estado:

- **`θ`** — cuánto más caliente está la superficie del neumático que el asfalto.
- **`d`** — qué fracción de la banda de rodadura se ha consumido (0 = nuevo,
  1 = agotado).

**Ninguna de las dos se puede observar.** Son datos propietarios de cada equipo.
Ahí está el nudo del problema, y la parte 5 explica cómo se resuelve.

---

# 5. Qué significa "physics-informed" — la idea central

Ésta es la parte que de verdad importa entender.

## 5.1 Cómo se entrena una red normal

Le das pares (entrada, respuesta correcta) y la castigas por cada error:

```
coste = Σ (predicción − valor_real)²
```

Y ya. **La red no sabe absolutamente nada más del mundo.** Fuera del rango donde
vio datos, hace lo que le dé la gana. No es que se equivoque: es que no tiene
ninguna referencia.

## 5.2 Cómo se entrena una PINN

Añades al coste un término que mide **cuánto incumple las leyes de la física**.

Toma la ecuación (E1) y pásalo todo a un lado:

```
residuo = dθ/dτ − [ A_gen·q·(1 + ζ·d) − (h₀ + h₁·v)·θ ]
```

Si la red respeta la física, ese residuo vale **cero**. Si no, mide exactamente
cuánto se está saltando la termodinámica. Así que lo metes en el coste:

```
coste = Σ (predicción − valor_real)²     ← ajustarse a los datos
      + Σ (residuo_E1)²                  ← respetar el balance térmico
      + Σ (residuo_E2)²                  ← respetar la ley de desgaste
```

**En términos de programación:** el primer término son tus *tests*, que
comprueban casos concretos que conoces. Los otros dos son **invariantes** o
*asserts*, que tienen que cumplirse siempre, en cualquier entrada, hayas escrito
un test para ella o no.

## 5.3 Y aquí viene lo importante

Fíjate en *dónde* se evalúa cada término.

El término de datos **solo puede evaluarse donde hay datos**: las vueltas que
realmente se corrieron.

Los términos de física **se pueden evaluar en cualquier punto**, porque una
ecuación no necesita mediciones para decirte si se cumple. Así que los evalúo en
**miles de puntos aleatorios** repartidos por todo el espacio de condiciones
posibles: cualquier combinación de compuesto, temperatura, carga, y cualquier
vuelta hasta la 45 — incluidas condiciones que nadie ha corrido nunca.

Esos puntos se llaman **puntos de colocación**. En este proyecto son unos 8 000
por iteración.

**Ésa es toda la ventaja de una PINN.** En la región donde no hay datos, la LSTM
no tiene nada que la restrinja y por eso predice que el neumático recupera
agarre. La PINN sigue teniendo la ecuación diferencial encima, así que sigue
comportándose como un neumático.

Y hay un segundo efecto, casi más sorprendente: **la red aprende a reconstruir
`d`, el desgaste, sin haberlo visto nunca**. Solo ve tiempos por vuelta. Pero
como se le exige cumplir una ecuación que relaciona `d` con `θ` y con lo
observable, la única forma de satisfacer todas las reglas a la vez es que `d`
tome el valor físicamente correcto. En la figura `03_latent_states.png` se ve
que la reconstrucción es casi exacta.

## 5.4 El precio

No sale gratis. Estás obligando a la red a cumplir reglas, así que si tus reglas
están mal, el modelo será peor que uno sin restricciones. **Una PINN es tan buena
como la física que le metas.**

Eso se nota en los resultados con datos reales: con solo 36 stints y ruido alto,
la PINN paga el coste de estar restringida sin poder cobrar todavía el beneficio.

---

# 6. De dónde salieron las ecuaciones

No las inventé. Cada término viene de física conocida. Ésta es la justificación
de cada uno, y las iteraciones que hicieron falta.

## 6.1 (E1), el balance térmico

```
dθ/dτ = generación − enfriamiento
```

**Generación:** al frenar y girar, el caucho roza contra el asfalto y esa
fricción se convierte en calor. Cuanta más energía friccional (`q`), más calor.

**Enfriamiento:** el neumático cede calor al aire y al asfalto. La física de
transferencia de calor dice que el ritmo de enfriamiento es **proporcional a la
diferencia de temperatura**: cuanto más caliente está respecto al entorno, más
rápido se enfría. De ahí el término `−h·θ`.

El coeficiente de enfriamiento tiene dos partes: `h₀` es el enfriamiento base, y
`h₁·v` es la **convección forzada** — a más velocidad, más aire pasando, más
enfriamiento. Por eso los neumáticos se enfrían en las rectas.

Esto se llama modelo de **capacitancia concentrada**: tratas el neumático como un
único bloque a temperatura uniforme, en vez de resolver la distribución de
temperatura dentro del caucho. Es una simplificación enorme, y es la que hace que
esto corra en milisegundos en vez de horas.

## 6.2 (E2), la ley de desgaste

Combina dos leyes clásicas.

**Ley de Archard** (desgaste por abrasión, 1953): el material que se pierde es
proporcional a la carga aplicada. De ahí el término `k_w · λ^m`, donde `λ` es la
carga mecánica. El exponente `m` no lo fijo: **lo aprende la red**.

**Ecuación de Arrhenius** (cinética química, 1889): la velocidad de un proceso
químico crece **exponencialmente** con la temperatura. Es una de las relaciones
más universales de la química física. El caucho se degrada químicamente al
calentarse, así que aplica directamente: de ahí `exp(E_a · temperatura)`.

Esa exponencial es el motivo de que la degradación de neumáticos sea tan
brutalmente no lineal. Un neumático 10 °C más caliente no se desgasta un poco
más: se desgasta *mucho* más.

## 6.3 Iteración: el factor `(1 − d)`

**Al probar el modelo básico encontré que `d` llegaba a 3.24.** No es físico: no
puedes consumir el 324 % de la banda de rodadura.

La solución fue multiplicar la velocidad de desgaste por `(1 − d)`. Cuando `d` se
acerca a 1, ese factor se acerca a 0 y el desgaste se frena. **`d` queda acotada
entre 0 y 1 estructuralmente.**

Lo elegante es que la cota **la impone la propia ecuación**, no un castigo en la
función de coste. En términos de programación: es la diferencia entre un tipo que
hace imposible representar un estado inválido, y una validación en tiempo de
ejecución que hay que acordarse de llamar. El primero siempre es mejor.

## 6.4 Iteración: el factor `(1 + ζ·d)` — la decisión clave del proyecto

Con la saturación anterior, **el cliff desapareció**. Ningún stint generaba ya la
caída brusca.

Aquí había dos caminos.

**El camino fácil:** subir `γ₂`, la constante que controla el término del cliff
en el observable. Lo descarté, porque convierte el cliff en un artefacto que yo
metí a mano. Si el cliff está en el modelo porque lo puse yo, el modelo no está
explicando nada.

**El camino correcto:** preguntarme *por qué existe el cliff de verdad*.

La respuesta es una realimentación positiva:

```
la banda se adelgaza
    → la misma energía friccional se reparte sobre menos masa de caucho
        → sube la temperatura superficial
            → por Arrhenius, el desgaste se acelera exponencialmente
                → la banda se adelgaza más rápido
                    → (y vuelta a empezar, cada vez más rápido)
```

Eso es un bucle que se retroalimenta. Al principio no se nota, y llegado un punto
se dispara. **Eso es exactamente lo que es un precipicio.**

Se modela con el factor `(1 + ζ·d)` en el término de generación de calor: a más
desgaste, más calentamiento para la misma energía. Con él, **el cliff emerge solo
de la dinámica**. No lo puse: sale de las ecuaciones.

Y es justamente la clase de mecanismo que una LSTM no tiene forma de descubrir
con 36 stints de datos ruidosos. Es el argumento central de por qué una PINN
aporta algo en este problema.

## 6.5 El observable: cómo se conecta con lo medible

`d` no se puede medir, pero su efecto sí: el coche va más lento. Modelo esa
relación como

```
δ = γ₁·d  +  γ₂·d⁸
```

- `γ₁·d` es la **degradación gradual**, proporcional al desgaste.
- `γ₂·d⁸` es el **colapso de agarre**. Con exponente 8, este término es
  prácticamente cero mientras `d` sea moderado (0.5⁸ ≈ 0.004) y se dispara cuando
  `d` se acerca a 1 (0.95⁸ ≈ 0.66).

Es el mismo truco que usarías para una rampa de activación en código: una
potencia alta actúa como un interruptor suave.

---

# 7. La arquitectura concreta, y qué hace DeepXDE

## 7.1 La decisión de diseño: una red paramétrica

Una PINN "de libro" resuelve **una sola trayectoria**: le das el tiempo, te
devuelve el estado. El problema es evidente para nuestro caso: habría que
**reentrenar la red para cada stint nuevo**. Treinta minutos por predicción. Con
un presupuesto de 500 ms, es absurdo.

Así que la red toma también las condiciones como entrada:

```
N(τ, q, λ, v, T_pista, compuesto) → (θ, d)
 └─tiempo──┘ └────── contexto ──────┘
```

Esto la convierte en un **operador solución**: en vez de aprender una solución,
aprende **la familia completa de soluciones** de la ecuación diferencial, para
todo el rango de condiciones de carrera posibles.

La consecuencia práctica es lo que hace viable todo el proyecto: predecir un
stint nuevo es **un paso forward**. Medido: **0.42 ms de media, 1.47 ms en p95**
para 45 vueltas. El presupuesto de 500 ms se lo come entero el transporte, no el
modelo.

Y permite el mapa de decisión de `06_cliff_map.png`: 1 728 predicciones para
barrer todo el espacio de condiciones, en una sola llamada por lotes.

**La contrapartida**, que hay que declarar: supongo el contexto constante dentro
de un stint (uso su mediana). La variación vuelta a vuelta se absorbe en el
término de datos.

## 7.2 Condiciones iniciales: imponerlas en vez de pedirlas

Sabemos dos cosas con certeza absoluta: un neumático nuevo tiene `d = 0`, y sale
de boxes a una temperatura conocida.

El enfoque habitual es añadir términos al coste que penalicen desviarse de eso.
Funciona, pero regular. Ahora tienes cinco términos compitiendo y **hay que
elegir cuánto pesa cada uno** — que es, con diferencia, la causa más común de que
una PINN no converja.

En vez de eso, hago que sea **imposible** violarlas:

```
θ(τ) = θ₀ + τ · N₀(x)          →  en τ=0 el segundo término se anula: θ(0) = θ₀
d(τ) = τ · softplus(N₁(x))     →  d(0) = 0, y softplus > 0 garantiza d ≥ 0
```

Da igual lo que la red produzca por dentro: la condición inicial se cumple
**exactamente, por construcción algebraica**.

De nuevo la misma filosofía que en 6.3: hacer los estados inválidos
irrepresentables, en lugar de validarlos después. Y de paso elimina dos términos
del coste y su ajuste de pesos.

## 7.3 Qué aporta DeepXDE

DeepXDE es la librería que el PDF del proyecto ya proponía. Concretamente
resuelve:

| Necesidad | Qué da DeepXDE |
|---|---|
| Derivar la salida respecto a la entrada | `dde.grad.jacobian(y, x, i, j)` — exacto, sin escribir la regla de la cadena |
| Generar puntos de colocación | `dde.geometry.Hypercube` los muestrea por el dominio |
| Conectar datos con estados latentes | `PointSetOperatorBC`, que compara una *función* de la salida contra observaciones |
| Estimar constantes físicas | `dde.Variable`, que las optimiza junto con los pesos |
| Orquestar el entrenamiento | Combina los cinco términos y gestiona Adam y L-BFGS |

Lo que más valor aporta es `PointSetOperatorBC`. Yo no observo `d`; observo
`δ = γ₁d + γ₂d⁸`. Con este mecanismo le digo a DeepXDE: *"la red predice `d`;
aplícale esta transformación y compara **eso** con lo medido"*. Los gradientes
fluyen hacia atrás a través de la transformación hasta los pesos. Es lo que hace
posible entrenar sobre una variable que nunca se ve.

## 7.4 Los cinco términos del coste

| Término | Qué impone | Dónde se evalúa |
|---|---|---|
| L1 | residuo de (E1), balance térmico | 8 000 puntos de todo el dominio |
| L2 | residuo de (E2), ley de desgaste | 8 000 puntos de todo el dominio |
| L3 | cota `d ≤ d_max` | 8 000 puntos de todo el dominio |
| L4 | ajuste a los tiempos medidos | solo vueltas reales |
| L5 | proxy de temperatura (solo sintético) | solo vueltas reales |

## 7.5 Dos optimizadores, y por qué hacen falta los dos

Entreno primero con **Adam** (15 000 iteraciones) y luego con **L-BFGS** (3 000).

Adam da pasos pequeños y robustos; es bueno para explorar cuando estás lejos de
la solución. L-BFGS usa información de curvatura, converge mucho más fino, pero
necesita estar ya cerca.

No es una convención copiada: **se ve el efecto en los datos**. En
`04_parameters.png`, los tres parámetros térmicos (`ζ`, `h₀`, `h₁`) se quedan
estancados durante las 15 000 iteraciones de Adam y **solo saltan a su valor
verdadero cuando entra L-BFGS**.

La razón es que son los peor condicionados del problema: solo influyen en lo
observable a través de dos capas de composición (temperatura → desgaste → tiempo
por vuelta), así que su gradiente es diminuto. Adam no los mueve. L-BFGS sí.

---

# 8. El problema inverso

## 8.1 Qué es

Las ecuaciones tienen nueve constantes (`ζ, h₀, h₁, k_w, m, E_a, κ, γ₁, γ₂`).
**No conozco ninguna.** Dependen del compuesto de Pirelli, del asfalto, del coche.

Así que no las fijo: **las estimo junto con los pesos de la red**. En DeepXDE son
`dde.Variable`, y el optimizador las trata como nueve parámetros más entre los
13 000. Esto se llama **problema inverso**: en vez de resolver la ecuación
conociendo las constantes, deduces las constantes observando el resultado.

Un detalle de implementación: las guardo en **logaritmo** y uso `exp()` para
recuperarlas. Así son **positivas por construcción**, que es lo que exige su
significado físico (no existe un coeficiente de enfriamiento negativo). Misma
filosofía que en 6.3 y 7.2.

## 8.2 Cómo se valida algo que no se puede validar

Aquí hay un problema metodológico serio: con datos reales **no existe la
verdad**. Si la red estima `E_a = 0.94`, no hay forma de saber si acertó.

Por eso construí un **banco de pruebas sintético**: integro numéricamente las
ecuaciones con constantes que yo elijo, añado ruido de medición realista, y
obtengo datos donde **sí** conozco la respuesta.

Entonces arranco la PINN desde valores iniciales deliberadamente equivocados y
compruebo si recupera los verdaderos usando **solo los tiempos por vuelta**.

**Resultado: 1.3 % de error medio en los nueve parámetros.** Ésa es la evidencia
de que el método funciona. En `04_parameters.png` se ve la convergencia.

Es, en esencia, un **test de integración con datos sintéticos** — exactamente lo
que harías para probar un sistema cuyas salidas reales no puedes verificar.

---

# 9. Los datos

## 9.1 El problema de fondo

**Nada de lo que el modelo necesita es público.** La temperatura interna, la
carga vertical y el espesor de banda son datos propietarios de cada equipo.

Lo que sí es público, vía la librería **FastF1** (datos oficiales de F1):

- Telemetría de a bordo: velocidad, acelerador, freno, marcha, RPM (~10 Hz).
- Posición GPS: X, Y, Z (~4 Hz).
- Cronometraje: tiempo por vuelta, stint, compuesto, edad del neumático.
- Meteorología: temperatura de aire y de pista.

Toda la ingeniería de características consiste en cerrar esa brecha.

## 9.2 Las variables proxy

| Variable | Cómo se construye | Papel en el modelo |
|---|---|---|
| `q_fric` | integral de \|aceleración\| × velocidad sobre la vuelta | generación de calor de (E1) |
| `load` | media de la aceleración total, en g | término de Archard de (E2) |
| `speed` | velocidad media | enfriamiento convectivo |
| `track_temp` | meteorología interpolada en el instante de la vuelta | temperatura ambiente |
| `compound` | 0 = blando, 0.5 = medio, 1 = duro | resistencia del compuesto |

**La aceleración lateral no viene en la telemetría** — y es la que más desgasta,
porque es la de las curvas. La reconstruyo derivando **dos veces la trayectoria
GPS**: la primera derivada de la posición es la velocidad, la segunda es la
aceleración, y su componente perpendicular al movimiento es la lateral.

El problema es que derivar dos veces **amplifica muchísimo el ruido de
muestreo**. Por eso hay que suavizar antes, con un filtro Savitzky-Golay.

**Detalle que costó una iteración:** la ventana de suavizado se fija en
*segundos*, no en número de muestras. FastF1 fusiona la telemetría del coche
(~10 Hz) con el GPS (~4 Hz) interpolando, así que la frecuencia efectiva varía
entre vueltas. Una ventana fija en muestras aplicaría un filtro físicamente
distinto en cada caso.

## 9.3 El observable, y la corrección obligatoria

La degradación se mide como pérdida de ritmo respecto a la mejor vuelta del
stint. Pero hay un efecto que la enmascara por completo:

**El coche se aligera unos 100 kg de combustible durante la carrera**, y eso vale
más de un segundo por vuelta. Sin corregirlo, el coche va acelerando a lo largo
del stint por aligeramiento **justo cuando el neumático lo está frenando por
desgaste**, y los dos efectos se cancelan visualmente.

Se corrige restando `k_fuel × (vueltas_restantes)`, con `k_fuel = 0.055 s/vuelta`.

## 9.4 Filtros de calidad

Se descartan las vueltas que son lentas por razones ajenas al neumático:

- Solo bandera verde (`TrackStatus == 1`) — un safety car cambia el tiempo varios
  segundos.
- Sin vueltas de entrada o salida de boxes.
- Solo vueltas marcadas `IsAccurate` por FastF1.
- Sin vueltas borradas por la FIA.
- Solo juegos nuevos — porque `d(0) = 0` solo tiene sentido con un neumático
  nuevo.

De Monza + Hungría 2023 sobreviven **36 stints con 740 vueltas**.

---

# 10. Los cuatro problemas serios

Ésta es probablemente la parte más útil para el informe. Ninguno era un bug de
programación: los cuatro eran problemas de fondo del planteamiento, y cada uno
enseña algo.

## Problema 1 — Los stints se cortaban demasiado pronto

**Síntoma:** el problema inverso daba 11.8 % de error medio, pero `γ₂` fallaba
por un **62 %**, y `k_w` y `γ₁` se compensaban mutuamente (uno un 18 % bajo, el
otro un 16 % alto).

**Diagnóstico:** el generador sintético cortaba el stint justo al llegar al
cliff. Solo 3 de 48 stints tenían un cliff detectable, y solo 1 alcanzaba
`d > 0.9`. Pero `γ₂` **solo hace algo cuando `d` se acerca a 1**. Sin datos en
ese régimen, no hay información de la que estimarlo. La red estaba adivinando.

**Solución:** dejar 2–5 vueltas *después* del cliff. Además es más realista: un
equipo no para en el instante exacto, pierde vueltas decidiendo o esperando hueco.

**Resultado:** error medio **11.8 % → 2.1 %**. `γ₂` de 62 % a 4.7 %.

**Lección:** un parámetro solo es estimable si los datos cubren el régimen donde
ese parámetro tiene efecto.

## Problema 2 — La normalización borraba la señal entre circuitos

**Síntoma:** entrenando con una carrera, las variables de contexto casi no
variaban (carga entre 0.95 y 1.04). La red no podía aprender a responder a las
condiciones porque no veía condiciones distintas.

**Diagnóstico:** estaba normalizando los proxies dividiendo por la **mediana de
la propia sesión**. Eso deja a Monza en 1.0 y a Hungría también en 1.0 —dos
circuitos radicalmente distintos— borrando justo la variación que hace falta.

**Solución:** normalizar contra **referencias físicas fijas** (1900 W/kg, 3.8 g,
58 m/s), calibradas midiendo carreras reales.

**Resultado:** con dos carreras la carga pasó a variar entre 0.84 y 1.59.

**Lección:** normalizar cada muestra contra sí misma destruye exactamente la
información que distingue unas muestras de otras.

## Problema 3 — El origen de la degradación estaba mal definido

**Síntoma:** con datos reales la PINN daba **RMSE 2.82 s** (los baselines, 0.59)
y 16.4 % de violaciones de monotonía. Los parámetros estaban degenerados: `E_a`
colapsó a 0.03 (sin activación térmica) y `m` a 0.13 (sin dependencia de carga).
La red había apagado la física para poder ajustar.

**Diagnóstico:** medí la `δ` de la primera vuelta válida de cada stint. Promedio:
**0.473 s**. Cada stint empezaba ya medio segundo por encima de su referencia.

La causa es física y evidente en retrospectiva: **un juego nuevo sale frío y se
hace *más rápido* durante dos o tres vueltas** antes de empezar a caer. Es la
fase de calentamiento. Mi modelo es monótono por construcción y **no puede
representarla**, así que cada stint arrancaba con un conflicto irresoluble entre
lo observado y lo predecible. La red lo compensaba distorsionando las constantes
físicas.

**Solución:** anclar `d = 0` en el **pico de rendimiento** del stint, no en su
primera vuelta, y descartar las vueltas de calentamiento.

**Resultado:** RMSE **2.82 → 0.567 s**. Violaciones **16.4 % → 0 %**.

**Lección:** si tu modelo no puede representar un fenómeno, no dejes ese fenómeno
dentro de los datos de entrenamiento. Se filtra, y no por donde esperas.

## Problema 4 — Degeneración exacta de escala (el más grave)

**Síntoma:** el entrenamiento completo con datos reales **divergió**:
`γ₂ = 2.5 × 10¹³`, `k_w = 0.026`, RMSE de prueba de **5 800 millones de
segundos**. Y lo desconcertante: **la pérdida de entrenamiento era baja** (0.117).
Según su propia métrica, el modelo iba bien.

**Diagnóstico:** el modelo tiene una **dirección exactamente degenerada**. Si
multiplicas `d` por cualquier factor `ε` y divides las gammas apropiadamente:

```
d → ε·d ,  γ₁ → γ₁/ε ,  γ₂ → γ₂/ε⁸     →  δ queda EXACTAMENTE igual
```

Hay infinitas combinaciones de parámetros que producen predicciones idénticas.
El optimizador no tiene forma de preferir una, así que se desliza por esa
dirección indefinidamente hasta desbordar la precisión numérica.

En el banco sintético esto no pasaba porque dos cosas lo impedían: el proxy
térmico y los stints que llegan a saturar en `d = 1`. **Con datos reales no
existe ninguna de las dos.**

**Intento 1:** acotar `γ₁ ∈ [0.2, 4.0]` y `γ₂ ∈ [0.2, 6.0]` con una sigmoide.
Detuvo el desbordamiento (RMSE 1.56), pero **las dos quedaron pegadas a sus
topes** — señal inequívoca de que el empuje seguía ahí y la cota solo lo tapaba.

**Diagnóstico final:** la escala absoluta de `d` **no es identificable a partir
de datos de carrera**. Lo único que podría anclarla es la saturación en `d = 1`,
y a eso solo se llega destruyendo el neumático. Los equipos paran mucho antes.
**No es un fallo del método: la información no está en los datos.**

**Solución:** la ley termo-mecánica se calibra en el banco sintético, donde sí
hay verdad de referencia. Sobre telemetría real se ajustan **solo `k_w` y `κ`**,
las dos cantidades que de verdad cambian entre circuitos y lotes de neumático.
"Un segundo de pérdida equivale a este desgaste" es una afirmación de
**calibración**, no algo que los tiempos por vuelta puedan responder.

**Resultado:** parámetros físicamente razonables (`k_w = 0.864`, `κ = 0.963`) y
ninguno pegado a una cota.

> **La lección más citable del proyecto:** en una PINN con problema inverso, una
> **pérdida de entrenamiento baja no garantiza absolutamente nada** si el modelo
> tiene direcciones degeneradas. Hay que enumerarlas explícitamente y cerrarlas
> una por una. Es el equivalente a tener el 100 % de cobertura de tests y aun así
> tener el sistema roto, porque los tests no comprueban lo que importa.

---

# 11. Resultados

## 11.1 Banco sintético — 64 stints (48 entrenamiento / 16 prueba)

| Modelo | RMSE [s] | MAE [s] | MaxErr [s] | Cliff MAE | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|---|---|
| **PINN** | **0.063** | **0.050** | **0.189** | n/d | **0.0 %** | **1.1 %** |
| Lineal (clásico) | 0.246 | 0.159 | 0.913 | n/d | 22.7 % | 22.2 % |
| LSTM (caja negra) | 0.070 | 0.057 | 0.219 | n/d | 0.8 % | 1.1 % |

**"Violaciones"** es la métrica que mejor resume el proyecto: el porcentaje de
vueltas en las que el modelo predice que el neumático **recupera** agarre. Es
físicamente imposible, y **ninguna métrica de error lo penaliza**, así que hay
que medirlo aparte.

La PINN es la más precisa y la más coherente físicamente: 0 % de violaciones
dentro del rango observado, frente al 22.7 % del modelo lineal.

> **Las columnas de cliff están vacías, y eso es un hallazgo, no un hueco.** Con
> una definición robusta al ruido (0.30 s/vuelta sostenido 4 vueltas, ver la
> parte 13), ningún stint del banco califica. La versión anterior de esta tabla
> reportaba "Cliff MAE 0.50, 2/2 detectados" usando un umbral de 0.15 s/vuelta en
> un solo punto — ese criterio dispara en el **100 %** de curvas sin cliff cuando
> hay ruido de cronometraje realista. Esos números medían ruido.

## 11.2 Recuperación de parámetros — 1.3 % de error medio

`ζ` 2.3 % · `h₀` 1.4 % · `h₁` 0.9 % · `k_w` 2.9 % · `m` 2.9 % · `E_a` 1.0 % ·
`κ` 0.5 % · `γ₁` 2.3 % · `γ₂` 4.7 %

Usando **solo los tiempos por vuelta**, sin ver jamás la temperatura ni el
desgaste.

## 11.3 Telemetría real — Monza + Hungría 2023, 36 stints

| Modelo | RMSE [s] | MAE [s] | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|
| PINN | 1.172 | 0.725 | 6.3 % | 6.3 % |
| Lineal (clásico) | **0.539** | **0.429** | 0.0 % | 0.0 % |
| LSTM (caja negra) | 0.536 | 0.423 | 0.6 % | 8.8 % |

Estas cifras se reproducen exactamente entre corridas repetidas, igual que las
sintéticas.

**Aquí la PINN no le gana a los baselines, y hay que decirlo sin adornos.**

El motivo es el que anticipaba la parte 5.4. Con 36 stints, un suelo de ruido de
~0.5 s por vuelta y poca variación de condiciones, la curva observada es **casi
lineal en el rango medido**, y ahí un ajuste lineal es difícil de batir. La PINN
paga el precio de estar restringida sin poder cobrar el beneficio, porque el
beneficio está en la extrapolación y en el cliff, regímenes que estos datos
apenas tocan.

Los parámetros ajustados sí son físicamente razonables (`k_w = 0.813`,
`κ = 0.968`) y ninguno queda pegado a una cota, que era el síntoma de la
degeneración.

Lo honesto es concluir que el camino de datos reales está **validado
mecánicamente pero no científicamente**: funciona de punta a punta y produce
parámetros interpretables, pero necesita bastantes más carreras.

## 11.4 Latencia

**0.42 ms de media, 1.47 ms en p95** para predecir un stint completo de 45
vueltas, en CPU. Tres órdenes de magnitud por debajo del presupuesto de 500 ms.

---

# 12. Limitaciones y siguientes pasos

## 12.1 Limitaciones

- **Contexto constante por stint.** Se usa la mediana; la variación vuelta a
  vuelta se absorbe en el término de datos. Es lo que hace tratable el operador
  paramétrico.
- **Evolución de pista sin modelar.** El circuito engoma y se hace más rápido
  durante la carrera. Ese efecto no está separado de la degradación y sesga la
  pendiente estimada.
- **`load` es un proxy relativo, no una medida.** Su magnitud absoluta (~3–4 g de
  media) está por encima de lo que mediría un acelerómetro real, porque sale de
  derivar GPS. Lo que importa es que sea monótono en la carga real y que
  discrimine circuitos, y ambas se cumplen.
- **Solo juegos nuevos.** Los stints con neumáticos usados se descartan.
- **La fase de calentamiento se descarta** en vez de modelarse.

## 12.2 Siguientes pasos, por impacto

1. **Entrenar sobre 8–10 carreras**, no dos. Es de largo lo que más cambiaría los
   resultados: ataca a la vez la falta de datos y la falta de variación de
   condiciones.
2. **Modelar la evolución de pista** como un término separado, para dejar de
   confundirla con degradación.
3. **Añadir la fase de calentamiento** al modelo, para no tener que tirar esas
   vueltas.
4. **Un término de efecto piloto/coche**, que hoy va entero al ruido.

---

# 13. Verificación contra datos publicados

Contrasté el modelo contra una fuente independiente: un análisis público de
degradación de neumáticos de F1 con tasas por compuesto y por circuito medidas
sobre datos de carrera
([Yahoo Sports, análisis de la temporada 2026](https://sports.yahoo.com/articles/f1-tyre-degradation-2026-data-112619253.html)).
Sus cifras principales son las tasas de 2026 —duro 0.071, medio 0.065, blando
0.063 s/vuelta— más los rangos por temporada y por circuito.

## 13.1 Lo que coincidió

Medí la degradación de la misma forma sobre mi propio dataset: ajuste lineal de
la pérdida de ritmo corregida por combustible contra la vuelta del stint.

| Magnitud | Publicado | Medido aquí | Δ |
|---|---|---|---|
| Spread entre compuestos, 2023 | 0.011 s/vuelta | 0.0102 s/vuelta (MEDIO − DURO) | ~7 % |
| Magnitud de la tasa | 2026 va de 0.022 (China) a 0.097 (Austria) | Monza 0.096, Hungría 0.067 | dentro del rango |
| La evolución de pista puede invertir el signo | Montreal −0.005 s/vuelta | 1 de 36 stints con pendiente negativa | consistente |

Las tasas que predice el modelo quedan cerca de lo observado:

| Compuesto | Modelo | Observado | Δ |
|---|---|---|---|
| MEDIO | 0.0892 s/vuelta | 0.0916 s/vuelta | −2.6 % |
| DURO | 0.0755 s/vuelta | 0.0814 s/vuelta | −7.2 % |

Y el banco sintético resulta estar bien calibrado en magnitud sin haber sido
ajustado para ello: un stint MEDIO nominal se degrada a **0.086 s/vuelta** contra
una mediana real medida de **0.090 s/vuelta**.

El spread de 2023 coincidiendo al ~7 % es la comprobación más fuerte, porque es
una comparación directa de lo mismo. Dos salvedades: mi muestra de blandos es de
un solo stint, así que el spread es medio-vs-duro; y dos carreras no pueden
replicar una cifra de temporada completa.

## 13.2 Lo que destapó — tres hallazgos

### 1. El detector de cliff estaba midiendo ruido

El criterio original —pendiente por encima de 0.15 s/vuelta en un solo punto—
dispara en el **100 %** de curvas que no tienen ningún cliff, en cuanto hay ruido
de cronometraje realista (σ ≈ 0.3–0.5 s). No es un fallo marginal:

| Ruido σ [s] | 0.00 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|
| "Cliff" falso detectado | 0 % | 0.6 % | 45 % | 98 % | 100 % | 100 % |

Explica un resultado que debería haberme parecido sospechoso: 34 de 36 stints
reales "tenían cliff" mientras su degradación mediana era un plácido 0.09
s/vuelta. El criterio ahora es 0.30 s/vuelta **sostenido durante 4 vueltas
seguidas**, que baja los falsos positivos a 0–1 % y sigue detectando el 96–100 %
de las rodillas genuinas. Con él, los cliffs reales son raros: **2 de 36 stints**.

### 2. Los cliffs son mucho más raros de lo que asume el planteamiento

El análisis publicado reporta la degradación como una única tasa lineal por
compuesto y nunca cuantifica un cliff. Medido aquí, la propia verdad de
referencia de mi banco sintético llega a 0.086 s/vuelta en un stint nominal y
solo alcanza 0.305 s/vuelta en el contexto más extremo. El colapso de agarre
existe en el modelo, pero como **evento detectable** apenas ocurre en esta era —
por eso las columnas de cliff quedaron vacías.

La lectura honesta es que el modelo predice bien **curvas de degradación**; hablar
de "predicción de la vuelta del cliff" promete más de lo que los datos sostienen.

### 3. El término de compuesto no puede representar 2026

En 2026 la jerarquía está **invertida**: el duro se degrada más rápido (0.071) y
el blando más lento (0.063). Mi ley de desgaste lleva el compuesto como
`exp(−κ·c)`, con `c` = 0 para blando y 1 para duro, y `κ` está parametrizada en
logaritmo, así que **`κ > 0` siempre** y el duro necesariamente se desgasta menos
que el blando.

Para 2023 ese orden es el correcto; para 2026 el modelo es estructuralmente
incapaz de ajustar los datos. El arreglo es de una línea —quitarle la
parametrización logarítmica a `κ` para que pueda ser negativa— y no cuesta nada,
porque a diferencia de un coeficiente de enfriamiento no hay ninguna razón física
para que `κ` sea positiva. No lo apliqué porque este proyecto trabaja con datos
de 2023.

---

# 14. Cómo reproducirlo

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
set DDE_BACKEND=pytorch

python run_train.py --source synthetic --stints 64          # ~30 min
python run_train.py --source synthetic --quick              # ~2 min, valida el pipeline
python run_train.py --source fastf1 --gp Monza Hungary      # telemetría real
python run_infer.py --compound SOFT --track-temp 0.8        # inferencia + latencia
```

## Estructura del código

```
src/tirepinn/
  config.py          hiperparámetros físicos, de red y de datos
  physics.py         el sistema de EDOs, integrador RK4, detección del cliff
  pinn.py            la PINN paramétrica (DeepXDE)
  dataset.py         Stint / StintDataset, partición, dominio
  data_synthetic.py  banco de pruebas con verdad conocida
  data_fastf1.py     telemetría real e ingeniería de características
  baselines.py       lineal clásico y LSTM
  evaluate.py        métricas
  plots.py           figuras
run_train.py         entrenamiento + comparación + figuras
run_infer.py         inferencia y latencia
```

## Referencias

- Raissi, Perdikaris & Karniadakis (2019). *Physics-informed neural networks*.
  Journal of Computational Physics, 378, 686–707. — el artículo fundacional.
- Lu, Meng, Mao & Karniadakis (2021). *DeepXDE: A deep learning library for
  solving differential equations*. SIAM Review, 63(1), 208–228.
- Archard, J.F. (1953). *Contact and rubbing of flat surfaces*. — la ley de
  desgaste.
- Arrhenius, S. (1889). — la dependencia exponencial con la temperatura.
- Oehrly, M. *FastF1: A Python package for F1 telemetry and timing data*.
- [F1 tyre degradation 2026 data](https://sports.yahoo.com/articles/f1-tyre-degradation-2026-data-112619253.html)
  — las cifras independientes usadas en la parte 13.
