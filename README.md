# PINN de degradación de neumáticos — v0

**Implementación inicial.** Una red neuronal a la que se le impone una ecuación
diferencial, aplicada a la degradación de un neumático de Fórmula 1. Es el
cimiento del proyecto: el mínimo que demuestra que el método funciona, escrito
para poder leerse entero de una sentada.

Sin frameworks de PINN — el residuo está escrito a mano con PyTorch — y con dos
fuentes de datos: un banco sintético donde se conoce la respuesta, y telemetría
real descargada de la API de Fórmula 1.

> La versión avanzada está en la rama `main`: dos EDO acopladas, el *cliff*
> emergiendo de la realimentación térmica, nueve parámetros estimados y una
> temporada completa de resultados. El camino de aquí hasta allí, paso a paso,
> está en [ROADMAP.md](ROADMAP.md).

![Ajuste y extrapolación](outputs/01_ajuste.png)

*Los tres compuestos. La zona gris es lo que el modelo vio; a la derecha de la
línea discontinua todos los modelos extrapolan. La curva roja del PINN queda
encima de la solución exacta (azul).*

---

## 1. La idea, en cuatro pasos

Una red neuronal es una función parametrizada, `d = N(τ, contexto ; W)`, y
entrenarla es buscar los pesos `W` que minimizan una pérdida. Lo que convierte
eso en un PINN es una sola observación:

1. La diferenciación automática puede derivar la salida de la red **respecto a
   sus entradas**, de forma exacta y barata.
2. Así que se puede calcular el **residuo** de la ecuación diferencial que la
   física dice que se cumple: `r = dd/dτ − velocidad_desgaste(d, contexto)`.
3. Evaluar ese residuo necesita **un punto del dominio y nada más**. No hace
   falta saber la respuesta correcta ahí.
4. Metiendo `r²` en la pérdida, la red obedece la ecuación — incluso en vueltas
   donde no hay ni un solo dato.

El paso 3 es todo el truco. El término de física se exige en 2 000 puntos
repartidos hasta la vuelta 45, mucho más allá del stint más largo del conjunto,
y en combinaciones de condiciones que no se dieron en ninguna carrera.

## 2. El modelo físico

Una sola EDO, con cinco variables de contexto y un observable:

```
(E)   dd/dτ = k(contexto) · (1 − d)

      k = k_w · (carga/carga_ref)^m
              · exp( E_a·(T_pista − T_ref)
                   + E_q·(q_fricción − q_ref)
                   − E_v·(velocidad − v_ref)
                   − κ·(compuesto − c_ref) )

obs   δ(τ) = γ₁ · d
```

**El tiempo y el estado**

- `τ` — tiempo adimensional del stint, `vuelta / 30`
- `d` — fracción de goma consumida, `0` nueva … `1` gastada · **estado latente, nunca se observa**

**El contexto** (constante dentro de un stint)

| Variable | Qué es | Efecto |
|---|---|---|
| `q_fricción` | energía de fricción por vuelta | más energía → más desgaste |
| `carga` | carga mecánica media en g | ley de Archard: `carga^m` |
| `velocidad` | velocidad media | más aire → más refrigeración → **menos** desgaste |
| `T_pista` | temperatura del asfalto | activación térmica |
| `compuesto` | 0 blando … 1 duro | más duro → **menos** desgaste |

**Lo único que se mide**

- `δ` — pérdida de ritmo en segundos contra la mejor vuelta del stint

### Las dos decisiones que cargan con el peso

**El factor `(1 − d)`** acota `d` a `[0, 1]` **estructuralmente**: no puedes
gastar más goma de la que hay. Y como mantiene la velocidad no negativa, la
monotonía (el neumático nunca se regenera) sale de la propia ecuación, no de una
restricción añadida después.

**Restar una referencia en cada término** hace que `k_w` signifique
literalmente *la velocidad de desgaste en condiciones normales*. No es
cosmética: sin centrar, `k_w` y `E_v` se pisan. Está medido en este mismo
proyecto — sin centrar, `k_w` salía con un 10 % de error y `E_v` con un 33 %,
pero la combinación `log(k_w) − E_v` se recuperaba con un error de **0,0098**.
El modelo sabía perfectamente cuánto se gastaba el neumático; lo que no sabía
era a cuál de las dos constantes atribuirlo.

### La propiedad que hace útil este modelo

Como el contexto es constante dentro de un stint, `k` también lo es, y entonces
hay **solución exacta escrita a mano**: `d(τ) = 1 − exp(−k·τ)`.

Eso es justamente lo que lo convierte en el punto de partida correcto: permite
comprobar el método contra una respuesta exacta antes de apuntarlo a un sistema
donde no existe ninguna. El integrador RK4 coincide con la forma cerrada con un
error de `1,3 × 10⁻¹¹`.

## 3. Qué hay dentro

| Fichero | Qué contiene |
|---|---|
| `physics.py` | La EDO, el observable, la solución exacta y un integrador RK4 |
| `data.py` | De dónde salen los stints: generador sintético **o** lector del CSV |
| `descargar_datos.py` | **Descarga telemetría real de la API y la deja en un CSV** |
| `pinn.py` | El PINN en PyTorch puro: red, residuo, colocación, entrenamiento |
| `baseline.py` | El modelo lineal clásico contra el que se compara |
| `evaluate.py` | RMSE, MAE, error máximo y violaciones de monotonía |
| `run.py` | Entrena, evalúa y dibuja |

La red es un perceptrón de `6 → 64 → 64 → 64 → 64 → 1` con `tanh`:
**12 993 pesos**. Se puede cambiar sin tocar código con `--neuronas` y `--capas`.

La pérdida tiene tres términos:

| Término | Qué impone | Dónde |
|---|---|---|
| física | residuo de la EDO | 2 000 puntos de colocación, haya datos o no |
| datos | ajuste al ritmo medido | solo vueltas observadas |
| condición inicial | `d(0) = 0` | en `τ = 0`, con contextos sorteados |

Y **seis constantes físicas** se estiman junto con los pesos de la red: un
problema inverso completo. Las cinco que tienen que ser positivas se
parametrizan como su logaritmo, para que lo sean por construcción. `κ` se deja
con el signo libre, porque es la única del sistema cuyo signo no está fijado por
la física.

`γ₁` **no** se estima, y hay un motivo: lo único que se mide es `δ = γ₁·d`, así
que dejar libres a la vez `d` y `γ₁` admite infinitas soluciones equivalentes.
Fijar `γ₁` ancla la escala. En `main` esta misma degeneración, sin cerrar, hizo
divergir un entrenamiento hasta un RMSE de miles de millones de segundos.

## 4. Cómo correrlo

```bash
pip install -r requirements.txt
```

### Con datos sintéticos

```bash
python run.py                     # 48 stints, 8 000 iteraciones, ~1 min en CPU
python run.py --quick             # versión corta para comprobar que arranca
python run.py --neuronas 96 --capas 5 --iteraciones 15000   # red más grande
```

### Con datos reales

Primero se descargan, y quedan en un CSV que puedes abrir en Excel:

```bash
# Una carrera
python descargar_datos.py --anio 2023 --carreras Monza --salida datos/monza.csv

# Varias, que es lo recomendable: con una sola, las condiciones apenas varían
python descargar_datos.py --anio 2023 \
    --carreras Monza Hungary Spa Silverstone \
    --salida datos/2023.csv

# Prueba rápida sin telemetría: segundos en vez de minutos
python descargar_datos.py --anio 2023 --carreras Monza --sin-telemetria \
    --salida datos/prueba.csv
```

Y después se entrena con ellos:

```bash
python run.py --fuente csv --csv datos/2023.csv
```

`python descargar_datos.py --ayuda-variables` explica de dónde sale cada columna
del CSV, y `--help` lista todas las opciones.

> **La primera descarga tarda.** Parsear la telemetría de una carrera lleva
> varios minutos, porque se baja y se deriva la de cada vuelta por separado.
> FastF1 cachea lo descargado en `cache/` (unos 200 MB), así que la segunda vez
> es mucho más rápida.

### Qué hace el descargador, y por qué

El problema de fondo es que **nada de lo que el modelo necesita es observable**.
La temperatura interna del neumático, la carga vertical y el estado de la banda
son datos propios de cada equipo. Lo público es la telemetría de a bordo y los
tiempos por vuelta. Así que las variables se reconstruyen:

- **`q_fricción`** integrando `|a|·v` a lo largo de la vuelta.
- **`carga`** como aceleración total media, en g.
- **La aceleración lateral no viene en la telemetría**: se reconstruye derivando
  dos veces la trayectoria GPS. Como eso amplifica el ruido, antes se suaviza con
  un filtro Savitzky-Golay cuya ventana se fija **en segundos y no en muestras**,
  porque FastF1 fusiona fuentes a 10 Hz y 4 Hz y la frecuencia efectiva cambia
  de una vuelta a otra.

Y hay dos correcciones sin las cuales los datos no sirven:

- **Combustible y evolución de pista.** El coche se hace más rápido a lo largo
  de la carrera por motivos que no son el neumático, y sin corregirlo eso tapa
  la degradación entera. Las dos causas no son separables entre sí, pero su
  suma sí se puede estimar: el script ajusta una pendiente por carrera
  controlando por piloto y por degradación. `--efecto-vuelta` permite fijarla a
  mano.
- **El origen de la degradación es el pico, no la primera vuelta.** Un juego
  nuevo sale frío y se hace *más rápido* dos o tres vueltas antes de empezar a
  caer. El modelo es monótono por construcción y no puede representar eso, así
  que esas vueltas se descartan y `d = 0` se define en el pico. En `main` esta
  corrección llevó el RMSE de 2,82 s a 0,567 s.

## 5. Resultados

### Por defecto: 48 stints, 8 000 iteraciones, ~55 s de CPU

```
python run.py
```

| Modelo | RMSE | MAE | ErrorMax | ViolDentro | ViolExtrap |
|---|---|---|---|---|---|
| **PINN** | **0,048** | **0,040** | **0,113** | **0,0 %** | **0,0 %** |
| Lineal clásico | 0,052 | 0,042 | 0,166 | 0,0 % | 0,8 % |

*ViolDentro / ViolExtrap = % de vueltas en las que el modelo predice que el
neumático recupera agarre. Es imposible; el valor correcto es 0 %.*

Con estos datos los dos modelos empatan prácticamente en RMSE, y **conviene
decirlo así**: a lo largo de 12–30 vueltas la curva real está suavemente
doblada, y una parábola ajusta muy bien una curva suavemente doblada. La
diferencia está en el error máximo y en la extrapolación.

Recuperación de las constantes físicas — **9,3 % de error medio**:

| Constante | Estimado | Real | Error |
|---|---|---|---|
| `k_w` | 0,5578 | 0,5500 | 1,4 % |
| `m` | 1,5439 | 1,5000 | 2,9 % |
| `E_a` | 0,8721 | 0,9500 | 8,2 % |
| `E_q` | 0,4102 | 0,4000 | 2,5 % |
| `E_v` | 0,2279 | 0,3500 | **34,9 %** |
| `κ` | 0,8003 | 0,8500 | 5,8 % |

### Ese 34,9 % de `E_v` no es un fallo: es falta de información

`E_v` es el coeficiente de refrigeración. Sobre el rango en el que varía la
velocidad, `0,6`–`1,4`, mueve el exponente de la ecuación solo **0,28**. Para
comparar, `κ` lo mueve `0,85` y `E_a` lo mueve `0,95`. Es decir: `E_v` es, con
diferencia, **el efecto más pequeño del modelo**, y por tanto el que peor se
distingue del ruido de cronometraje.

Se comprueba dándole más datos y menos ruido:

```
python run.py --stints 140 --ruido 0.02 --iteraciones 10000
```

| Modelo | RMSE | MAE | ErrorMax | ViolDentro | ViolExtrap |
|---|---|---|---|---|---|
| **PINN** | **0,021** | **0,017** | **0,062** | **0,0 %** | **0,0 %** |
| Lineal clásico | 0,048 | 0,036 | 0,222 | 2,5 % | **11,7 %** |

| Constante | Estimado | Real | Error |
|---|---|---|---|
| `k_w` | 0,5505 | 0,5500 | 0,1 % |
| `m` | 1,4935 | 1,5000 | 0,4 % |
| `E_a` | 0,9554 | 0,9500 | 0,6 % |
| `E_q` | 0,3863 | 0,4000 | 3,4 % |
| `E_v` | 0,3463 | 0,3500 | **1,1 %** |
| `κ` | 0,8446 | 0,8500 | 0,6 % |
| | | **media** | **1,0 %** |

Las seis constantes caen al **1,0 % de error medio**, y `E_v` pasa del 34,9 % al
1,1 %. Con suficientes datos el problema inverso funciona; con pocos, lo primero
que se pierde es el efecto más débil.

Y aquí sí se separan los dos modelos: **el lineal extrapola mal en el 11,7 % de
las vueltas** — dice que el neumático mejora — mientras que el PINN no lo hace
nunca. Eso no es cuestión de ajustar mejor: es que el PINN tiene dentro una
ecuación que se lo prohíbe, y el lineal no tiene nada.

> **La lección que generaliza:** una constante solo es estimable si los datos
> cubren el régimen donde esa constante tiene efecto, y con una amplitud que
> destaque sobre el ruido. Es el mismo problema que en `main` obligó a fijar
> siete de los nueve parámetros cuando se entrena con telemetría real.

## 6. Lo que esta versión NO hace

Está escrito para que se vea el hueco, no para disimularlo:

- **No hay cliff.** No hay temperatura como estado propio, así que no hay
  realimentación entre el desgaste y el calor, y la curva de ritmo solo puede
  doblarse en un sentido. Es el paso 1 del ROADMAP y lo más importante que le
  falta.
- **La condición inicial es un término de pérdida**, no una restricción exacta.
  Se cumple aproximadamente y compite por gradiente con los otros dos términos.
- **Solo Adam.** Sin L-BFGS, que en `main` resulta ser lo único que mueve los
  parámetros peor condicionados.
- **La corrección de vuelta de carrera es una recta.** `main` ajusta un spline
  lineal a trozos, porque la forma de esa curva no tiene por qué ser lineal.
- **Sin baseline LSTM.** Aquí solo compite el modelo lineal clásico.
