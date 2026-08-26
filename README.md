# F1 Tire PINN

Physics-Informed Neural Network para predecir la degradación de neumáticos de
Fórmula 1 y la vuelta del *cliff*, implementada con **DeepXDE** sobre PyTorch.

Corresponde a la parte de modelado del proyecto *Real-Time Prediction of Formula 1
Tire Degradation using Physics-Informed Neural Networks*. **No incluye la capa
cloud**: aquí está el sistema físico, la red, el entrenamiento offline, los
baselines y la evaluación.

---

## 1. El modelo físico

Dos EDOs adimensionales acopladas describen la vida de un stint:

```
(E1)  dθ/dτ = A_gen · q · (1 + ζ·d)  −  (h₀ + h₁·v) · θ        balance térmico
(E2)  dd/dτ = k_w · λ^m · exp(E_a·(θ + T_trk) − κ·c) · (1 − d)  desgaste
```

| Símbolo | Significado | Origen |
|---|---|---|
| `τ` | vuelta del stint / `L_ref` | tiempo adimensional |
| `θ` | `(T_superficie − T_pista) / ΔT_ref` | **estado latente**, no observable |
| `d` | fracción de banda de rodadura consumida | **estado latente**, no observable |
| `q` | energía friccional específica por vuelta | telemetría |
| `λ` | carga mecánica media en g | telemetría |
| `v` | velocidad media (enfriamiento convectivo) | telemetría |
| `T_trk` | temperatura de pista normalizada | meteorología de la sesión |
| `c` | dureza del compuesto (0 blando … 1 duro) | timing |

**(E1)** es un balance térmico de capacitancia concentrada: generación
friccional menos decaimiento exponencial superficial. **(E2)** combina la ley de
Archard con una activación térmica tipo Arrhenius: el desgaste crece
exponencialmente con la temperatura.

Dos factores hacen el trabajo importante:

- **`(1 − d)` en (E2)** acota `d ∈ [0,1]` estructuralmente. No se puede consumir
  más banda de la que hay. La cota física la impone la propia EDO, no una
  penalización.
- **`(1 + ζ·d)` en (E1)** es el motor del *cliff*: al adelgazarse la banda, la
  misma energía se deposita sobre menos caucho → sube la temperatura → por
  Arrhenius se acelera el desgaste → la banda se adelgaza más. Es una
  realimentación positiva, así que **el cliff emerge de la dinámica acoplada en
  lugar de imponerse a mano**. Es exactamente la clase de restricción que una
  LSTM no tiene forma de conocer.

El observable medible no es `d` sino la pérdida de ritmo:

```
δ(τ) = γ₁·d + γ₂·d^p        (p = 8)
```

`γ₁·d` es la degradación lineal; `γ₂·d^p` es despreciable hasta que `d` se
acerca a 1 y entonces domina: el colapso de agarre.

---

## 2. Por qué esta PINN es paramétrica

Una PINN clásica resuelve **una** trayectoria: la red toma `t` y devuelve el
estado. Eso obligaría a reentrenar por cada stint, lo cual es inservible para
inferencia en vivo. Aquí la red es un **operador solución**:

```
N(τ, q, λ, v, T_trk, c) → (θ, d)
```

Aprende de una sola vez la familia completa de soluciones de la EDO para todo el
rango de condiciones de carrera. Predecir un stint nuevo es **un paso forward**,
sin reentrenar y sin integrar nada. Eso es lo que hace viable el desacople
entrenamiento-offline / inferencia-online.

### Función de pérdida

| Término | Qué impone | Dónde |
|---|---|---|
| `L1` | residuo de (E1) | todo el hipercubo de condiciones |
| `L2` | residuo de (E2) | todo el hipercubo de condiciones |
| `L3` | cota `d ≤ d_max` | todo el hipercubo |
| `L4` | ajuste a la pérdida de ritmo medida | puntos observados |
| `L5` | proxy de temperatura (opcional) | puntos observados |

`L1`–`L3` se imponen **también donde no hay datos**, hasta el horizonte de
decisión completo (45 vueltas por defecto), no solo hasta donde llegó el stint
más largo observado. Ahí está la ventaja sobre una caja negra: fuera de su
distribución de entrenamiento, una LSTM no tiene nada que la ate a la
termodinámica.

### Condiciones iniciales duras

Se imponen por transformación de salida, no como términos de pérdida:

```
θ(τ) = θ₀ + τ · N₀(x)          ⟹  θ(0) = θ₀ exacto
d(τ) = τ · softplus(N₁(x))     ⟹  d(0) = 0 exacto y d ≥ 0 siempre
```

Esto elimina dos términos de la pérdida y con ellos el problema de balancear sus
pesos, que es la principal fuente de fallos de convergencia en PINNs.

### Problema inverso

Los coeficientes físicos (`ζ, h₀, h₁, k_w, m, E_a, κ, γ₁, γ₂`) no se conocen: se
estiman **junto con** los pesos de la red como `dde.Variable`. Se parametrizan en
logaritmo, así que son positivos por construcción, que es lo que exige su
significado físico.

### Las dos degeneraciones del problema (y cómo se cierran)

Este modelo tiene **dos direcciones exactamente degeneradas**. Ignorarlas no
produce un ajuste mediocre: produce divergencia.

**1. Escala de temperatura.** Si `A_gen`, la escala de `θ` y `E_a` fueran todos
libres, duplicar `A_gen` y partir `E_a` a la mitad dejaría el desgaste
invariante. Se cierra **fijando `A_gen`**, que ancla la escala térmica. Con
`--source synthetic` la supervisión débil de temperatura (`L5`) ayuda además;
con datos reales `L5` se desactiva automáticamente, porque la temperatura interna
del neumático no es pública.

**2. Escala de desgaste.** Ésta es la peligrosa:

```
d → ε·d ,  γ₁ → γ₁/ε ,  γ₂ → γ₂/ε^p     deja δ exactamente igual
```

En el banco sintético la rompen dos cosas: el proxy térmico y los stints que
llegan a saturar en `d = 1`. **Con datos reales no existe ninguna de las dos**, y
el optimizador se va por esa dirección hasta desbordar. Ocurrió literalmente: un
entrenamiento sobre Monza + Hungría terminó con `γ₂ = 2.5 × 10¹³`, `k_w = 0.026`
y un RMSE de prueba de 5.8 × 10⁹ s — con una **pérdida de entrenamiento baja**
(0.117), porque en la dirección degenerada el ajuste es perfecto.

Se cierra acotando `γ₁ ∈ [0.2, 4.0] s` y `γ₂ ∈ [0.2, 6.0] s` mediante una
sigmoide (`gamma1_bounds`, `gamma2_bounds` en `PhysicsConfig`). No es una
precaución numérica: es afirmar algo que sí sabemos con certeza, que un neumático
destruido cuesta unos segundos por vuelta y no millones.

> La lección general: en una PINN con problema inverso, **una pérdida de
> entrenamiento baja no garantiza nada** si el modelo tiene direcciones
> degeneradas. Hay que enumerarlas y cerrarlas explícitamente.

---

## 3. Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

DeepXDE necesita saber el backend:

```bash
set DDE_BACKEND=pytorch
```

---

## 4. Uso

Entrenamiento sobre el banco sintético (no necesita red ni API):

```bash
python run_train.py --source synthetic --stints 64
```

Validación rápida del pipeline (unos 2 minutos):

```bash
python run_train.py --source synthetic --quick
```

Entrenamiento sobre telemetría real. **Usa varias carreras**: en una sola, las
variables de contexto casi no varían (mismo circuito, mismo clima), así que la
red solo alcanza a ver el efecto del compuesto y del tiempo.

```bash
python run_train.py --source fastf1 --year 2023 --gp Monza Hungary Bahrain Spain
```

Inferencia y medición de latencia con el modelo ya entrenado:

```bash
python run_infer.py --compound SOFT --track-temp 0.8 --load 1.2
```

### Salidas en `outputs/`

| Archivo | Contenido |
|---|---|
| `01_stints.png` | curvas predichas vs observadas, stints de prueba |
| `02_extrapolacion.png` | comportamiento más allá de los datos: PINN vs caja negra |
| `03_estados_latentes.png` | `θ` y `d` reconstruidos vs verdad sintética |
| `04_parametros.png` | convergencia del problema inverso |
| `05_perdida.png` | evolución de cada término de la pérdida |
| `06_mapa_cliff.png` | mapa de decisión: vuelta del cliff por compuesto y condiciones |
| `report.txt` | tabla de métricas y recuperación de parámetros |
| `pinn_weights.pt`, `pinn_params.json` | modelo entrenado para inferencia |

---

## 5. El banco sintético

`data_synthetic.py` integra (E1)–(E2) con parámetros conocidos
(`physics.GROUND_TRUTH`) y añade ruido de medición. Sirve para dos cosas:

1. Que el pipeline corra sin depender de la red ni de FastF1.
2. **Validar el problema inverso**: la PINN arranca de valores iniciales
   distintos y debe *recuperar* los parámetros verdaderos usando solo la pérdida
   de ritmo observada. Con datos reales esa verificación es imposible porque no
   existe la verdad de referencia.

El generador imita la decisión de un estratega: el stint se corta entre dos y
cinco vueltas **después** del cliff. Ningún equipo rueda con el neumático
agotado, pero tampoco para en el instante exacto: pierde vueltas decidiendo,
esperando hueco en boxes o cubriendo a un rival.

Ese margen importa más de lo que parece. Son las únicas vueltas que informan
sobre el régimen `d → 1`, del que dependen `γ₂` y la escala de `k_w`. Cortando
el stint justo en el cliff, el error medio de recuperación de parámetros era
**11.8 %** (`γ₂` al 62 %); dejando esas pocas vueltas extra baja a **2.1 %**
(`γ₂` al 4.7 %).

---

## 6. Datos reales: qué se observa y qué no

**Nada de lo que el modelo necesita es directamente observable.** La temperatura
interna, la carga vertical y el estado de la banda son datos propietarios de cada
equipo. Lo público es la telemetría de a bordo y los tiempos por vuelta.
`data_fastf1.py` cierra la brecha:

- **`q_fric`** — potencia friccional específica, integrando `|a|·v` sobre la
  vuelta. Es el término de generación de calor de (E1).
- **`load`** — aceleración total media en g. Es el término de Archard de (E2).
- **`speed`** — velocidad media, gobierna el enfriamiento convectivo.

La **aceleración lateral no viene en la telemetría**: se reconstruye derivando
dos veces la trayectoria GPS, con suavizado Savitzky-Golay previo porque la doble
derivada numérica amplifica el ruido de muestreo.

El observable de degradación es la pérdida de ritmo **corregida por
combustible**: un coche se aligera ~100 kg durante la carrera y eso vale más de
un segundo por vuelta. Sin corregirlo, el aligeramiento enmascara por completo la
degradación.

### El origen de la degradación es el pico, no la primera vuelta

Un juego nuevo sale frío y se hace *más rápido* durante dos o tres vueltas antes
de empezar a caer. El modelo es monótono por construcción, así que no puede
representar esa fase de calentamiento. La solución es anclar `d = 0` en el **pico
de rendimiento** del stint y descartar las vueltas anteriores.

No es un detalle cosmético. Sin ese anclaje, cada stint arranca con ~0.5 s de
desfase sistemático entre lo observado y lo que el modelo puede predecir, y la
red compensa ese conflicto degenerando los parámetros físicos: en una prueba
sobre Monza + Hungría, `E_a` colapsó a 0.03 (sin activación térmica), `m` a 0.13
(sin dependencia de la carga) y `γ₂` se disparó a 8.35. Corregir el anclaje bajó
el RMSE de la PINN de **2.82 s a 0.57 s** y las violaciones de monotonía de
**16.4 % a 0 %**.

Filtros de calidad aplicados: solo bandera verde (`TrackStatus == 1`), sin
vueltas de entrada/salida de boxes, `IsAccurate`, sin vueltas borradas, y solo
juegos nuevos (`FreshTyre`), porque `d(0) = 0` únicamente vale para un neumático
nuevo.

Los proxies se adimensionalizan contra **referencias fijas** (`q_fric_ref`,
`load_ref`, `speed_ref` en `DataConfig`), no contra la mediana de cada sesión.
Normalizar cada carrera contra sí misma dejaría a Monza y a Hungría ambas en 1.0
y borraría justo la variación entre circuitos que el modelo necesita. Las
constantes están calibradas sobre carreras de 2023 (Monza 1877 W/kg · 3.39 g ·
66.4 m/s; Hungría 1968 · 4.35 · 51.6).

### Limitaciones conocidas con datos reales

- **Contexto constante por stint.** El modelo supone `q`, `λ`, `v` constantes
  dentro del stint y usa su mediana. La variación vuelta a vuelta queda absorbida
  por el residuo de datos. Es la simplificación que hace tratable el operador
  paramétrico.
- **`load` es un proxy relativo, no una medida.** Sale de derivar dos veces la
  trayectoria GPS, y su magnitud absoluta (≈3–4 g de media) está por encima de lo
  que mediría un acelerómetro real. Lo que importa es que sea monótono en la
  carga real y que discrimine circuitos, y ambas cosas se cumplen; el valor
  absoluto se cancela al dividir por `load_ref`.
- **Evolución de pista.** El circuito engoma y se hace más rápido durante la
  carrera. Ese efecto no está separado de la degradación y sesga a la baja la
  pendiente estimada.
- **Tráfico y aire sucio.** Se mitigan con el filtro `max_delta_s`, no se
  eliminan.
- **`γ₂` está débilmente identificado.** Solo entra en juego cuando `d → 1`, y
  los equipos paran antes de llegar ahí, así que hay pocos datos en ese régimen.
  Es una limitación real del problema, no del método: la información sobre el
  colapso final simplemente no está en los datos de carrera. En el banco
  sintético, ~1 de cada 4 stints llega al cliff, y `γ₂` es el parámetro con peor
  recuperación por esa misma razón.
- **`k_w` y `γ₁` se compensan parcialmente.** `δ ≈ γ₁·d` y `d` escala con `k_w`,
  así que subestimar uno y sobrestimar el otro deja la curva de ritmo casi igual.
  Lo que rompe la degeneración es la saturación `(1−d)`: cuando un stint se acerca
  a `d = 1`, la escala de `d` queda anclada. Otra razón por la que los stints
  largos son valiosos.

---

## 7. Resultados

Banco sintético, 64 stints (48 entrenamiento / 16 prueba, partición por stint),
15 000 iteraciones Adam + 3 000 L-BFGS, ~30 min en CPU:

| Modelo | RMSE [s] | MAE [s] | MaxErr [s] | Cliff MAE | Cliff detectados | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|---|---|---|
| **PINN** | **0.066** | **0.052** | **0.180** | **0.50** | **2/2** | **0.0 %** | **0.0 %** |
| Lineal (clásico) | 0.247 | 0.167 | 1.078 | — | 0/2 | 15.0 % | 12.5 % |
| LSTM (caja negra) | 0.081 | 0.058 | 0.641 | 1.00 | 2/2 | 1.3 % | 11.5 % |

La LSTM es competitiva **dentro** del rango observado (0.081 vs 0.066) pero se
rompe al extrapolar: 11.5 % de las vueltas predice que el neumático recupera
agarre. El modelo lineal ni siquiera detecta el cliff, porque no puede
representarlo. La PINN es la única con 0 % de violaciones en ambos regímenes.

> El `Cliff MAE` se calcula sobre los **2 stints de prueba (de 16) que llegan al
> cliff**. Es una muestra pequeña y el número no debe leerse como un intervalo
> estrecho: solo ~1 de cada 4 stints alcanza ese régimen, por la misma razón por
> la que `γ₂` es el parámetro peor identificado.

### Recuperación de parámetros físicos

La PINN arranca de valores iniciales deliberadamente distintos y recupera los
verdaderos usando **solo la pérdida de ritmo observada**:

| Parámetro | Estimado | Verdadero | Error |
|---|---|---|---|
| `ζ` (acople cliff) | 0.879 | 0.900 | 2.3 % |
| `h₀` (enfriamiento base) | 5.917 | 6.000 | 1.4 % |
| `h₁` (conv. forzada) | 3.962 | 4.000 | 0.9 % |
| `k_w` (desgaste) | 0.534 | 0.550 | 2.9 % |
| `m` (exp. de carga) | 1.544 | 1.500 | 2.9 % |
| `E_a` (activación térmica) | 0.941 | 0.950 | 1.0 % |
| `κ` (dureza compuesto) | 0.846 | 0.850 | 0.5 % |
| `γ₁` (pérdida lineal) | 1.382 | 1.350 | 2.3 % |
| `γ₂` (magnitud cliff) | 2.722 | 2.600 | 4.7 % |
| | | **media** | **2.1 %** |

En `04_parametros.png` se ve que `ζ`, `h₀` y `h₁` se quedan estancados durante
las 15 000 iteraciones de Adam y solo saltan a su valor verdadero en la fase
L-BFGS. Los parámetros térmicos son los peor condicionados del problema —solo
influyen en `δ` a través de dos capas de composición— y necesitan un optimizador
de segundo orden. Es la razón concreta por la que el régimen Adam → L-BFGS no es
opcional aquí.

### Sobre telemetría real (Monza + Hungría 2023, 38 stints)

Aquí el resultado es **peor**, y conviene decirlo sin adornos:

| Modelo | RMSE [s] | MAE [s] | Cliff detectados | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|---|
| PINN | 0.697 | 0.476 | 9/9 | 18.3 % | 9.6 % |
| Lineal (clásico) | **0.529** | **0.424** | 0/9 | 0.0 % | 0.0 % |
| LSTM (caja negra) | 0.522 | 0.414 | 4/9 | 0.0 % | 5.6 % |

**La PINN no le gana a los baselines con dos carreras.** El motivo es que con
38 stints, un suelo de ruido de ~0.5 s por vuelta y muy poca variación de
condiciones, la curva de degradación observada es casi lineal en el rango
medido, y ahí un ajuste lineal es difícil de batir. La PINN paga el precio de
estar restringida sin poder cobrar todavía el beneficio.

Los parámetros ajustados sí son físicamente razonables (`k_w = 0.864`,
`κ = 0.963`) y ninguno queda pegado a una cota, que era el síntoma de la
degeneración. Pero el residuo de la EDO de desgaste se queda en ~0.15 —dos
órdenes de magnitud peor que en el banco sintético—, y de ahí vienen las
violaciones de monotonía que quedan.

Lo honesto es concluir que **el camino de datos reales está validado
mecánicamente pero no científicamente**: funciona de punta a punta, produce
parámetros interpretables, y necesita bastantes más carreras para que la física
empiece a rendir. Es la continuación natural del trabajo.

### Latencia de inferencia

Predecir un stint completo de 45 vueltas cuesta **0.42 ms de media, 1.47 ms en
p95** (CPU, 500 repeticiones). El presupuesto de 500 ms del proyecto lo consume
íntegramente el transporte, no el modelo: la red paramétrica resuelve la EDO de
un solo paso forward.

---

## 8. Evaluación

Tres dimensiones, porque responden a preguntas distintas:

- **RMSE / MAE** sobre la pérdida de ritmo: cuánto se equivoca en la vuelta que
  está viendo.
- **Error de vuelta del cliff**: cuánto se equivoca en la única predicción que
  cambia una decisión de estrategia. Un modelo puede tener buen RMSE global y aun
  así fallar el cliff por cinco vueltas.
- **Violaciones de monotonía**: con qué frecuencia predice que el neumático
  *recupera* agarre. Es físicamente imposible y ninguna métrica de error la
  penaliza, así que se mide aparte. Lo correcto es 0 %.

Los baselines son los dos extremos del estado del arte descritos en el proyecto:
`LinearDegBaseline` (el modelo empírico de los equipos, ampliado con término
cuadrático e interacciones para no hacerlo de paja) y `LSTMBaseline` (la caja
negra recurrente).

La partición es **por stint completo**, nunca por vuelta: partir por vuelta
filtraría información del mismo stint entre train y test.

---

## 9. Estructura

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

---

## 10. Referencias

- Raissi, Perdikaris & Karniadakis (2019). *Physics-informed neural networks*.
  Journal of Computational Physics, 378, 686–707.
- Lu, Meng, Mao & Karniadakis (2021). *DeepXDE: A deep learning library for
  solving differential equations*. SIAM Review, 63(1), 208–228.
- Oehrly, M. *FastF1: A Python package for F1 telemetry and timing data*.
