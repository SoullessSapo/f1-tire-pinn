# F1 Tire PINN — Documentación del proceso completo

**Autor del proyecto:** Esteban Valencia
**Alcance implementado:** la red neuronal (PINN) y el uso de la librería DeepXDE.
**Explícitamente fuera de alcance:** toda la capa cloud de AWS (Kinesis, microservicios, almacenamiento).

Este documento explica de dónde salen los datos, qué herramientas se usaron, cómo
se llegó a cada decisión de diseño, y —sobre todo— los cuatro problemas serios que
aparecieron durante el desarrollo y cómo se diagnosticaron. Esa última parte es la
más útil para el informe, porque son problemas de fondo del método, no de código.

---

## 1. Punto de partida

El PDF del proyecto propone una arquitectura cloud dirigida por eventos que ingiere
telemetría de F1 y la procesa con una Physics-Informed Neural Network para predecir
la degradación de neumáticos y la vuelta del *cliff*. El encargo aquí fue construir
**solo la parte de modelado**: el sistema físico, la red, el entrenamiento offline,
los modelos de comparación y la evaluación.

El PDF fijaba tres cosas que se respetaron:

1. **PINN con DeepXDE**, para no calcular gradientes a mano.
2. **Entrenamiento desacoplado de la inferencia** — se entrena offline, se infiere en vivo.
3. **Criterios de evaluación**: RMSE frente a un baseline clásico y a una LSTM, y
   latencia por debajo de 500 ms.

---

## 2. Herramientas y por qué cada una

| Herramienta | Versión | Para qué |
|---|---|---|
| **DeepXDE** | 1.15.0 | Framework de PINNs. Aporta la geometría del dominio, el muestreo de puntos de colocación, la diferenciación automática de los residuos (`dde.grad.jacobian`), las condiciones de contorno tipo `PointSetOperatorBC` y el problema inverso vía `dde.Variable`. |
| **PyTorch** | 2.13.0 (CPU) | Backend de DeepXDE. Se eligió sobre TensorFlow porque el proyecto no necesita GPU y la instalación CPU es mucho más liviana. |
| **FastF1** | 3.8.3 | Telemetría y cronometraje oficiales de F1. Es la fuente que el PDF ya proponía. |
| **NumPy / SciPy / pandas** | 2.5 / 1.17 / 2.3 | Integrador RK4 de referencia, filtro Savitzky-Golay, manejo de vueltas. |
| **Matplotlib** | 3.10 | Las seis figuras del informe. |
| **poppler (`pdftotext`)** | — | Extraer el texto del PDF del proyecto al inicio. |
| **ruff** | — | Verificación de imports muertos y errores de sintaxis. |

Todo corre en un entorno virtual aislado (`.venv`) con Python 3.13.2, sobre CPU.
No hace falta GPU: el entrenamiento completo son ~30 minutos.

---

## 3. De dónde salen los datos

### 3.1 El problema de fondo

**Nada de lo que el modelo necesita es observable públicamente.** La temperatura
interna del neumático, la carga vertical y el espesor restante de la banda de
rodadura son datos propietarios de cada equipo. Lo público es:

- Telemetría de a bordo: velocidad, acelerador, freno, marcha, RPM, DRS (~10 Hz).
- Posición GPS del coche: X, Y, Z (~4 Hz).
- Cronometraje: tiempo por vuelta, número de stint, compuesto, edad del neumático.
- Meteorología de sesión: temperatura de aire y de pista.

Toda la ingeniería de características consiste en cerrar esa brecha.

### 3.2 Variables proxy construidas

| Variable | Cómo se calcula | Qué representa en el modelo |
|---|---|---|
| `q_fric` | Integral de `\|a\| · v` sobre la vuelta, dividida por el tiempo de vuelta | Potencia friccional específica = término de generación de calor |
| `load` | Media de `√(a_lat² + a_long²)` en g | Carga mecánica = término de Archard |
| `speed` | Velocidad media | Gobierna el enfriamiento convectivo |
| `track_temp` | Interpolación de la meteorología en el instante de la vuelta | Temperatura ambiente del caucho |
| `compound` | 0 = blando, 0.5 = medio, 1 = duro | Resistencia del compuesto |

**La aceleración lateral no viene en la telemetría.** Se reconstruye derivando dos
veces la trayectoria GPS: si `(x', y')` es la velocidad planar y `(x'', y'')` la
aceleración, la componente normal es `|x'y'' − y'x''| / √(x'² + y'²)`. Como la doble
derivada numérica amplifica el ruido de muestreo, se suaviza antes con
Savitzky-Golay.

**Detalle que costó una iteración:** la ventana de suavizado se fija en *segundos*,
no en número de muestras. FastF1 fusiona telemetría de coche (~10 Hz) con posición
GPS (~4 Hz) interpolando, así que la frecuencia efectiva varía entre vueltas. Una
ventana fija en muestras aplicaría un filtro distinto en cada caso.

### 3.3 El observable de degradación

Lo único que se puede medir de la degradación es cuánto ritmo se pierde:

```
δ(vuelta) = tiempo_corregido(vuelta) − tiempo_corregido(vuelta_de_referencia)
```

Con dos correcciones obligatorias:

- **Combustible.** Un coche se aligera ~100 kg durante la carrera, lo que vale más
  de un segundo por vuelta. Sin corregirlo, el aligeramiento *cancela visualmente*
  la degradación. Se corrige con `t − k_fuel · (vueltas_totales − vuelta)`, con
  `k_fuel = 0.055 s/vuelta`.
- **Filtros de calidad.** Solo bandera verde (`TrackStatus == 1`, para excluir
  safety cars), sin vueltas de entrada/salida de boxes, solo vueltas marcadas como
  `IsAccurate`, sin vueltas borradas por la FIA, y solo juegos nuevos (`FreshTyre`),
  porque `d(0) = 0` solo tiene sentido para un neumático nuevo.

De Monza + Hungría 2023 sobreviven **38 stints con 803 vueltas útiles**.

### 3.4 El banco sintético

Además de los datos reales hay un generador que integra las ecuaciones con
parámetros **conocidos** y añade ruido de medición. Sirve para dos cosas:

1. El pipeline corre sin depender de red ni de la API.
2. **Validar el problema inverso**: la PINN arranca de valores iniciales distintos
   y debe *recuperar* los verdaderos usando solo `δ`. Con datos reales esa
   verificación es imposible: no existe la verdad de referencia.

---

## 4. Cómo se llegó al modelo físico

Este fue el trabajo con más iteraciones. El sistema final es:

```
(E1)  dθ/dτ = A_gen · q · (1 + ζ·d)  −  (h₀ + h₁·v) · θ        balance térmico
(E2)  dd/dτ = k_w · λ^m · exp(E_a·(θ + T_trk) − κ·c) · (1 − d)  desgaste
      δ(τ)  = γ₁·d + γ₂·d^p                                     observable
```

Donde `θ` es el exceso térmico de la banda y `d` la fracción de banda consumida.
Ambos son **estados latentes**: nunca se observan.

### Iteración 1 — el sistema básico

Se empezó con dos ecuaciones desacopladas: un balance térmico de capacitancia
concentrada (generación friccional menos decaimiento exponencial) y una ley de
desgaste tipo Archard con activación térmica de Arrhenius. Es lo estándar.

**Problema encontrado al probarlo:** con condiciones exigentes, `d` alcanzaba 3.24.
No físico: no se puede consumir el 324 % de la banda de rodadura.

### Iteración 2 — saturación

Se añadió el factor **`(1 − d)`** a (E2). Acota `d ∈ [0,1]` **estructuralmente**:
la cota física la impone la propia ecuación diferencial, no una penalización en la
función de pérdida. Es una ventaja doble, porque además la monotonía del desgaste
(`dd/dτ ≥ 0`) queda implicada por el residuo de la EDO.

**Problema encontrado:** con la saturación, el *cliff* desapareció. Ningún stint
generaba la caída brusca característica.

### Iteración 3 — el cliff como fenómeno emergente

Aquí está la decisión de diseño más importante del proyecto.

La opción fácil era imponer el cliff a mano subiendo `γ₂`. Se descartó porque
convierte el cliff en un artefacto del observable, no en física.

La opción correcta es preguntarse **por qué existe el cliff realmente**: al
adelgazarse la banda de rodadura, la misma energía friccional se deposita sobre
menos masa de caucho, la temperatura superficial sube, y por Arrhenius el desgaste
se acelera, lo que adelgaza más la banda. Es una **realimentación positiva**.

Eso se modela con el factor **`(1 + ζ·d)`** en (E1). Con él, el cliff *emerge* de
la dinámica acoplada. Y es exactamente la clase de restricción termodinámica que
una LSTM no tiene forma de conocer, así que también es el argumento central de por
qué una PINN aporta algo aquí.

### Iteración 4 — calibración

Se barrió `k_w ∈ {0.45, 0.55, 0.65}` buscando una dispersión realista de stints.
Con `k_w = 0.55` unos llegan al cliff entre las vueltas 8 y 18 y otros aguantan
todo el horizonte sin llegar — que es lo que pasa en una carrera real.

---

## 5. Decisiones de diseño de la red

### 5.1 PINN paramétrica, no por stint

Una PINN clásica resuelve **una** trayectoria: la red toma `t` y devuelve el estado.
Eso obligaría a reentrenar por cada stint, lo cual es **inservible para inferencia
en vivo** y rompería el requisito de latencia del proyecto.

La red aquí es un **operador solución**:

```
N(τ, q, λ, v, T_trk, c) → (θ, d)
```

Aprende de una vez la familia completa de soluciones de la EDO para todo el rango
de condiciones de carrera. Predecir un stint nuevo es **un paso forward**: sin
reentrenar y sin integrar nada. Medido: **0.42 ms de media, 1.47 ms en p95** para
45 vueltas. El presupuesto de 500 ms del proyecto lo consume íntegramente el
transporte, no el modelo.

La contrapartida es una simplificación explícita: el contexto `(q, λ, v)` se toma
constante dentro del stint (su mediana). La variación vuelta a vuelta queda
absorbida por el término de datos.

### 5.2 Condiciones iniciales duras, no como pérdida

En lugar de añadir términos de pérdida para `θ(0) = θ₀` y `d(0) = 0`, se imponen
por **transformación de salida**:

```
θ(τ) = θ₀ + τ · N₀(x)          ⟹  θ(0) = θ₀ exacto
d(τ) = τ · softplus(N₁(x))     ⟹  d(0) = 0 exacto y d ≥ 0 siempre
```

Esto elimina dos términos de la función de pérdida, y con ellos el problema de
balancear sus pesos — que es la principal fuente de fallos de convergencia en PINNs.
Se implementa con `net.apply_output_transform()` de DeepXDE.

### 5.3 Dónde se impone la física

Los residuos de las EDOs se imponen sobre **todo el hipercubo de condiciones**, y
en `τ` **hasta el horizonte de decisión completo** (45 vueltas), no solo hasta donde
llegó el stint más largo observado.

Esa elección es deliberada: **el dominio físico lo define la pregunta que se le va
a hacer al modelo, no la longitud de los datos que se alcanzaron a recoger.** Es
precisamente lo que permite que la PINN extrapole con sentido donde una caja negra
no tiene nada que la ate.

### 5.4 Problema inverso

Los nueve coeficientes físicos se estiman junto con los pesos de la red, usando
`dde.Variable`. Se parametrizan en **logaritmo**, de modo que son positivos por
construcción — que es lo que exige su significado físico.

### 5.5 Función de pérdida

| Término | Qué impone | Dónde |
|---|---|---|
| `L1` | residuo de (E1) | todo el hipercubo |
| `L2` | residuo de (E2) | todo el hipercubo |
| `L3` | cota `d ≤ d_max` | todo el hipercubo |
| `L4` | ajuste a la pérdida de ritmo medida | puntos observados |
| `L5` | proxy de temperatura (solo sintético) | puntos observados |

Régimen de entrenamiento: **Adam (15 000) → L-BFGS (3 000)**. No es opcional: en la
figura de convergencia se ve que `ζ`, `h₀` y `h₁` se quedan estancados durante todo
Adam y solo saltan a su valor verdadero en L-BFGS. Los parámetros térmicos son los
peor condicionados —solo influyen en `δ` a través de dos capas de composición— y
necesitan un optimizador de segundo orden.

---

## 6. Los cuatro problemas serios (y cómo se diagnosticaron)

Esta es la parte más valiosa del proceso. Ninguno era un bug de programación: los
cuatro eran problemas de fondo del planteamiento.

### Problema 1 — Los stints se cortaban demasiado pronto

**Síntoma:** el problema inverso recuperaba los parámetros con un 11.8 % de error
medio, pero `γ₂` fallaba por un **62 %** y `k_w` y `γ₁` se compensaban entre sí
(uno subestimado un 18 %, el otro sobrestimado un 16 %).

**Diagnóstico:** el generador cortaba el stint justo en el cliff. Solo 3 de 48
stints llegaban a tener un cliff detectable, y solo 1 alcanzaba `d > 0.9`. Pero
`γ₂` **solo entra en juego cuando `d → 1`**: sin datos en ese régimen, no hay
información para estimarlo.

**Solución:** dejar 2–5 vueltas *después* del cliff. Es además más realista —ningún
equipo para en el instante exacto: pierde vueltas decidiendo, esperando hueco en
boxes o cubriendo a un rival.

**Resultado:** error medio de **11.8 % → 2.1 %**, y `γ₂` de 62 % a 4.7 %.

### Problema 2 — La normalización borraba la señal entre circuitos

**Síntoma:** entrenando con una sola carrera, las variables de contexto casi no
variaban (carga entre 0.95 y 1.04).

**Diagnóstico:** los proxies se estaban normalizando por la **mediana de la propia
sesión**. Eso deja a Monza y a Hungría **ambas en 1.0**, borrando exactamente la
variación entre circuitos que el modelo necesita aprender.

**Solución:** normalizar contra **referencias físicas fijas** (`q_fric_ref = 1900`
W/kg, `load_ref = 3.8` g, `speed_ref = 58` m/s), calibradas midiendo carreras reales.

**Resultado:** con dos carreras, la carga pasó a variar entre 0.84 y 1.59 — señal
real. Se añadió además soporte para entrenar sobre varias carreras a la vez.

### Problema 3 — El origen de la degradación estaba mal definido

**Síntoma:** con datos reales, la PINN daba **RMSE 2.82 s** (los baselines, 0.59) y
un 16.4 % de violaciones de monotonía. Los parámetros estaban degenerados: `E_a`
colapsó a 0.03 (sin activación térmica), `m` a 0.13 (sin dependencia de la carga).

**Diagnóstico:** se midió la `δ` de la primera vuelta válida de cada stint: promedio
**0.473 s**. Es decir, el stint empezaba ya medio segundo por encima de su
referencia. La causa es física: **un juego nuevo sale frío y se hace más rápido
durante dos o tres vueltas** antes de empezar a caer. El modelo es monótono por
construcción y no puede representar ese calentamiento, así que cada stint arrancaba
con un conflicto sistemático que la red compensaba distorsionando la física.

**Solución:** anclar `d = 0` en el **pico de rendimiento** del stint, no en su
primera vuelta, y descartar las vueltas de calentamiento.

**Resultado:** RMSE **2.82 → 0.567 s**, violaciones **16.4 % → 0 %**.

### Problema 4 — Degeneración exacta de escala (el más grave)

**Síntoma:** el entrenamiento completo sobre datos reales **divergió**:
`γ₂ = 2.5 × 10¹³`, `k_w = 0.026`, RMSE de prueba de **5.8 × 10⁹ s**. Y lo más
llamativo: **la pérdida de entrenamiento era baja** (0.117).

**Diagnóstico:** el modelo tiene una dirección exactamente degenerada:

```
d → ε·d ,  γ₁ → γ₁/ε ,  γ₂ → γ₂/ε^p     deja δ exactamente igual
```

En el banco sintético la rompen dos cosas: el proxy térmico y los stints que llegan
a saturar en `d = 1`. **Con datos reales no existe ninguna de las dos**, así que el
optimizador se desliza por esa dirección hasta desbordar.

**Intento 1:** acotar `γ₁ ∈ [0.2, 4.0]` y `γ₂ ∈ [0.2, 6.0]` con una sigmoide.
Detuvo el desbordamiento (RMSE 1.56) pero **ambas quedaron pegadas a sus topes** —
señal de que el empuje seguía ahí.

**Conclusión de fondo:** la escala absoluta de `d` **no es identificable desde datos
de carrera**. El único observable es `δ`, y lo único que ancla `d` es la saturación
en `d = 1`, a la que solo se llega destruyendo el neumático. Los equipos paran mucho
antes. **No es un problema del método: la información sencillamente no está en los
datos.**

**Solución final:** la ley termo-mecánica se calibra en el banco físico, donde sí
existe verdad de referencia, y sobre telemetría real se ajustan **solo `k_w` y `κ`**,
las dos cantidades que de verdad cambian entre circuitos y lotes de neumático.
"Un segundo de pérdida de ritmo equivale a este desgaste" es una afirmación de
calibración, no algo que los tiempos por vuelta puedan responder.

**Resultado:** parámetros físicamente razonables (`k_w = 0.864`, `κ = 0.963`) y
ninguno pegado a una cota.

> **La lección general, y probablemente lo más citable del proyecto:** en una PINN
> con problema inverso, **una pérdida de entrenamiento baja no garantiza nada** si
> el modelo tiene direcciones degeneradas. Hay que enumerarlas explícitamente y
> cerrarlas una por una.

---

## 7. Resultados

### 7.1 Banco sintético (64 stints, 48 entrenamiento / 16 prueba)

| Modelo | RMSE [s] | MAE [s] | MaxErr [s] | Cliff MAE | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|---|---|
| **PINN** | **0.066** | **0.052** | **0.180** | **0.50** | **0.0 %** | **0.0 %** |
| Lineal (clásico) | 0.247 | 0.167 | 1.078 | — | 15.0 % | 12.5 % |
| LSTM (caja negra) | 0.081 | 0.058 | 0.641 | 1.00 | 1.3 % | 11.5 % |

La LSTM es competitiva **dentro** del rango observado, pero se rompe al extrapolar:
un 11.5 % de las vueltas predice que el neumático **recupera agarre**, lo cual es
termodinámicamente imposible. El modelo lineal ni siquiera detecta el cliff. **La
PINN es la única con 0 % de violaciones en ambos regímenes.**

### 7.2 Recuperación de parámetros — error medio **2.1 %**

Los nueve coeficientes se recuperan usando **solo la pérdida de ritmo observada**:
`ζ` 2.3 %, `h₀` 1.4 %, `h₁` 0.9 %, `k_w` 2.9 %, `m` 2.9 %, `E_a` 1.0 %, `κ` 0.5 %,
`γ₁` 2.3 %, `γ₂` 4.7 %.

### 7.3 Telemetría real (Monza + Hungría 2023, 38 stints)

| Modelo | RMSE [s] | MAE [s] | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|
| PINN | 0.697 | 0.476 | 18.3 % | 9.6 % |
| Lineal (clásico) | **0.529** | **0.424** | 0.0 % | 0.0 % |
| LSTM (caja negra) | 0.522 | 0.414 | 0.0 % | 5.6 % |

**Aquí la PINN no le gana a los baselines, y conviene decirlo sin adornos.** Con 38
stints, un suelo de ruido de ~0.5 s por vuelta y poca variación de condiciones, la
curva observada es casi lineal en el rango medido, y ahí un ajuste lineal es difícil
de batir. La PINN paga el precio de estar restringida sin poder cobrar todavía el
beneficio.

Lo honesto es concluir que **el camino de datos reales está validado mecánicamente
pero no científicamente**: funciona de punta a punta y produce parámetros
interpretables, pero necesita bastantes más carreras para que la física rinda. Es la
continuación natural del trabajo.

### 7.4 Latencia

**0.42 ms de media, 1.47 ms en p95** para predecir un stint completo de 45 vueltas
en CPU. Tres órdenes de magnitud por debajo del presupuesto de 500 ms.

---

## 8. Limitaciones conocidas

- **Contexto constante por stint.** Se usa la mediana; la variación vuelta a vuelta
  queda en el residuo de datos.
- **Evolución de pista.** El circuito engoma y se hace más rápido durante la
  carrera. Ese efecto no está separado de la degradación y sesga la pendiente.
- **`load` es un proxy relativo, no una medida.** Su magnitud absoluta (~3–4 g de
  media) está por encima de lo que mediría un acelerómetro real. Lo que importa es
  que sea monótono en la carga real y que discrimine circuitos, y ambas se cumplen.
- **Solo juegos nuevos.** Los stints con neumáticos usados se descartan, porque
  `d(0) = 0` no aplica.
- **Tráfico y aire sucio** se mitigan con un filtro, no se eliminan.

## 9. Siguientes pasos naturales

1. **Entrenar sobre 8–10 carreras**, no dos. Es lo que más impacto tendría.
2. Modelar explícitamente la **evolución de pista** como un término separado.
3. Añadir la **fase de calentamiento** al modelo, para no tener que descartarla.
4. Un término de **efecto del piloto o del coche**, que hoy va todo al ruido.

---

## 10. Cómo reproducirlo

```bash
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

## 11. Referencias

- Raissi, Perdikaris & Karniadakis (2019). *Physics-informed neural networks*.
  Journal of Computational Physics, 378, 686–707.
- Lu, Meng, Mao & Karniadakis (2021). *DeepXDE: A deep learning library for solving
  differential equations*. SIAM Review, 63(1), 208–228.
- Oehrly, M. *FastF1: A Python package for F1 telemetry and timing data*.
