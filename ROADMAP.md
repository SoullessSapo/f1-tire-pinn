# De v0 a la versión actual

Esta rama es el punto de partida. La rama `main` es donde llegó el proyecto.
Este documento es el camino entre las dos: ocho pasos, cada uno motivado por un
problema concreto que la versión anterior no podía resolver.

El orden importa. Ninguno de estos pasos se decidió por adelantado; cada uno
salió de medir que algo no funcionaba.

---

## Panorama

| | v0 (esta rama) | `main` |
|---|---|---|
| Framework | PyTorch puro | DeepXDE |
| Física | 1 EDO (desgaste) | 2 EDO acopladas (térmica + desgaste) |
| Cliff | no representable | emerge de la realimentación |
| Entradas de la red | `(τ, q, λ, v, T_pista, c)` | `(τ, q, λ, v, T_pista, c)` |
| Parámetros estimados | 6 | 9 |
| Condición inicial | término de pérdida | transformación de salida (exacta) |
| Optimizador | Adam | Adam → L-BFGS |
| Datos | sintéticos **o FastF1** | FastF1, 11 carreras, 434 stints |
| Corrección de vuelta de carrera | pendiente lineal | spline lineal a trozos |
| Baselines | lineal | lineal + LSTM |
| Métricas | RMSE, MAE, MaxErr, monotonía | + vuelta del cliff |
| Tamaño de la red | 12 993 pesos | 13 058 pesos |
| Código | 912 líneas de código + 901 de comentarios | ~2 800 líneas |

Respecto a la v0 original han cambiado cuatro filas: las entradas de la red,
los parámetros estimados, los datos y el tamaño de la red. Es decir, **el paso 2
ya está dado y el paso 6 está a medias**.

Lo que sigue separando esta rama de `main` es, sobre todo, **la física**: aquí
no hay temperatura como estado propio, y por tanto no hay cliff. Ése es el
paso 1, y es el que de verdad importa.

---

## Paso 1 · Acoplar la temperatura, para que exista el cliff

**El problema.** El modelo de v0 solo puede doblarse en un sentido. Un neumático
real no se degrada suavemente hasta el final: en algún momento cae por un
precipicio. Con una sola EDO y un observable lineal, ese comportamiento
sencillamente no está en el espacio de funciones que el modelo puede expresar.

**El cambio.** Una segunda EDO, un balance térmico de capacitancia concentrada
—la misma ecuación de un condensador cargándose a través de una resistencia—
acoplada a la del desgaste:

```
(E1)  dθ/dτ = A_gen·q·(1 + ζ·d) − (h₀ + h₁·v)·θ
(E2)  dd/dτ = k_w·λ^m·exp(E_a·(θ + T_pista) − κ·c)·(1 − d)
```

Lo decisivo es el factor `(1 + ζ·d)` de E1: cuando la banda de rodadura
adelgaza, la misma energía de fricción se deposita en menos masa de goma, así
que la temperatura sube; y por Arrhenius, más temperatura significa más
desgaste. Es realimentación positiva, y **el cliff emerge de ella** en vez de
estar codificado a mano. El observable pasa a ser `δ = γ₁·d + γ₂·d⁸`, donde el
segundo término es despreciable hasta que `d → 1` y entonces domina.

Coste: se pierde la solución analítica. Por eso el integrador RK4 de
`physics.py` ya está aquí en v0, validado contra la forma cerrada mientras
todavía era barato comprobarlo.

## Paso 2 · Hacer la red paramétrica  ✅ HECHO EN ESTA RAMA

> Este paso ya está dado aquí. La red toma las seis entradas y estima seis
> constantes físicas. Se deja el texto porque explica **por qué** hacía falta.

**El problema.** La v0 original tomaba `(τ, c)`. Eso permite tres curvas, una por compuesto.
Pero la degradación depende de la temperatura de pista, de la carga del
circuito, de la energía de fricción. Un PINN de libro de texto resolvería una
trayectoria por cada combinación, o sea reentrenar para cada stint: media hora
de CPU por predicción, inútil en carrera.

**El cambio.** Las condiciones entran como **entradas de la red**:

```
N(τ, q, λ, v, T_pista, c) → (θ, d)
```

Con eso la red deja de aprender una curva y aprende un **operador solución**: la
familia entera de soluciones de la EDO en todo el rango de condiciones, de una
vez. Predecir un stint nuevo pasa a ser un forward pass. Medido en `main`:
**0,42 ms** para 45 vueltas.

## Paso 3 · Condiciones iniciales duras

**El problema.** En v0, `L_ic` es un término de pérdida. Se cumple
aproximadamente, compite por gradiente con los otros dos, y hay que elegirle un
peso a mano. Ese balanceo entre términos que compiten es la causa más común de
que un PINN no converja.

**El cambio.** Imponerlas por transformación de la salida:

```
θ(τ) = θ₀ + τ · N₀(x)        ⟹  θ(0) = θ₀ exactamente
d(τ) = τ · softplus(N₁(x))   ⟹  d(0) = 0 exactamente y d ≥ 0 siempre
```

Como `τ` multiplica la salida de la red, en `τ = 0` el término desaparece sea
cual sea la red: la condición se cumple con error cero en toda iteración. Dos
términos de pérdida eliminados y un problema de convergencia evitado.

## Paso 4 · El problema inverso completo, y sus degeneraciones

**El problema.** v0 estima un parámetro y sale bien. Con nueve aparece algo que
con uno no puede pasar: **direcciones degeneradas**. Si multiplicas `d` por ε y
divides `γ₁` por ε y `γ₂` por `ε^p`, el observable queda *idéntico*. Cualquier
punto de esa recta ajusta los datos igual de bien, y el optimizador se desliza
por ella hasta desbordar.

Pasó de verdad: una ejecución terminó con `γ₂ = 2,5 × 10¹³` y un RMSE de test de
5,8 × 10⁹ s — **con una pérdida de entrenamiento baja, 0,117**, porque a lo largo
de la dirección degenerada el ajuste es perfecto.

**El cambio.** Enumerar las degeneraciones y cerrarlas explícitamente: fijar
`A_gen` para anclar la escala térmica, y acotar `γ₁ ∈ [0,2; 4,0]` y
`γ₂ ∈ [0,2; 6,0]` con una sigmoide. No es cautela numérica: es afirmar algo que
sabemos, que un neumático destruido cuesta unos segundos por vuelta, no
millones.

> **La lección que generaliza, y es la más importante de todo el proyecto:** en
> un PINN con problema inverso, una pérdida de entrenamiento baja **no garantiza
> nada** si el modelo tiene direcciones degeneradas.

## Paso 5 · Adam → L-BFGS

**El problema.** En v0 Adam basta. Con los parámetros térmicos deja de bastar:
`ζ`, `h₀` y `h₁` se quedan clavados durante las 15 000 iteraciones de Adam y
**solo saltan a su valor correcto cuando entra L-BFGS**. Son los peor
condicionados del problema — solo llegan al observable a través de dos capas de
composición — y necesitan información de curvatura para moverse.

**El cambio.** Adam para explorar, L-BFGS (cuasi-Newton) para refinar. Con eso,
los nueve parámetros se recuperan con **1,3 % de error medio**.

## Paso 6 · Datos reales  ◐ A MEDIAS EN ESTA RAMA

> `download_data.py` ya baja telemetría real, reconstruye los proxies y
> escribe un CSV que `run.py --source csv` sabe entrenar. Lo que sigue siendo
> más simple aquí es la corrección de vuelta de carrera: esta rama ajusta una
> **pendiente lineal** por carrera, `main` ajusta un **spline lineal a trozos**
> de 4 nudos, porque la forma de esa curva no tiene por qué ser una recta.

**El problema.** Nada de lo que el modelo necesita es observable. Temperatura
interna, carga vertical y estado de la goma son propiedad de cada equipo. Lo
público es telemetría de a bordo y tiempos por vuelta.

**El cambio.** Un módulo entero (`data_fastf1.py`, ~490 líneas) para construir
proxies desde telemetría:

- `q_fric` — potencia de fricción específica, integrando `|a|·v` sobre la vuelta.
- `load` — aceleración total media en g.
- `speed` — velocidad media, que gobierna el enfriamiento convectivo.

La aceleración lateral no está en la telemetría: se reconstruye derivando dos
veces la trayectoria GPS, con filtro Savitzky-Golay previo y ventana definida en
segundos y no en muestras, porque FastF1 fusiona fuentes a frecuencias
distintas.

Más los filtros de calidad (bandera verde, sin vueltas de entrada/salida, juego
nuevo) y la corrección del efecto de vuelta de carrera: combustible y evolución
de pista, que **no son separables entre sí** y se estiman conjuntamente con una
regresión con spline.

## Paso 7 · Los fallos que solo aparecen con datos reales

Tres hallazgos que ninguna cantidad de trabajo sobre datos sintéticos habría
producido:

1. **El origen de la degradación es el pico, no la primera vuelta.** Un juego
   nuevo sale frío y va *más rápido* dos o tres vueltas antes de empezar a caer.
   El modelo es monótono y no puede representar eso. Anclar `d = 0` en el pico
   llevó el RMSE de **2,82 s a 0,57 s** y las violaciones de monotonía del
   **16,4 % al 0 %**.
2. **El detector de cliff medía ruido.** El criterio original (0,15 s/vuelta en
   un punto) disparaba en el **100 %** de curvas sin ningún cliff, en cuanto se
   le añadía el ruido de cronometraje real. Ahora exige 0,30 s/vuelta sostenido
   4 vueltas.
3. **Más de la mitad de la varianza del contexto era ruido.** El 53 % de la
   varianza de `q_fric` y el 61 % de `load` es intra-circuito, y esa parte
   correlaciona con la degradación a nivel de r ≈ 0,001. Colapsarlas a la
   mediana por carrera mejoró el PINN un 25 % y a los baselines nada — porque el
   PINN impone la física *en función del contexto*, así que una coordenada
   ruidosa corrompe la restricción en todas partes.

## Paso 8 · Evaluación seria

**El cambio.** Un baseline LSTM además del lineal (la caja negra, para tener los
dos extremos del estado del arte), la métrica de error en la vuelta del cliff, y
resultados sobre una temporada completa:

| Modelo | RMSE [s] | MAE [s] | Viol. extrap. |
|---|---|---|---|
| **PINN** | **0,694** | **0,494** | **5,7 %** |
| Lineal clásico | 0,801 | 0,602 | 26,0 % |
| LSTM caja negra | 0,746 | 0,541 | 11,2 % |

Y una tabla donde el PINN **pierde**, con solo dos carreras (36 stints): 1,172
de RMSE contra 0,539 del lineal. Está publicada en el README de `main` sin
adornos, porque el resultado honesto es que el método necesita datos suficientes
antes de que la física empiece a rendir.

---

## Qué queda por hacer

- Más temporadas. La ventaja de 2026 es real pero depende de una decisión de
  preprocesado; hace falta más datos para separarlas.
- `γ₂` está débilmente identificado: solo entra en juego con `d → 1` y los
  equipos paran antes. Es una limitación del problema, no del método.
- La capa cloud (inferencia en tiempo real durante la carrera) está fuera del
  alcance de ambas ramas.
