# Cómo ajustar el modelo

Esta rama ya tiene las **dos** ecuaciones acopladas. Lo que no tiene son los
valores correctos de las constantes: ésos los eliges tú. Este documento es el
manual para hacerlo con criterio en vez de a tientas.

La idea de fondo, y es la única que hay que interiorizar:

> **Hay dos ajustes distintos y van en este orden.**
> Primero decides **qué mundo describen las ecuaciones** (las 12 constantes
> físicas). Después decides **cómo entrena la red** (pesos, learning rate,
> iteraciones). Si haces el segundo antes que el primero, estás enseñándole a
> la red a reproducir un neumático que no se parece a un neumático, y ninguna
> cantidad de iteraciones arregla eso.

Para el primer ajuste hay una herramienta que no entrena nada y tarda segundos:

```bash
python tune.py
```

Para el segundo, `run.py` con sus banderas.

---

## 1. Qué ajustas y qué no

Las 12 constantes viven en `TireParams`, en `physics.py`.

| | constante | qué significa | quién la fija |
|---|---|---|---|
| **E2** | `kw` | ritmo de desgaste base, gomas frías, condiciones de referencia | la red |
| | `m` | exponente de carga (ley de Archard) | la red |
| | `Ea` | Arrhenius: cuánto acelera el calor **total** el desgaste | la red |
| | `Eq` | cuánto acelera la energía de fricción | la red |
| | `Ev` | cuánto frena el enfriamiento por velocidad | la red |
| | `kappa` | cuánto resiste un compuesto más duro | la red |
| **E1** | `A_gen` | calor generado por unidad de `q_fric` | **fija — tú** |
| | `zeta` | la realimentación: cuánto se recalienta una goma gastada | la red |
| | `h0` | enfriamiento que no depende de la velocidad | la red |
| | `h1` | enfriamiento extra por flujo de aire | la red |
| **E3** | `gamma1` | segundos perdidos por unidad de desgaste | **fija — tú** |
| | `gamma2` | segundos perdidos en el acantilado | la red (acotada) |

Las dos fijas no son comodidad. Son **direcciones degeneradas**: moverte a lo
largo de ellas cambia el estado latente y deja el observable idéntico, así que
ninguna cantidad de datos puede decidir dónde pararse, y el optimizador se
desliza por ahí hasta desbordar.

- `gamma1` ancla la escala de `d`. Duplicas `d`, divides `gamma1` entre dos, y
  `delta` no cambia.
- `A_gen` ancla la escala de `theta`. Multiplicas `theta` por 3, divides `Ea`
  entre 3, y E1 sigue cuadrando.

En `main` este mismo problema a tamaño completo fue el peor bug del proyecto:
una corrida terminó con `gamma2 = 2,5 × 10¹³` y un RMSE de test de 5,8 × 10⁹ s
**con la pérdida de entrenamiento baja**, 0,117 — porque a lo largo de una
dirección degenerada el ajuste de verdad es perfecto.

Los valores que hay ahora mismo en el archivo **son un punto de partida, no una
calibración**. `python tune.py` te dirá sin rodeos qué está mal con ellos.

---

## 2. El número que decide todo: β

Antes de tocar nada, esto. Se hace en papel y te ahorra corridas enteras.

La temperatura se estabiliza mucho más rápido de lo que se gasta la goma, así
que `theta` va siempre pegada a su valor de equilibrio para el `d` del momento:

```
theta_ss(d) = A_gen · q_fric · (1 + zeta·d) / (h0 + h1·speed)
```

Metes eso en E2 y todo se derrumba en una sola expresión:

```
dd/dτ = k₀ · exp(β·d) · (1 − d)

              Ea · A_gen · q_fric · zeta
con    β  =  ───────────────────────────
                   h0 + h1 · speed
```

Ahora deriva el lado derecho respecto de `d`. El desgaste **acelera** —que es
literalmente lo que es un acantilado— exactamente mientras

```
β · (1 − d) > 1        es decir        d < 1 − 1/β
```

De ahí salen las dos únicas conclusiones que importan:

- **β ≤ 1 → NO PUEDE HABER ACANTILADO.** Ni débil ni tardío: ninguno. El
  factor `(1 − d)` siempre gana, la curva sólo se aplana, y no hay
  entrenamiento que encuentre un fenómeno que las ecuaciones no saben escribir.
- **β > 1 →** la curva se empina hasta `d = 1 − 1/β` y luego se aplana. Más β,
  antes y más brusco.

`kw` y `gamma2` sólo cambian **cuándo** y **cuánto**. β decide **si**.

**El truco está en que β y la temperatura se mueven juntos.** β sube con
`A_gen` y con `zeta`, y esas dos también empujan `theta` hacia arriba. Si sólo
persigues β acabas con un acantilado precioso en un neumático que va a 200 °C
por encima de la pista. Hay que acertar las dos cosas a la vez, y eso es lo que
revisa el bloque DIAGNOSIS de `tune.py`.

### Por qué esto no es teoría

Lo medí en esta rama, con todo lo demás idéntico:

| | β = 0,71 | β = 1,94 |
|---|---|---|
| Subir `zeta` un 50 % mueve la curva de pace, como mucho | **0,469 s** | **1,445 s** |
| `zeta` aprendida (arranca en 0,50) | 0,350 → error **61 %** | 1,677 → error **24 %** |
| Error medio de las 10 constantes | 14,6 % | 8,6 % |

Con β por debajo de 1, `zeta` casi no deja huella en lo único que se mide, así
que no hay gradiente que seguir y se queda donde empezó. **No era el
optimizador: era la física.** Calibrar no es cosmética, decide si el problema
inverso tiene solución.

---

## 3. Qué mueve cada constante

Para que puedas girar un mando sabiendo qué esperar. `tune.py --sweep` hace
exactamente esto, una constante a la vez:

```bash
python tune.py --sweep zeta 0.5 4.0 8
python tune.py --sweep kw 0.10 0.60 6
```

| constante | sube β | sube `theta` frío | sube `theta` gastado | adelanta el acantilado | sube los segundos |
|---|---|---|---|---|---|
| `Ea` | sí | no | no | sí | sí |
| `A_gen` | sí | **sí** | sí | sí | sí |
| `zeta` | sí | no | **sí** | sí | sí |
| `h0`, `h1` | baja | baja | baja | retrasa | baja |
| `kw` | no | no | no | **sí** | **sí** |
| `gamma2` | no | no | no | un poco | sólo al final |
| `m`, `Eq`, `Ev`, `kappa` | no | no | no | según el contexto | según el contexto |

Las dos columnas de temperatura son la clave para desenredar:

- **`A_gen` es el único que mueve la goma fría.** Si un juego nuevo ya va
  demasiado caliente, ése es el culpable.
- **`zeta` es el único que mueve la gastada sin tocar la fría.** Si el problema
  aparece sólo al final del stint, es `zeta`.

---

## 4. Receta para calibrar la física

Cada paso es un `python tune.py ...`. No entrena nada; tarda segundos.

**Paso 1 — mira dónde estás.**

```bash
python tune.py
```

Lee el bloque DIAGNOSIS de arriba abajo y el de WHAT TO TURN NEXT. Las bandas
objetivo están en `TARGETS`, dentro de `tune.py`, y están puestas para que las
discutas: si crees que un neumático de F1 aguanta más, cámbialas.

**Paso 2 — arregla β primero, ignorando todo lo demás.**

Mientras β ≤ 1 las otras filas no significan nada, porque todas se mueven
cuando β se mueve. Sube `Ea`, `A_gen` o `zeta`, o baja `h0`/`h1`.

Un apunte que ayuda: `Ea` en estas unidades no es arbitrario. Un grado de goma
son 1/40 de unidad, y para caucho la degradación se duplica más o menos cada
10 °C, lo que da un factor 16 sobre el rango completo — o sea `Ea ≈ ln(16) ≈
2,8`. El 0,95 que hay puesto es **físicamente bajo**, y subirlo sube β sin
tocar la temperatura. Es el primer sitio donde mirar.

**Paso 3 — cuadra la temperatura.**

Con β ya por encima de 1, ajusta `A_gen` (goma fría) y `zeta` (goma gastada)
hasta que las dos filas `theta_ss` caigan en banda. Cada vez que las toques, β
se mueve: vuelve a mirarlo.

**Paso 4 — cuadra la magnitud con `kw`.**

`kw` no toca β ni la temperatura, así que es el mando limpio para llevar
`median d at lap 45` y `median pace loss at lap 30` a su sitio. Si demasiados
stints terminan con la goma muerta (`fraction of dead tires`), baja `kw`: esas
curvas son planas y no le enseñan nada a la red.

**Paso 5 — ajusta la altura del acantilado con `gamma2`.**

Sólo actúa cuando `d → 1`, así que muévelo mirando `s@45` y dejando `s@15`
quieto.

**Paso 6 — mira la forma, no sólo los números.**

```bash
python tune.py --plot outputs/tuning.png
```

Tres paneles: `theta`, `d` y los segundos. Las tablas pueden estar todas en
verde con la forma mal —una temperatura que nunca se estabiliza, una curva de
desgaste con un codo, un «acantilado» que en realidad es una esquina. La forma
es justo lo que las tablas esconden.

**Paso 7 — copia los valores.**

La última cosa que imprime `tune.py` es el bloque exacto para pegar en
`TireParams`. Pégalo; transcribir doce números a mano es como uno acaba siendo
distinto del que probaste.

**Paso 8 — comprueba que el integrador sigue sano.**

```bash
python tune.py --check-integrator
```

Poniendo `A_gen = 0` el acoplamiento se apaga, `theta` se queda en 0 y el
sistema vuelve a ser la única ecuación que sí tiene solución cerrada. Ahí RK4
se puede comparar contra la verdad exacta. Ahora mismo da **3,05 × 10⁻¹²**.
Cualquier cosa por encima de 1e-8 es que el integrador está mal, no la física.

---

## 5. Los mandos del entrenamiento

Ya con la física calibrada, `run.py`.

### Los pesos de la pérdida

```
loss = w_physics · ‖r_desgaste‖² + w_thermal · ‖r_térmico‖²
     + w_data · ‖error en segundos‖² + w_ic · ‖condiciones iniciales‖²
```

**No son importancias: son escalas.** Los residuos son adimensionales y
pequeños; el término de datos está en segundos. Sin los factores, el término
que de verdad ancla el modelo a lo medido queda por debajo del ruido numérico
de los otros.

`w_thermal` está separado de `w_physics` por una razón concreta y medible.
`dtheta/dτ` es del orden de `A_gen` (unidades enteras) mientras `dd/dτ` es del
orden de `kw` (una fracción), así que sus cuadrados se llevan órdenes de
magnitud. En la primera iteración de una corrida real:

```
wear 0.03279    heat 55.48579
```

El residuo térmico nace **1700 veces más grande**. Con `w_thermal = 1` la red
dedica casi todo su esfuerzo a la ecuación térmica, que no tiene ni un dato que
la sujete, y descuida la que sí se compara contra algo. Medido, con todo lo
demás igual:

| | por defecto | `--w-thermal 0.02` |
|---|---|---|
| `Ea` | 10,5 % de error | **3,5 %** |
| `gamma2` | 18,3 % | **2,8 %** |
| media de las 10 | 16,3 % | **13,2 %** |

Empieza por `--w-thermal 0.02` y muévelo de ahí.

### Condiciones iniciales: `--ic hard` (por defecto) o `--ic soft`

`hard` las impone transformando la salida de la red:

```
theta(τ) = τ · N₀                      ⟹  theta(0) = 0 exacto
d(τ)     = 1 − exp(−τ · softplus(N₁))  ⟹  d(0) = 0 exacto, y 0 ≤ d < 1 siempre
```

Como `τ` multiplica la salida, en `τ = 0` el término desaparece diga lo que
diga la red: se cumple con error cero en toda iteración y no cuesta nada. La
segunda merece leerse dos veces, porque no sólo fija `d(0)`: hace que la
saturación sea **estructural**. `softplus` es no negativo, el exponente es no
positivo, y `d` no puede salirse de [0,1). Tres modos de fallo eliminados por
una línea, y sin perder nada — toda curva que arranca en 0 y se queda por
debajo de 1 se sigue pudiendo escribir así.

`soft` lo mete como un término más de la pérdida. Es la formulación de libro de
texto, se cumple aproximadamente y nunca exacto, y compite por gradiente con
todo lo demás. Con dos estados esa competencia se duplica. **Está ahí para que
midas la diferencia, no porque sea buena idea.**

### `--lbfgs N`: Adam explora, L-BFGS refina

Las constantes térmicas (`zeta`, `h0`, `h1`) son las peor condicionadas del
problema: sólo llegan al observable a través de dos capas de composición —
mueven `theta`, `theta` mueve el ritmo de desgaste, el desgaste mueve `d`, y
sólo entonces pasa algo en segundos. Adam, que escala cada parámetro por su
propio historial de gradiente, tiende a dejarlas donde empezaron. Un método
cuasi-Newton usa la curvatura y sí las mueve. Medido:

| | Adam solo | `--lbfgs 400` |
|---|---|---|
| `h1` | 34,4 % de error | **3,8 %** |
| `kw` | 3,3 % | **0,1 %** |

Ojo: en la misma corrida `Ev` empeoró (2,4 % → 22 %). L-BFGS no es gratis,
hay que mirar la tabla entera.

Detalle de implementación que importa si tocas el código: **los puntos de
colocación se congelan durante L-BFGS**. Construye un modelo interno de la
superficie de pérdida a partir de evaluaciones sucesivas, y resamplear entre
ellas haría que cada evaluación fuese una función distinta — la aproximación
sería de puro ruido. Adam lo tolera; L-BFGS no.

### El resto

| bandera | qué hace | cuándo tocarla |
|---|---|---|
| `--iterations` | iteraciones de Adam | si las constantes aún se mueven al final |
| `--lr` | learning rate | bájalo si la pérdida va a saltos |
| `--collocation` | puntos donde se imponen las ecuaciones | súbelo si el residuo baja pero extrapola mal |
| `--width` / `--layers` | tamaño de la red | de los últimos; casi nunca es el cuello de botella |
| `--stints` | cuántos stints simular | súbelo antes que las iteraciones |
| `--noise` | ruido de cronometraje | bájalo para aislar si un fallo es de ruido o de modelo |
| `INITIAL_VALUES` en `pinn.py` | de dónde arrancan las 10 constantes | si una nunca se mueve |

---

## 6. Síntomas → qué tocar

| lo que ves | lo que significa | qué hacer |
|---|---|---|
| `Cliff 0%` en la tabla de resultados | las ecuaciones no pueden expresar un acantilado | mira β. Casi seguro es ≤ 1 |
| Una constante es una línea plana en `03_parameters.png` | nunca recibió gradiente usable | ¿deja huella en el observable? `--sweep` te lo dice. Si no, es la física |
| `zeta`, `h0` o `h1` no se mueven | mal condicionadas | `--lbfgs 400` |
| La pérdida total baja pero `data` se queda quieto | el término de datos está aplastado | baja `w_thermal`, sube `w_data` |
| `heat` domina desde la iteración 1 | escalas descompensadas | `--w-thermal 0.02` |
| `ViolExtrap` > 0 | predice que el neumático **recupera** agarre | imposible: `dd/dτ ≥ 0` siempre. Es bug, no ajuste |
| RMSE bueno pero `CliffErr` grande | acierta los segundos y falla el sitio | es *el* fallo que importa. Sube `--w-data`, y revisa que haya acantilados dentro de los stints |
| `theta` sale absurda en `04_state.png` | acierta por razones equivocadas | mira ahí antes que en el RMSE |
| Pérdida de entrenamiento baja y RMSE de test enorme | dirección degenerada | ¿destapaste `gamma1` o `A_gen`? Vuelve a fijarlas |
| NaN | el exponente de Arrhenius desbordó | `EXPONENT_CAP` en `physics.py` lo tapa; si chocas contra él, las térmicas se dispararon |

---

## 7. Cómo saber si quedó bien

Cuatro comprobaciones, en orden de qué tan fácil es engañarse:

1. **`tune.py --check-integrator` pasa.** Si RK4 está mal, todo lo demás mide
   ruido. Ahora mismo: 3,05 × 10⁻¹².
2. **`ViolIn` y `ViolExtrap` en 0,0 %.** Predecir que la goma recupera agarre
   es imposible y **ninguna métrica de error lo penaliza**. Un modelo puede
   tener un RMSE estupendo y decirte que en la vuelta 34 el coche irá más
   rápido que en la 33 — y si dice eso en la 34, tampoco te vas a fiar de lo
   que diga en la 20.
3. **`04_state.png`: `theta` y `d` pegadas a la verdad.** Ninguna de las dos se
   mide nunca; lo único que sujeta `theta` es la ecuación. Si sale mal aquí, el
   ajuste en segundos es una coincidencia.

   Con las constantes que vienen puestas ya se ve el caso: **`d` sale casi
   perfecta y `θ` sale alta**. La red compensa `zeta` y `h0` demasiado bajas con
   una trayectoria de temperatura distinta que da el mismo `d`. RMSE 0,061 y por
   dentro un neumático que corre más caliente que el real. Ninguna métrica de
   error puede enseñarte eso; esta figura sí.
4. **La tabla de recuperación de constantes.** Es la prueba de que el método
   funciona, y es la única que **es imposible con datos reales**, porque ahí no
   hay verdad contra la que comparar. Por eso existe el banco sintético: si no
   recupera las constantes aquí, apuntarlo a datos reales sólo produce números
   equivocados con más esfuerzo.

Y una advertencia que vale por todo lo demás:

> **Una pérdida de entrenamiento baja no garantiza nada** si el modelo tiene
> direcciones degeneradas. Es la lección más cara del proyecto.

---

## 8. Lo que ya medí, para que no lo repitas

Todo con 48 stints sintéticos, ruido 0,05 s, semilla 0, 3000 iteraciones.

| corrida | media de error de las 10 | detalle |
|---|---|---|
| por defecto | 16,3 % | `zeta` 52,9 %, `h1` 34,4 % |
| `--w-thermal 0.02` | **13,2 %** | `Ea` 10,5 → 3,5 %, `gamma2` 18,3 → 2,8 % |
| `--w-thermal 0.02 --lbfgs 400` | 14,6 % | `h1` 34,4 → 3,8 %, `kw` → 0,1 %, pero `Ev` 2,4 → 22 % |
| β = 1,94 en vez de 0,71 | **8,6 %** | `zeta` 61 % → 24 % |

Léelo así: los tres primeros son ajustes de entrenamiento y mueven la media
unos pocos puntos. El cuarto es un ajuste de **física** y la mueve casi el
doble. Por eso el orden del principio no es un capricho.

Las constantes de desgaste que llegan directas al observable (`kw`, `kappa`,
`m`) se recuperan bien en todas las configuraciones — 0,1 % a 4 %. Las que
llegan a través de `theta` son las difíciles. Ahí es donde vas a pasar el rato.
