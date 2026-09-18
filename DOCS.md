# Reference documentation

Every module, every function: what it does, how it works, and how it connects to
the rest. Written to be read alongside the code, not instead of it — the code
carries the *why*, this carries the *map*.

For the project's story and results, see [README.md](README.md). For the path
from this branch to `main`, see [ROADMAP.md](ROADMAP.md).

**A note on shape.** 87 functions, median 10 lines of code each, longest 46 —
and that one is a flat list of command-line declarations. Nothing is split for
the sake of it; things are split so that reading a caller tells you *what*
happens and opening one callee tells you *how*, without ever holding more than
a screen in your head. Every function and class has a docstring.

---

## 1. How it all fits together

### The flow of data

```
                physics.py                  ← the equation
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  data.py                  pinn.py          ← the network
  (stints)                 baseline.py      ← the rival
        │                       │
        │    download_data.py   │
        │    (API → CSV) ───────┤
        │                       ▼
        └──────────────► evaluate.py        ← the yardstick
                              │
                              ▼
                           run.py           ← orchestrates
                              │
                              ▼
                          outputs/
```

### The import graph

Modules import downwards only. There is no cycle and no sideways import.

| Module | Imports from the project | Third-party |
|---|---|---|
| `physics.py` | — | numpy, torch *(optional)* |
| `data.py` | `physics` | numpy |
| `pinn.py` | `physics` | numpy, torch |
| `baseline.py` | `physics` | numpy |
| `evaluate.py` | `physics` | numpy |
| `run.py` | all of the above | numpy, matplotlib |
| `download_data.py` | `physics` *(only `COMPOUND_INDEX`)* | numpy, pandas, scipy, fastf1 |

Two things worth noticing:

- **`download_data.py` barely touches `physics.py`.** Building variables from
  telemetry is a signal-processing job; mixing it with the model would mean that
  changing an equation forces you to rebuild the dataset.
- **`baseline.py` never sees the equation.** If the classic rival could look at
  the physics, the comparison would stop measuring what it claims to measure.

### The contract everything depends on

`physics.py` defines the canonical order of the network's inputs. Every other
module assumes it:

```
index:     0      1        2      3       4            5
name:     tau   q_fric   load   speed   track_temp   compound
          └──┘  └──────────── context (5) ───────────────────┘
```

`Context` is a dataclass with those five fields, so code reads
`context.track_temp` rather than `context[3]`. `Context.vector()` produces the
row in canonical order; `data.flatten()` glues `tau` in front of it.

---

## 2. The physics, in words

One differential equation and one observation operator.

```
   dd/dtau = k(context) · (1 − d)              the equation

   k = kw · (load/load_ref)^m
          · exp( Ea·(track_temp − temp_ref)
               + Eq·(q_fric     − q_ref)
               − Ev·(speed      − speed_ref)
               − kappa·(compound − compound_ref) )

   delta = gamma1 · d                          the observation operator
```

| Symbol | Meaning | Units / range | Status |
|---|---|---|---|
| `tau` | dimensionless stint time = `lap / 30` | 0 … 1.5 | input |
| `d` | fraction of tread consumed | 0 … 1 | **latent, never measured** |
| `delta` | pace loss vs the stint's best lap | seconds | **the only observable** |
| `q_fric` | frictional energy per lap, normalised | ≈ 0.4 … 1.6 | input |
| `load` | mean mechanical load in g, normalised | ≈ 0.5 … 1.5 | input |
| `speed` | mean speed, normalised | ≈ 0.6 … 1.4 | input |
| `track_temp` | track temperature, rescaled | 0 … 1 | input |
| `compound` | 0 soft, 0.5 medium, 1 hard | 0 … 1 | input |
| `kw` | wear rate at reference conditions | — | **estimated** |
| `m` | load exponent (Archard) | — | **estimated** |
| `Ea` | thermal activation | — | **estimated** |
| `Eq` | frictional-energy sensitivity | — | **estimated** |
| `Ev` | cooling by speed | — | **estimated** |
| `kappa` | compound resistance | — | **estimated** |
| `gamma1` | seconds lost per unit of wear | s | **fixed** |

Three design decisions carry the weight, and each has its own reason:

**`(1 − d)` bounds `d` structurally.** Wear stops when there is no rubber left.
Because the rate is non-negative while `d ≤ 1`, both monotonic wear and the
bound `d ≤ 1` fall out of the equation itself rather than being constraints
bolted on afterwards.

**Every variable is centred on a reference.** At reference conditions the
exponent is zero, so `kw` means "the wear rate under normal conditions".
Without centring, `kw` and `Ev` trade off against each other — measured here:
uncentred, `kw` came out 10 % wrong and `Ev` 33 % wrong, while the combination
`log(kw) − Ev` was recovered to 0.0098.

**`gamma1` is fixed, not estimated.** The only measurement is `delta = gamma1·d`.
Leaving both `d` and `gamma1` free admits infinitely many equivalent solutions —
double `d`, halve `gamma1`, nothing observable changes. Fixing `gamma1` anchors
the scale. In `main` this same degeneracy, left open, diverged a training run to
an RMSE of billions of seconds while the training loss looked fine.

---

## 3. `physics.py` — the model

### Constants

| Name | Value | What it is |
|---|---|---|
| `LAP_REF` | `30.0` | Laps per unit of `tau`. Keeps network inputs near 1. |
| `STRATEGY_HORIZON` | `45` | How far ahead predictions and the equation reach. |
| `COMPOUND_INDEX` | dict | Compound name → number, 0 soft … 1 hard. |
| `CONTEXT_NAMES` | tuple | The five field names, in canonical order. |
| `N_CONTEXT` / `N_INPUTS` | `5` / `6` | Sizes derived from `Context`. |
| `CONTEXT_RANGES` | dict | Physically sensible range of each variable. Used for sampling synthetic stints and for placing collocation points. |
| `REFERENCE` | `Context` | The centre of each range. Where the exponent vanishes. |
| `GROUND_TRUTH` | `TireParams` | The true constants the synthetic generator uses. |
| `LEARNABLE_PARAMS` | 6 names | Which constants the PINN estimates. |

### `Context` — the conditions of a stint

A frozen dataclass with five float fields. Constant within a stint, because one
set of tires runs one circuit at roughly one temperature on one compound.

| Member | What it does |
|---|---|
| `vector()` | The five numbers as a numpy array, canonical order. |
| `from_vector(v)` | Class method. The inverse of `vector()`. |
| `compound_name` | Property. The nearest compound name, for plot labels. |

### `TireParams` — the physical constants

A frozen dataclass holding `kw, m, Ea, Eq, Ev, kappa, gamma1`. Defaults are the
values used by the synthetic generator, which is why `GROUND_TRUTH =
TireParams()` needs no arguments.

The same class holds both plain floats (when integrating) and torch tensors
(when training). Python does not enforce the type hints, and that is used
deliberately so one class serves both paths.

### Functions

| Function | What it does | How it works |
|---|---|---|
| `wear_constant(q_fric, load, speed, track_temp, compound, p)` | `k(context)` — everything that does **not** depend on `d`. | Dispatches `exp`/`power` on whether the first argument is a torch tensor, so the same code serves numpy and torch. Each variable enters as a deviation from `REFERENCE`. Load is clamped at `1e-6` because a negative base makes `power` return NaN. |
| `wear_rate(d, ..., p)` | The right-hand side, `dd/dtau`. | `wear_constant(...) * (1 - d)`. Non-negative while `d ≤ 1`. |
| `pace_loss(d, p)` | The observation operator: latent `d` → seconds. | `gamma1 * d`. This is how `d` gets compared against data without ever appearing in the loss directly. |
| `exact_solution(tau, context, p)` | The closed-form answer. | `1 - exp(-k·tau)`, from separating variables with `d(0)=0`. Only exists because `k` is constant within a stint. |
| `integrate_stint(n_laps, context, p, steps_per_lap=8)` | Solve step by step, returns `(laps, d)`. | Runge-Kutta 4 with 8 sub-steps per lap. Averages four slopes across each step rather than trusting one. Euler would accumulate a systematic bias that the inverse problem would read as physics. |
| `_is_tensor(x)` | Internal. True for torch tensors. | Guards the numpy/torch dispatch and returns False cleanly when torch is not installed. |

**Why `integrate_stint` exists at all** when `exact_solution` is right there:
so the integrator can be validated against the exact answer *now*, while doing so
is cheap. In `main` two coupled equations kill the closed form and this
integrator is all that remains.

---

## 4. `data.py` — where the stints come from

### `Stint` — the unit of learning

| Field | Type | What it holds |
|---|---|---|
| `stint_id` | str | Unique name, e.g. `SYN03` or `MON-VER-S2`. |
| `context` | `Context` | The five conditions. |
| `laps` | array | Lap number within the stint: 1, 2, 3 … |
| `delta` | array | **Measured** pace loss in seconds. Carries noise. |
| `d_true` | array or None | True wear. Synthetic only. **Checking, never training.** |
| `delta_clean` | array or None | Noise-free pace curve. Synthetic only. |

| Property | What it gives |
|---|---|
| `n_laps` | How many laps the stint lasted. |
| `tau` | `laps / LAP_REF` — the dimensionless time. |
| `compound` | The compound name, for grouping and labels. |

### Functions

| Function | What it does | How it works |
|---|---|---|
| `generate_synthetic(n_stints=48, p=GROUND_TRUTH, noise_s=0.05, min_laps=12, max_laps=30, seed=0)` | Simulate stints with known constants. | For each stint: pick a compound (cycling), draw a context from `CONTEXT_RANGES`, integrate the equation, convert to seconds with `pace_loss`, then add Gaussian noise. The noise matters — without it the fit is trivial and the physics term has nothing to do. |
| `load_csv(path, min_laps=8)` | Read the CSV from `download_data.py`. | Groups rows by `stint_id`, sorts by `stint_lap`, checks the required columns are present and drops short stints. Raises `FileNotFoundError` with the exact download command if the file is missing. |
| `split(stints, test_fraction=0.25, seed=0)` | Train/test split. | Permutes stint indices and takes a slice. **By whole stint, never by lap** — splitting by lap leaks, and it is the worst kind of leak because it flatters every model equally and cannot be spotted by comparing them. |
| `flatten(stints)` | Stints → the two matrices that train the network. | Returns `inputs (N, 6)` and `delta (N, 1)`. The constant context is tiled across the stint's laps with `np.tile`, which is what lets the network learn the effect of time and of conditions at once. |
| `_read_rows(path)` | Internal. Read the CSV and check the required columns exist. | Raises `FileNotFoundError` carrying the exact download command, or `ValueError` naming the missing columns. |
| `_stint_from_rows(id, rows)` | Internal. Build one `Stint` from its sorted rows. | Reads the context off the first row; every row of a stint carries the same value. |
| `describe(stints)` | A paragraph summarising the set. | Counts, lengths, pace-loss range and compound breakdown. Printed before training and written into `report.txt`. |
| `_sample_context(rng, compound)` | Internal. Draw plausible conditions. | Uniform within each `CONTEXT_RANGES` entry, except `compound`, which the caller decides. |

---

## 5. `pinn.py` — the network with physics inside

### `MLP` — just the network

`6 → width × layers → 1`, `tanh` between layers, nothing after the last one.
Default `64 × 4` gives **12,993 weights**.

| Member | What it does | How it works |
|---|---|---|
| `__init__(width=64, layers=4)` | Build the stack. | `nn.Sequential` of `Linear`/`Tanh` pairs. The last layer has no activation, so the output is unbounded. |
| `_glorot_init()` | Initialise the weights. | `xavier_normal_` on weights, zeros on biases. Keeps signal variance steady across layers; without it, four layers of tanh saturate on iteration 1 and the network never learns. |
| `forward(tau, context)` | One batch through the network. | Concatenates to `(N, 6)` and runs the stack. `tau` arrives separately so it can stay a leaf of the autograd graph. |

**Why tanh and not ReLU:** the derivative of a ReLU network is piecewise
constant, so it would predict a stepped, discontinuous `dd/dtau` that cannot
match a smooth right-hand side. (The usual argument — "ReLU's second derivative
is zero" — applies to second-order equations, not to this one.)

### `PINN` — the network plus the six constants

| Member | What it does | How it works |
|---|---|---|
| `__init__(width, layers, gamma1, tau_max, seed)` | Create everything. | Builds the `MLP` and one `nn.Parameter` per learnable constant, stored in **raw** form: `log(value)` for the five that must be positive, the value itself for `kappa`. |
| `_value(name)` | Raw → physical value. | `exp(raw)` for the positive ones, `raw` for `kappa`. |
| `params_tensor()` | The constants as tensors. | Used inside the residual so the constants enter the autograd graph and receive gradients. |
| `learned_params()` | The constants as plain floats. | `.detach()`-ed, for printing and reporting. |
| `n_weights()` | Count the network weights. | Excludes the six constants. |
| **`residual(tau, context)`** | **`r = dd/dtau − wear_rate(d, context)`** | The heart of the method. Four steps: mark `tau` with `requires_grad_`, run the network, call `torch.autograd.grad` to differentiate the output w.r.t. the input, subtract what the physics says. `create_graph=True` is essential — without it the physics term contributes no gradient at all. |
| `collocation_points(n)` | Where the equation is enforced. | Uniform `tau` out to `tau_max` (the full 45-lap horizon) and uniform draws within each `CONTEXT_RANGES` entry. **No labels.** That is the entire reason this term can cover ground the data never does. |
| `_physics_loss(n)` | Term 1 of the loss. | Samples `n` collocation points and returns the mean squared residual. |
| `_data_loss(tau, context, delta)` | Term 2 of the loss. | Forwards the observed laps, converts `d` to seconds with `gamma1`, compares against the measurement. |
| `_initial_condition_loss(n=64)` | Term 3 of the loss. | Forwards `tau=0` with `n` random contexts; the answer should be zero. |
| `train(inputs, delta, ...)` | The training loop. | Adam over network weights and the six constants together. Each iteration calls the three methods above, weights them, backpropagates and steps. Splitting the terms out is what lets the loop read like the formula. |
| `_record(...)` | Save the state of an iteration. | Uses `.item()` rather than `float()`, because the tensors are still attached to the graph and `float()` would warn. |
| `_log_line(r)` | Format a progress line. | Reads from the recorded dict, so printing never touches the graph. |
| `predict_stint(context, laps)` | Predicted pace loss. | Builds `tau`, tiles the context, forwards under `no_grad`, multiplies by `gamma1`. **Same signature as the baseline**, which is what makes the comparison fair. |
| `predict_wear(context, laps)` | The latent state `d`. | `predict_stint(...) / gamma1`. Never measured; it exists only because the equation was imposed. |
| `_columns(context)` | Internal. Split `(N,5)` into 5 columns. | Keeps the residual readable. |

### The three loss terms

| Term | Default weight | Where it applies | What it imposes |
|---|---|---|---|
| physics | 1.0 | 2,000 collocation points, labels not needed | The equation holds |
| data | 10.0 | Observed laps only | Match measured pace |
| initial condition | 10.0 | `tau = 0`, 64 random contexts | `d(0) = 0` |

The weights are **scales, not importances**. The equation residual is
dimensionless and small; the data residual is in seconds. Without the factor,
the term that anchors the model to reality would sit below the other's numerical
noise.

### `INITIAL_VALUES`

Starting points, deliberately wrong, so recovering `GROUND_TRUTH` proves
something:

| Constant | Starts at | True value |
|---|---|---|
| `kw` | 0.30 | 0.55 |
| `m` | 1.00 | 1.50 |
| `Ea` | 0.50 | 0.95 |
| `Eq` | 0.20 | 0.40 |
| `Ev` | 0.20 | 0.35 |
| `kappa` | 0.30 | 0.85 |

---

## 6. `baseline.py` — the rival

`LinearBaseline` — least squares on 13 features of `(tau, context)`.

| Member | What it does | How it works |
|---|---|---|
| `__init__(ridge=1e-6)` | Set up an unfitted model. | `ridge` adds a tiny term to the diagonal of `XᵀX` so the solve stays stable if two features are nearly identical. |
| `_features(tau, context)` | Build the design matrix. | 13 columns: `1`, `tau`, `tau²`, the 5 context variables, and `tau × ` each of the 5. The last group is what lets it say "on a hot track it degrades faster". |
| `fit(inputs, delta)` | Solve for the coefficients. | Normal equations: `np.linalg.solve(XᵀX + ridge·I, Xᵀy)`. |
| `predict_stint(context, laps)` | Predicted pace loss. | Same signature as the PINN's. |

It is given the quadratic term and the interactions **on purpose**. A straw man
that loses for lack of flexibility proves nothing — and in fact, inside the
measured range it is very hard to beat, because a parabola fits a gently bent
curve well. It separates from the PINN only when extrapolating.

---

## 7. `evaluate.py` — the yardstick

### `Metrics`

| Field | What it measures |
|---|---|
| `rmse` | `sqrt(mean(error²))`. Large errors weighted more. |
| `mae` | `mean(abs(error))`. Easier to read, less sensitive to outliers. |
| `max_error` | The single worst lap. |
| `violations_inside` | % of laps inside the stint where the model says the tire improves. |
| `violations_extrapolating` | The same, over the full 45-lap horizon. |

### Functions

| Function | What it does | How it works |
|---|---|---|
| `evaluate(name, predict_stint, stints)` | Measure any model. | For each stint: predict over the observed laps (error + violations), then predict over the full horizon (violations again). Accumulates and returns `Metrics`. Takes the prediction function, not the model, so it cannot accidentally depend on model internals. |
| `_count_violations(delta, tolerance=1e-3)` | Count lap-to-lap improvements. | `np.diff` then count entries below `−tolerance`. The tolerance avoids flagging a millionth of a second of numerical noise. |
| `parameter_recovery(learned, truth, names)` | Compare estimates to truth. | Returns `(name, estimated, true, relative error %)`. Synthetic data only — with real data there is no truth to compare against. |

**Why monotonicity is measured separately.** A tire only degrades, so a model
predicting recovery is making a physically impossible claim — and no error
metric penalises it. A model that says lap 34 will be faster than lap 33 will not
be trusted on lap 20 either. The correct value is 0 %.

---

## 8. `run.py` — the orchestrator

`main()` is a list of five calls, one per step. Reading `main()` tells you what
happens; opening one function tells you how.

| Step | Function | What it does |
|---|---|---|
| 1 | `get_stints(args)` | Simulate stints, or read the CSV. Returns `None` on a read failure, which `main` turns into a clean exit instead of a traceback. |
| 2 | `train_pinn(inputs, delta, args)` | Build the network and train it. |
| 3 | `train_baseline(inputs, delta)` | Fit the rival on exactly the same data. |
| 4 | `build_report(model, linear, test, synthetic)` | Measure both and assemble the text report. Delegates to `_results_table`, and then to `_recovery_table` (synthetic) or `_estimates_list` (real). |
| 5 | `save_outputs(...)` | Write the three figures and `report.txt`. |

With synthetic data step 4 adds the inverse-problem table — checking whether
the true constants were recovered — which is impossible with real data.

### Command-line options

| Flag | Default | What it does |
|---|---|---|
| `--source {synthetic,csv}` | `synthetic` | Where the stints come from. |
| `--csv PATH` | `data/races.csv` | The file `download_data.py` produced. |
| `--stints N` | `48` | How many stints to simulate. |
| `--noise S` | `0.05` | Timing noise in seconds, synthetic only. |
| `--min-laps N` | `8` | Discard shorter stints, CSV only. |
| `--width N` | `64` | Neurons per layer. |
| `--layers N` | `4` | Hidden layers. |
| `--iterations N` | `8000` | Adam iterations. |
| `--collocation N` | `2000` | Collocation points per iteration. |
| `--lr F` | `3e-3` | Learning rate. |
| `--seed N` | `0` | Controls data generation, the split and the weight init. |
| `--out DIR` | `outputs` | Where figures and the report go. |
| `--quick` | off | 18 stints, 1,500 iterations. Checks it runs. |

### Plot functions

| Function | Output | What it shows |
|---|---|---|
| `plot_fit` | `01_fit.png` | One panel per compound, each drawn by `_draw_stint_panel`: measurements, the exact curve, both models. The shaded band is where data exists; to its right everything is extrapolating. |
| `plot_training` | `02_training.png` | The three loss terms on a log scale. |
| `plot_parameters` | `03_parameters.png` | The six constants over training, with the true value as a dashed line. |

`matplotlib.use("Agg")` is called **before** importing pyplot, because pyplot
picks its backend at import time. Without it the script fails on a headless
machine.

---

## 9. `download_data.py` — real data from the API

### What it produces

One CSV, one row per lap, ready for `data.load_csv()`.

| Column | What it is |
|---|---|
| `race`, `driver`, `stint_id` | Identification. |
| `race_lap` | Lap number within the race. |
| `stint_lap` | Lap number within the stint. **1 = the performance peak.** |
| `lap_time` | Raw lap time in seconds. |
| `corrected_time` | Minus the fuel + track-evolution effect. |
| `delta` | Pace loss vs the stint's best lap. **The observable.** |
| `q_fric`, `load`, `speed`, `track_temp`, `compound` | The model's five variables, already normalised. Constant per stint (the stint median). |
| `compound_name`, `tyre_life` | For inspecting the file by hand. |

### The pipeline

```
FastF1 session
   ↓  lap_is_usable()          drop safety car, in/out laps, deleted, used rubber
   ↓  lap_dynamics()           telemetry → q_fric, load, speed
   ↓  normalise                divide by the FIXED references
   ↓  estimate_race_lap_effect()   fuel + track evolution
   ↓  assemble_stints()        anchor d=0 at the peak, compute delta
   ↓  CSV
```

### Functions

| Function | What it does | How it works |
|---|---|---|
| `lap_dynamics(telemetry)` | One lap's telemetry → three variables. | Four steps: clean, smooth, differentiate both ways, average over the lap. Returns `None` if the lap cannot be used. |
| `_clean_time_and_speed(telemetry)` | Internal. Time and speed, with bad samples dropped. | Converts km/h to m/s and removes repeated timestamps, which merged telemetry occasionally carries and which break differentiation. |
| `_lateral_acceleration(telemetry, time, window)` | Internal. The component the telemetry does not carry. | Differentiates the GPS trajectory twice and takes `\|v × a\| / \|v\|`, the part perpendicular to the velocity. Clipped at 6 g — an F1 car's real ceiling, so anything above is a numerical artefact. Returns zeros when there is no position channel. |
| `_smoothing_window(time, seconds=1.0)` | Window size in samples. | Fixed **in seconds, not samples**: FastF1 merges car telemetry (~10 Hz) with GPS (~4 Hz), so the effective rate varies per lap, and a fixed sample count would filter each lap differently. `\| 1` forces an odd window. |
| `_smooth(values, window)` | Savitzky-Golay filter. | Fits a local parabola and takes its centre. Preserves peaks, unlike a moving average, which would flatten the braking zones — the most informative part. Falls back to a moving average if scipy is missing. |
| `lap_is_usable(lap, only_fresh_tyres)` | Quality filter. | Rejects laps with no time, flagged inaccurate, under yellow/safety car, in/out laps, deleted laps and (optionally) used rubber. Uses `bool(...)` rather than `is False`, because pandas may hand back `numpy.bool_` and `is` would silently fail. |
| `estimate_race_lap_effect(rows)` | Seconds per lap the car gains for reasons other than the tire. | Least squares of `lap_time ~ driver dummies + slope·race_lap + compound×tyre_life`. Returns the slope. The degradation terms exist only to stop the slope absorbing them. Returns `0.0` rather than failing when there is too little data. |
| `assemble_stints(rows, args)` | Group laps into stints. | Groups by `(driver, stint)` and hands each group to `_one_stint`. The interesting decision lives in `_find_peak`. |
| `_find_peak(times, search_laps)` | Internal. **Where `d = 0` is defined.** | The fastest of the first few laps, not the first lap. See the section below on why. |
| `_one_stint(laps, args)` | Internal. One group of laps → CSV rows. | Drops warm-up, measures `delta` against the peak, drops laps losing more than `--max-delta`, attaches the context. Returns `None` if the stint does not qualify. |
| `_stint_context(laps)` | Internal. The five variables for a stint. | The median over the surviving laps only — the warm-up laps must not sneak back in through the median. |
| `process_race(name, args)` | One race, end to end. | Three calls: load the session, collect the laps, apply the correction, assemble. |
| `_load_session(name, args)` | Internal. Download or read from cache. | Enables the FastF1 cache and loads laps, telemetry and weather. |
| `_track_temp_series(session)` / `_track_temp_at(lap, weather)` | Internal. Track temperature. | The first pulls the session trace, the second interpolates it at a lap's start time, falling back to 35 °C when unrecorded. |
| `_lap_variables(lap, args)` | Internal. The three telemetry variables. | Calls `lap_dynamics`, or returns the reference values when `--no-telemetry` is set. |
| `_lap_row(lap, race, driver, dynamics, temp)` | Internal. Build one raw row. | Where normalisation against the **fixed** references happens, and where temperature goes from degrees to 0..1. |
| `_collect_laps(session, race, args)` | Internal. The driver/lap loop. | Applies `lap_is_usable`, builds rows, reports how many were dropped. |
| `_apply_race_lap_correction(rows, args)` | Internal. Subtract the fuel + track effect. | Modifies rows in place, adding `corrected_time`. Also fills in missing tire ages. |
| `_download_all(args)` | Internal. Loop over the requested races. | One race failing does not stop the rest; reasons are collected and reported at the end, which matters when a season takes an hour to parse. |
| `_write_csv(rows, out)` / `_print_summary(...)` | Internal. Output. | Write the file, then print what was downloaded and the command to train on it. |
| `describe_columns()` | The `--explain-columns` text. | |

### Why the origin is the peak, not lap 1

A new set comes out cold and gets **faster** for two or three laps before it
starts falling away. The model is monotonic by construction and cannot represent
that, so leaving it in would give every stint an irreconcilable conflict between
what was observed and what can be predicted — and the network would compensate
by distorting the physical constants. In `main` this correction took RMSE from
2.82 s to 0.567 s and violations from 16.4 % to 0 %.

### Why the references are fixed constants

Normalising each race against its own median would put Monza at 1.0 and Hungary
at 1.0 too, erasing exactly the between-circuit variation the model has to learn.
This was a real mistake in the project and cost a redesign.

### Command-line options

| Flag | Default | What it does |
|---|---|---|
| `--year N` | `2023` | Season. |
| `--races A B C` | `Monza` | One or more Grands Prix. |
| `--session S` | `R` | `R` race, `Q` qualifying, `FP1`–`FP3` practice. |
| `--drivers ...` | all | Three-letter codes. |
| `--max-drivers N` | all | Limit for quick tests. |
| `--min-laps N` | `8` | Discard shorter stints. |
| `--max-delta S` | `6.0` | Discard laps losing more than this (traffic). |
| `--reference-laps N` | `3` | How many laps to search for the peak. |
| `--only-fresh-tyres` / `--include-used-tyres` | fresh only | `d(0)=0` only holds on new rubber. |
| `--race-lap-effect` | `auto` | Estimate per race, or give a fixed number like `-0.055`. |
| `--no-telemetry` | off | Skip telemetry: seconds instead of minutes, but leaves `q_fric`, `load` and `speed` pinned at 1.0. |
| `--cache DIR` | `cache` | FastF1 cache, fills itself, ~200 MB. |
| `--out PATH` | `data/races.csv` | Output CSV. |
| `--explain-columns` | — | Print the column reference and exit. |

---

## 10. How to check it still works

| Check | Command | Expected |
|---|---|---|
| RK4 matches the closed form | see snippet below | error ~`1e-11` |
| The pipeline runs | `python run.py --quick` | finishes in well under a minute |
| Default results | `python run.py` | PINN RMSE ≈ 0.048, 0 % violations |
| Constants recover | `python run.py --stints 140 --noise 0.02 --iterations 10000` | ≈ 1 % mean error |
| Every symbol documented | `python -c "..."` (docstring audit) | 65 / 65 |

```python
import numpy as np
from physics import *

rng = np.random.default_rng(1)
worst = 0.0
for _ in range(20):
    ctx = Context(*[rng.uniform(*CONTEXT_RANGES[n]) for n in CONTEXT_NAMES])
    laps, d_rk4 = integrate_stint(30, ctx, GROUND_TRUTH)
    d_exact = exact_solution(laps / LAP_REF, ctx, GROUND_TRUTH)
    worst = max(worst, float(np.max(np.abs(d_rk4 - d_exact))))
print(worst)          # ~1.3e-11
```

---

## 11. Known limits of this version

Stated plainly, because the gap is the point of the branch:

- **No cliff.** There is no tire temperature as a state, so no feedback between
  wear and heat, so the pace curve can only bend one way. This is step 1 of the
  ROADMAP and the most important thing missing.
- **The initial condition is a loss term**, not an exact constraint. It is
  satisfied approximately and competes for gradient.
- **Adam only.** No L-BFGS, which in `main` turns out to be the only thing that
  moves the worst-conditioned parameters.
- **The race-lap correction is a straight line.** `main` fits a piecewise-linear
  spline, because that curve has no reason to be straight.
- **`Ev` is weakly identified at default settings** (≈ 35 % error). It is not a
  bug: over the range `speed` varies, `Ev` moves the exponent by only 0.28,
  against 0.85 for `kappa` and 0.95 for `Ea`. It is the smallest effect in the
  model and therefore the first thing lost to noise. With more data it recovers
  to ≈ 1 %.
- **No LSTM baseline.** Only the classic linear model competes here.
