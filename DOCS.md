# Reference documentation

Every module, every function: what it does, how it works, and how it connects to
the rest. Written to be read alongside the code, not instead of it — the code
carries the *why*, this carries the *map*.

For the project's story and results, see [README.md](README.md). For **how to
choose the physical constants**, see [TUNING.md](TUNING.md) — the code ships
with the equations, not with a calibration. For the path from this branch to
`main`, see [ROADMAP.md](ROADMAP.md).

**A note on shape.** 120 functions, median 9 lines of code each, longest 53 —
and the two longest are flat lists of command-line declarations. Nothing is
split for the sake of it; things are split so that reading a caller tells you
*what* happens and opening one callee tells you *how*, without ever holding more
than a screen in your head. All 127 functions and classes have a docstring.

---

## 1. How it all fits together

### The flow of data

```
                physics.py                  ← the two equations
                    │
        ┌───────────┼───────────┬──────────────┐
        ▼           │           ▼              ▼
  data.py           │      pinn.py         tune.py    ← calibrate the
  (stints)          │      baseline.py                   constants, no
        │           │           │                        training
        │  download_data.py     │
        │  (API → CSV) ─────────┤
        │                       ▼
        └──────────────► evaluate.py        ← the yardstick
                              │
                              ▼
                           run.py           ← orchestrates
                              │
                              ▼
                          outputs/
```

`tune.py` hangs off `physics.py` alone and touches nothing downstream. That is
the point of it: it answers *what world do these constants describe* before any
network exists to be confused by the answer.

### The import graph

Modules import downwards only. There is no cycle and no sideways import.

| Module | Imports from the project | Third-party |
|---|---|---|
| `physics.py` | — | numpy, torch *(optional)* |
| `data.py` | `physics` | numpy |
| `pinn.py` | `physics` | numpy, torch |
| `baseline.py` | `physics` | numpy |
| `evaluate.py` | `physics` | numpy |
| `tune.py` | `physics`, `evaluate` | numpy, matplotlib *(only with `--plot`)* |
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

Two coupled differential equations and one observation operator.

```
   (E1)  dtheta/dtau = A_gen·q_fric·(1 + zeta·d) − (h0 + h1·speed)·theta

   (E2)  dd/dtau     = k(context, theta) · (1 − d)

         k = kw · (load/load_ref)^m
                · exp( Ea·(track_temp − temp_ref + theta)
                     + Eq·(q_fric     − q_ref)
                     − Ev·(speed      − speed_ref)
                     − kappa·(compound − compound_ref) )

   (E3)  delta = gamma1·d + gamma2·d^8          the observation operator
```

| Symbol | Meaning | Units / range | Status |
|---|---|---|---|
| `tau` | dimensionless stint time = `lap / 30` | 0 … 1.5 | input |
| `theta` | rubber temperature over the reference state | 1 unit = 40 °C | **latent, never measured** |
| `d` | fraction of tread consumed | 0 … 1 | **latent, never measured** |
| `delta` | pace loss vs the stint's best lap | seconds | **the only observable** |
| `q_fric` | frictional energy per lap, normalised | ≈ 0.4 … 1.6 | input |
| `load` | mean mechanical load in g, normalised | ≈ 0.5 … 1.5 | input |
| `speed` | mean speed, normalised | ≈ 0.6 … 1.4 | input |
| `track_temp` | track temperature, rescaled | 0 … 1 | input |
| `compound` | 0 soft, 0.5 medium, 1 hard | 0 … 1 | input |
| `kw` | wear rate at reference conditions, cold tire | — | **estimated** |
| `m` | load exponent (Archard) | — | **estimated** |
| `Ea` | Arrhenius activation on the **total** heat | — | **estimated** |
| `Eq` | frictional-energy sensitivity | — | **estimated** |
| `Ev` | cooling by speed | — | **estimated** |
| `kappa` | compound resistance | — | **estimated** |
| `A_gen` | heat generated per unit of `q_fric` | — | **fixed** |
| `zeta` | the feedback: how much a worn tire overheats | — | **estimated** |
| `h0` | cooling independent of speed | — | **estimated** |
| `h1` | extra cooling from airflow | — | **estimated** |
| `gamma1` | seconds lost per unit of wear | s | **fixed** |
| `gamma2` | seconds lost to the cliff | s | **estimated, boxed** |

Five design decisions carry the weight, and each has its own reason:

**The cliff is not written down anywhere — it emerges.** The factor `(1 + zeta·d)`
in E1 says that as the tread thins, the same frictional energy goes into less
mass of rubber, so the temperature rises; and by the Arrhenius term in E2, more
temperature means faster wear, which thins the tread further. Nothing in the
code says "fall off after lap N". The single-equation version of this model
could only bend one way, and a real tire does not do that.

**`theta` shares `Ea` with `track_temp`.** Physically, Arrhenius activation
depends on the absolute temperature of the rubber, which is the track's heat
plus whatever friction added — splitting them would claim a degree from the
track wears differently from a degree from friction. It also pins the SCALE of
`theta`: nobody measures it, so on its own the model could shrink `theta` and
grow `Ea` by the same factor unnoticed. Tying `Ea` to `track_temp`, which *is*
measured, closes that escape.

**`(1 − d)` bounds `d` structurally.** Wear stops when there is no rubber left.
Because the rate stays non-negative while `d ≤ 1` — the thermal feedback does
not change that — both monotonic wear and the bound `d ≤ 1` fall out of the
equation itself. The cliff is a *steepening*, never a reversal.

**Every context variable is centred on a reference.** At reference conditions
with a cold tire the exponent is zero, so `kw` means "the wear rate under normal
conditions". Without centring, `kw` and `Ev` trade off against each other —
measured here: uncentred, `kw` came out 10 % wrong and `Ev` 33 % wrong, while
the combination `log(kw) − Ev` was recovered to 0.0098.

**`gamma1` and `A_gen` are fixed, not estimated.** Both are degenerate
directions: moving along them changes the latent state and leaves the observable
identical, so no amount of data can pick a point and the optimiser slides until
it overflows. `gamma1` anchors the scale of `d` (double `d`, halve `gamma1`,
nothing observable changes); `A_gen` anchors the scale of `theta`. In `main`
this same degeneracy, left open, diverged a run to an RMSE of billions of
seconds while the training loss looked fine.

### The one number that decides whether a cliff can exist

Substituting the quasi-steady temperature `theta_ss(d)` into E2 collapses the
system to `dd/dtau = k0·exp(beta·d)·(1 − d)` with

```
beta = Ea · A_gen · q_fric · zeta / (h0 + h1·speed)
```

Wear *accelerates* exactly while `beta·(1 − d) > 1`, i.e. `d < 1 − 1/beta`. So
**`beta ≤ 1` means no cliff can exist at all**, whatever else is tuned. This is
the first thing `tune.py` prints and the first thing to fix. Measured on this
branch: at `beta = 0.71`, `zeta` was recovered with 61 % error; at `beta = 1.94`,
24 %, because below 1 the constant barely reaches the observable. See
[TUNING.md](TUNING.md).

---

## 3. `physics.py` — the model

### Constants

| Name | Value | What it is |
|---|---|---|
| `LAP_REF` | `30.0` | Laps per unit of `tau`. Keeps network inputs near 1. |
| `STRATEGY_HORIZON` | `45` | How far ahead predictions and the equations reach. |
| `CLIFF_EXPONENT` | `8` | The exponent in E3. Fixed, never fitted: its only job is to keep the cliff term invisible until `d` nears 1. |
| `EXPONENT_CAP` | `12.0` | Numerical guard on the Arrhenius exponent. **Not physics** — it only stops a runaway `theta` producing `inf` and then `NaN`. |
| `COMPOUND_INDEX` | dict | Compound name → number, 0 soft … 1 hard. |
| `CONTEXT_NAMES` | tuple | The five field names, in canonical order. |
| `N_CONTEXT` / `N_INPUTS` | `5` / `6` | Sizes derived from `Context`. |
| `N_STATES` | `2` | The network predicts `(theta, d)`. |
| `CONTEXT_RANGES` | dict | Physically sensible range of each variable. Used for sampling synthetic stints and for placing collocation points. |
| `REFERENCE` | `Context` | The centre of each range. Where the exponent vanishes. |
| `GROUND_TRUTH` | `TireParams` | The constants the synthetic generator uses. **A starting point, not a calibration** — see TUNING.md. |
| `LEARNABLE_PARAMS` | 10 names | Which constants the PINN estimates. |
| `FIXED_PARAMS` | `A_gen`, `gamma1` | The two degenerate directions, held still. |
| `GAMMA2_RANGE` | `(0.20, 6.00)` | The box `gamma2` is confined to. |

### `Context` — the conditions of a stint

A frozen dataclass with five float fields. Constant within a stint, because one
set of tires runs one circuit at roughly one temperature on one compound.

| Member | What it does |
|---|---|
| `vector()` | The five numbers as a numpy array, canonical order. |
| `from_vector(v)` | Class method. The inverse of `vector()`. |
| `compound_name` | Property. The nearest compound name, for plot labels. |

### `TireParams` — the physical constants

A frozen dataclass holding the twelve constants, grouped by which equation they
belong to. Defaults are the values the synthetic generator uses, which is why
`GROUND_TRUTH = TireParams()` needs no arguments.

The same class holds both plain floats (when integrating) and torch tensors
(when training). Python does not enforce the type hints, and that is used
deliberately so one class serves both paths.

### Functions

| Function | What it does | How it works |
|---|---|---|
| `wear_constant(theta, q_fric, load, speed, track_temp, compound, p)` | `k(context, theta)` — everything in E2 that does **not** depend on `d`. | This is where the coupling lives: `theta` is added to the centred `track_temp` inside the same Arrhenius bracket. Load is clamped at `1e-6` because a negative base raised to a real power is NaN; the exponent is clamped at `EXPONENT_CAP`. With `theta = 0` it reduces exactly to the single-equation model. |
| `wear_rate(d, theta, ..., p)` | The right-hand side of E2, `dd/dtau`. | `wear_constant(...) * (1 - d)`. Non-negative while `d ≤ 1`, so the cliff is a steepening and never a reversal. |
| `thermal_rate(theta, d, q_fric, speed, p)` | The right-hand side of E1, `dtheta/dtau`. | Generation `A_gen·q_fric·(1 + zeta·d)` minus cooling `(h0 + h1·speed)·theta`. `d` enters here and `theta` enters E2: that mutual dependence is what makes the pair coupled. |
| `steady_temperature(d, q_fric, speed, p)` | Where E1 settles if `d` were held still. | `A_gen·q_fric·(1+zeta·d)/(h0+h1·speed)`. Not used in training — a reading instrument, and the quickest way to tell whether the thermal constants are sane. |
| `pace_loss(d, p)` | E3: latent `d` → seconds. | `gamma1·d + gamma2·d^8`. The even exponent means a slightly negative `d` gives a small positive contribution rather than a NaN. |
| `exact_solution_isothermal(tau, context, p)` | The closed form of E2 **alone**. | `1 - exp(-k·tau)`. Valid *only* when `A_gen = 0`, which kills generation so `theta` stays at 0 and `k` becomes constant. **Raises** if `A_gen ≠ 0` rather than quietly returning a plausible wrong curve. |
| `integrate_stint(n_laps, context, p, steps_per_lap=8)` | Solve both equations, returns `(laps, theta, d)`. | Runge-Kutta 4 with 8 sub-steps per lap. Both states advance **together** within each stage — E1 needs `d` and E2 needs `theta`, so stepping one then the other would silently make the scheme first-order. |
| `_is_tensor(x)` | Internal. True for torch tensors. | Returns False cleanly when torch is not installed. |
| `_backend(*values)` | Internal. Returns `torch` or `numpy`. | Every equation is written once and runs on both, so the same code defines the ground truth and the residual. Only `exp` and `clip` are needed, and both libraries spell those identically; powers use `**`. |

**What adding E1 cost:** the exact solution. With `theta` in the loop, `k` is no
longer constant in time and no closed form exists.
`exact_solution_isothermal` is what survives, and it exists for exactly one
reason — `integrate_stint` is now the only route to the truth, so it had better
be validated. `python tune.py --check-integrator` does that, and currently
reports a worst discrepancy of **3.05e-12**.

---

## 4. `data.py` — where the stints come from

### `Stint` — the unit of learning

| Field | Type | What it holds |
|---|---|---|
| `stint_id` | str | Unique name, e.g. `SYN03` or `MON-VER-S2`. |
| `context` | `Context` | The five conditions. |
| `laps` | array | Lap number within the stint: 1, 2, 3 … |
| `delta` | array | **Measured** pace loss in seconds. Carries noise. |
| `driver` | str or None | Three-letter code. Real stints only — a simulated stint was driven by nobody. |
| `d_true` | array or None | True wear. Synthetic only. **Checking, never training.** |
| `theta_true` | array or None | True temperature. Synthetic only. Kept so a run can be asked not just whether it got the seconds right but whether it got them right **for the right reason**. |
| `delta_clean` | array or None | Noise-free pace curve. Synthetic only. |

| Property | What it gives |
|---|---|
| `n_laps` | How many laps the stint lasted. |
| `tau` | `laps / LAP_REF` — the dimensionless time. |
| `compound` | The compound name, for grouping and labels. |

### Functions

| Function | What it does | How it works |
|---|---|---|
| `generate_synthetic(n_stints=48, p=GROUND_TRUTH, noise_s=0.05, min_laps=12, max_laps=30, seed=0)` | Simulate stints with known constants. | For each stint: pick a compound (cycling), draw a context from `CONTEXT_RANGES`, integrate **both** equations, convert `d` to seconds with `pace_loss`, then add Gaussian noise. There is no closed form for the coupled system, so this really is a numerical integration now. The noise matters — without it the fit is trivial and the physics term has nothing to do. |
| `load_csv(path, min_laps=8, drivers=())` | Read the CSV from `download_data.py`. | Groups rows by `stint_id`, sorts by `stint_lap`, checks the required columns are present and drops short stints. Raises `FileNotFoundError` with the exact download command if the file is missing. With `drivers`, keeps only those drivers' rows **before** the length check, so the discarded count and the errors speak about the drivers asked for; warns when a requested driver loses every stint to that check. |
| `split(stints, test_fraction=0.25, seed=0)` | Train/test split. | Permutes stint indices and takes a slice. **By whole stint, never by lap** — splitting by lap leaks, and it is the worst kind of leak because it flatters every model equally and cannot be spotted by comparing them. |
| `flatten(stints)` | Stints → the two matrices that train the network. | Returns `inputs (N, 6)` and `delta (N, 1)`. The constant context is tiled across the stint's laps with `np.tile`, which is what lets the network learn the effect of time and of conditions at once. |
| `_read_rows(path)` | Internal. Read the CSV and check the required columns exist. | Raises `FileNotFoundError` carrying the exact download command, or `ValueError` naming the missing columns. |
| `_stint_from_rows(id, rows)` | Internal. Build one `Stint` from its sorted rows. | Reads the context and the driver off the first row; every row of a stint carries the same value. |
| `_only_drivers(rows, drivers, path)` | Internal. The driver filter. | Case-insensitive. A requested driver who is not in the file raises `ValueError` listing who is — skipping it would let a typo (`VER HMA`) quietly train on VER alone and answer a different question. |
| `describe(stints)` | A paragraph summarising the set. | Counts, lengths, pace-loss range, compound breakdown and, on real data, stints per driver. Printed before training and written into `report.txt`. |
| `_sample_context(rng, compound)` | Internal. Draw plausible conditions. | Uniform within each `CONTEXT_RANGES` entry, except `compound`, which the caller decides. |

---

## 5. `pinn.py` — the network with physics inside

### `MLP` — just the network

`6 → width × layers → 2`, `tanh` between layers, nothing after the last one.
Default `64 × 4` gives **13,058 weights**.

| Member | What it does | How it works |
|---|---|---|
| `__init__(width=64, layers=4)` | Build the stack. | `nn.Sequential` of `Linear`/`Tanh` pairs. The last layer has no activation; whatever shaping the outputs need happens in the output transform, where it can be justified per state. |
| `_glorot_init()` | Initialise the weights. | `xavier_normal_` on weights, zeros on biases. Keeps signal variance steady across layers; without it, four layers of tanh saturate on iteration 1 and the network never learns. |
| `forward(tau, context)` | One batch through the network, raw `(N, 2)`. | Concatenates to `(N, 6)` and runs the stack. `tau` arrives separately so it can stay a leaf of the autograd graph. |

**One network with two outputs, not two networks.** The states are coupled, so
the features that explain one largely explain the other; sharing the hidden
layers learns them once instead of twice from the same data.

**Why tanh and not ReLU:** the derivative of a ReLU network is piecewise
constant, so it would predict a stepped, discontinuous `dtheta/dtau` that cannot
match a smooth right-hand side. (The usual argument — "ReLU's second derivative
is zero" — applies to second-order equations, not to these.)

### `PINN` — the network plus the ten constants

| Member | What it does | How it works |
|---|---|---|
| `__init__(width, layers, gamma1, A_gen, ic, tau_max, seed)` | Create everything. | Builds the `MLP` and one `nn.Parameter` per learnable constant in **raw** form. `gamma1` and `A_gen` arrive as arguments and stay fixed. `ic` is `"hard"` or `"soft"`. |
| `_to_raw(name, value)` | Physical value → the number the optimiser moves. | `log` for the positive ones, `logit` of the position in the box for `gamma2`, identity for `kappa`. |
| `_value(name)` | The inverse. | `exp`, or `low + (high−low)·sigmoid`, or identity. |
| `params_tensor()` | The constants as tensors. | Used inside the residuals so the constants enter the autograd graph and receive gradients. |
| `learned_params()` | The constants as plain floats. | `.detach()`-ed, for printing and reporting. |
| `n_weights()` | Count the network weights. | Excludes the ten constants. |
| **`state(tau, context)`** | **Raw output → the physical `(theta, d)`.** | Where the hard initial conditions live. See the table below. |
| **`residuals(tau, context)`** | **`(r_theta, r_d)`, one per equation.** | The heart of the method. Mark `tau` with `requires_grad_`, run the network and the transform, differentiate **each output separately**, subtract what the physics says. |
| `_d_dtau(output, tau)` | Exact derivative of one column w.r.t. `tau`. | `torch.autograd.grad` with `create_graph=True`, which is essential — without it the physics terms contribute no gradient at all. Called **twice**, once per state: one call with `grad_outputs=ones` over both columns would return `d(theta + d)/dtau`, the sum, which is not what either equation needs. |
| `collocation_points(n)` | Where the equations are enforced. | Uniform `tau` out to `tau_max` (the full 45-lap horizon) and uniform draws within each `CONTEXT_RANGES` entry. **No labels.** That is the entire reason these terms can cover ground the data never does — and it matters twice over here, because `theta` is never measured anywhere. |
| `_physics_losses(n)` | Terms 1 and 2. | Both residuals from the **same** points in one call: drawing two sets would double the cost, and coupled residuals have to be measured at the same place to be traded off. |
| `_data_loss(tau, context, delta)` | Term 3. | Forwards the observed laps, converts `d` to seconds with `pace_loss` (so the learned `gamma2` is fitted here too — it is the only place `gamma2` reaches the loss), compares against the measurement. |
| `_initial_condition_loss(n=64)` | Term 4. | Forwards `tau=0` with `n` random contexts. With `ic="hard"` it is identically zero and skipped entirely. |
| `train(inputs, delta, ...)` | Adam, then optionally L-BFGS. | Converts the inputs, assembles the weights, delegates to the two phase methods. |
| `_total_loss(...)` | Assemble the weighted sum. | Returns the pieces too, so logging needs no second pass. |
| `_run_adam(...)` | Phase 1: explore. | Cheap steps, collocation points **redrawn every iteration** — that is what stops the network satisfying the equations at a fixed finite set of places and nowhere between. |
| `_run_lbfgs(...)` | Phase 2: refine. | Quasi-Newton, uses curvature. The collocation points are **frozen**, drawn once before the loop: L-BFGS builds an internal model of the loss surface from successive evaluations, and resampling between them would make every evaluation a different function. Adam tolerates that; L-BFGS does not. |
| `_record(...)` | Save the state of an iteration. | Uses `.item()` rather than `float()`, because the tensors are still attached to the graph and `float()` would warn. |
| `_log_line(r)` | Format a progress line. | Reads from the recorded dict, so printing never touches the graph. |
| `predict_state(context, laps)` | Both latent states. | The window into what the model believes is happening inside the tire. `run.py` draws it as `04_state.png`. |
| `predict_stint(context, laps)` | Predicted pace loss. | `pace_loss` of the predicted `d`. **Same signature as the baseline**, which is what makes the comparison fair. |
| `predict_wear(context, laps)` | Just `d`. | For callers that do not want the pair. |
| `_columns(context)` | Internal. Split `(N,5)` into 5 columns. | Keeps the residuals readable. |

### The initial conditions, hard or soft

Two states start at zero: a tire leaving the pits is unworn and sits at the
reference thermal state.

| | How | Cost | When satisfied |
|---|---|---|---|
| `ic="hard"` *(default)* | `theta = tau·N₀`, `d = 1 − exp(−tau·softplus(N₁))` | none | **exactly, every iteration** |
| `ic="soft"` | one more loss term | competes for gradient | approximately, never exactly |

Because `tau` multiplies the output, at `tau = 0` the term vanishes whatever the
network says. The second transform is worth reading twice: it does not merely
pin `d(0)`, it makes the saturation bound **structural** — `softplus ≥ 0`, so the
exponent is `≤ 0`, so `d` can never leave `[0, 1)`. Three failure modes deleted
by one line, and nothing is lost: every curve from 0 that stays under 1 can
still be written this way.

`soft` is kept so the difference can be measured, not because it is a good idea.

### The loss terms

| Term | Default weight | Where it applies | What it imposes |
|---|---|---|---|
| `w_physics` | 1.0 | collocation points, no labels | E2 holds |
| `w_thermal` | 1.0 | the same points | E1 holds |
| `w_data` | 10.0 | observed laps only | match measured pace |
| `w_ic` | 10.0 | `tau = 0` | the initial conditions — **ignored when `ic="hard"`** |

The weights are **scales, not importances**. The residuals are dimensionless and
small; the data residual is in seconds. `w_thermal` is separate from `w_physics`
because `dtheta/dtau` is of order `A_gen` (units) while `dd/dtau` is of order
`kw` (a fraction). Measured on iteration 1 of a real run: `wear 0.03279` against
`heat 55.48579` — the thermal residual is born **1700× larger**. See
[TUNING.md](TUNING.md) section 5.

### How the ten constants are parametrised

| Form | Constants | Why |
|---|---|---|
| `log(value)` | `kw, m, Ea, Eq, Ev, zeta, h0, h1` | There is no negative wear coefficient or cooling rate. For `h0` it is load-bearing twice: `h0 ≤ 0` makes E1 unstable and `theta` runs away. |
| free | `kappa` | The one constant whose sign physics does **not** fix — there is analysis arguing 2026 inverted the compound ordering. Log-parametrising it would make the model unable to express that. |
| boxed sigmoid | `gamma2` → `GAMMA2_RANGE` | The most weakly identified constant: it only reaches the observable once `d` nears 1, and teams pit before that. The box asserts something we know — a destroyed tire costs a few seconds a lap, not millions. |

### `INITIAL_VALUES`

Starting points, deliberately away from the truth, so recovering `GROUND_TRUTH`
proves something. They are also a **tuning knob**: a coupled system has a much
rougher loss surface than a single equation, and where you start decides which
basin you land in.

| Constant | Starts at | | Constant | Starts at |
|---|---|---|---|---|
| `kw` | 0.30 | | `kappa` | 0.30 |
| `m` | 1.00 | | `zeta` | 0.50 |
| `Ea` | 0.50 | | `h0` | 3.00 |
| `Eq` | 0.20 | | `h1` | 1.50 |
| `Ev` | 0.20 | | `gamma2` | 1.00 |

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

### Constants

| Name | Value | What it is |
|---|---|---|
| `CLIFF_THRESHOLD` | `0.30` | Seconds per lap that count as falling off. |
| `CLIFF_SUSTAINED` | `4` | For how many consecutive laps. |

Both numbers were bought with a mistake. The original criterion was 0.15 s/lap
at a **single point**, and it fired on 100 % of curves that had no cliff at all
once real timing noise was added: it was measuring noise. Requiring the slope to
be *sustained* is what makes the detector a detector.

### `Metrics`

| Field | What it measures |
|---|---|
| `rmse` | `sqrt(mean(error²))`. Large errors weighted more. |
| `mae` | `mean(abs(error))`. Easier to read, less sensitive to outliers. |
| `max_error` | The single worst lap. |
| `violations_inside` | % of laps inside the stint where the model says the tire improves. |
| `violations_extrapolating` | The same, over the full 45-lap horizon. |
| `cliff_rate` | Fraction of stints where the model predicts a cliff at all. |
| `cliff_error` | Mean error, in laps, in **where** the cliff falls. `nan` with real data. |

### Functions

| Function | What it does | How it works |
|---|---|---|
| `cliff_lap(delta, threshold, sustained)` | The lap the tire falls off, or `None`. | Finds the first run of `sustained` consecutive laps each losing more than `threshold`. Returns a **1-based lap number**, which is why the index gets `+2`: `np.diff` shifts by one and laps start at one. |
| `evaluate(name, predict_stint, stints, true_stint=None)` | Measure any model. | For each stint: predict over the observed laps (error + violations), then over the full horizon (violations + cliff). Takes the prediction function, not the model, so it cannot accidentally depend on model internals. `true_stint` is passed only on the synthetic bench. |
| `_count_violations(delta, tolerance=1e-3)` | Count lap-to-lap improvements. | `np.diff` then count entries below `−tolerance`. The tolerance avoids flagging a millionth of a second of numerical noise. |
| `parameter_recovery(learned, truth, names)` | Compare estimates to truth. | Returns `(name, estimated, true, relative error %)`. Synthetic data only — with real data there is no truth to compare against. |

**Why monotonicity is measured separately.** A tire only degrades, so a model
predicting recovery is making a physically impossible claim — and no error
metric penalises it. A model that says lap 34 will be faster than lap 33 will not
be trusted on lap 20 either. The correct value is 0 %. Note that the coupled
model did not change this: `dd/dtau = k(1−d)` with `k > 0` stays non-negative
even with the feedback on, so the cliff is a *steepening*, and a predicted cliff
that dips is still a bug.

**Why the cliff lap is measured separately.** A model can have an excellent RMSE
and still put the cliff five laps late, because the cliff occupies few laps and
contributes little to a mean squared error. That is the failure that matters
most to a strategist, and the only way to see it is to measure it directly.
`cliff_error` only scores stints where the truth *has* a cliff and the model
found one — counting a miss as zero error would flatter a model that never
predicts one, which is what `cliff_rate` is there to catch.

---

---

## 8. `run.py` — the orchestrator

`main()` is a list of five calls, one per step. Reading `main()` tells you what
happens; opening one function tells you how.

| Step | Function | What it does |
|---|---|---|
| 1 | `get_stints(args)` | Simulate stints, or read the CSV (filtered by `--drivers` if given). Returns `None` on a read failure, or when fewer than 2 stints survive — the split needs one on each side, and one driver in one race is often only two or three. `main` turns `None` into a clean exit instead of a traceback. |
| 2 | `train_pinn(inputs, delta, args)` | Build the network and train it. |
| 3 | `train_baseline(inputs, delta)` | Fit the rival on exactly the same data. |
| 4 | `build_report(model, linear, test, synthetic)` | Measure both and assemble the text report. Delegates to `_results_table`, and then to `_recovery_table` (synthetic) or `_estimates_list` (real). |
| 5 | `save_outputs(...)` | Write the four figures and `report.txt`. |

With synthetic data step 4 adds the inverse-problem table — checking whether
the true constants were recovered — which is impossible with real data.

| Helper | What it does |
|---|---|
| `true_state(context, laps)` | `(theta, d)` from the equations with the true constants. Really integrates: there is no closed form for the coupled system. |
| `true_pace(context, laps)` | The noiseless pace loss. Same signature as any model's `predict_stint`, so it can be handed straight to `evaluate`. Passed **only** on the synthetic bench — with real data there is no true cliff lap, so that column stays blank rather than being filled with something invented. |
| `_representative_stints(stints)` | One stint per compound, SOFT/MEDIUM/HARD where possible. Shared by both stint figures. |

### Command-line options

| Flag | Default | What it does |
|---|---|---|
| `--source {synthetic,csv}` | `synthetic` | Where the stints come from. |
| `--csv PATH` | `data/races.csv` | The file `download_data.py` produced. |
| `--stints N` | `48` | How many stints to simulate. |
| `--noise S` | `0.05` | Timing noise in seconds, synthetic only. |
| `--min-laps N` | `8` | Discard shorter stints, CSV only. |
| `--drivers CODE ...` | everyone | Train and test only on these drivers' stints, CSV only. Case-insensitive. Asking for it with `--source synthetic` is an error, not a silent no-op. |
| `--width N` | `64` | Neurons per layer. |
| `--layers N` | `4` | Hidden layers. |
| `--iterations N` | `8000` | Adam iterations (phase 1). |
| `--lbfgs N` | `0` | L-BFGS iterations after Adam (phase 2). Try 500 when `zeta`, `h0` or `h1` refuse to move. |
| `--collocation N` | `2000` | Collocation points per iteration. |
| `--lr F` | `3e-3` | Learning rate. |
| `--ic {hard,soft}` | `hard` | How the initial conditions are imposed. |
| `--w-physics F` | `1.0` | Weight on the wear residual. |
| `--w-thermal F` | `1.0` | Weight on the thermal residual. Its natural scale is ~1700× larger, so this usually wants to be much smaller. |
| `--w-data F` | `10.0` | Weight on the fit to measured pace. |
| `--w-ic F` | `10.0` | Weight on the initial conditions. Ignored when `--ic hard`. |
| `--seed N` | `0` | Controls data generation, the split and the weight init. |
| `--out DIR` | `outputs` | Where figures and the report go. |
| `--quick` | off | 18 stints, 1,500 iterations. Checks it runs. |

### Plot functions

| Function | Output | What it shows |
|---|---|---|
| `plot_fit` | `01_fit.png` | One panel per compound, each drawn by `_draw_stint_panel`: measurements, the true curve, both models. The shaded band is where data exists; to its right everything is extrapolating. |
| `plot_training` | `02_training.png` | The loss terms on a log scale. A term that is identically zero (the IC term under `--ic hard`) is left out rather than drawn as a flat line a log axis cannot render. |
| `plot_parameters` | `03_parameters.png` | The ten constants over training, with the true value as a dashed line. A curve that is flat at its starting value is the signature of a constant that never received a usable gradient. |
| `plot_state` | `04_state.png` | **`theta` and `d`, the two latent states.** The most useful figure for calibration: a model can produce the right seconds for entirely wrong reasons, and the seconds plot cannot show that. Only the equation holds `theta` in place; if `theta` is wrong here, the fit is a coincidence. |

`matplotlib.use("Agg")` is called **before** importing pyplot, because pyplot
picks its backend at import time. Without it the script fails on a headless
machine.

---

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
| `--drivers ...` | all | Three-letter codes. Filters the **download**. To compare drivers, download everyone once and filter at training time with `run.py --drivers` instead: no second download. |
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

## 10. `tune.py` — the calibration bench

It trains nothing. It answers one question: *given these twelve constants, what
world do the equations describe?* Because before asking a network to recover
constants from noisy lap times, the constants have to describe a tire that
behaves like a tire. Nothing here writes back to `physics.py` — the last thing
it prints is the block to paste yourself.

[TUNING.md](TUNING.md) is the guide that goes with it.

| Function | What it does | How it works |
|---|---|---|
| `feedback_strength(p, context)` | **`beta`** — whether a cliff can exist at all. | `Ea·A_gen·q_fric·zeta / (h0 + h1·speed)`. Derived by substituting the quasi-steady `theta` into E2. |
| `acceleration_ends_at(beta)` | The wear level where the steepening stops. | `1 − 1/beta`, or 0 below `beta = 1`. |
| `thermal_time_constant(p, context)` | How many **laps** the temperature takes to settle. | `LAP_REF/(h0 + h1·speed)`. In laps rather than tau, because that is the only form anybody can sanity-check: 0.2 or 40 tells you the cooling constants are wrong. |
| `scenario_rows(p)` | One line per corner of the context space. | Integrates worst / middle / best over the full horizon. |
| `sample_contexts(n, seed)` | Contexts drawn the way `data.generate_synthetic` draws them. | Deliberately the same distribution — constants that behave on three hand-picked corners and badly on the sampled space would produce a training set nothing can learn from. |
| `population_statistics(p, n, seed)` | Summarise `n` integrated stints. | The corners give the range; only this gives the **bulk**, and the bulk is what the network learns from. |
| `measurements(p, n)` | Every graded number, in one dict. | Closed-form diagnostics plus the population stats. |
| `diagnosis_lines(numbers)` | Grade each against its target band. | OK / LOW / HIGH. The point is not the verdict but that a failing row names the direction. |
| `advice_lines(numbers)` | What to turn next, in order. | Returns early on `beta ≤ 1`, because every other row moves when `beta` does, and guards the `kw` suggestions so two of them cannot ask for opposite edits in the same breath. |
| `paste_block(p)` | The exact text to paste into `TireParams`. | Copyable rather than instructional: transcribing twelve numbers by hand is how one ends up different from what was tested. |
| `sweep(p, name, low, high, steps)` | Move ONE constant and tabulate. | In a coupled system nearly every constant moves nearly every output, so moving one at a time is the only way to build intuition. |
| `check_integrator(p, n)` | Validate RK4. | Sets `A_gen = 0`, which collapses the system back to the one equation that has a closed form, and compares. Currently **3.05e-12**. |
| `plot_curves(p, path)` | Three panels: `theta`, `d`, seconds. | The tables can all be green with the *shape* wrong — a temperature that never settles, a wear curve with a kink. Shape is what tables hide. |
| `params_from_args(args)` | Apply the flags on top of `GROUND_TRUTH`. | Returns the params and a list of what changed, so the run states its own inputs. |

### Command-line options

One `--<name>` flag per constant; any you leave out keeps the value in
`physics.py`. Plus:

| Flag | What it does |
|---|---|
| `--sweep NAME LOW HIGH STEPS` | Move one constant across a range and tabulate it. |
| `--check-integrator` | Validate RK4 against the exact isothermal solution. |
| `--plot PATH` | Draw `theta`, `d` and `delta` to a PNG. |
| `--samples N` | Stints to integrate for the population stats (default 240). |

### `TARGETS`

The bands the diagnosis grades against — `beta`, both `theta_ss` figures, the
thermal time constant, median `d`, the dead-tire fraction, two pace-loss
percentiles and two cliff rates. **They are not laws.** They are what this
project decided to aim for, they live at the top of `tune.py`, and they are
there to be argued with.

---

## 11. How to check it still works

| Check | Command | Expected |
|---|---|---|
| RK4 matches the closed form | `python tune.py --check-integrator` | `PASS`, ~`3e-12` |
| The pipeline runs | `python run.py --quick` | finishes in well under a minute |
| The bench runs | `python tune.py --samples 60` | a diagnosis table |
| A sweep runs | `python tune.py --sweep zeta 0.5 4.0 8` | eight rows |
| Every symbol documented | docstring audit (below) | 126 / 126 |

```python
import ast, pathlib
total = missing = 0
for f in sorted(pathlib.Path(".").glob("*.py")):
    for node in ast.walk(ast.parse(f.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            total += 1
            missing += not ast.get_docstring(node)
print(f"{total - missing}/{total}")
```

The default `python run.py` numbers are deliberately **not** listed here. The
constants in `physics.py` are a starting point, not a calibration, so any
reference result would be a reference to an arbitrary world. Calibrate first —
see TUNING.md — and then the numbers are yours to record.

---

## 12. Known limits of this version

Stated plainly, because the gap is the point of the branch:

- **The constants are uncalibrated.** `beta = 0.71 < 1` as shipped, so the
  equations *can* express a cliff but these particular values do not produce
  one. That is deliberate: the equations are the deliverable, the values are
  yours. `python tune.py` says so on the first line.
- **The thermal constants are the hard ones.** `zeta`, `h0` and `h1` only reach
  the observable through two layers of composition, and they are the worst
  recovered. Measured: `zeta` at 53 % error with Adam alone. `--lbfgs` fixes
  `h1` (34 % → 3.8 %) but not `zeta`; raising `beta` above 1 is what moves
  `zeta` (61 % → 24 %), because below 1 it barely reaches the observable at all.
- **`gamma2` is weakly identified by construction.** It only acts once `d`
  nears 1 and teams pit before that. This is a limitation of the problem, not of
  the method.
- **The race-lap correction is a straight line.** `main` fits a piecewise-linear
  spline, because that curve has no reason to be straight.
- **`Ev` is weakly identified at default settings.** Over the range `speed`
  varies, `Ev` moves the exponent by only 0.28, against 0.85 for `kappa` and
  0.95 for `Ea`. It is the smallest effect in the model and the first thing lost
  to noise.
- **No LSTM baseline.** Only the classic linear model competes here.
