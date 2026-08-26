# F1 Tire PINN

A Physics-Informed Neural Network that predicts Formula 1 tire degradation and
the lap at which the performance **cliff** hits, built with **DeepXDE** on
PyTorch.

This is the modelling half of the project *Real-Time Prediction of Formula 1 Tire
Degradation using Physics-Informed Neural Networks*. **The AWS cloud layer is out
of scope**: what lives here is the physical system, the network, offline
training, the comparison baselines and the evaluation.

![Extrapolation: PINN vs black box](outputs/02_extrapolation.png)

*The figure that sums up the whole project. The grey band is the observed laps;
to the right of the dashed line every model is extrapolating. In the left panel
the LSTM (blue) **bends downwards** — it predicts the tire regains grip, which is
thermodynamically impossible. In the middle panel the linear model (grey) goes
negative: it predicts the tire is faster than new. The PINN (red) keeps
integrating the differential equation and saturates where it should.*

---

## 1. The physical model

Two coupled dimensionless ODEs describe the life of a stint:

```
(E1)  dθ/dτ = A_gen · q · (1 + ζ·d)  −  (h₀ + h₁·v) · θ        thermal balance
(E2)  dd/dτ = k_w · λ^m · exp(E_a·(θ + T_trk) − κ·c) · (1 − d)  wear
```

| Symbol | Meaning | Source |
|---|---|---|
| `τ` | stint lap / `L_ref` | dimensionless time |
| `θ` | `(T_surface − T_track) / ΔT_ref` | **latent state**, never observed |
| `d` | fraction of tread consumed | **latent state**, never observed |
| `q` | specific frictional energy per lap | telemetry |
| `λ` | mean mechanical load in g | telemetry |
| `v` | mean speed (convective cooling) | telemetry |
| `T_trk` | normalised track temperature | session weather |
| `c` | compound hardness (0 soft … 1 hard) | timing |

**(E1)** is a lumped-capacitance heat balance: frictional generation minus
exponential surface decay. **(E2)** combines Archard's wear law with an
Arrhenius thermal activation — wear grows exponentially with temperature.

Two factors do the heavy lifting:

- **`(1 − d)` in (E2)** bounds `d ∈ [0,1]` *structurally*. You cannot consume more
  tread than exists. The physical bound is enforced by the equation itself, not by
  a penalty term.
- **`(1 + ζ·d)` in (E1)** is what drives the cliff: as the tread thins, the same
  frictional energy is deposited into less rubber → temperature rises → Arrhenius
  accelerates wear → the tread thins faster. It is positive feedback, so **the
  cliff emerges from the coupled dynamics instead of being hard-coded**. This is
  exactly the kind of constraint an LSTM has no way of knowing.

The measurable observable is not `d` but the pace loss:

```
δ(τ) = γ₁·d + γ₂·d^p        (p = 8)
```

`γ₁·d` is gradual degradation; `γ₂·d^p` is negligible until `d` approaches 1 and
then dominates — the grip collapse.

---

## 2. Why this PINN is parametric

A textbook PINN solves **one** trajectory: the network takes `t` and returns the
state. That would mean retraining for every stint, which is useless for live
inference. Here the network is a **solution operator**:

```
N(τ, q, λ, v, T_trk, c) → (θ, d)
```

It learns the entire *family* of solutions to the ODE across the full range of
race conditions, once. Predicting a new stint is **a single forward pass** — no
retraining, no integration. That is what makes the offline-training /
online-inference split viable.

### Loss function

| Term | What it enforces | Where |
|---|---|---|
| `L1` | residual of (E1) | the whole condition hypercube |
| `L2` | residual of (E2) | the whole condition hypercube |
| `L3` | bound `d ≤ d_max` | the whole hypercube |
| `L4` | fit to measured pace loss | observed points only |
| `L5` | temperature proxy (optional) | observed points only |

`L1`–`L3` are enforced **even where there is no data**, out to the full decision
horizon (45 laps by default) rather than just as far as the longest observed
stint. That is the edge over a black box: outside its training distribution, an
LSTM has nothing tying it to thermodynamics.

### Hard initial conditions

Imposed by output transform, not as loss terms:

```
θ(τ) = θ₀ + τ · N₀(x)          ⟹  θ(0) = θ₀ exactly
d(τ) = τ · softplus(N₁(x))     ⟹  d(0) = 0 exactly and d ≥ 0 always
```

This removes two loss terms, and with them the weight-balancing problem that is
the single most common cause of PINN convergence failures.

### Inverse problem

The physical coefficients (`ζ, h₀, h₁, k_w, m, E_a, κ, γ₁, γ₂`) are unknown: they
are estimated **jointly** with the network weights as `dde.Variable`. They are
parametrised in log space, so they are positive by construction — which is what
their physical meaning demands.

### The two degeneracies (and how they are closed)

This model has **two exactly degenerate directions**. Ignoring them does not
produce a mediocre fit — it produces divergence.

**1. Temperature scale.** If `A_gen`, the scale of `θ` and `E_a` were all free,
doubling `A_gen` and halving `E_a` would leave wear unchanged. Closed by
**fixing `A_gen`**, which anchors the thermal scale. Under `--source synthetic`
the weak temperature supervision (`L5`) helps too; with real data `L5` is
disabled automatically, because internal tire temperature is not public.

**2. Wear scale.** This is the dangerous one:

```
d → ε·d ,  γ₁ → γ₁/ε ,  γ₂ → γ₂/ε^p     leaves δ exactly unchanged
```

On the synthetic bench two things break it: the thermal proxy, and the stints
that saturate at `d = 1`. **With real data neither exists**, and the optimiser
slides along that direction until it overflows. This happened literally: a run on
Monza + Hungary ended with `γ₂ = 2.5 × 10¹³`, `k_w = 0.026` and a test RMSE of
5.8 × 10⁹ s — with a **low training loss** (0.117), because along the degenerate
direction the fit is perfect.

Closed by bounding `γ₁ ∈ [0.2, 4.0] s` and `γ₂ ∈ [0.2, 6.0] s` through a sigmoid
(`gamma1_bounds`, `gamma2_bounds` in `PhysicsConfig`). This is not numerical
caution: it asserts something we genuinely know — a destroyed tire costs a few
seconds a lap, not millions.

> The general lesson: in a PINN with an inverse problem, **a low training loss
> guarantees nothing** if the model has degenerate directions. You have to
> enumerate them and close them explicitly.

---

## 3. Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

DeepXDE needs to know its backend:

```bash
set DDE_BACKEND=pytorch
```

---

## 4. Usage

Train on the synthetic bench (needs no network access and no API):

```bash
python run_train.py --source synthetic --stints 64
```

Quick pipeline check (about 2 minutes):

```bash
python run_train.py --source synthetic --quick
```

Train on real telemetry. **Use several races**: within a single one the context
variables barely move (same circuit, same weather), so the network only really
sees the effect of compound and time.

```bash
python run_train.py --source fastf1 --year 2023 --gp Monza Hungary Bahrain Spain
```

Inference and latency measurement with a trained model:

```bash
python run_infer.py --compound SOFT --track-temp 0.8 --load 1.2
```

### Outputs in `outputs/`

| File | Contents |
|---|---|
| `01_stints.png` | predicted vs observed curves, test stints |
| `02_extrapolation.png` | behaviour beyond the data: PINN vs black box |
| `03_latent_states.png` | `θ` and `d` reconstructed vs synthetic ground truth |
| `04_parameters.png` | inverse-problem convergence |
| `05_loss.png` | evolution of each loss term |
| `06_cliff_map.png` | decision map: cliff lap by compound and conditions |
| `report.txt` | metrics table and parameter recovery |
| `pinn_weights.pt`, `pinn_params.json` | trained model, ready for inference |

---

## 5. The synthetic bench

`data_synthetic.py` integrates (E1)–(E2) with **known** parameters
(`physics.GROUND_TRUTH`) and adds measurement noise. It serves two purposes:

1. The pipeline runs without depending on the network or on FastF1.
2. **It validates the inverse problem**: the PINN starts from deliberately wrong
   initial values and must *recover* the true ones using only the observed pace
   loss. With real data that check is impossible — there is no ground truth.

The generator imitates a strategist's decision: the stint is cut **two to five
laps after** the cliff. No team runs a destroyed tire, but no team stops at the
exact instant either — they lose laps deciding, waiting for a pit window, or
covering a rival.

That margin matters more than it looks. Those are the only laps that carry
information about the `d → 1` regime, which is what `γ₂` and the scale of `k_w`
depend on. Cutting the stint exactly at the cliff gave a mean parameter-recovery
error of **11.8 %** (`γ₂` off by 62 %); leaving those few extra laps brings it
down to **2.1 %** (`γ₂` off by 4.7 %).

---

## 6. Real data: what is observable and what is not

**Nothing the model needs is directly observable.** Internal temperature,
vertical load and tread state are proprietary to each team. What is public is
onboard telemetry and lap timing. `data_fastf1.py` bridges that gap:

- **`q_fric`** — specific frictional power, integrating `|a|·v` over the lap. This
  is the heat-generation term of (E1).
- **`load`** — mean total acceleration in g. This is the Archard term of (E2).
- **`speed`** — mean speed, which governs convective cooling.

**Lateral acceleration is not in the telemetry**: it is reconstructed by
differentiating the GPS trajectory twice, with Savitzky-Golay smoothing first
because numerical second derivatives amplify sampling noise.

The degradation observable is pace loss **corrected for fuel**: a car sheds
~100 kg over a race and that is worth more than a second a lap. Uncorrected, the
weight loss completely masks degradation.

Quality filters applied: green flag only (`TrackStatus == 1`), no in/out laps,
`IsAccurate` only, no deleted laps, and fresh sets only (`FreshTyre`) — because
`d(0) = 0` only holds for a new tire.

The proxies are made dimensionless against **fixed references** (`q_fric_ref`,
`load_ref`, `speed_ref` in `DataConfig`), not against each session's median.
Normalising each race against itself would put both Monza and Hungary at 1.0 and
erase precisely the between-circuit variation the model needs. The constants are
calibrated from 2023 races (Monza 1877 W/kg · 3.39 g · 66.4 m/s; Hungary 1968 ·
4.35 · 51.6).

### The degradation origin is the peak, not the first lap

A new set comes out cold and gets *faster* for two or three laps before it starts
falling away. The model is monotone by construction and cannot represent that
warm-up phase. The fix is to anchor `d = 0` at the **performance peak** of the
stint and discard the laps before it.

This is not cosmetic. Without that anchoring every stint starts ~0.5 s
systematically offset between what is observed and what the model can predict,
and the network absorbs the conflict by degenerating the physical parameters: in
a test on Monza + Hungary, `E_a` collapsed to 0.03 (no thermal activation), `m`
to 0.13 (no load dependence) and `γ₂` blew up to 8.35. Fixing the anchoring
dropped PINN RMSE from **2.82 s to 0.57 s** and monotonicity violations from
**16.4 % to 0 %**.

### Known limitations with real data

- **Constant context per stint.** The model assumes `q`, `λ`, `v` are constant
  within a stint and uses their median. Lap-to-lap variation is absorbed by the
  data residual. This is the simplification that makes the parametric operator
  tractable.
- **`load` is a relative proxy, not a measurement.** It comes from
  double-differentiating GPS, and its absolute magnitude (≈3–4 g mean) sits above
  what a real accelerometer would read. What matters is that it is monotone in
  true load and discriminates between circuits, and both hold; the absolute value
  cancels when dividing by `load_ref`.
- **Track evolution.** The circuit rubbers in and gets faster during the race.
  That effect is not separated from degradation and biases the estimated slope.
- **Traffic and dirty air** are mitigated by the `max_delta_s` filter, not
  eliminated.
- **`γ₂` is weakly identified.** It only comes into play as `d → 1`, and teams pit
  before that, so there is little data in that regime. That is a real limitation
  of the problem, not of the method: the information about the final collapse
  simply is not in race data.
- **`k_w` and `γ₁` partially compensate.** `δ ≈ γ₁·d` and `d` scales with `k_w`,
  so underestimating one and overestimating the other leaves the pace curve almost
  identical. What breaks the degeneracy is the `(1−d)` saturation: once a stint
  approaches `d = 1`, the scale of `d` is pinned. Another reason long stints are
  valuable.

---

## 7. Results

Synthetic bench, 64 stints (48 train / 16 test, split by stint), 15 000 Adam
iterations + 3 000 L-BFGS, ~30 min on CPU:

| Model | RMSE [s] | MAE [s] | MaxErr [s] | Cliff MAE | Cliffs found | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|---|---|---|
| **PINN** | **0.066** | **0.052** | **0.180** | **0.50** | **2/2** | **0.0 %** | **0.0 %** |
| Linear (classic) | 0.247 | 0.167 | 1.078 | — | 0/2 | 15.0 % | 12.5 % |
| LSTM (black box) | 0.081 | 0.058 | 0.641 | 1.00 | 2/2 | 1.3 % | 11.5 % |

The LSTM is competitive **inside** the observed range (0.081 vs 0.066) but breaks
down when extrapolating: 11.5 % of laps predict the tire regaining grip. The
linear model does not even detect the cliff, because it cannot represent one.
**The PINN is the only model with 0 % violations in both regimes.**

> `Cliff MAE` is computed over the **2 test stints (of 16) that reach the cliff**.
> That is a small sample and the number should not be read as a tight interval:
> only ~1 stint in 4 reaches that regime, for the same reason `γ₂` is the
> worst-identified parameter.

![Per-stint prediction](outputs/01_stints.png)

*Degradation curves on the test set. The PINN and the LSTM both track the points
well; the linear model drifts systematically — in `SYN007` it even crosses into
negative values.*

### Latent states: what the network reconstructs without ever seeing it

![Latent states](outputs/03_latent_states.png)

*Conceptually the most important figure. `d` (bottom row) is the fraction of
tread consumed and **never appears in the training data** — the network only sees
lap times. The red curve landing on the black points means the network
reconstructed wear **purely because we forced it to satisfy the differential
equation**. A network without physics has no way to recover a state it never
observes.*

### Physical parameter recovery

![Parameter convergence](outputs/04_parameters.png)

*All nine physical constants converge to their true value (dashed black line)
starting from deliberately wrong initial values. Note the jump in `zeta`, `h0`
and `h1` past iteration 15 000: that is L-BFGS taking over.*

The PINN starts from deliberately different initial values and recovers the true
ones using **only the observed pace loss**:

| Parameter | Estimated | True | Error |
|---|---|---|---|
| `ζ` (cliff coupling) | 0.879 | 0.900 | 2.3 % |
| `h₀` (base cooling) | 5.917 | 6.000 | 1.4 % |
| `h₁` (forced convection) | 3.962 | 4.000 | 0.9 % |
| `k_w` (wear rate) | 0.534 | 0.550 | 2.9 % |
| `m` (load exponent) | 1.544 | 1.500 | 2.9 % |
| `E_a` (thermal activation) | 0.941 | 0.950 | 1.0 % |
| `κ` (compound hardness) | 0.846 | 0.850 | 0.5 % |
| `γ₁` (linear pace loss) | 1.382 | 1.350 | 2.3 % |
| `γ₂` (cliff magnitude) | 2.722 | 2.600 | 4.7 % |
| | | **mean** | **2.1 %** |

In `04_parameters.png` you can see that `ζ`, `h₀` and `h₁` stall through all
15 000 Adam iterations and only jump to their true value during the L-BFGS phase.
The thermal parameters are the worst-conditioned in the problem — they only reach
`δ` through two layers of composition — and need a second-order optimiser. That
is the concrete reason the Adam → L-BFGS regime is not optional here.

### On real telemetry (Monza + Hungary 2023, 36 stints)

Here the result is **worse**, and it is worth saying so plainly:

| Model | RMSE [s] | MAE [s] | Cliffs found | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|---|
| PINN | 1.172 | 0.725 | 7/9 | 6.3 % | 6.3 % |
| Linear (classic) | **0.539** | **0.429** | 0/9 | 0.0 % | 0.0 % |
| LSTM (black box) | 0.536 | 0.423 | 3/9 | 0.6 % | 8.8 % |

**The PINN does not beat the baselines on two races.** With 36 stints, a ~0.5 s
per lap noise floor and very little variation in conditions, the observed
degradation curve is close to linear over the measured range — and a linear fit
is hard to beat there. The PINN pays the price of being constrained without yet
being able to collect the benefit.

The fitted parameters are physically reasonable (`k_w = 0.813`, `κ = 0.968`) and
neither is pinned against a bound, which was the symptom of the degeneracy. But
the wear-ODE residual settles around 0.15 — two orders of magnitude worse than on
the synthetic bench — and that is where the remaining monotonicity violations
come from.

> **These real-data numbers are not stable run to run.** An earlier run of the
> same configuration gave PINN RMSE 0.697 with 18.3 % interpolation violations.
> The synthetic results, by contrast, reproduce to the digit. That contrast is
> itself a measurement: on real data the fit sits close to the degenerate
> direction described above, so it is badly conditioned and small numerical
> differences move the answer. Treat the real-data row as an order of magnitude,
> not as a precise figure.

The honest conclusion is that **the real-data path is mechanically validated but
not scientifically validated**: it runs end to end and produces interpretable
parameters, but it needs considerably more races before the physics starts paying
off. That is the natural continuation of this work.

### The end product: the decision map

![Cliff decision map](outputs/06_cliff_map.png)

*Expected cliff lap as a function of compound, track temperature and mechanical
load. Red = the cliff arrives early; grey = the set survives the full 45-lap
horizon. This map is 1 728 predictions and **is only possible because the network
is parametric**: each cell is a forward pass, not a retrain. It is the output a
race strategist would actually use on the pit wall.*

### Inference latency

Predicting a full 45-lap stint costs **0.42 ms on average, 1.47 ms at p95** (CPU,
500 repetitions). The project's 500 ms budget is consumed entirely by transport,
not by the model: the parametric network solves the ODE in a single forward pass.

---

## 8. Evaluation

Three dimensions, because they answer different questions:

- **RMSE / MAE** on pace loss: how wrong the model is on the lap it is looking at.
- **Cliff lap error**: how wrong it is on the one prediction that changes a
  strategy decision. A model can have a good global RMSE and still miss the cliff
  by five laps.
- **Monotonicity violations**: how often it predicts the tire *regaining* grip.
  That is physically impossible and no error metric penalises it, so it is
  measured separately. The correct value is 0 %.

The baselines are the two ends of the state of the art described in the project:
`LinearDegBaseline` (the empirical model teams use, generously extended with a
quadratic term and lap-context interactions so it is not a straw man) and
`LSTMBaseline` (the recurrent black box).

The split is **by whole stint**, never by lap: splitting by lap would leak
information from the same stint between train and test.

---

## 9. Structure

```
src/tirepinn/
  config.py          physical, network and data hyperparameters
  physics.py         the ODE system, RK4 integrator, cliff detection
  pinn.py            the parametric PINN (DeepXDE)
  dataset.py         Stint / StintDataset, splitting, domain bounds
  data_synthetic.py  test bench with known ground truth
  data_fastf1.py     real telemetry and feature engineering
  baselines.py       classic linear and LSTM
  evaluate.py        metrics
  plots.py           figures
run_train.py         training + comparison + figures
run_infer.py         inference and latency
```

A full walkthrough of the reasoning, the modelling choices and the four
substantive problems found during development is in
[DOCUMENTACION.md](DOCUMENTACION.md) *(in Spanish)*.

---

## 10. References

- Raissi, Perdikaris & Karniadakis (2019). *Physics-informed neural networks*.
  Journal of Computational Physics, 378, 686–707.
- Lu, Meng, Mao & Karniadakis (2021). *DeepXDE: A deep learning library for
  solving differential equations*. SIAM Review, 63(1), 208–228.
- Archard, J.F. (1953). *Contact and rubbing of flat surfaces*.
- Oehrly, M. *FastF1: A Python package for F1 telemetry and timing data*.
