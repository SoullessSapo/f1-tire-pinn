# F1 Tire PINN — Complete Project Guide

**Author:** Esteban Valencia
**Repository:** https://github.com/SoullessSapo/F1-CNN-IA
**Scope:** the neural network and the use of DeepXDE. The AWS cloud layer is **not** included.

---

## How to read this document

This is written for someone who knows how to program but does not necessarily
come from a mathematical-modelling or machine-learning background. Parts 1 to 5
therefore build the concepts from scratch, using software-engineering analogies,
before getting into the concrete decisions.

| Part | What it covers |
|---|---|
| **1** | Important clarification: this is **not** an LLM |
| **2** | The problem, in engineering terms |
| **3** | What the neural network actually is here, and what "training" means |
| **4** | What a differential equation is and why model this way |
| **5** | What "physics-informed" means — the central idea |
| **6** | Where the equations came from (the real physics) |
| **7** | The concrete architecture, and what DeepXDE does for us |
| **8** | The inverse problem: estimating physical constants from data |
| **9** | The data: where it comes from and what had to be invented |
| **10** | The four substantive problems encountered |
| **11** | Results |
| **12** | Limitations and next steps |

---

# 1. First, a clarification: this is not an LLM

Worth getting out of the way, because it changes all the intuitions.

An **LLM** (GPT, Claude) is a transformer with billions of parameters, trained on
terabytes of text, that predicts the next token. It costs millions of dollars to
train and needs data centres full of GPUs.

What is here is a **small dense network**: 6 inputs → 4 layers of 64 neurons →
2 outputs. That is roughly **13,000 parameters**, four orders of magnitude smaller
than the smallest thing you would call an LLM. It trains in 30 minutes on a
laptop CPU, no GPU required.

It is also not a **CNN** (convolutional network). CNNs are for data with spatial
structure — images, signals — where sliding a filter makes sense. Here the inputs
are six loose numbers with no spatial relationship between them, so a convolution
would contribute nothing.

The correct category is **PINN**: *Physics-Informed Neural Network*. What makes it
distinctive is not the network architecture, which is about as simple as they
come. What makes it distinctive is **how it is trained**, and that is what Part 5
explains.

---

# 2. The problem, in engineering terms

A Formula 1 tire degrades over a stint. At first the pace loss is gradual — a few
tenths per lap — and at some point it falls off a cliff. Grip collapses and the
car loses several seconds a lap at once.

Predicting the exact lap of that cliff is **the** strategic decision of a race.
Getting it wrong by one lap costs positions you do not get back.

Framed as an engineering problem:

- **Input:** onboard telemetry and lap timing, live.
- **Output:** which lap does the cliff arrive on this set, in these conditions?
- **Hard constraint:** the answer has to arrive in under 500 ms.
- **Awkward constraint:** the thing you want to predict (wear) **cannot be
  measured**. All you see is lap times.

That last point is what makes the problem interesting and what motivates
everything else.

## 2.1 Why the obvious approaches are not enough

The project PDF already identified three families and their limits:

**Empirical linear models** (what many teams use): fit a straight line of "seconds
lost per lap" per compound. Fast and interpretable, but a straight line
**cannot represent a cliff** — that is what being linear means.

**Pure deep learning** (LSTM, GRU): plenty of capacity to capture the
nonlinearity. The problem is that **nothing forces it to respect physics**. It
will happily predict that a worn tire regains grip. And it does: in my
measurements, 11.5 % of laps when extrapolating.

**Finite-element simulation**: physically exact but takes hours. Good for
designing the tire, useless for deciding during a race.

The PINN tries to keep the good parts of the last two: the flexibility of a neural
network with the guarantees of physics.

---

# 3. What the neural network is here, in programming terms

## 3.1 A network is a parametrised function

Forget the neuron analogy for a moment. For our purposes, a neural network is
simply **a mathematical function with a lot of adjustable parameters**:

```python
def network(input, parameters):    # parameters = ~13,000 numbers
    ...
    return output
```

The internal structure here is the simplest possible: multiply by a matrix, apply
a nonlinear function (`tanh`), repeat four times. The nonlinearity is what lets it
represent curves rather than only straight lines.

The only thing that matters conceptually is this: **by changing those 13,000
parameters, that function can take almost any shape**. It is a universal mould.

## 3.2 "Training" is an optimisation problem

There is nothing magical about training. It is this:

1. Define a cost function that measures **how badly** the network is doing.
2. Compute the derivative of the cost with respect to each of the 13,000
   parameters.
3. Nudge each parameter slightly in the direction that reduces the cost.
4. Repeat 18,000 times.

That is gradient descent. The same thing you would do to minimise any function,
except in a 13,000-dimensional space.

**The analogy that probably helps most:** training is like a *linter with
autofix*. You define a set of rules (the cost function), and the optimiser keeps
rewriting the parameters until it satisfies them as well as it can. **The entire
design of this project consists of choosing the rules well.**

## 3.3 Automatic differentiation: the piece that makes it possible

Step 2 requires differentiating the cost with respect to 13,000 parameters. By
hand this is hopeless.

PyTorch does it for you, using **automatic differentiation**: every operation you
execute is recorded in a graph, and at the end the graph is walked backwards
applying the chain rule. The result is **exact** — it is not a numerical
approximation like `(f(x+h) − f(x)) / h`.

This is central to the project, and not only for the parameters. I also need to
differentiate **the network's output with respect to its own input** (`dθ/dτ`, the
rate at which temperature changes). Without exact automatic differentiation, PINNs
simply would not work.

---

# 4. What a differential equation is and why model with them

## 4.1 The idea

A differential equation does not describe **how much** something is, but **how
fast it changes**. Instead of saying "the temperature on lap 12 is 95 °C", it says
"temperature rises at this rate per lap, and that rate depends on the current
temperature".

In pseudocode, a differential equation is essentially the body of a simulation
loop:

```python
state = initial_state
for step in range(n):
    rate_of_change = f(state, inputs)   # <-- this is the differential equation
    state = state + rate_of_change * dt
```

## 4.2 Why this fits so well here

Because degradation is **cumulative and historical**. Wear on lap 20 depends on
everything that happened in the previous 19. A differential equation expresses
exactly that: the present state is the integral of the entire history.

And there is an additional advantage that turned out to be decisive. I can write
rules about `dd/dτ` (the *rate* of wear) that guarantee properties of `d`
(accumulated wear). Specifically: **if the wear rate is never negative, wear can
never decrease**. That is the physical guarantee the LSTM cannot give, and it
comes for free from the structure of the model.

## 4.3 The full system

The model is two coupled equations — "coupled" meaning each depends on the other,
so they have to be solved together:

```
(E1)  dθ/dτ = A_gen · q · (1 + ζ·d)  −  (h₀ + h₁·v) · θ
(E2)  dd/dτ = k_w · λ^m · exp(E_a·(θ + T_trk) − κ·c) · (1 − d)
```

With two state variables:

- **`θ`** — how much hotter the tire surface is than the asphalt.
- **`d`** — what fraction of the tread has been consumed (0 = new, 1 = gone).

**Neither can be observed.** They are proprietary to each team. That is the crux
of the problem, and Part 5 explains how it is resolved.

---

# 5. What "physics-informed" means — the central idea

This is the part that really matters.

## 5.1 How a normal network is trained

You give it (input, correct answer) pairs and punish it for every error:

```
cost = Σ (prediction − actual)²
```

That is all. **The network knows absolutely nothing else about the world.**
Outside the range where it saw data, it does whatever it likes. It is not that it
gets things wrong — it is that it has no reference at all.

## 5.2 How a PINN is trained

You add a term to the cost that measures **how much it violates the laws of
physics**.

Take equation (E1) and move everything to one side:

```
residual = dθ/dτ − [ A_gen·q·(1 + ζ·d) − (h₀ + h₁·v)·θ ]
```

If the network respects the physics, that residual is **zero**. If not, it
measures exactly how much thermodynamics is being ignored. So you put it in the
cost:

```
cost = Σ (prediction − actual)²      ← fit the data
     + Σ (residual_E1)²              ← respect the thermal balance
     + Σ (residual_E2)²              ← respect the wear law
```

**In programming terms:** the first term is your *tests*, which check specific
cases you know about. The other two are **invariants** or *asserts*, which must
hold always, on any input, whether or not you wrote a test for it.

## 5.3 And here is the important bit

Look at *where* each term is evaluated.

The data term **can only be evaluated where there is data**: the laps that were
actually driven.

The physics terms **can be evaluated anywhere**, because an equation does not need
measurements to tell you whether it holds. So I evaluate them at **thousands of
random points** scattered across the entire space of possible conditions: any
combination of compound, temperature, load, and any lap out to 45 — including
conditions nobody has ever driven.

Those points are called **collocation points**. In this project there are about
8,000 per iteration.

**That is the entire advantage of a PINN.** In the region without data, the LSTM
has nothing constraining it, which is why it predicts the tire regaining grip. The
PINN still has the differential equation sitting on top of it, so it keeps
behaving like a tire.

There is a second effect, almost more surprising: **the network learns to
reconstruct `d`, the wear, without ever having seen it**. It only sees lap times.
But because it is required to satisfy an equation relating `d` to `θ` and to the
observable, the only way to satisfy all the rules simultaneously is for `d` to
take the physically correct value. The figure `03_estados_latentes.png` shows the
reconstruction is nearly exact.

## 5.4 The price

It is not free. You are forcing the network to obey rules, so if your rules are
wrong, the model will be worse than an unconstrained one. **A PINN is only as good
as the physics you put into it.**

That shows up in the real-data results: with only 38 stints and high noise, the
PINN pays the cost of being constrained without yet being able to collect the
benefit.

---

# 6. Where the equations came from

I did not invent them. Every term comes from established physics. Here is the
justification for each, and the iterations it took.

## 6.1 (E1), the thermal balance

```
dθ/dτ = generation − cooling
```

**Generation:** braking and cornering rub rubber against asphalt, and that
friction becomes heat. The more frictional energy (`q`), the more heat.

**Cooling:** the tire gives heat back to the air and the asphalt. Heat-transfer
physics says the cooling rate is **proportional to the temperature difference**:
the hotter it is relative to its surroundings, the faster it cools. Hence the
`−h·θ` term.

The cooling coefficient has two parts: `h₀` is baseline cooling, and `h₁·v` is
**forced convection** — more speed means more air flowing, means more cooling.
That is why tires cool down on the straights.

This is called a **lumped-capacitance** model: you treat the tire as a single
block at uniform temperature instead of solving the temperature distribution
inside the rubber. It is an enormous simplification, and it is what makes this run
in milliseconds rather than hours.

## 6.2 (E2), the wear law

This combines two classical laws.

**Archard's law** (abrasive wear, 1953): material lost is proportional to the
applied load. Hence the `k_w · λ^m` term, where `λ` is mechanical load. I do not
fix the exponent `m` — **the network learns it**.

**The Arrhenius equation** (chemical kinetics, 1889): the rate of a chemical
process grows **exponentially** with temperature. It is one of the most universal
relationships in physical chemistry. Rubber degrades chemically as it heats, so it
applies directly: hence `exp(E_a · temperature)`.

That exponential is why tire degradation is so brutally nonlinear. A tire 10 °C
hotter does not wear a little faster — it wears *much* faster.

## 6.3 Iteration: the `(1 − d)` factor

**Testing the basic model, I found `d` reaching 3.24.** Not physical: you cannot
consume 324 % of the tread.

The fix was to multiply the wear rate by `(1 − d)`. As `d` approaches 1 that
factor approaches 0 and wear stalls. **`d` is bounded between 0 and 1
structurally.**

The elegant part is that the bound is **enforced by the equation itself**, not by
a penalty in the cost function. In programming terms: it is the difference between
a type that makes an invalid state unrepresentable, and a runtime validation you
have to remember to call. The former is always better.

## 6.4 Iteration: the `(1 + ζ·d)` factor — the key decision of the project

With that saturation in place, **the cliff disappeared**. No stint produced the
sharp drop any more.

There were two paths here.

**The easy path:** raise `γ₂`, the constant controlling the cliff term in the
observable. I rejected it, because that turns the cliff into an artefact I put in
by hand. If the cliff is in the model because I put it there, the model is not
explaining anything.

**The correct path:** ask *why the cliff actually exists*.

The answer is positive feedback:

```
the tread thins
    → the same frictional energy is spread over less rubber mass
        → surface temperature rises
            → by Arrhenius, wear accelerates exponentially
                → the tread thins faster
                    → (and round again, faster each time)
```

That is a self-reinforcing loop. Unnoticeable at first, and at some point it takes
off. **That is exactly what a cliff is.**

It is modelled with the `(1 + ζ·d)` factor on the heat-generation term: more wear
means more heating for the same energy. With it, **the cliff emerges by itself
from the dynamics**. I did not put it there: it falls out of the equations.

And it is precisely the kind of mechanism an LSTM has no way of discovering from
38 noisy stints. It is the central argument for why a PINN contributes something
to this problem.

## 6.5 The observable: connecting to what is measurable

`d` cannot be measured, but its effect can: the car goes slower. I model that
relationship as

```
δ = γ₁·d  +  γ₂·d⁸
```

- `γ₁·d` is **gradual degradation**, proportional to wear.
- `γ₂·d⁸` is the **grip collapse**. With exponent 8, this term is essentially zero
  while `d` is moderate (0.5⁸ ≈ 0.004) and explodes as `d` approaches 1
  (0.95⁸ ≈ 0.66).

Same trick you would use for a soft activation ramp in code: a high power acts as
a smooth switch.

---

# 7. The concrete architecture, and what DeepXDE does

## 7.1 The design decision: a parametric network

A textbook PINN solves **one single trajectory**: you give it the time, it returns
the state. The problem is obvious in our case: you would have to **retrain the
network for every new stint**. Thirty minutes per prediction. With a 500 ms
budget, absurd.

So the network takes the conditions as input too:

```
N(τ, q, λ, v, T_track, compound) → (θ, d)
 └─time──┘ └───────  context  ───────┘
```

This turns it into a **solution operator**: instead of learning one solution, it
learns **the entire family of solutions** to the differential equation, across the
full range of possible race conditions.

The practical consequence is what makes the whole project viable: predicting a new
stint is **one forward pass**. Measured: **0.42 ms mean, 1.47 ms at p95** for 45
laps. The 500 ms budget goes entirely to the network transport, not the model.

It also enables the decision map in `06_mapa_cliff.png`: 1,728 predictions
sweeping the entire condition space, in a single batched call.

**The trade-off**, which has to be declared: I assume context is constant within a
stint (I use its median). Lap-to-lap variation is absorbed into the data term.

## 7.2 Initial conditions: enforce them rather than ask for them

We know two things with absolute certainty: a new tire has `d = 0`, and it leaves
the pit at a known temperature.

The usual approach is to add cost terms penalising deviation from that. It works,
but only so-so. Now you have five terms competing and **you have to choose how
much each weighs** — which is by far the most common cause of a PINN failing to
converge.

Instead, I make it **impossible** to violate them:

```
θ(τ) = θ₀ + τ · N₀(x)          →  at τ=0 the second term vanishes: θ(0) = θ₀
d(τ) = τ · softplus(N₁(x))     →  d(0) = 0, and softplus > 0 guarantees d ≥ 0
```

Whatever the network produces internally, the initial condition holds
**exactly, by algebraic construction**.

Same philosophy as in 6.3: make invalid states unrepresentable rather than
validating them afterwards. And as a bonus it removes two cost terms and their
weight tuning.

## 7.3 What DeepXDE contributes

DeepXDE is the library the project PDF already proposed. Concretely it solves:

| Need | What DeepXDE gives |
|---|---|
| Differentiate output w.r.t. input | `dde.grad.jacobian(y, x, i, j)` — exact, no hand-written chain rule |
| Generate collocation points | `dde.geometry.Hypercube` samples them across the domain |
| Connect data to latent states | `PointSetOperatorBC`, which compares a *function* of the output against observations |
| Estimate physical constants | `dde.Variable`, optimised alongside the weights |
| Orchestrate training | Combines the five terms and manages Adam and L-BFGS |

The highest-value piece is `PointSetOperatorBC`. I do not observe `d`; I observe
`δ = γ₁d + γ₂d⁸`. This mechanism lets me tell DeepXDE: *"the network predicts `d`;
apply this transformation to it and compare **that** against the measurements"*.
Gradients flow backwards through the transformation all the way to the weights. It
is what makes it possible to train on a variable that is never seen.

## 7.4 The five cost terms

| Term | What it enforces | Where it is evaluated |
|---|---|---|
| L1 | residual of (E1), thermal balance | 8,000 points across the domain |
| L2 | residual of (E2), wear law | 8,000 points across the domain |
| L3 | bound `d ≤ d_max` | 8,000 points across the domain |
| L4 | fit to measured lap times | real laps only |
| L5 | temperature proxy (synthetic only) | real laps only |

## 7.5 Two optimisers, and why both are needed

I train first with **Adam** (15,000 iterations) and then with **L-BFGS** (3,000).

Adam takes small robust steps; it is good for exploring when you are far from the
solution. L-BFGS uses curvature information and converges much more finely, but
needs to already be close.

This is not a copied convention: **you can see the effect in the data**. In
`04_parametros.png`, the three thermal parameters (`ζ`, `h₀`, `h₁`) stall through
all 15,000 Adam iterations and **only jump to their true value once L-BFGS takes
over**.

The reason is that they are the worst-conditioned parameters in the problem: they
only influence the observable through two layers of composition (temperature →
wear → lap time), so their gradient is tiny. Adam does not move them. L-BFGS does.

---

# 8. The inverse problem

## 8.1 What it is

The equations have nine constants (`ζ, h₀, h₁, k_w, m, E_a, κ, γ₁, γ₂`).
**I know none of them.** They depend on Pirelli's compound, the asphalt, the car.

So I do not fix them: **I estimate them jointly with the network weights**. In
DeepXDE they are `dde.Variable`, and the optimiser treats them as nine more
parameters among the 13,000. This is called an **inverse problem**: instead of
solving the equation knowing the constants, you deduce the constants by observing
the result.

An implementation detail: I store them in **log space** and use `exp()` to recover
them. That makes them **positive by construction**, which their physical meaning
demands (there is no such thing as a negative cooling coefficient). Same
philosophy as 6.3 and 7.2.

## 8.2 How to validate something that cannot be validated

There is a serious methodological problem here: with real data **there is no
ground truth**. If the network estimates `E_a = 0.94`, there is no way to know
whether it was right.

That is why I built a **synthetic test bench**: I numerically integrate the
equations with constants I choose, add realistic measurement noise, and get data
where I **do** know the answer.

Then I start the PINN from deliberately wrong initial values and check whether it
recovers the true ones using **only lap times**.

**Result: 2.1 % mean error across all nine parameters.** That is the evidence that
the method works. `04_parametros.png` shows the convergence.

It is essentially an **integration test with synthetic data** — exactly what you
would do to test a system whose real outputs you cannot verify.

---

# 9. The data

## 9.1 The underlying problem

**Nothing the model needs is public.** Internal temperature, vertical load and
tread thickness are proprietary to each team.

What *is* public, via the **FastF1** library (official F1 data):

- Onboard telemetry: speed, throttle, brake, gear, RPM (~10 Hz).
- GPS position: X, Y, Z (~4 Hz).
- Timing: lap time, stint, compound, tire age.
- Weather: air and track temperature.

All the feature engineering consists of bridging that gap.

## 9.2 The proxy variables

| Variable | How it is built | Role in the model |
|---|---|---|
| `q_fric` | integral of \|acceleration\| × speed over the lap | heat generation in (E1) |
| `load` | mean total acceleration, in g | Archard term in (E2) |
| `speed` | mean speed | convective cooling |
| `track_temp` | weather interpolated at the lap's timestamp | ambient temperature |
| `compound` | 0 = soft, 0.5 = medium, 1 = hard | compound resistance |

**Lateral acceleration is not in the telemetry** — and it is the one that wears
tires most, because it is the cornering load. I reconstruct it by
**differentiating the GPS trajectory twice**: the first derivative of position is
velocity, the second is acceleration, and its component perpendicular to the
direction of travel is the lateral one.

The problem is that differentiating twice **massively amplifies sampling noise**.
So it has to be smoothed first, with a Savitzky-Golay filter.

**A detail that cost one iteration:** the smoothing window is fixed in *seconds*,
not in number of samples. FastF1 merges car telemetry (~10 Hz) with GPS (~4 Hz) by
interpolating, so the effective sampling rate varies between laps. A window fixed
in samples would apply a physically different filter in each case.

## 9.3 The observable, and the mandatory correction

Degradation is measured as pace loss relative to the best lap of the stint. But
there is an effect that completely masks it:

**The car sheds around 100 kg of fuel over a race**, and that is worth more than a
second a lap. Uncorrected, the car is speeding up through the stint from getting
lighter **exactly while the tire is slowing it down from wear**, and the two
effects cancel visually.

It is corrected by subtracting `k_fuel × (laps_remaining)`, with
`k_fuel = 0.055 s/lap`.

## 9.4 Quality filters

Laps that are slow for reasons unrelated to the tire are discarded:

- Green flag only (`TrackStatus == 1`) — a safety car changes lap time by seconds.
- No pit in or pit out laps.
- Only laps FastF1 marks as `IsAccurate`.
- No laps deleted by the FIA.
- Fresh sets only — because `d(0) = 0` only makes sense on a new tire.

From Monza + Hungary 2023, **38 stints with 803 laps** survive.

---

# 10. The four substantive problems

This is probably the most useful part for the report. None was a programming bug:
all four were problems with the framing itself, and each teaches something.

## Problem 1 — Stints were being cut too early

**Symptom:** the inverse problem gave 11.8 % mean error, but `γ₂` was off by
**62 %**, and `k_w` and `γ₁` were compensating for each other (one 18 % low, the
other 16 % high).

**Diagnosis:** the synthetic generator cut the stint right at the cliff. Only 3 of
48 stints had a detectable cliff, and only 1 reached `d > 0.9`. But `γ₂`
**only does anything as `d` approaches 1**. With no data in that regime, there is
no information to estimate it from. The network was guessing.

**Fix:** leave 2–5 laps *after* the cliff. It is also more realistic: a team does
not stop at the exact instant, it loses laps deciding or waiting for a pit window.

**Result:** mean error **11.8 % → 2.1 %**. `γ₂` from 62 % to 4.7 %.

**Lesson:** a parameter is only estimable if the data covers the regime where that
parameter has an effect.

## Problem 2 — Normalisation was erasing the between-circuit signal

**Symptom:** training on one race, the context variables barely varied (load
between 0.95 and 1.04). The network could not learn to respond to conditions
because it never saw different conditions.

**Diagnosis:** I was normalising the proxies by dividing by the **median of the
session itself**. That puts Monza at 1.0 and Hungary also at 1.0 — two radically
different circuits — erasing exactly the variation that is needed.

**Fix:** normalise against **fixed physical references** (1900 W/kg, 3.8 g,
58 m/s), calibrated by measuring real races.

**Result:** with two races, load went from varying 0.95–1.04 to 0.84–1.59.

**Lesson:** normalising each sample against itself destroys precisely the
information that distinguishes samples from one another.

## Problem 3 — The origin of degradation was badly defined

**Symptom:** on real data the PINN gave **RMSE 2.82 s** (baselines: 0.59) and
16.4 % monotonicity violations. The parameters were degenerate: `E_a` collapsed to
0.03 (no thermal activation) and `m` to 0.13 (no load dependence). The network had
switched the physics off in order to fit.

**Diagnosis:** I measured `δ` on the first valid lap of every stint. Average:
**0.473 s**. Every stint was starting half a second above its own reference.

The cause is physical and obvious in hindsight: **a new set comes out cold and
gets *faster* for two or three laps** before it starts falling away. That is the
warm-up phase. My model is monotone by construction and **cannot represent it**,
so every stint started with an irresolvable conflict between what was observed and
what was predictable. The network absorbed the conflict by distorting the physical
constants.

**Fix:** anchor `d = 0` at the **performance peak** of the stint, not at its first
lap, and discard the warm-up laps.

**Result:** RMSE **2.82 → 0.567 s**. Violations **16.4 % → 0 %**.

**Lesson:** if your model cannot represent a phenomenon, do not leave that
phenomenon in the training data. It leaks, and not where you expect.

## Problem 4 — Exact scale degeneracy (the most serious)

**Symptom:** the full run on real data **diverged**: `γ₂ = 2.5 × 10¹³`,
`k_w = 0.026`, test RMSE of **5.8 billion seconds**. And the disconcerting part:
**the training loss was low** (0.117). By its own metric, the model was doing
fine.

**Diagnosis:** the model has an **exactly degenerate direction**. Multiply `d` by
any factor `ε` and divide the gammas appropriately:

```
d → ε·d ,  γ₁ → γ₁/ε ,  γ₂ → γ₂/ε⁸     →  δ stays EXACTLY the same
```

There are infinitely many parameter combinations producing identical predictions.
The optimiser has no way to prefer one, so it slides along that direction
indefinitely until it overflows numerical precision.

On the synthetic bench this did not happen because two things prevented it: the
thermal proxy, and the stints that saturate at `d = 1`. **With real data neither
exists.**

**Attempt 1:** bound `γ₁ ∈ [0.2, 4.0]` and `γ₂ ∈ [0.2, 6.0]` with a sigmoid. It
stopped the overflow (RMSE 1.56), but **both ended pinned against their ceilings**
— unmistakable evidence that the push was still there and the bound was only
covering it up.

**Final diagnosis:** the absolute scale of `d` **is not identifiable from race
data**. The only thing that could anchor it is the saturation at `d = 1`, and you
only reach that by destroying the tire. Teams pit long before. **This is not a
failure of the method: the information is not in the data.**

**Fix:** the thermo-mechanical law is calibrated on the synthetic bench, where
ground truth exists. On real telemetry only **`k_w` and `κ`** are fitted — the two
quantities that genuinely change between circuits and tire batches. "One second of
pace loss corresponds to this much wear" is a **calibration** statement, not
something lap times can answer.

**Result:** physically reasonable parameters (`k_w = 0.864`, `κ = 0.963`) and none
pinned against a bound.

> **The most quotable lesson of the project:** in a PINN with an inverse problem, a
> **low training loss guarantees absolutely nothing** if the model has degenerate
> directions. You have to enumerate them explicitly and close them one by one. It
> is the equivalent of having 100 % test coverage and still having a broken
> system, because the tests do not check what matters.

---

# 11. Results

## 11.1 Synthetic bench — 64 stints (48 train / 16 test)

| Model | RMSE [s] | MAE [s] | MaxErr [s] | Cliff MAE | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|---|---|
| **PINN** | **0.066** | **0.052** | **0.180** | **0.50** | **0.0 %** | **0.0 %** |
| Linear (classic) | 0.247 | 0.167 | 1.078 | — | 15.0 % | 12.5 % |
| LSTM (black box) | 0.081 | 0.058 | 0.641 | 1.00 | 1.3 % | 11.5 % |

**"Violations"** is the metric that best summarises the project: the percentage of
laps where the model predicts the tire **regaining** grip. It is physically
impossible, and **no error metric penalises it**, so it has to be measured
separately.

The LSTM is competitive inside the observed range (0.081 vs 0.066). But when
extrapolating, **11.5 % of its predictions are thermodynamically impossible**. The
linear model does not even detect the cliff, because a straight line cannot
represent one. **The PINN is the only one at 0 % in both regimes.**

## 11.2 Parameter recovery — 2.1 % mean error

`ζ` 2.3 % · `h₀` 1.4 % · `h₁` 0.9 % · `k_w` 2.9 % · `m` 2.9 % · `E_a` 1.0 % ·
`κ` 0.5 % · `γ₁` 2.3 % · `γ₂` 4.7 %

Using **only lap times**, never seeing temperature or wear.

## 11.3 Real telemetry — Monza + Hungary 2023, 38 stints

| Model | RMSE [s] | MAE [s] | Viol. interp. | Viol. extrap. |
|---|---|---|---|---|
| PINN | 0.697 | 0.476 | 18.3 % | 9.6 % |
| Linear (classic) | **0.529** | **0.424** | 0.0 % | 0.0 % |
| LSTM (black box) | 0.522 | 0.414 | 0.0 % | 5.6 % |

**Here the PINN does not beat the baselines, and that has to be said plainly.**

The reason is what Part 5.4 anticipated. With 38 stints, a ~0.5 s per lap noise
floor and little variation in conditions, the observed curve is **close to linear
over the measured range**, and a linear fit is hard to beat there. The PINN pays
the price of being constrained without collecting the benefit, because the benefit
lives in extrapolation and in the cliff — regimes this data barely touches.

The honest conclusion is that the real-data path is **mechanically validated but
not scientifically validated**: it runs end to end and produces interpretable
parameters, but it needs considerably more races.

## 11.4 Latency

**0.42 ms mean, 1.47 ms at p95** to predict a full 45-lap stint, on CPU. Three
orders of magnitude below the 500 ms budget.

---

# 12. Limitations and next steps

## 12.1 Limitations

- **Constant context per stint.** The median is used; lap-to-lap variation is
  absorbed into the data term. This is what makes the parametric operator
  tractable.
- **Track evolution not modelled.** The circuit rubbers in and gets faster during
  the race. That effect is not separated from degradation and biases the estimated
  slope.
- **`load` is a relative proxy, not a measurement.** Its absolute magnitude
  (≈3–4 g mean) is above what a real accelerometer would read, because it comes
  from differentiating GPS. What matters is that it is monotone in true load and
  discriminates between circuits, and both hold.
- **Fresh sets only.** Stints on used tires are discarded.
- **The warm-up phase is discarded** rather than modelled.

## 12.2 Next steps, by impact

1. **Train on 8–10 races**, not two. By far the change that would move the results
   most: it attacks both the data shortage and the lack of condition variation at
   once.
2. **Model track evolution** as a separate term, to stop confusing it with
   degradation.
3. **Add the warm-up phase** to the model, so those laps do not have to be thrown
   away.
4. **A driver/car effect term**, which currently all goes into the noise.

---

# 13. Reproducing it

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
set DDE_BACKEND=pytorch

python run_train.py --source synthetic --stints 64          # ~30 min
python run_train.py --source synthetic --quick              # ~2 min, pipeline check
python run_train.py --source fastf1 --gp Monza Hungary      # real telemetry
python run_infer.py --compound SOFT --track-temp 0.8        # inference + latency
```

## Code structure

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

## References

- Raissi, Perdikaris & Karniadakis (2019). *Physics-informed neural networks*.
  Journal of Computational Physics, 378, 686–707. — the foundational paper.
- Lu, Meng, Mao & Karniadakis (2021). *DeepXDE: A deep learning library for
  solving differential equations*. SIAM Review, 63(1), 208–228.
- Archard, J.F. (1953). *Contact and rubbing of flat surfaces*. — the wear law.
- Arrhenius, S. (1889). — the exponential temperature dependence.
- Oehrly, M. *FastF1: A Python package for F1 telemetry and timing data*.
