"""Central configuration.

Every hyperparameter (physical, network and data) lives here so experiments are
reproducible and the report can quote concrete values.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

# Canonical order of the network input vector: (tau, context...)
CONTEXT_NAMES = ("q_fric", "load", "speed", "track_temp", "compound")
INPUT_NAMES = ("tau", *CONTEXT_NAMES)
INPUT_DIM = len(INPUT_NAMES)
OUTPUT_NAMES = ("theta", "d")
OUTPUT_DIM = len(OUTPUT_NAMES)

# Physical parameters the network can estimate as an inverse problem.
LEARNABLE_PARAMS = ("zeta", "h0", "h1", "kw", "m", "Ea", "kappa", "gamma1", "gamma2")

# Subset estimated when training on real telemetry. The rest stay pinned at the
# values calibrated on the physics bench.
#
# The reason is identifiability, not convenience. One race yields a few dozen
# stints with a ~0.5 s per lap noise floor and very little variation in
# conditions, which is not enough to separate nine coefficients: attempting it
# makes the thermal exponents collapse to zero.
#
# And there is something deeper. The absolute scale of d is NOT identifiable
# from race data, because the only observable is delta and nothing anchors d
# except the saturation at d = 1, which you only reach by destroying the tire.
# Teams pit long before that. Leaving gamma free makes the fit push d towards
# zero and gamma upwards until it hits the bound: verified on Monza + Hungary,
# where gamma1 and gamma2 both ended pinned against their ceilings.
#
# So gamma is taken from calibration too: "one second of pace loss corresponds
# to this much wear" is a calibration statement, not something lap times can
# answer. What stays free are the two quantities that genuinely do vary from one
# circuit or compound to another.
REAL_DATA_FREE_PARAMS = ("kw", "kappa")

# Compound hardness index: 0 = softest, 1 = hardest.
COMPOUND_INDEX = {
    "HYPERSOFT": 0.0,
    "ULTRASOFT": 0.0,
    "SUPERSOFT": 0.0,
    "SOFT": 0.0,
    "MEDIUM": 0.5,
    "HARD": 1.0,
    "INTERMEDIATE": 0.75,
    "WET": 1.0,
}


@dataclass(frozen=True)
class ContextRanges:
    """Physically admissible range of each (already normalised) context variable.

    These define the hypercube over which the ODE residuals are enforced, i.e.
    the region where physics regularises the network even without data.
    """

    tau: tuple[float, float] = (0.0, 1.40)
    q_fric: tuple[float, float] = (0.40, 1.60)
    load: tuple[float, float] = (0.50, 1.50)
    speed: tuple[float, float] = (0.60, 1.40)
    track_temp: tuple[float, float] = (0.00, 1.00)
    compound: tuple[float, float] = (0.00, 1.00)

    def lows(self) -> list[float]:
        return [getattr(self, n)[0] for n in INPUT_NAMES]

    def highs(self) -> list[float]:
        return [getattr(self, n)[1] for n in INPUT_NAMES]


@dataclass
class PhysicsConfig:
    """Non-dimensionalisation scales and thermo-mechanical model parameters."""

    # --- scales ---
    lap_ref: float = 30.0          # L_ref: laps per unit of tau
    dT_ref: float = 40.0           # Delta T_ref [K], the temperature scale
    theta_init: float = 0.5        # theta(tau=0): leaving the pits off blankets
    d_max: float = 1.10            # admissible upper bound on degradation

    # --- parameter that anchors the temperature scale (see README: identifiability) ---
    A_gen: float = 8.0
    train_A_gen: bool = False

    # --- initial values of the learnable parameters (inverse problem) ---
    zeta_init: float = 0.50        # wear -> temperature coupling (drives the cliff)
    h0_init: float = 4.0           # baseline cooling
    h1_init: float = 2.0           # speed-driven forced convection
    kw_init: float = 0.80          # wear coefficient (Archard)
    m_init: float = 1.00           # mechanical load exponent
    Ea_init: float = 0.60          # thermal activation (linearised Arrhenius)
    kappa_init: float = 0.50       # compound resistance to wear
    gamma1_init: float = 1.00      # linear pace loss [s]
    gamma2_init: float = 1.50      # cliff pace loss [s]

    # Hard physical bounds on the observable parameters. These are necessary,
    # not cosmetic: (d, gamma1, gamma2) have an exact scale degeneracy, because
    # d -> e*d with gamma1 -> gamma1/e and gamma2 -> gamma2/e^p leaves delta
    # unchanged. On synthetic data the degeneracy is broken by the thermal proxy
    # and by stints that saturate at d = 1; with real data neither exists and
    # the optimiser slides along that direction until it overflows. What closes
    # it is a simple physical assertion: a destroyed tire costs a few seconds a
    # lap, not millions.
    gamma1_bounds: tuple[float, float] = (0.2, 4.0)   # [s]
    gamma2_bounds: tuple[float, float] = (0.2, 6.0)   # [s]

    # kappa is the one physical coefficient whose SIGN is not fixed by physics.
    # It says how much a harder compound resists wear, via exp(-kappa*c) with
    # c = 0 soft, 1 hard. Positive kappa is the classic ordering, harder wears
    # less. Published analysis claims 2026 inverted that.
    #
    # Log-parametrising kappa, as the other ODE coefficients are, would force
    # kappa > 0 and make the model structurally incapable of expressing such an
    # inversion. Symmetric bounds are used instead, so the SIGN IS DECIDED BY THE
    # DATA rather than by the parametrisation. |kappa| <= 1.5 caps the
    # soft-to-hard wear ratio at e^1.5 ~ 4.5x either way, which is generous.
    # Unlike a cooling coefficient, nothing physical demands kappa > 0.
    #
    # Having freed it, the 2026 fit returns a small positive kappa. That is not
    # evidence against the reported inversion: measuring the rates directly,
    # controlling for driver and for the race-lap effect, gives HARD - MEDIUM =
    # -0.006 +/- 0.006 s/lap over 11 circuits, t = -0.96, with 5 of 11 circuits
    # going the other way. The 95% interval contains the published +0.006, zero,
    # and the classic ordering alike. Circuit-to-circuit variability is 3x the
    # effect, so a season cannot resolve it. See the 2026 section of the README.
    kappa_bounds: tuple[float, float] = (-1.5, 1.5)

    # --- pace-loss observable ---
    cliff_exponent: float = 8.0    # p in delta = g1*d + g2*d^p

    # --- operational definition of the cliff ---
    # The threshold and the minimum run length are set by a false-positive
    # experiment, not by taste. Real F1 lap timing carries sigma ~0.3-0.5 s of
    # noise from traffic, wind and driving. Against a purely linear curve with
    # NO cliff at the measured median rate of 0.09 s/lap, a single-point test at
    # 0.15 s/lap fires 100% of the time -- it detects noise, not cliffs.
    # Requiring 0.30 s/lap sustained over 4 consecutive laps drops false
    # positives to 0-1% while still catching 96-100% of genuine knees.
    # 0.30 s/lap is also ~4x the normal degradation rate, which is a defensible
    # physical definition of "falling off a cliff".
    cliff_slope_s_per_lap: float = 0.30
    cliff_min_run: int = 4
    d_crit: float = 0.85
    # Strategic decision horizon. It defines how far the ODE is enforced: the
    # physical domain is set by the question the model will be asked, not by how
    # long the observed stints happened to run.
    strategy_horizon: int = 45
    exp_clamp: float = 6.0         # cap on the Arrhenius exponent (stability)


@dataclass
class PINNConfig:
    """PINN architecture and training regime (DeepXDE)."""

    hidden: Sequence[int] = (64, 64, 64, 64)
    activation: str = "tanh"
    initializer: str = "Glorot normal"

    lr: float = 1e-3
    adam_iters: int = 15000
    lbfgs_iters: int = 3000
    display_every: int = 1000

    num_domain: int = 4000         # uniform collocation across the hypercube
    tau_collocation: int = 40      # points in tau per observed context

    # weights of the individual loss components
    w_pde_theta: float = 1.0
    w_pde_wear: float = 1.0
    w_bound: float = 10.0
    w_data_delta: float = 20.0
    w_data_theta: float = 1.0

    use_theta_proxy: bool = True   # weak supervision on temperature
    # Physical parameters that get estimated. Anything not listed here stays
    # fixed at its `PhysicsConfig` value.
    free_params: tuple[str, ...] = LEARNABLE_PARAMS
    seed: int = 42
    float64: bool = False


@dataclass
class DataConfig:
    """Stint source and preprocessing."""

    source: str = "synthetic"      # {"synthetic", "fastf1"}

    # --- synthetic generator (test bench with known physical ground truth) ---
    n_stints: int = 40
    min_stint: int = 12
    max_stint: int = 34
    noise_delta_s: float = 0.06
    noise_theta: float = 0.05

    # --- FastF1 ---
    year: int = 2023
    gp: str = "Monza"
    session: str = "R"
    drivers: tuple[str, ...] = ()
    cache_dir: str = "cache"
    # Correction for everything that changes lap time with race lap and is not
    # the tire: fuel burn plus track evolution. Estimating it per race beats
    # assuming a constant, because the effect scales with lap length and with how
    # fast a circuit rubbers in -- see `data_fastf1._estimate_race_lap_effect`.
    # The fixed figure below is only the fallback when estimation is disabled.
    estimate_race_lap_effect: bool = True
    fuel_effect_s_per_lap: float = 0.055   # fallback fuel correction [s/lap]

    # Fixed references for making the telemetry proxies dimensionless.
    # Deliberately NOT the session median: normalising each race against itself
    # would put both Monza and Hungary at 1.0 and erase precisely the
    # between-circuit variation the model needs to learn.
    # Calibrated on 2023 races (Monza: 1877 / 3.39 / 66.4;
    # Hungary: 1968 / 4.35 / 51.6).
    q_fric_ref: float = 1900.0     # specific frictional power [W/kg]
    load_ref: float = 3.8          # mean mechanical load [g]
    speed_ref: float = 58.0        # mean speed [m/s]

    min_stint_laps: int = 8
    ref_window: int = 3            # laps considered for the stint reference pace
    max_delta_s: float = 6.0       # discards laps lost to traffic/incidents
    only_fresh_tyres: bool = True  # d(0)=0 only holds for a brand-new set

    test_fraction: float = 0.25


@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    pinn: PINNConfig = field(default_factory=PINNConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ranges: ContextRanges = field(default_factory=ContextRanges)
    out_dir: str = "outputs"

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)
