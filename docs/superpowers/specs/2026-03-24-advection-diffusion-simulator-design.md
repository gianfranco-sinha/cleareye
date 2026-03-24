# Advection-Diffusion Simulator Design Spec

## Purpose

A 1D advection-diffusion solver for the ClearEye dual-sensor pipe geometry. Solves Fick's law with advection to model particle/colloid transport through the measurement pipe. Used in the training pipeline for synthetic data generation, data cleansing (outlier detection via physics plausibility), and physics-informed feature extraction. Not used during real-time inference.

## Governing Equation

```
∂C/∂t + v·∂C/∂x = D_eff·∂²C/∂x²
```

Where:
- C(x, t) — concentration field
- v — mean flow velocity (m/s)
- D_eff — effective dispersion coefficient (m²/s), includes Taylor dispersion

### Taylor Dispersion

In laminar pipe flow, the parabolic velocity profile and radial diffusion combine to produce enhanced axial mixing:

```
D_eff = D_molecular + (R²·v²) / (48·D_molecular)
```

Where R is the pipe inner radius. At higher flow rates, D_eff increases — faster flow means more shear-driven mixing.

### Temperature Correction

D_molecular scales with temperature via Stokes-Einstein:

```
viscosity_ratio = 1.0 + 0.02 * (20.0 - temperature)
D_corrected = D_molecular / max(viscosity_ratio, 0.1)
```

Reference temperature: 20 deg C. This linear approximation is accurate within ~5% for 10-30 deg C and degrades at extremes (underestimates viscosity at 1 deg C by ~20%). Acceptable for training data generation — not metrology. Intentionally consistent with the same formula in `transport_physics.py`.

### Peclet Number

```
Pe = v·L_sensor / D_eff
```

Where `L_sensor` is the sensor spacing (default 0.200m, from `pipe_geometry.yaml`: downstream position minus upstream position). This matches the definition in `transport_physics.py`.

The simulator computes ground-truth Pe for each run, which maps directly to turbidity regime. Default thresholds (configurable in `pipe_geometry.yaml` under a `regime_thresholds` key):
- Pe > 1000: suspension (advection dominates, particles carried by flow)
- 10 < Pe < 1000: colloid (mixed transport)
- Pe < 10: solution (diffusion dominates)

These thresholds are initial estimates to be calibrated against real data or OpenFOAM validation. The trained ML models learn the signal-to-Pe-to-regime mapping from simulator-generated ground truth.

## Pipe Geometry

All geometry is defined in `pipe_geometry.yaml` (project root, alongside `quantities.yaml`):

```yaml
pipe:
  length_mm: 300
  inner_radius_mm: 25
  mesh_opening_mm: 0.5

sensors:
  - name: upstream
    position_mm: 50
  - name: downstream
    position_mm: 250

flow:
  regime: laminar
  velocity_range:
    min_m_s: 0.001
    max_m_s: 0.5

perturbation_zones: []
# Future: sensor head protrusion correction
# - position_mm: 50
#   dispersion_multiplier: 1.2

regime_thresholds:
  suspension_pe: 1000    # Pe above this → suspension
  solution_pe: 10        # Pe below this → solution
  # Between solution_pe and suspension_pe → colloid
```

`pipe_geometry.yaml` is the single source of truth for geometry. Loading is handled exclusively by `simulator/geometry.py`, which exposes a `PipeGeometry` dataclass and a module-level `load_geometry()` function. If the file is missing, `load_geometry()` raises `ConfigError` (from `app/exceptions.py`) — no silent fallback.

The existing `DEFAULT_SENSOR_SPACING` and `DEFAULT_PIPE_LENGTH` constants in `transport_physics.py` remain as hardcoded defaults for backward compatibility. They are not migrated to read from the YAML — the simulator uses `PipeGeometry` directly, and `transport_physics.py` functions continue to accept `sensor_spacing` as an optional parameter with the same defaults.

### Copper Mesh Boundaries

The mesh openings (0.5mm) are orders of magnitude larger than the particles and colloids being measured (sub-micron to tens of microns). The mesh is transparent to concentration transport. Its only effect is a minor flow perturbation at the pipe ends, which is negligible for 1D modelling.

### Sensor Head Protrusion

The sensor probes protrude into the pipe, creating a local flow disturbance (wake/recirculation zone). In 1D, this is approximated as a local dispersion enhancement at the sensor position. Implemented as a placeholder via `perturbation_zones` in the YAML config — a list of (position, dispersion_multiplier) tuples. Default is empty (no perturbation). To be populated later if OpenFOAM validation shows the effect is significant.

## Numerical Method

### Crank-Nicolson Finite Difference

Semi-implicit scheme, second-order accurate in both space and time. Unconditionally stable (no CFL constraint on timestep).

At each timestep, the discretised PDE produces a tridiagonal linear system. Define:

```
r_d = D_eff·dt / dx²          (diffusion number)
r_a = v·dt / dx                (Courant number)
```

**Full Crank-Nicolson discretisation with upwind advection (v > 0):**

Implicit side (time level n+1):
```
a_i · C[i-1]^(n+1) + b_i · C[i]^(n+1) + c_i · C[i+1]^(n+1)
```

Explicit side (time level n):
```
= d_i · C[i-1]^n + e_i · C[i]^n + f_i · C[i+1]^n
```

**Coefficients (upwind advection blended with Crank-Nicolson):**

```
a_i = -r_d/2 - r_a/2           (sub-diagonal: diffusion + upwind advection)
b_i =  1 + r_d + r_a/2         (main diagonal)
c_i = -r_d/2                   (super-diagonal: diffusion only, upwind has no downstream term)

d_i =  r_d/2 + r_a/2           (explicit sub-diagonal)
e_i =  1 - r_d - r_a/2         (explicit main diagonal)
f_i =  r_d/2                   (explicit super-diagonal)
```

The matrix is tridiagonal but asymmetric due to the advection terms. For v < 0 (reverse flow), the upwind direction flips: the advection contribution moves from `a_i` to `c_i`.

Solved via `scipy.linalg.solve_banded` (vectorised tridiagonal solve) — O(N) per timestep, numpy-level performance. No pure Python loops in the inner solver.

### Timestep Selection

Crank-Nicolson is unconditionally stable, but accuracy requires the Courant number `r_a` to be O(1). Guidance:

```
dt = dx / v   (when v > 0)
```

At v=0.5 m/s, dx=5mm: dt=0.01s. For a 60s simulation: 6000 timesteps. The solver computes `dt` automatically from `dx` and `v` unless the caller overrides it.

### Grid Resolution

Configurable via `nx` parameter:
- Default `nx=60` (5mm spacing) — fast batch runs, milliseconds per simulation
- Fine `nx=300` (1mm spacing) — accurate parameter fitting

### Boundary Conditions

- **Inlet (x=0):** Dirichlet — C(0, t) = prescribed input function
- **Outlet (x=L):** Neumann — ∂C/∂x = 0 (zero gradient, open outflow)

## Input Conditions

### Boundary Condition Types (inlet)

| Type | Function | Use case |
|------|----------|----------|
| `pulse` | C₀·exp(-(t-t₀)²/2sigma²) | Transient events, cross-correlation analysis |
| `step` | 0 for t < t₀, C₀ for t >= t₀ | Sustained turbidity, steady-state calibration |
| `ramp` | Linear 0 to C₀ over duration tau | Gradual onset (algal bloom) |
| `arbitrary` | User-supplied C(t) array, linearly interpolated to solver timestep grid | Replay real upstream sensor data |

### Initial Conditions

C(x, t=0) — configurable:
- Default: uniform zero (clean pipe)
- Uniform C₀ (pre-contaminated pipe)
- Arbitrary C(x) array (pre-existing concentration gradient)

### Parameter Ranges for Synthetic Generation

| Parameter | Range | Distribution | Physical meaning |
|-----------|-------|--------------|-----------------|
| D_molecular | 1e-12 to 1e-5 m²/s | Log-uniform | 1e-12 = large sand, 1e-5 = dissolved ions |
| velocity | 0.001 to 0.5 m/s | Log-uniform | Laminar pipe flow range |
| C₀ | 1 to 5000 (arb. units) | Log-uniform | Turbidity event magnitude |
| temperature | 1 to 35 deg C | Uniform | Affects D via Stokes-Einstein |
| pipe_radius | From YAML | Fixed | For Taylor dispersion calculation |

## Output

### SimulationParams

```python
@dataclass
class SimulationParams:
    d_molecular: float          # molecular diffusion coefficient (m²/s)
    velocity: float             # mean flow velocity (m/s)
    c0: float                   # inlet concentration magnitude (arb. units)
    temperature: float          # water temperature (deg C)
    nx: int                     # spatial grid points
    dt: float | None            # timestep (s), None = auto from Courant condition
    duration: float             # simulation duration (s)
    boundary_type: str          # "pulse", "step", "ramp", or "arbitrary"
    geometry: PipeGeometry      # pipe geometry reference
```

### SimulationResult

```python
@dataclass
class SimulationResult:
    concentration: np.ndarray   # C(x, t) — shape (nt, nx)
    upstream: np.ndarray        # C(t) at upstream sensor position
    downstream: np.ndarray      # C(t) at downstream sensor position
    time: np.ndarray            # time array (seconds)
    x: np.ndarray               # spatial grid (metres)
    params: SimulationParams    # input parameters
    peclet_number: float        # Pe = v·L_sensor/D_eff
    d_effective: float          # D_eff with Taylor dispersion
    transit_time: float         # L_sensor/v (seconds)
    mass_conservation_error: float  # relative error in total mass (diagnostic)
```

**Naming convention:** sensor outputs use `upstream`/`downstream` throughout the simulator, matching `pipe_geometry.yaml`. When passing to existing `transport_physics.py` functions (which use `signal_a`/`signal_b`), the caller maps: `upstream` = `signal_a`, `downstream` = `signal_b`.

Output is raw concentration at sensor positions. No sensor transfer function applied — the simulator is a pure physics engine. Conversion to ADC/voltage is done by the caller using existing `sensor_physics.py` functions.

## Parameter Fitting (Inverse Solver)

Given observed sensor_a(t) and sensor_b(t) from real readings, find best-fit (D_molecular, velocity, C₀).

### Method

L-BFGS-B optimisation (bounded, gradient-free-friendly) via `scipy.optimize.minimize`.

### Cost Function

```
cost(D, v, C₀) = w_up·||sim_upstream(t) - obs_upstream(t)||² + w_down·||sim_downstream(t) - obs_downstream(t)||²
```

Where w_up, w_down are weights (default equal). The optimiser runs the forward solver at each iteration.

### Bounds

Same physical ranges as synthetic generation. Prevents non-physical parameter space.

### FitResult

```python
from app.regime import TurbidityRegime

@dataclass
class FitResult:
    d_molecular: float              # best-fit molecular diffusion
    d_effective: float              # with Taylor dispersion
    velocity: float                 # best-fit flow velocity
    c0: float                       # best-fit concentration
    peclet_number: float            # derived from fit
    regime: TurbidityRegime         # derived from Pe thresholds
    residual: float                 # final cost value (goodness of fit)
    converged: bool                 # optimiser converged
    at_bound: bool                  # True if any parameter hit its bound
    simulation: SimulationResult    # best-fit forward run
```

### Failure Modes

- **Non-convergence** (L-BFGS-B hits max iterations): `converged=False`, `residual` is populated with the last value, all other fields are the best parameters found so far. The caller should treat these results with low confidence.
- **At-bound solution** (`at_bound=True`): the optimiser pushed a parameter to its physical bound, suggesting the true optimum may lie outside the modelled range. Results are valid but flagged — useful as an outlier signal in data cleansing.
- **Flat cost function** (e.g. pure noise input): the optimiser converges but `residual` is high. No special handling — the high residual naturally flags this as a poor fit.
- **Numerical instability** (NaN in forward solve): returns `converged=False`, `residual=inf`. The solver catches NaN and aborts early.

### Usage in Training Pipeline

- **Data cleansing (`clean` stage):** if `residual` exceeds threshold, the reading pair is physically implausible — flag as outlier
- **Feature engineering (`feature_engineer` stage):** `d_effective`, `peclet_number`, and `residual` become ML input features
- **Causal event database (future):** fitted parameter set = event fingerprint

### Performance

Each fit runs the forward solver ~20-50 L-BFGS-B iterations. Each iteration requires ~7 forward solves (1 function eval + 6 finite-difference gradient evals over 3 parameters). At nx=60 with `scipy.linalg.solve_banded` (vectorised, no Python loops), each forward solve takes ~0.3ms, giving ~100ms per fit. For 50K readings: ~80 minutes as a batch job. Fits are independent per reading pair, so embarrassingly parallel — the solver is stateless and safe for `concurrent.futures.ProcessPoolExecutor`.

## Synthetic Data Generation

`generate_synthetic_dataset(n_samples, ...)` function that:

1. Samples parameters from configured ranges (log-uniform for D, velocity, C₀; uniform for temperature)
2. Runs forward solver for each parameter set
3. Extracts sensor_a(t) and sensor_b(t) time series
4. Computes ground-truth Pe and regime label
5. Returns a DataFrame with columns:
   - `upstream_signal`, `downstream_signal` — sensor time series (stored as arrays)
   - `d_molecular`, `d_effective`, `velocity`, `c0`, `temperature` — input parameters
   - `peclet_number_true` — ground-truth Pe from known simulation parameters
   - `regime` — ground-truth TurbidityRegime label (from true Pe)
   - `cross_correlation_peak`, `cross_correlation_lag`, `peak_attenuation`, `signal_broadening` — extracted features (via existing `transport_physics.dual_sensor_features()`). Note: `dual_sensor_features()` also returns an estimated `peclet_number` derived from attenuation — this differs from `peclet_number_true` and is intentionally included as a separate feature for the ML to learn the estimation error.

This feeds the training pipeline as augmentation data or pre-training data for the dual-sensor regime classifier.

## File Structure

```
simulator/
    __init__.py
    solver.py                 # AdvectionDiffusionSolver, Crank-Nicolson, Thomas algorithm
    conditions.py             # BoundaryCondition types, initial conditions
    taylor_dispersion.py      # effective_diffusion(), Stokes-Einstein correction
    fitting.py                # InverseSolver, FitResult, cost function
    synthetic.py              # generate_synthetic_dataset(), parameter sampling
    geometry.py               # PipeGeometry dataclass, load pipe_geometry.yaml

pipe_geometry.yaml            # Project root

tests/unit/test_simulator/
    __init__.py
    test_solver.py
    test_conditions.py
    test_taylor_dispersion.py
    test_fitting.py
    test_synthetic.py
    test_geometry.py
```

## Integration Points

### With existing code

- `pipe_geometry.yaml` is the single source of truth for geometry. The simulator reads it via `simulator.geometry.load_geometry()`. The existing hardcoded constants in `transport_physics.py` (`DEFAULT_SENSOR_SPACING`, `DEFAULT_PIPE_LENGTH`) remain unchanged for backward compatibility — the simulator uses `PipeGeometry` directly and does not modify `transport_physics.py`.

### With training pipeline

- `training/pipeline/` clean stage imports `simulator.fitting.InverseSolver` for outlier detection (dual-sensor data only)
- `training/pipeline/` feature_engineer stage imports fitted parameters as ML features
- `simulator.synthetic.generate_synthetic_dataset()` callable from training pipeline or standalone

### With app/ (real-time inference)

No dependency. `app/` never imports from `simulator/`. The physics knowledge flows through trained models and the existing analytical approximations in `transport_physics.py`.

## Dependency

`scipy` — added explicitly to `requirements.txt`. Used for:
- `scipy.linalg.solve_banded` (tridiagonal solve, alternative to custom Thomas algorithm)
- `scipy.optimize.minimize` (L-BFGS-B for parameter fitting)

Already available transitively via scikit-learn but should be declared directly.

## Future Extensions (documented, not built)

### Offline Inference Pipeline

A batch job that runs periodically on recent dual-sensor readings:
1. Fetch recent reading pairs from data store
2. Run inverse solver on each pair
3. Produce enriched results: fitted D, Pe, residual, regime, event fingerprint
4. Feed back into real-time system as updated priors or correction factors

The simulator's fitting interface is designed to support this — `InverseSolver.fit()` accepts raw sensor arrays and returns `FitResult`. The pipeline orchestration (scheduling, data fetching, result storage) is not built yet.

**When to build:** once dual-sensor data is flowing from the rig.

### Causal Event Database

A lookup/similarity table mapping fitted parameter fingerprints to event types:
- Each entry: fitted (D, v, C₀, Pe) + event label (storm runoff, algal bloom, dredging, etc.)
- Seeded by domain expert labelling of a small set of fitted events
- Over time, the ML learns the fingerprint-to-event mapping

**When to build:** once parameter fitting is validated against real dual-sensor data and a seed set of labelled events exists.

### OpenFOAM Validation

Run 20-50 full 3D CFD simulations across the parameter space to validate the 1D model:
- Compare 1D vs 3D concentration at sensor positions
- If agreement is within sensor noise, the 1D model is sufficient
- If sensor protrusion effect is significant, derive a correction lookup table

**When to build:** once the 1D simulator is working and producing results that can be compared.
