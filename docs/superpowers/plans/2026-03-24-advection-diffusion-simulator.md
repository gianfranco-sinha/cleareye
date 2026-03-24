# Advection-Diffusion Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 1D advection-diffusion PDE solver for ClearEye's dual-sensor pipe geometry, with parameter fitting and synthetic data generation for the ML training pipeline.

**Architecture:** Crank-Nicolson finite difference solver in a top-level `simulator/` package. Six modules: geometry (YAML config), Taylor dispersion, boundary conditions, forward solver, inverse solver (parameter fitting), and synthetic dataset generator. No dependency on `app/` — the simulator is a pure physics engine.

**Tech Stack:** Python 3.12, NumPy, SciPy (`linalg.solve_banded`, `optimize.minimize`), pandas (synthetic output)

---

## File Structure

| File | Responsibility |
|------|---------------|
| **Create:** `pipe_geometry.yaml` | Pipe dimensions, sensor positions, flow config, Pe thresholds |
| **Create:** `simulator/__init__.py` | Package init, public API re-exports |
| **Create:** `simulator/geometry.py` | `PipeGeometry` dataclass, `load_geometry()` from YAML |
| **Create:** `simulator/taylor_dispersion.py` | `effective_diffusion()`, `temperature_correct_diffusion()` |
| **Create:** `simulator/conditions.py` | `BoundaryCondition` base + `Pulse`, `Step`, `Ramp`, `Arbitrary` subclasses; `InitialCondition` |
| **Create:** `simulator/solver.py` | `AdvectionDiffusionSolver`, `SimulationParams`, `SimulationResult` dataclasses, Crank-Nicolson solve |
| **Create:** `simulator/fitting.py` | `InverseSolver`, `FitResult` dataclass, L-BFGS-B cost function |
| **Create:** `simulator/synthetic.py` | `generate_synthetic_dataset()`, parameter sampling |
| **Modify:** `requirements.txt` | Add `scipy>=1.11.0` explicitly |
| **Create:** `tests/unit/test_simulator/__init__.py` | Test package init |
| **Create:** `tests/unit/test_simulator/test_geometry.py` | Tests for geometry loading |
| **Create:** `tests/unit/test_simulator/test_taylor_dispersion.py` | Tests for dispersion and temperature correction |
| **Create:** `tests/unit/test_simulator/test_conditions.py` | Tests for boundary/initial conditions |
| **Create:** `tests/unit/test_simulator/test_solver.py` | Tests for forward solver (conservation, analytical limits) |
| **Create:** `tests/unit/test_simulator/test_fitting.py` | Tests for inverse solver |
| **Create:** `tests/unit/test_simulator/test_synthetic.py` | Tests for synthetic generation |

---

### Task 1: Pipe Geometry YAML and Loader

**Files:**
- Create: `pipe_geometry.yaml`
- Create: `simulator/__init__.py`
- Create: `simulator/geometry.py`
- Create: `tests/unit/test_simulator/__init__.py`
- Create: `tests/unit/test_simulator/test_geometry.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for pipe geometry loading."""

import pytest

from simulator.geometry import PipeGeometry, load_geometry


class TestPipeGeometry:
    def test_load_default_geometry(self):
        geo = load_geometry()
        assert geo.pipe_length == pytest.approx(0.300)
        assert geo.inner_radius == pytest.approx(0.025)
        assert geo.mesh_opening == pytest.approx(0.0005)

    def test_sensor_positions(self):
        geo = load_geometry()
        assert geo.upstream_position == pytest.approx(0.050)
        assert geo.downstream_position == pytest.approx(0.250)

    def test_sensor_spacing_derived(self):
        geo = load_geometry()
        assert geo.sensor_spacing == pytest.approx(0.200)

    def test_velocity_range(self):
        geo = load_geometry()
        assert geo.velocity_min == pytest.approx(0.001)
        assert geo.velocity_max == pytest.approx(0.5)

    def test_regime_thresholds(self):
        geo = load_geometry()
        assert geo.suspension_pe == 1000
        assert geo.solution_pe == 10

    def test_missing_file_raises_config_error(self, tmp_path):
        from app.exceptions import ConfigError
        with pytest.raises(ConfigError):
            load_geometry(tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_raises_config_error(self, tmp_path):
        from app.exceptions import ConfigError
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("pipe:\n  length_mm: 300\n")  # missing sensors
        with pytest.raises(ConfigError):
            load_geometry(bad_file)

    def test_perturbation_zones_default_empty(self):
        geo = load_geometry()
        assert geo.perturbation_zones == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_simulator/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simulator'`

- [ ] **Step 3: Create empty `__init__.py` files**

Create `tests/unit/test_simulator/__init__.py` (empty file).

- [ ] **Step 4: Create `pipe_geometry.yaml`**

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

regime_thresholds:
  suspension_pe: 1000
  solution_pe: 10
```

- [ ] **Step 5: Implement `simulator/__init__.py` and `simulator/geometry.py`**

`simulator/__init__.py`:
```python
"""Advection-diffusion simulator for ClearEye dual-sensor pipe geometry."""
```

`simulator/geometry.py`:
```python
"""Pipe geometry configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.exceptions import ConfigError

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "pipe_geometry.yaml"


@dataclass(frozen=True)
class PipeGeometry:
    """Pipe geometry and sensor configuration.

    All lengths stored in metres (converted from mm in YAML).
    """

    pipe_length: float
    inner_radius: float
    mesh_opening: float
    upstream_position: float
    downstream_position: float
    velocity_min: float
    velocity_max: float
    suspension_pe: float
    solution_pe: float
    perturbation_zones: list[dict] = field(default_factory=list)

    @property
    def sensor_spacing(self) -> float:
        return self.downstream_position - self.upstream_position


def load_geometry(path: Path | None = None) -> PipeGeometry:
    """Load pipe geometry from YAML file.

    Args:
        path: Path to YAML file. Defaults to project root pipe_geometry.yaml.

    Raises:
        ConfigError: If the file is missing or malformed.
    """
    path = path or _DEFAULT_PATH
    if not path.exists():
        raise ConfigError(f"Pipe geometry file not found: {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid pipe geometry YAML: {e}") from e

    try:
        pipe = data["pipe"]
        sensors = {s["name"]: s["position_mm"] for s in data["sensors"]}
        flow = data["flow"]["velocity_range"]
        thresholds = data.get("regime_thresholds", {})

        return PipeGeometry(
            pipe_length=pipe["length_mm"] / 1000.0,
            inner_radius=pipe["inner_radius_mm"] / 1000.0,
            mesh_opening=pipe["mesh_opening_mm"] / 1000.0,
            upstream_position=sensors["upstream"] / 1000.0,
            downstream_position=sensors["downstream"] / 1000.0,
            velocity_min=flow["min_m_s"],
            velocity_max=flow["max_m_s"],
            suspension_pe=thresholds.get("suspension_pe", 1000),
            solution_pe=thresholds.get("solution_pe", 10),
            perturbation_zones=data.get("perturbation_zones", []),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"Malformed pipe geometry: {e}") from e
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_simulator/test_geometry.py -v`
Expected: 8 PASSED

- [ ] **Step 7: Commit**

```bash
git add pipe_geometry.yaml simulator/__init__.py simulator/geometry.py tests/unit/test_simulator/__init__.py tests/unit/test_simulator/test_geometry.py
git commit -m "feat(simulator): add pipe geometry YAML and loader"
```

---

### Task 2: Taylor Dispersion and Temperature Correction

**Files:**
- Create: `simulator/taylor_dispersion.py`
- Create: `tests/unit/test_simulator/test_taylor_dispersion.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for Taylor dispersion and temperature correction."""

import pytest

from simulator.taylor_dispersion import (
    effective_diffusion,
    temperature_correct_diffusion,
)


class TestTemperatureCorrection:
    def test_reference_temp_no_change(self):
        d = temperature_correct_diffusion(1e-9, temperature=20.0)
        assert d == pytest.approx(1e-9)

    def test_cold_water_reduces_diffusion(self):
        d_cold = temperature_correct_diffusion(1e-9, temperature=5.0)
        d_ref = temperature_correct_diffusion(1e-9, temperature=20.0)
        assert d_cold < d_ref

    def test_warm_water_increases_diffusion(self):
        d_warm = temperature_correct_diffusion(1e-9, temperature=30.0)
        d_ref = temperature_correct_diffusion(1e-9, temperature=20.0)
        assert d_warm > d_ref

    def test_extreme_cold_clamps_viscosity(self):
        # viscosity_ratio clamped at 0.1 minimum
        d = temperature_correct_diffusion(1e-9, temperature=-100.0)
        assert d > 0


class TestEffectiveDiffusion:
    def test_zero_velocity_returns_molecular(self):
        d_eff = effective_diffusion(d_molecular=1e-9, velocity=0.0, pipe_radius=0.025)
        assert d_eff == pytest.approx(1e-9)

    def test_taylor_dispersion_increases_with_velocity(self):
        d_slow = effective_diffusion(d_molecular=1e-9, velocity=0.01, pipe_radius=0.025)
        d_fast = effective_diffusion(d_molecular=1e-9, velocity=0.1, pipe_radius=0.025)
        assert d_fast > d_slow

    def test_taylor_dominates_at_high_velocity(self):
        # At high v, Taylor term R²v²/(48D) >> D_molecular
        d_eff = effective_diffusion(d_molecular=1e-9, velocity=0.5, pipe_radius=0.025)
        taylor_term = (0.025**2 * 0.5**2) / (48 * 1e-9)
        assert d_eff == pytest.approx(1e-9 + taylor_term)

    def test_with_temperature(self):
        d_cold = effective_diffusion(
            d_molecular=1e-9, velocity=0.1, pipe_radius=0.025, temperature=5.0
        )
        d_warm = effective_diffusion(
            d_molecular=1e-9, velocity=0.1, pipe_radius=0.025, temperature=30.0
        )
        assert d_warm > d_cold

    def test_perturbation_zone_multiplier(self):
        d_base = effective_diffusion(
            d_molecular=1e-9, velocity=0.1, pipe_radius=0.025
        )
        d_perturbed = effective_diffusion(
            d_molecular=1e-9, velocity=0.1, pipe_radius=0.025,
            perturbation_multiplier=1.5,
        )
        assert d_perturbed == pytest.approx(d_base * 1.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_simulator/test_taylor_dispersion.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `simulator/taylor_dispersion.py`**

```python
"""Taylor dispersion and temperature-corrected diffusion."""

from __future__ import annotations


def temperature_correct_diffusion(
    d_molecular: float,
    temperature: float = 20.0,
) -> float:
    """Apply Stokes-Einstein temperature correction to diffusion coefficient.

    Uses a linear viscosity approximation relative to 20 deg C.
    Accurate within ~5% for 10-30 deg C.

    Args:
        d_molecular: Molecular diffusion coefficient at 20 deg C (m^2/s).
        temperature: Water temperature in deg C.

    Returns:
        Temperature-corrected diffusion coefficient (m^2/s).
    """
    viscosity_ratio = 1.0 + 0.02 * (20.0 - temperature)
    return d_molecular / max(viscosity_ratio, 0.1)


def effective_diffusion(
    d_molecular: float,
    velocity: float,
    pipe_radius: float,
    temperature: float = 20.0,
    perturbation_multiplier: float = 1.0,
) -> float:
    """Compute effective dispersion coefficient with Taylor dispersion.

    D_eff = D_corrected + (R^2 * v^2) / (48 * D_corrected)

    Taylor dispersion captures the enhanced axial mixing caused by
    the parabolic velocity profile in laminar pipe flow.

    Args:
        d_molecular: Molecular diffusion coefficient at 20 deg C (m^2/s).
        velocity: Mean flow velocity (m/s).
        pipe_radius: Pipe inner radius (m).
        temperature: Water temperature in deg C.
        perturbation_multiplier: Local dispersion multiplier (for sensor
            head protrusion correction). Default 1.0 (no perturbation).

    Returns:
        Effective dispersion coefficient (m^2/s).
    """
    d_corrected = temperature_correct_diffusion(d_molecular, temperature)

    if velocity == 0.0 or d_corrected == 0.0:
        return d_corrected * perturbation_multiplier

    taylor_term = (pipe_radius**2 * velocity**2) / (48.0 * d_corrected)
    return (d_corrected + taylor_term) * perturbation_multiplier
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_simulator/test_taylor_dispersion.py -v`
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add simulator/taylor_dispersion.py tests/unit/test_simulator/test_taylor_dispersion.py
git commit -m "feat(simulator): add Taylor dispersion and temperature correction"
```

---

### Task 3: Boundary and Initial Conditions

**Files:**
- Create: `simulator/conditions.py`
- Create: `tests/unit/test_simulator/test_conditions.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for boundary and initial conditions."""

import numpy as np
import pytest

from simulator.conditions import (
    Arbitrary,
    InitialCondition,
    Pulse,
    Ramp,
    Step,
)


class TestPulse:
    def test_peak_at_t0(self):
        bc = Pulse(c0=100.0, t0=5.0, sigma=1.0)
        assert bc(5.0) == pytest.approx(100.0)

    def test_decays_away_from_peak(self):
        bc = Pulse(c0=100.0, t0=5.0, sigma=1.0)
        assert bc(5.0) > bc(8.0)
        assert bc(5.0) > bc(2.0)

    def test_array_evaluation(self):
        bc = Pulse(c0=100.0, t0=5.0, sigma=1.0)
        t = np.array([0.0, 5.0, 10.0])
        result = bc(t)
        assert result.shape == (3,)
        assert result[1] == pytest.approx(100.0)


class TestStep:
    def test_zero_before_t0(self):
        bc = Step(c0=50.0, t0=3.0)
        assert bc(2.0) == pytest.approx(0.0)

    def test_c0_after_t0(self):
        bc = Step(c0=50.0, t0=3.0)
        assert bc(5.0) == pytest.approx(50.0)

    def test_at_t0(self):
        bc = Step(c0=50.0, t0=3.0)
        assert bc(3.0) == pytest.approx(50.0)


class TestRamp:
    def test_zero_before_t0(self):
        bc = Ramp(c0=100.0, t0=2.0, tau=4.0)
        assert bc(1.0) == pytest.approx(0.0)

    def test_c0_after_ramp(self):
        bc = Ramp(c0=100.0, t0=2.0, tau=4.0)
        assert bc(10.0) == pytest.approx(100.0)

    def test_halfway(self):
        bc = Ramp(c0=100.0, t0=0.0, tau=10.0)
        assert bc(5.0) == pytest.approx(50.0)


class TestArbitrary:
    def test_interpolation(self):
        t_data = np.array([0.0, 1.0, 2.0])
        c_data = np.array([0.0, 50.0, 100.0])
        bc = Arbitrary(t_data, c_data)
        assert bc(0.5) == pytest.approx(25.0)

    def test_clamps_outside_range(self):
        t_data = np.array([0.0, 1.0])
        c_data = np.array([10.0, 20.0])
        bc = Arbitrary(t_data, c_data)
        assert bc(-1.0) == pytest.approx(10.0)
        assert bc(5.0) == pytest.approx(20.0)


class TestInitialCondition:
    def test_zeros(self):
        ic = InitialCondition.zeros(nx=10)
        assert ic.values.shape == (10,)
        assert np.all(ic.values == 0.0)

    def test_uniform(self):
        ic = InitialCondition.uniform(c0=42.0, nx=10)
        assert np.all(ic.values == pytest.approx(42.0))

    def test_from_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        ic = InitialCondition.from_array(arr)
        np.testing.assert_array_equal(ic.values, arr)

    def test_from_array_wrong_type_raises(self):
        with pytest.raises(TypeError):
            InitialCondition.from_array([1, 2, 3])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_simulator/test_conditions.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `simulator/conditions.py`**

```python
"""Boundary and initial conditions for the advection-diffusion solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class BoundaryCondition(Protocol):
    """Protocol for inlet boundary conditions. Callable(t) -> concentration."""

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray: ...


class Pulse:
    """Gaussian pulse inlet boundary condition.

    C(t) = c0 * exp(-(t - t0)^2 / (2 * sigma^2))
    """

    def __init__(self, c0: float, t0: float = 5.0, sigma: float = 1.0) -> None:
        self.c0 = c0
        self.t0 = t0
        self.sigma = sigma

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.c0 * np.exp(-((t - self.t0) ** 2) / (2.0 * self.sigma**2))


class Step:
    """Step function inlet boundary condition.

    C(t) = 0 for t < t0, c0 for t >= t0.
    """

    def __init__(self, c0: float, t0: float = 0.0) -> None:
        self.c0 = c0
        self.t0 = t0

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.c0 * np.heaviside(np.asarray(t) - self.t0, 1.0)


class Ramp:
    """Linear ramp inlet boundary condition.

    C(t) = 0 for t < t0, linear 0->c0 over duration tau, c0 after.
    """

    def __init__(self, c0: float, t0: float = 0.0, tau: float = 5.0) -> None:
        self.c0 = c0
        self.t0 = t0
        self.tau = tau

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        elapsed = (t_arr - self.t0) / self.tau
        return self.c0 * np.clip(elapsed, 0.0, 1.0)


class Arbitrary:
    """Arbitrary inlet boundary condition from user-supplied data.

    Linearly interpolated to solver timestep grid. Values outside
    the supplied time range are clamped to the nearest endpoint.
    """

    def __init__(self, t_data: np.ndarray, c_data: np.ndarray) -> None:
        self.t_data = t_data
        self.c_data = c_data

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.interp(t, self.t_data, self.c_data)


@dataclass
class InitialCondition:
    """Initial concentration field C(x, t=0)."""

    values: np.ndarray

    @classmethod
    def zeros(cls, nx: int) -> InitialCondition:
        return cls(values=np.zeros(nx))

    @classmethod
    def uniform(cls, c0: float, nx: int) -> InitialCondition:
        return cls(values=np.full(nx, c0))

    @classmethod
    def from_array(cls, arr: np.ndarray) -> InitialCondition:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(arr).__name__}")
        return cls(values=arr.copy())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_simulator/test_conditions.py -v`
Expected: 13 PASSED

- [ ] **Step 5: Commit**

```bash
git add simulator/conditions.py tests/unit/test_simulator/test_conditions.py
git commit -m "feat(simulator): add boundary and initial conditions"
```

---

### Task 4: Forward Solver — Crank-Nicolson

**Files:**
- Create: `simulator/solver.py`
- Create: `tests/unit/test_simulator/test_solver.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add scipy to requirements.txt**

Add `scipy>=1.11.0` to `requirements.txt` (after `scikit-learn` line).

- [ ] **Step 2: Write the failing tests**

```python
"""Tests for the 1D advection-diffusion forward solver."""

import numpy as np
import pytest

from simulator.conditions import InitialCondition, Pulse, Step
from simulator.geometry import load_geometry
from simulator.solver import (
    AdvectionDiffusionSolver,
    SimulationParams,
    SimulationResult,
)


@pytest.fixture
def geometry():
    return load_geometry()


@pytest.fixture
def solver(geometry):
    return AdvectionDiffusionSolver(geometry)


class TestSimulationResult:
    def test_result_has_expected_fields(self, solver):
        params = SimulationParams(
            d_molecular=1e-9, velocity=0.05, c0=100.0,
            temperature=20.0, nx=30, dt=None, duration=20.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=100.0, t0=5.0, sigma=1.0))
        assert isinstance(result, SimulationResult)
        assert result.concentration.ndim == 2
        assert len(result.upstream) == len(result.time)
        assert len(result.downstream) == len(result.time)
        assert result.peclet_number > 0
        assert result.d_effective > 0
        assert result.transit_time > 0

    def test_auto_timestep(self, solver):
        params = SimulationParams(
            d_molecular=1e-9, velocity=0.1, c0=100.0,
            temperature=20.0, nx=60, dt=None, duration=10.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=100.0))
        # dt should be auto-computed from Courant condition: dx / v
        dx = solver.geometry.pipe_length / (params.nx - 1)
        expected_dt = dx / params.velocity
        actual_dt = result.time[1] - result.time[0]
        assert actual_dt == pytest.approx(expected_dt, rel=0.01)


class TestMassConservation:
    def test_pulse_mass_conservation(self, solver):
        """Total mass in domain + outflow should be tracked."""
        params = SimulationParams(
            d_molecular=1e-7, velocity=0.05, c0=100.0,
            temperature=20.0, nx=60, dt=None, duration=30.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=100.0, t0=5.0, sigma=1.0))
        # Mass conservation error should be small (< 5%)
        assert abs(result.mass_conservation_error) < 0.05


class TestPhysicalBehaviour:
    def test_pulse_arrives_at_downstream_sensor(self, solver):
        """A pulse should appear at the downstream sensor after a delay."""
        params = SimulationParams(
            d_molecular=1e-9, velocity=0.1, c0=100.0,
            temperature=20.0, nx=60, dt=None, duration=15.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=100.0, t0=2.0, sigma=0.5))
        # Downstream peak should be after upstream peak
        upstream_peak_t = result.time[np.argmax(result.upstream)]
        downstream_peak_t = result.time[np.argmax(result.downstream)]
        assert downstream_peak_t > upstream_peak_t

    def test_high_diffusion_broadens_peak(self, solver):
        """Higher diffusion → broader downstream peak."""
        bc = Pulse(c0=100.0, t0=2.0, sigma=0.3)
        params_low_d = SimulationParams(
            d_molecular=1e-9, velocity=0.05, c0=100.0,
            temperature=20.0, nx=60, dt=None, duration=30.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        params_high_d = SimulationParams(
            d_molecular=1e-5, velocity=0.05, c0=100.0,
            temperature=20.0, nx=60, dt=None, duration=30.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result_low = solver.solve(params_low_d, bc)
        result_high = solver.solve(params_high_d, bc)
        # Downstream peak should be lower with higher diffusion
        assert np.max(result_high.downstream) < np.max(result_low.downstream)

    def test_step_input_reaches_steady_state(self, solver):
        """A step input should eventually fill the pipe uniformly."""
        params = SimulationParams(
            d_molecular=1e-6, velocity=0.1, c0=50.0,
            temperature=20.0, nx=60, dt=None, duration=60.0,
            boundary_type="step", geometry=solver.geometry,
        )
        result = solver.solve(params, Step(c0=50.0))
        # At the end, downstream should be close to c0
        assert result.downstream[-1] == pytest.approx(50.0, rel=0.1)

    def test_zero_velocity_pure_diffusion(self, solver):
        """With v=0, transport is purely diffusive — symmetric spreading."""
        ic = InitialCondition.zeros(60)
        # Place concentration in the middle
        ic.values[28:32] = 100.0
        params = SimulationParams(
            d_molecular=1e-5, velocity=0.0, c0=0.0,
            temperature=20.0, nx=60, dt=0.01, duration=5.0,
            boundary_type="step", geometry=solver.geometry,
        )
        result = solver.solve(params, Step(c0=0.0), initial_condition=ic)
        # Concentration should spread symmetrically
        final = result.concentration[-1, :]
        center = 30
        # Check approximate symmetry around center
        left_mass = np.sum(final[:center])
        right_mass = np.sum(final[center:])
        assert left_mass == pytest.approx(right_mass, rel=0.15)

    def test_nan_returns_early(self, solver):
        """Extreme parameters that would cause NaN should be caught."""
        params = SimulationParams(
            d_molecular=1e-20, velocity=100.0, c0=1e30,
            temperature=20.0, nx=10, dt=0.001, duration=1.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        # Should not raise — solver catches NaN
        result = solver.solve(params, Pulse(c0=1e30))
        # Result may have NaN/inf or early termination but shouldn't crash
        assert isinstance(result, SimulationResult)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_simulator/test_solver.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement `simulator/solver.py`**

```python
"""1D advection-diffusion solver using Crank-Nicolson finite differences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from simulator.conditions import BoundaryCondition, InitialCondition
from simulator.geometry import PipeGeometry
from simulator.taylor_dispersion import effective_diffusion


@dataclass
class SimulationParams:
    """Input parameters for a single simulation run."""

    d_molecular: float
    velocity: float
    c0: float
    temperature: float
    nx: int
    dt: float | None
    duration: float
    boundary_type: str
    geometry: PipeGeometry


@dataclass
class SimulationResult:
    """Output of a forward simulation run."""

    concentration: np.ndarray
    upstream: np.ndarray
    downstream: np.ndarray
    time: np.ndarray
    x: np.ndarray
    params: SimulationParams
    peclet_number: float
    d_effective: float
    transit_time: float
    mass_conservation_error: float


class AdvectionDiffusionSolver:
    """1D advection-diffusion PDE solver.

    Uses Crank-Nicolson (semi-implicit) finite differences with
    first-order upwind advection. Tridiagonal system solved via
    scipy.linalg.solve_banded.

    The solver is stateless — safe for concurrent use.
    """

    def __init__(self, geometry: PipeGeometry) -> None:
        self.geometry = geometry

    def solve(
        self,
        params: SimulationParams,
        boundary_condition: BoundaryCondition,
        initial_condition: InitialCondition | None = None,
    ) -> SimulationResult:
        """Run the forward simulation.

        Args:
            params: Simulation parameters.
            boundary_condition: Callable(t) returning inlet concentration.
            initial_condition: C(x, t=0). Defaults to zeros.

        Returns:
            SimulationResult with full concentration field and sensor traces.
        """
        geo = self.geometry
        nx = params.nx
        L = geo.pipe_length

        # Spatial grid
        x = np.linspace(0, L, nx)
        dx = x[1] - x[0]

        # Effective diffusion (with Taylor dispersion)
        d_eff = effective_diffusion(
            params.d_molecular, params.velocity,
            geo.inner_radius, params.temperature,
        )

        # Timestep: auto from Courant condition or user-specified
        if params.dt is not None:
            dt = params.dt
        elif params.velocity > 0:
            dt = dx / params.velocity
        else:
            # Pure diffusion: accuracy heuristic (CN is unconditionally stable,
            # but accuracy requires the diffusion number r_d = D*dt/dx^2 ~ O(1))
            dt = 0.4 * dx**2 / max(d_eff, 1e-20)

        nt = int(params.duration / dt) + 1
        time = np.linspace(0, params.duration, nt)
        dt = time[1] - time[0] if nt > 1 else dt

        # Initial condition
        if initial_condition is not None:
            c = initial_condition.values.copy()
            if len(c) != nx:
                c = np.interp(x, np.linspace(0, L, len(c)), c)
        else:
            c = np.zeros(nx)

        # Store full field
        concentration = np.zeros((nt, nx))
        concentration[0, :] = c

        # Sensor indices (nearest grid point)
        upstream_idx = int(round(geo.upstream_position / dx))
        downstream_idx = int(round(geo.downstream_position / dx))
        upstream_idx = min(max(upstream_idx, 0), nx - 1)
        downstream_idx = min(max(downstream_idx, 0), nx - 1)

        # Crank-Nicolson coefficients
        r_d = d_eff * dt / dx**2
        v = params.velocity
        r_a = v * dt / dx if dx > 0 else 0.0

        # Build tridiagonal bands for implicit side (interior points only)
        # For solve_banded: ab[0] = super-diagonal, ab[1] = diagonal, ab[2] = sub-diagonal
        n_interior = nx - 2  # exclude boundary points

        if v >= 0:
            # Upwind: advection from left
            a_sub = -r_d / 2 - r_a / 2   # sub-diagonal
            b_diag = 1 + r_d + r_a / 2    # main diagonal
            c_sup = -r_d / 2               # super-diagonal
            d_sub = r_d / 2 + r_a / 2      # explicit sub
            e_diag = 1 - r_d - r_a / 2     # explicit main
            f_sup = r_d / 2                 # explicit super
        else:
            # Downwind advection (upwind flips to right)
            a_sub = -r_d / 2
            b_diag = 1 + r_d + abs(r_a) / 2
            c_sup = -r_d / 2 - abs(r_a) / 2
            d_sub = r_d / 2
            e_diag = 1 - r_d - abs(r_a) / 2
            f_sup = r_d / 2 + abs(r_a) / 2

        # Banded matrix for solve_banded (shape: 3 x n_interior)
        ab = np.zeros((3, n_interior))
        ab[0, 1:] = c_sup       # super-diagonal (shifted)
        ab[1, :] = b_diag       # main diagonal
        ab[2, :-1] = a_sub      # sub-diagonal (shifted)

        # Neumann outlet BC: ghost point C[nx] = C[nx-1] means the
        # implicit c_sup term for the last interior point folds into
        # the diagonal. Apply once — constant across all timesteps.
        ab[1, -1] += c_sup

        # Track mass for conservation check
        initial_mass = np.trapezoid(c, x)
        mass_in = 0.0
        mass_out = 0.0

        # Time stepping
        nan_detected = False
        for n in range(nt - 1):
            t_now = time[n + 1]

            # Inlet boundary (Dirichlet)
            c_inlet = float(boundary_condition(t_now))
            mass_in += c_inlet * abs(v) * dt

            # Build RHS — vectorised, no Python loops
            # Interior points: indices 1..nx-2 in the full grid
            rhs = d_sub * c[:-2] + e_diag * c[1:-1] + f_sup * c[2:]

            # Left boundary: implicit side needs c_inlet at n+1
            rhs[0] -= a_sub * c_inlet

            # Right boundary (Neumann): ghost C[nx] = C[nx-1]
            # Explicit side: the vectorised RHS already used c[nx-1]
            # via f_sup * c[2:] for the last interior point (which
            # accesses c[nx-1]). The ghost adds another f_sup * c[nx-1].
            # But c[nx-1] == c[nx-2] (Neumann), and the vectorised
            # expression already included f_sup * c[nx-1]. The ghost
            # contribution on the explicit side is:
            rhs[-1] += f_sup * c[-1]

            # Solve tridiagonal system
            try:
                c_new_interior = solve_banded((1, 1), ab, rhs)
            except np.linalg.LinAlgError:
                nan_detected = True
                break

            if np.any(np.isnan(c_new_interior)) or np.any(np.isinf(c_new_interior)):
                nan_detected = True
                break

            # Assemble full solution
            c[0] = c_inlet
            c[1:-1] = c_new_interior
            c[-1] = c[-2]  # Neumann BC

            # Track outlet mass flux
            mass_out += c[-1] * abs(v) * dt

            concentration[n + 1, :] = c

        # Sensor traces
        upstream_trace = concentration[:, upstream_idx]
        downstream_trace = concentration[:, downstream_idx]

        # Peclet number and transit time
        sensor_spacing = geo.sensor_spacing
        pe = (params.velocity * sensor_spacing / d_eff) if d_eff > 0 else float("inf")
        transit_time = sensor_spacing / params.velocity if params.velocity > 0 else float("inf")

        # Mass conservation error (accounts for inlet and outlet flux)
        final_mass = np.trapezoid(c, x)
        total_expected = initial_mass + mass_in - mass_out
        if abs(initial_mass + mass_in) > 0:
            mass_error = (final_mass - total_expected) / (initial_mass + mass_in)
        else:
            mass_error = 0.0

        return SimulationResult(
            concentration=concentration,
            upstream=upstream_trace,
            downstream=downstream_trace,
            time=time,
            x=x,
            params=params,
            peclet_number=pe,
            d_effective=d_eff,
            transit_time=transit_time,
            mass_conservation_error=mass_error,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_simulator/test_solver.py -v`
Expected: 7 PASSED

- [ ] **Step 6: Commit**

```bash
git add requirements.txt simulator/solver.py tests/unit/test_simulator/test_solver.py
git commit -m "feat(simulator): add Crank-Nicolson forward solver"
```

---

### Task 5: Inverse Solver (Parameter Fitting)

**Files:**
- Create: `simulator/fitting.py`
- Create: `tests/unit/test_simulator/test_fitting.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for inverse solver (parameter fitting)."""

import numpy as np
import pytest

from app.regime import TurbidityRegime
from simulator.conditions import Pulse
from simulator.fitting import FitResult, InverseSolver
from simulator.geometry import load_geometry
from simulator.solver import AdvectionDiffusionSolver, SimulationParams


@pytest.fixture
def geometry():
    return load_geometry()


@pytest.fixture
def solver(geometry):
    return AdvectionDiffusionSolver(geometry)


@pytest.fixture
def inverse_solver(geometry):
    return InverseSolver(geometry)


class TestFitResult:
    def test_fit_recovers_known_parameters(self, solver, inverse_solver):
        """Fit should recover parameters from a synthetic signal."""
        # Generate a known signal
        true_d = 1e-7
        true_v = 0.05
        true_c0 = 100.0
        params = SimulationParams(
            d_molecular=true_d, velocity=true_v, c0=true_c0,
            temperature=20.0, nx=60, dt=None, duration=30.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=true_c0, t0=5.0, sigma=1.0))

        # Fit from the sensor traces
        fit = inverse_solver.fit(
            observed_upstream=result.upstream,
            observed_downstream=result.downstream,
            time=result.time,
            temperature=20.0,
        )
        assert fit.converged
        # Should recover velocity within 50% (fitting is approximate)
        assert fit.velocity == pytest.approx(true_v, rel=0.5)

    def test_fit_returns_regime(self, solver, inverse_solver):
        """FitResult should contain a TurbidityRegime."""
        params = SimulationParams(
            d_molecular=1e-7, velocity=0.05, c0=100.0,
            temperature=20.0, nx=60, dt=None, duration=30.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=100.0, t0=5.0, sigma=1.0))
        fit = inverse_solver.fit(
            observed_upstream=result.upstream,
            observed_downstream=result.downstream,
            time=result.time,
        )
        assert isinstance(fit.regime, TurbidityRegime)

    def test_noise_produces_high_residual(self, inverse_solver):
        """Pure noise input should produce a high residual."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 30, 500)
        noise_up = rng.standard_normal(500)
        noise_down = rng.standard_normal(500)
        fit = inverse_solver.fit(
            observed_upstream=noise_up,
            observed_downstream=noise_down,
            time=t,
        )
        # Residual should be relatively high — fit is poor
        assert fit.residual > 0.1

    def test_at_bound_flag(self, inverse_solver):
        """FitResult should have at_bound field."""
        t = np.linspace(0, 10, 100)
        fit = inverse_solver.fit(
            observed_upstream=np.zeros(100),
            observed_downstream=np.zeros(100),
            time=t,
        )
        assert isinstance(fit.at_bound, bool)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_simulator/test_fitting.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `simulator/fitting.py`**

```python
"""Inverse solver — fit advection-diffusion parameters to observed data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from app.regime import TurbidityRegime
from simulator.conditions import Arbitrary, Pulse
from simulator.geometry import PipeGeometry
from simulator.solver import AdvectionDiffusionSolver, SimulationParams, SimulationResult


@dataclass
class FitResult:
    """Result of parameter fitting."""

    d_molecular: float
    d_effective: float
    velocity: float
    c0: float
    peclet_number: float
    regime: TurbidityRegime
    residual: float
    converged: bool
    at_bound: bool
    simulation: SimulationResult


# Parameter bounds (physical ranges)
_D_BOUNDS = (1e-12, 1e-5)
_V_BOUNDS = (0.001, 0.5)
_C0_BOUNDS = (1.0, 5000.0)


class InverseSolver:
    """Fit advection-diffusion parameters to observed sensor data.

    Uses L-BFGS-B to minimise the squared difference between
    simulated and observed sensor traces.

    Note: currently uses a Pulse boundary condition for fitting.
    For fitting real data where the upstream signal is available,
    pass it as observed_upstream and the solver will use a Pulse
    approximation. Future: support Arbitrary BC to replay the
    actual upstream signal as the inlet condition.
    """

    def __init__(
        self,
        geometry: PipeGeometry,
        nx: int = 60,
        w_upstream: float = 1.0,
        w_downstream: float = 1.0,
    ) -> None:
        self.geometry = geometry
        self.nx = nx
        self.w_upstream = w_upstream
        self.w_downstream = w_downstream
        self._solver = AdvectionDiffusionSolver(geometry)

    def _make_bc(self, c0: float, duration: float) -> Pulse:
        """Create the boundary condition used during fitting."""
        return Pulse(c0=c0, t0=duration * 0.15, sigma=duration * 0.05)

    def fit(
        self,
        observed_upstream: np.ndarray,
        observed_downstream: np.ndarray,
        time: np.ndarray,
        temperature: float = 20.0,
    ) -> FitResult:
        """Fit D, v, C0 to observed sensor traces.

        Args:
            observed_upstream: Observed upstream concentration time series.
            observed_downstream: Observed downstream concentration time series.
            time: Time array matching the observations (seconds).
            temperature: Water temperature in deg C.

        Returns:
            FitResult with best-fit parameters and diagnostics.
        """
        duration = float(time[-1] - time[0])

        # Normalise observations for numerical stability
        obs_scale = max(
            np.max(np.abs(observed_upstream)),
            np.max(np.abs(observed_downstream)),
            1.0,
        )
        obs_up_norm = observed_upstream / obs_scale
        obs_down_norm = observed_downstream / obs_scale

        def cost(params_vec: np.ndarray) -> float:
            log_d, log_v, log_c0 = params_vec
            d_mol = np.exp(log_d)
            vel = np.exp(log_v)
            c0 = np.exp(log_c0)

            sim_params = SimulationParams(
                d_molecular=d_mol, velocity=vel, c0=c0,
                temperature=temperature, nx=self.nx, dt=None,
                duration=duration, boundary_type="pulse",
                geometry=self.geometry,
            )
            # Use c0 directly (not scaled) — compare in normalised space
            bc = self._make_bc(c0, duration)

            try:
                result = self._solver.solve(sim_params, bc)
            except Exception:
                return 1e10

            # Interpolate simulation to observation times, normalise
            sim_up = np.interp(time, result.time, result.upstream) / obs_scale
            sim_down = np.interp(time, result.time, result.downstream) / obs_scale

            if np.any(np.isnan(sim_up)) or np.any(np.isnan(sim_down)):
                return 1e10

            cost_val = (
                self.w_upstream * np.mean((sim_up - obs_up_norm) ** 2)
                + self.w_downstream * np.mean((sim_down - obs_down_norm) ** 2)
            )
            return float(cost_val)

        # Initial guess (log-space): mid-range
        x0 = np.array([
            np.log(1e-8),   # D
            np.log(0.05),   # v
            np.log(100.0),  # C0
        ])
        bounds = [
            (np.log(_D_BOUNDS[0]), np.log(_D_BOUNDS[1])),
            (np.log(_V_BOUNDS[0]), np.log(_V_BOUNDS[1])),
            (np.log(_C0_BOUNDS[0]), np.log(_C0_BOUNDS[1])),
        ]

        opt_result = minimize(
            cost, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-8},
        )

        # Extract best-fit parameters
        best_d = np.exp(opt_result.x[0])
        best_v = np.exp(opt_result.x[1])
        best_c0 = np.exp(opt_result.x[2])

        # Check if any parameter hit its bound
        at_bound = any(
            abs(opt_result.x[i] - bounds[i][0]) < 1e-6
            or abs(opt_result.x[i] - bounds[i][1]) < 1e-6
            for i in range(3)
        )

        # Run final forward simulation with best-fit params
        # Uses the same BC as the cost function for consistency
        final_params = SimulationParams(
            d_molecular=best_d, velocity=best_v, c0=best_c0,
            temperature=temperature, nx=self.nx, dt=None,
            duration=duration, boundary_type="pulse",
            geometry=self.geometry,
        )
        bc = self._make_bc(best_c0, duration)
        final_sim = self._solver.solve(final_params, bc)

        # Determine regime from Pe
        pe = final_sim.peclet_number
        geo = self.geometry
        if pe > geo.suspension_pe:
            regime = TurbidityRegime.SUSPENSION
        elif pe < geo.solution_pe:
            regime = TurbidityRegime.SOLUTION
        else:
            regime = TurbidityRegime.COLLOID

        return FitResult(
            d_molecular=best_d,
            d_effective=final_sim.d_effective,
            velocity=best_v,
            c0=best_c0,
            peclet_number=pe,
            regime=regime,
            residual=float(opt_result.fun),
            converged=opt_result.success,
            at_bound=at_bound,
            simulation=final_sim,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_simulator/test_fitting.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add simulator/fitting.py tests/unit/test_simulator/test_fitting.py
git commit -m "feat(simulator): add inverse solver for parameter fitting"
```

---

### Task 6: Synthetic Data Generation

**Files:**
- Create: `simulator/synthetic.py`
- Create: `tests/unit/test_simulator/test_synthetic.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for synthetic data generation."""

import numpy as np
import pytest

from app.regime import TurbidityRegime
from simulator.synthetic import generate_synthetic_dataset


class TestSyntheticGeneration:
    def test_returns_dataframe_with_expected_columns(self):
        df = generate_synthetic_dataset(n_samples=5, seed=42)
        expected_cols = {
            "upstream_signal", "downstream_signal",
            "d_molecular", "d_effective", "velocity", "c0", "temperature",
            "peclet_number_true", "regime",
            "cross_correlation_peak", "cross_correlation_lag",
            "peak_attenuation", "signal_broadening",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_correct_number_of_samples(self):
        df = generate_synthetic_dataset(n_samples=10, seed=42)
        assert len(df) == 10

    def test_all_regimes_represented(self):
        # With enough samples across the parameter space, all 3 regimes should appear
        df = generate_synthetic_dataset(n_samples=50, seed=42)
        regimes = set(df["regime"].values)
        assert len(regimes) >= 2  # at least 2 of 3 regimes

    def test_signals_are_arrays(self):
        df = generate_synthetic_dataset(n_samples=3, seed=42)
        for _, row in df.iterrows():
            assert isinstance(row["upstream_signal"], np.ndarray)
            assert isinstance(row["downstream_signal"], np.ndarray)
            assert len(row["upstream_signal"]) > 0

    def test_reproducible_with_seed(self):
        df1 = generate_synthetic_dataset(n_samples=5, seed=123)
        df2 = generate_synthetic_dataset(n_samples=5, seed=123)
        np.testing.assert_array_equal(
            df1["d_molecular"].values, df2["d_molecular"].values
        )

    def test_parameters_within_bounds(self):
        df = generate_synthetic_dataset(n_samples=20, seed=42)
        assert (df["d_molecular"] >= 1e-12).all()
        assert (df["d_molecular"] <= 1e-5).all()
        assert (df["velocity"] >= 0.001).all()
        assert (df["velocity"] <= 0.5).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_simulator/test_synthetic.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `simulator/synthetic.py`**

```python
"""Synthetic dataset generation for ML training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.regime import TurbidityRegime
from app.transport_physics import dual_sensor_features
from simulator.conditions import Pulse
from simulator.geometry import PipeGeometry, load_geometry
from simulator.solver import AdvectionDiffusionSolver, SimulationParams


def generate_synthetic_dataset(
    n_samples: int = 1000,
    seed: int | None = None,
    geometry: PipeGeometry | None = None,
    nx: int = 60,
    duration: float = 30.0,
) -> pd.DataFrame:
    """Generate a synthetic dataset of dual-sensor signals with ground truth.

    Samples D_molecular, velocity, C0, and temperature from physical ranges,
    runs the forward solver for each, and extracts features.

    Args:
        n_samples: Number of synthetic samples to generate.
        seed: Random seed for reproducibility.
        geometry: Pipe geometry. Defaults to loading from YAML.
        nx: Spatial grid points for solver.
        duration: Simulation duration in seconds.

    Returns:
        DataFrame with columns: upstream_signal, downstream_signal,
        d_molecular, d_effective, velocity, c0, temperature,
        peclet_number_true, regime, and transport physics features.
    """
    rng = np.random.default_rng(seed)
    geo = geometry or load_geometry()
    solver = AdvectionDiffusionSolver(geo)

    records: list[dict] = []

    for _ in range(n_samples):
        # Sample parameters (log-uniform for D, v, c0; uniform for temp)
        d_mol = float(np.exp(rng.uniform(np.log(1e-12), np.log(1e-5))))
        velocity = float(np.exp(rng.uniform(np.log(0.001), np.log(0.5))))
        c0 = float(np.exp(rng.uniform(np.log(1.0), np.log(5000.0))))
        temperature = float(rng.uniform(1.0, 35.0))

        params = SimulationParams(
            d_molecular=d_mol, velocity=velocity, c0=c0,
            temperature=temperature, nx=nx, dt=None,
            duration=duration, boundary_type="pulse",
            geometry=geo,
        )
        bc = Pulse(c0=c0, t0=duration * 0.15, sigma=duration * 0.05)

        try:
            result = solver.solve(params, bc)
        except Exception:
            continue  # skip failed simulations

        if np.any(np.isnan(result.downstream)):
            continue

        # Ground-truth regime from Pe
        pe = result.peclet_number
        if pe > geo.suspension_pe:
            regime = TurbidityRegime.SUSPENSION
        elif pe < geo.solution_pe:
            regime = TurbidityRegime.SOLUTION
        else:
            regime = TurbidityRegime.COLLOID

        # Extract transport features using existing analytical functions
        features = dual_sensor_features(
            signal_a=result.upstream,
            signal_b=result.downstream,
            velocity=velocity,
            temperature=temperature,
            sensor_spacing=geo.sensor_spacing,
        )

        records.append({
            "upstream_signal": result.upstream,
            "downstream_signal": result.downstream,
            "d_molecular": d_mol,
            "d_effective": result.d_effective,
            "velocity": velocity,
            "c0": c0,
            "temperature": temperature,
            "peclet_number_true": pe,
            "regime": regime.value,
            "cross_correlation_peak": features["cross_correlation_peak"],
            "cross_correlation_lag": features["cross_correlation_lag"],
            "peak_attenuation": features["peak_attenuation"],
            "signal_broadening": features["signal_broadening"],
        })

    return pd.DataFrame(records)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_simulator/test_synthetic.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add simulator/synthetic.py tests/unit/test_simulator/test_synthetic.py
git commit -m "feat(simulator): add synthetic dataset generation"
```

---

### Task 7: Package Init and Full Integration Test

**Files:**
- Modify: `simulator/__init__.py`

- [ ] **Step 1: Update `simulator/__init__.py` with public API re-exports**

```python
"""Advection-diffusion simulator for ClearEye dual-sensor pipe geometry.

Public API:
    - AdvectionDiffusionSolver, SimulationParams, SimulationResult
    - InverseSolver, FitResult
    - generate_synthetic_dataset
    - PipeGeometry, load_geometry
    - Pulse, Step, Ramp, Arbitrary, InitialCondition
    - effective_diffusion, temperature_correct_diffusion
"""

from simulator.conditions import Arbitrary, BoundaryCondition, InitialCondition, Pulse, Ramp, Step
from simulator.fitting import FitResult, InverseSolver
from simulator.geometry import PipeGeometry, load_geometry
from simulator.solver import AdvectionDiffusionSolver, SimulationParams, SimulationResult
from simulator.synthetic import generate_synthetic_dataset
from simulator.taylor_dispersion import effective_diffusion, temperature_correct_diffusion

__all__ = [
    "AdvectionDiffusionSolver",
    "Arbitrary",
    "BoundaryCondition",
    "FitResult",
    "InitialCondition",
    "InverseSolver",
    "PipeGeometry",
    "Pulse",
    "Ramp",
    "SimulationParams",
    "SimulationResult",
    "Step",
    "effective_diffusion",
    "generate_synthetic_dataset",
    "load_geometry",
    "temperature_correct_diffusion",
]
```

- [ ] **Step 2: Run the full test suite**

Run: `python3 -m pytest -v`
Expected: All tests pass (54 existing + ~46 new simulator tests = ~100 total)

- [ ] **Step 3: Commit**

```bash
git add simulator/__init__.py
git commit -m "feat(simulator): add public API re-exports and verify full test suite"
```
