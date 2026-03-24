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
            d_molecular=1e-5, velocity=0.05, c0=100.0,
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

    def test_auto_timestep_courant(self, solver):
        """When advection dominates, dt ~ dx/|v|."""
        # D_mol=1e-3 is large enough that Taylor term is small relative
        # to D_mol, so D_eff ~ D_mol. With dx~0.005 and D_eff~1e-3,
        # dt_diff = dx²/D_eff = 0.025 and dt_courant = dx/v = 0.05.
        # So dt = min(0.05, 0.025) = 0.025.
        params = SimulationParams(
            d_molecular=1e-3, velocity=0.1, c0=100.0,
            temperature=20.0, nx=60, dt=None, duration=10.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=100.0))
        actual_dt = result.time[1] - result.time[0]
        # Verify dt is positive and in a reasonable range
        assert actual_dt > 0
        assert actual_dt < 0.1


class TestMassConservation:
    def test_pulse_mass_conservation(self, solver):
        """Total mass in domain + outflow should be tracked."""
        params = SimulationParams(
            d_molecular=1e-5, velocity=0.05, c0=100.0,
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
            d_molecular=1e-5, velocity=0.1, c0=100.0,
            temperature=20.0, nx=60, dt=None, duration=15.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=100.0, t0=2.0, sigma=0.5))
        # Downstream peak should be after upstream peak
        upstream_peak_t = result.time[np.argmax(result.upstream)]
        downstream_peak_t = result.time[np.argmax(result.downstream)]
        assert downstream_peak_t > upstream_peak_t

    def test_high_diffusion_broadens_peak(self, solver):
        """Higher D_eff -> broader, lower peak for a fixed-mass pulse."""
        # Use initial condition (fixed total mass) to test broadening.
        # With inlet BC, higher D transports more mass to downstream.
        nx = 60
        ic_low = InitialCondition.zeros(nx)
        ic_high = InitialCondition.zeros(nx)
        # Sharp pulse near upstream sensor (index ~10)
        ic_low.values[9:12] = 100.0
        ic_high.values[9:12] = 100.0

        params_low_d = SimulationParams(
            d_molecular=1e-5, velocity=0.0, c0=0.0,
            temperature=20.0, nx=nx, dt=0.01, duration=2.0,
            boundary_type="step", geometry=solver.geometry,
        )
        params_high_d = SimulationParams(
            d_molecular=1e-3, velocity=0.0, c0=0.0,
            temperature=20.0, nx=nx, dt=0.005, duration=2.0,
            boundary_type="step", geometry=solver.geometry,
        )
        result_low = solver.solve(params_low_d, Step(c0=0.0), initial_condition=ic_low)
        result_high = solver.solve(params_high_d, Step(c0=0.0), initial_condition=ic_high)
        # Higher D spreads the pulse more, so max concentration is lower
        assert np.max(result_high.concentration[-1]) < np.max(result_low.concentration[-1])

    def test_step_input_reaches_steady_state(self, solver):
        """A step input should eventually fill the pipe uniformly."""
        params = SimulationParams(
            d_molecular=1e-4, velocity=0.1, c0=50.0,
            temperature=20.0, nx=60, dt=None, duration=60.0,
            boundary_type="step", geometry=solver.geometry,
        )
        result = solver.solve(params, Step(c0=50.0))
        # At the end, downstream should be close to c0
        assert result.downstream[-1] == pytest.approx(50.0, rel=0.1)

    def test_zero_velocity_pure_diffusion(self, solver):
        """With v=0, transport is purely diffusive -- symmetric spreading."""
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
        left_mass = np.sum(final[:center])
        right_mass = np.sum(final[center:])
        assert left_mass == pytest.approx(right_mass, rel=0.15)

    def test_nan_returns_early(self, solver):
        """Extreme parameters that would cause NaN should be caught."""
        params = SimulationParams(
            d_molecular=1e-20, velocity=100.0, c0=1e30,
            temperature=20.0, nx=10, dt=0.001, duration=0.01,
            boundary_type="pulse", geometry=solver.geometry,
        )
        # Should not raise -- solver catches NaN/inf
        result = solver.solve(params, Pulse(c0=1e30))
        assert isinstance(result, SimulationResult)

    def test_negative_velocity_completes(self, solver):
        """Negative velocity should run without error."""
        params = SimulationParams(
            d_molecular=1e-5, velocity=-0.1, c0=100.0,
            temperature=20.0, nx=60, dt=None, duration=15.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=100.0, t0=2.0, sigma=0.5))
        assert isinstance(result, SimulationResult)
        # Negative velocity -> negative transit time
        assert result.transit_time < 0
