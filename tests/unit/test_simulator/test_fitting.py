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
    return InverseSolver(geometry, nx=20)


class TestFitResult:
    def test_fit_recovers_known_parameters(self, solver, inverse_solver):
        """Fit should recover parameters from a synthetic signal."""
        true_d = 1e-5
        true_v = 0.05
        true_c0 = 100.0
        params = SimulationParams(
            d_molecular=true_d, velocity=true_v, c0=true_c0,
            temperature=20.0, nx=60, dt=None, duration=30.0,
            boundary_type="pulse", geometry=solver.geometry,
        )
        result = solver.solve(params, Pulse(c0=true_c0, t0=5.0, sigma=1.0))

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
            d_molecular=1e-5, velocity=0.05, c0=100.0,
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
        # Residual should be relatively high -- fit is poor
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
