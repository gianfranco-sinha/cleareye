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
        # Uses default temperature=20.0 (reference), so D_corrected = D_molecular
        d_eff = effective_diffusion(d_molecular=1e-9, velocity=0.5, pipe_radius=0.025)
        taylor_term = (0.025**2 * 0.5**2) / (48 * 1e-9)
        assert d_eff == pytest.approx(1e-9 + taylor_term)

    def test_with_temperature_no_flow(self):
        # At v=0, no Taylor term — D_eff = D_corrected, so warm > cold
        d_cold = effective_diffusion(
            d_molecular=1e-9, velocity=0.0, pipe_radius=0.025, temperature=5.0
        )
        d_warm = effective_diffusion(
            d_molecular=1e-9, velocity=0.0, pipe_radius=0.025, temperature=30.0
        )
        assert d_warm > d_cold

    def test_taylor_inverts_temperature_at_high_velocity(self):
        # At practical velocities, Taylor term R²v²/(48·D) dominates.
        # Lower D_corrected (cold) → larger Taylor term → larger D_eff.
        d_cold = effective_diffusion(
            d_molecular=1e-9, velocity=0.1, pipe_radius=0.025, temperature=5.0
        )
        d_warm = effective_diffusion(
            d_molecular=1e-9, velocity=0.1, pipe_radius=0.025, temperature=30.0
        )
        assert d_cold > d_warm

    def test_perturbation_zone_multiplier(self):
        d_base = effective_diffusion(
            d_molecular=1e-9, velocity=0.1, pipe_radius=0.025
        )
        d_perturbed = effective_diffusion(
            d_molecular=1e-9, velocity=0.1, pipe_radius=0.025,
            perturbation_multiplier=1.5,
        )
        assert d_perturbed == pytest.approx(d_base * 1.5)
