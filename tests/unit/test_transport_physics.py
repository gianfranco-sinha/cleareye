"""Tests for transport physics — dual-sensor pipe geometry calculations."""

import numpy as np
import pytest

from app.transport_physics import (
    correlation_peak,
    cross_correlation,
    dual_sensor_features,
    estimate_diffusion_coefficient,
    peak_attenuation,
    peclet_number,
    signal_broadening,
)


class TestPecletNumber:
    def test_high_pe_means_particles(self):
        # Fast flow, low diffusion → high Pe → particles
        pe = peclet_number(velocity=0.1, diffusion_coeff=1e-9)
        assert pe > 1000

    def test_low_pe_means_colloids(self):
        # Slow flow, high diffusion → low Pe → colloids
        pe = peclet_number(velocity=0.001, diffusion_coeff=1e-3)
        assert pe < 1

    def test_zero_diffusion_returns_inf(self):
        pe = peclet_number(velocity=0.1, diffusion_coeff=0.0)
        assert pe == float("inf")

    def test_custom_spacing(self):
        pe1 = peclet_number(velocity=0.1, diffusion_coeff=1e-6, sensor_spacing=0.2)
        pe2 = peclet_number(velocity=0.1, diffusion_coeff=1e-6, sensor_spacing=0.3)
        assert pe2 > pe1


class TestCrossCorrelation:
    def test_identical_signals_peak_at_zero_lag(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        _, corr_lag = correlation_peak(signal, signal)
        assert corr_lag == 0

    def test_delayed_signal_positive_lag(self):
        t = np.linspace(0, 4 * np.pi, 200)
        signal_a = np.zeros(200)
        signal_a[50:70] = 1.0  # pulse at sensor A
        signal_b = np.zeros(200)
        signal_b[60:80] = 1.0  # same pulse, delayed by 10 samples
        peak_val, lag = correlation_peak(signal_a, signal_b, max_lag=30)
        assert lag == 10
        assert peak_val > 0.5

    def test_uncorrelated_signals_low_peak(self):
        rng = np.random.default_rng(42)
        signal_a = rng.standard_normal(200)
        signal_b = rng.standard_normal(200)
        peak_val, _ = correlation_peak(signal_a, signal_b)
        assert peak_val < 0.3


class TestPeakAttenuation:
    def test_identical_signals_attenuation_one(self):
        signal = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0], dtype=float)
        atten = peak_attenuation(signal, signal)
        assert atten == pytest.approx(1.0)

    def test_damped_signal_low_attenuation(self):
        signal_a = np.array([0, 0, 0, 10, 0, 0, 0], dtype=float)
        signal_b = np.array([0, 0, 0, 3, 0, 0, 0], dtype=float)
        atten = peak_attenuation(signal_a, signal_b)
        assert atten == pytest.approx(0.3, rel=0.1)

    def test_no_signal_returns_zero(self):
        flat = np.ones(10)
        atten = peak_attenuation(flat, flat)
        assert atten == 0.0


class TestSignalBroadening:
    def test_identical_peaks_broadening_one(self):
        signal = np.array([0, 0, 5, 10, 5, 0, 0], dtype=float)
        b = signal_broadening(signal, signal)
        assert b == pytest.approx(1.0)

    def test_broader_downstream_peak(self):
        # Upstream: narrow peak (3 samples wide above half-max)
        signal_a = np.array([0, 0, 0, 5, 10, 5, 0, 0, 0, 0], dtype=float)
        # Downstream: broader peak (5 samples wide above half-max)
        signal_b = np.array([0, 0, 3, 5, 8, 5, 3, 0, 0, 0], dtype=float)
        b = signal_broadening(signal_a, signal_b)
        assert b >= 1.0


class TestDiffusionEstimate:
    def test_high_attenuation_low_diffusion(self):
        # attenuation near 1 → relatively low diffusion (compared to low attenuation)
        d_high = estimate_diffusion_coefficient(attenuation=0.95, velocity=0.05)
        d_low = estimate_diffusion_coefficient(attenuation=0.1, velocity=0.05)
        assert d_high > 0
        assert d_high < d_low  # higher attenuation = less diffusion

    def test_low_attenuation_high_diffusion(self):
        # attenuation << 1 → strong diffusion
        d = estimate_diffusion_coefficient(attenuation=0.1, velocity=0.05)
        assert d > 0

    def test_zero_attenuation_returns_zero(self):
        d = estimate_diffusion_coefficient(attenuation=0.0, velocity=0.05)
        assert d == 0.0

    def test_temperature_affects_diffusion(self):
        d_cold = estimate_diffusion_coefficient(attenuation=0.5, velocity=0.05, temperature=5.0)
        d_warm = estimate_diffusion_coefficient(attenuation=0.5, velocity=0.05, temperature=25.0)
        assert d_warm > d_cold


class TestDualSensorFeatures:
    def test_returns_all_features(self):
        rng = np.random.default_rng(42)
        signal_a = rng.uniform(400, 600, 100)
        signal_b = rng.uniform(400, 600, 100)
        features = dual_sensor_features(
            signal_a, signal_b, velocity=0.05, temperature=18.0
        )
        expected_keys = {
            "cross_correlation_peak", "cross_correlation_lag",
            "peak_attenuation", "signal_broadening",
            "peclet_number", "effective_diffusion",
        }
        assert set(features.keys()) == expected_keys

    def test_particle_like_signals(self):
        # Sharp pulse transported with minimal spreading → particles
        signal_a = np.zeros(100)
        signal_a[30:35] = 100.0
        signal_b = np.zeros(100)
        signal_b[40:45] = 95.0  # delayed, minimal attenuation
        features = dual_sensor_features(
            signal_a, signal_b, velocity=0.05, temperature=18.0
        )
        assert features["peak_attenuation"] > 0.8
        assert features["cross_correlation_lag"] > 0

    def test_colloid_like_signals(self):
        # Broad, damped downstream signal → colloids
        signal_a = np.zeros(100)
        signal_a[30:35] = 100.0
        signal_b = np.zeros(100)
        signal_b[35:55] = 20.0  # delayed, attenuated, broadened
        features = dual_sensor_features(
            signal_a, signal_b, velocity=0.05, temperature=18.0
        )
        assert features["peak_attenuation"] < 0.5
        assert features["signal_broadening"] > 1.0
