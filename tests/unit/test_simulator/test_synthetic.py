"""Tests for synthetic data generation."""

import numpy as np
import pytest

from app.regime import TurbidityRegime
from simulator.synthetic import generate_synthetic_dataset

# Use minimal grid and short duration for fast tests
_NX = 10
_DURATION = 5.0


class TestSyntheticGeneration:
    def test_returns_dataframe_with_expected_columns(self):
        df = generate_synthetic_dataset(n_samples=3, seed=42, nx=_NX, duration=_DURATION)
        expected_cols = {
            "upstream_signal", "downstream_signal",
            "d_molecular", "d_effective", "velocity", "c0", "temperature",
            "peclet_number_true", "regime",
            "cross_correlation_peak", "cross_correlation_lag",
            "peak_attenuation", "signal_broadening",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_correct_number_of_samples(self):
        df = generate_synthetic_dataset(n_samples=5, seed=42, nx=_NX, duration=_DURATION)
        assert len(df) == 5

    def test_regime_values_are_valid(self):
        # Taylor dispersion dominates D_eff for small D_mol, so most samples
        # have low Pe (solution regime). Verify regimes are valid enum values.
        df = generate_synthetic_dataset(n_samples=5, seed=42, nx=_NX, duration=_DURATION)
        valid_regimes = {"solution", "colloid", "suspension"}
        assert set(df["regime"].values).issubset(valid_regimes)

    def test_signals_are_arrays(self):
        df = generate_synthetic_dataset(n_samples=3, seed=42, nx=_NX, duration=_DURATION)
        for _, row in df.iterrows():
            assert isinstance(row["upstream_signal"], np.ndarray)
            assert isinstance(row["downstream_signal"], np.ndarray)
            assert len(row["upstream_signal"]) > 0

    def test_reproducible_with_seed(self):
        df1 = generate_synthetic_dataset(n_samples=3, seed=123, nx=_NX, duration=_DURATION)
        df2 = generate_synthetic_dataset(n_samples=3, seed=123, nx=_NX, duration=_DURATION)
        np.testing.assert_array_equal(
            df1["d_molecular"].values, df2["d_molecular"].values
        )

    def test_parameters_within_bounds(self):
        df = generate_synthetic_dataset(n_samples=5, seed=42, nx=_NX, duration=_DURATION)
        assert (df["d_molecular"] >= 1e-12).all()
        assert (df["d_molecular"] <= 1e-5).all()
        assert (df["velocity"] >= 0.001).all()
        assert (df["velocity"] <= 0.5).all()
