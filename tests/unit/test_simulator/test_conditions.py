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

    def test_nonzero_t0_offset(self):
        bc = Ramp(c0=100.0, t0=5.0, tau=10.0)
        assert bc(5.0) == pytest.approx(0.0)    # at t0, ramp just starts
        assert bc(10.0) == pytest.approx(50.0)   # halfway through ramp
        assert bc(15.0) == pytest.approx(100.0)  # end of ramp


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
