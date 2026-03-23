"""Tests for sensor physics — ADC conversion, SEN0189 transfer function."""

import pytest

from app.sensor_physics import (
    adc_to_voltage,
    sen0189_adc_to_ntu,
    sen0189_voltage_to_ntu,
    temperature_compensate,
)


class TestADCToVoltage:
    def test_zero_adc(self):
        assert adc_to_voltage(0) == 0.0

    def test_max_adc(self):
        assert adc_to_voltage(1023, v_ref=5.0) == pytest.approx(4.9951, rel=1e-3)

    def test_mid_adc(self):
        assert adc_to_voltage(512, v_ref=5.0) == pytest.approx(2.5, rel=1e-2)

    def test_esp32_ref(self):
        assert adc_to_voltage(512, v_ref=3.3) == pytest.approx(1.65, rel=1e-2)


class TestSEN0189VoltageToNTU:
    def test_clear_water(self):
        assert sen0189_voltage_to_ntu(4.5) == 0.0

    def test_high_turbidity(self):
        assert sen0189_voltage_to_ntu(1.5) == 4000.0

    def test_midrange(self):
        ntu = sen0189_voltage_to_ntu(3.5)
        assert 0 < ntu < 500

    def test_monotonic_decreasing_voltage(self):
        """Higher voltage → lower NTU (sensor characteristic)."""
        voltages = [4.0, 3.5, 3.0, 2.5, 2.0]
        ntus = [sen0189_voltage_to_ntu(v) for v in voltages]
        for i in range(len(ntus) - 1):
            assert ntus[i] <= ntus[i + 1]


class TestTemperatureCompensation:
    def test_reference_temp_no_change(self):
        assert temperature_compensate(100.0, 25.0) == pytest.approx(100.0)

    def test_cold_water_increases(self):
        assert temperature_compensate(100.0, 15.0) > 100.0

    def test_warm_water_decreases(self):
        assert temperature_compensate(100.0, 35.0) < 100.0
