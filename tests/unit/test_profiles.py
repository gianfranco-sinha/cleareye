"""Tests for sensor profiles and calibration standards."""

from app.profiles import sensor_registry, standards_registry

# Ensure builtin profiles are registered
import app.builtin_profiles  # noqa: F401


class TestSensorProfiles:
    def test_sen0189_registered(self):
        p = sensor_registry.get("sen0189")
        assert p.quantity == "turbidity"

    def test_tds_registered(self):
        p = sensor_registry.get("tds_meter")
        assert p.quantity == "tds"

    def test_ds18b20_registered(self):
        p = sensor_registry.get("ds18b20")
        assert p.quantity == "water_temperature"

    def test_sen0189_transfer(self):
        p = sensor_registry.get("sen0189")
        ntu = p.transfer(500.0)
        assert ntu >= 0


class TestCalibrationStandards:
    def test_environment_agency_loaded(self):
        s = standards_registry.get("environment_agency")
        assert s.unit == "NTU"
        assert len(s.categories) == 5

    def test_iso7027_loaded(self):
        s = standards_registry.get("iso7027")
        assert s.unit == "FNU"

    def test_classify_high_quality(self):
        s = standards_registry.get("environment_agency")
        cat = s.classify(5.0)
        assert cat is not None
        assert cat.name == "High"

    def test_classify_poor_quality(self):
        s = standards_registry.get("environment_agency")
        cat = s.classify(150.0)
        assert cat is not None
        assert cat.name == "Poor"
