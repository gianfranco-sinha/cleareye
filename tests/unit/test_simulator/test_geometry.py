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
