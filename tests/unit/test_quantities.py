"""Tests for the quantity registry."""

import pytest

from app.exceptions import ConfigError, ReadingOutOfRange
from app.quantities import registry


class TestQuantityRegistry:
    def test_load_turbidity(self):
        q = registry.get("turbidity")
        assert q.canonical_unit == "NTU"
        assert q.valid_range == (0, 4000)

    def test_alias_lookup(self):
        q = registry.get("turb")
        assert q.name == "turbidity"

    def test_unknown_quantity(self):
        with pytest.raises(ConfigError):
            registry.get("nonexistent")

    def test_validate_in_range(self):
        registry.validate("turbidity", 100.0)  # Should not raise

    def test_validate_out_of_range(self):
        with pytest.raises(ReadingOutOfRange):
            registry.validate("turbidity", 5000.0)

    def test_all_quantities_loaded(self):
        names = [q.name for q in registry.all()]
        assert "turbidity" in names
        assert "tds" in names
        assert "water_temperature" in names
