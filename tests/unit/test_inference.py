"""Tests for the inference engine."""

from datetime import datetime

from app.inference import CalibratedReading, InferenceEngine, Reading


class TestInferenceEngine:
    def setup_method(self):
        self.engine = InferenceEngine()

    def _make_reading(self, adc: int = 500, tds: float = 300.0) -> Reading:
        return Reading(
            timestamp=datetime(2023, 10, 7, 19, 12, 5),
            rig_id="test-rig",
            turbidity_adc=adc,
            tds=tds,
            water_temperature=18.0,
        )

    def test_predict_returns_calibrated_reading(self):
        result = self.engine.predict(self._make_reading())
        assert isinstance(result, CalibratedReading)

    def test_predict_preserves_metadata(self):
        reading = self._make_reading()
        result = self.engine.predict(reading)
        assert result.rig_id == "test-rig"
        assert result.timestamp == reading.timestamp

    def test_predict_clear_water(self):
        result = self.engine.predict(self._make_reading(adc=900))
        assert result.turbidity_ntu < 100
        assert result.regime.value == "solution"

    def test_predict_turbid_water(self):
        result = self.engine.predict(self._make_reading(adc=200))
        assert result.turbidity_ntu > 100
        assert result.regime.value == "suspension"

    def test_biofouling_defaults(self):
        result = self.engine.predict(self._make_reading())
        assert result.biofouling_factor == 1.0
        assert result.cleaning_alert is False

    def test_confidence_in_range(self):
        result = self.engine.predict(self._make_reading())
        assert 0.0 <= result.confidence <= 1.0
