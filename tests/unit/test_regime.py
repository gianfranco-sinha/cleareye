"""Tests for regime classification."""

from app.regime import RuleBasedRegimeClassifier, TurbidityRegime


class TestRuleBasedRegimeClassifier:
    def setup_method(self):
        self.classifier = RuleBasedRegimeClassifier()

    def test_clear_water_is_solution(self):
        result = self.classifier.classify(turbidity_adc=900, tds=200, temperature=18.0)
        assert result.regime == TurbidityRegime.SOLUTION

    def test_turbid_water_is_suspension(self):
        result = self.classifier.classify(turbidity_adc=200, tds=100, temperature=18.0)
        assert result.regime == TurbidityRegime.SUSPENSION

    def test_midrange_is_colloid(self):
        result = self.classifier.classify(turbidity_adc=600, tds=400, temperature=18.0)
        assert result.regime == TurbidityRegime.COLLOID

    def test_confidence_range(self):
        result = self.classifier.classify(turbidity_adc=500, tds=300, temperature=18.0)
        assert 0.0 <= result.confidence <= 1.0
