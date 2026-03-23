"""TurbidityRegime enum, RegimeClassifier protocol, and rule-based classifier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class TurbidityRegime(str, Enum):
    """Water turbidity regime — determines which calibration model applies."""

    SOLUTION = "solution"
    COLLOID = "colloid"
    SUSPENSION = "suspension"


@dataclass(frozen=True)
class RegimeResult:
    """Result of regime classification."""

    regime: TurbidityRegime
    confidence: float  # 0.0–1.0


class RegimeClassifier(Protocol):
    """Protocol for regime classifiers — rule-based and ML implementations."""

    def classify(
        self, turbidity_adc: int, tds: float, temperature: float
    ) -> RegimeResult: ...


class RuleBasedRegimeClassifier:
    """Rule-based regime classifier using ADC/TDS thresholds.

    Classifies based on the optical signature of each regime:
    - Solution: high ADC (clear water), low turbidity, dissolved substances dominate
    - Colloid: moderate ADC, TDS and turbidity partially correlated
    - Suspension: low ADC (turbid water), TDS and turbidity decouple
    """

    # Thresholds derived from SEN0189 characteristics and domain knowledge.
    # ADC thresholds (10-bit, 0–1023): higher ADC → clearer water for SEN0189.
    ADC_SOLUTION_THRESHOLD = 800   # Above this → likely solution regime
    ADC_SUSPENSION_THRESHOLD = 400  # Below this → likely suspension regime

    # TDS can help disambiguate: in colloid regime, TDS and turbidity are
    # partially correlated; in suspension, they decouple.
    TDS_HIGH_THRESHOLD = 600  # ppm

    def classify(
        self,
        turbidity_adc: int,
        tds: float,
        temperature: float,
    ) -> RegimeResult:
        """Classify the turbidity regime from raw sensor features.

        Args:
            turbidity_adc: Raw 10-bit ADC reading from SEN0189.
            tds: Total dissolved solids in ppm.
            temperature: Water temperature in °C (reserved for future use).
        """
        if turbidity_adc >= self.ADC_SOLUTION_THRESHOLD:
            return RegimeResult(TurbidityRegime.SOLUTION, confidence=0.85)

        if turbidity_adc <= self.ADC_SUSPENSION_THRESHOLD:
            # High TDS with low ADC strongly indicates suspension (sediment event)
            if tds < self.TDS_HIGH_THRESHOLD:
                return RegimeResult(TurbidityRegime.SUSPENSION, confidence=0.80)
            # High TDS + low ADC could be either — lower confidence
            return RegimeResult(TurbidityRegime.SUSPENSION, confidence=0.65)

        # Mid-range ADC → colloid
        return RegimeResult(TurbidityRegime.COLLOID, confidence=0.70)
