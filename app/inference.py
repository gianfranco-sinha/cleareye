"""InferenceEngine — three-stage prediction pipeline.

Stage 1: Regime classification (solution / colloid / suspension)
Stage 2: Regime-specific calibration (ADC → voltage → NTU)
Stage 3: Biofouling correction (drift detection, correction factor)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.biofouling import BiofoulingMonitor
from app.calibration import CalibratorBank
from app.config import settings
from app.regime import RegimeClassifier, RuleBasedRegimeClassifier, TurbidityRegime
from app.sensor_physics import adc_to_voltage


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Reading(BaseModel):
    """Raw sensor reading from a deployed rig."""

    timestamp: datetime
    rig_id: str
    turbidity_adc: int = Field(ge=0, le=1023)
    tds: float = Field(ge=0)
    water_temperature: float
    depth: float | None = None
    flow_rate: float | None = None
    ph: float | None = None
    dissolved_oxygen: float | None = None
    seq: int | None = None


class CalibratedReading(BaseModel):
    """Output of the three-stage inference pipeline."""

    timestamp: datetime
    rig_id: str
    regime: TurbidityRegime
    turbidity_ntu: float
    turbidity_voltage: float
    calibration_method: str = "datasheet"
    biofouling_factor: float = 1.0
    cleaning_alert: bool = False
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Three-stage inference pipeline for water quality prediction."""

    def __init__(
        self,
        regime_classifier: RegimeClassifier | None = None,
        calibrator_bank: CalibratorBank | None = None,
        biofouling_monitor: BiofoulingMonitor | None = None,
        v_ref: float | None = None,
    ) -> None:
        self.regime_classifier = regime_classifier or RuleBasedRegimeClassifier()
        self.calibrator_bank = calibrator_bank or CalibratorBank()
        self.biofouling_monitor = biofouling_monitor or BiofoulingMonitor()
        self.v_ref = v_ref or settings.default_v_ref

    def predict(self, reading: Reading) -> CalibratedReading:
        """Run the three-stage pipeline on a single reading."""
        # Stage 1: classify regime
        regime_result = self.regime_classifier.classify(
            turbidity_adc=reading.turbidity_adc,
            tds=reading.tds,
            temperature=reading.water_temperature,
        )

        # Stage 2: calibrate to NTU
        voltage = adc_to_voltage(reading.turbidity_adc, v_ref=self.v_ref)
        calibrator = self.calibrator_bank.get(regime_result.regime)
        ntu = calibrator.calibrate(
            voltage=voltage,
            temperature=reading.water_temperature,
            tds=reading.tds,
        )

        # Stage 3: biofouling correction
        fouling = self.biofouling_monitor.assess(
            rig_id=reading.rig_id,
            calibrated_ntu=ntu,
        )
        corrected_ntu = ntu * fouling.correction_factor

        return CalibratedReading(
            timestamp=reading.timestamp,
            rig_id=reading.rig_id,
            regime=regime_result.regime,
            turbidity_ntu=round(corrected_ntu, 2),
            turbidity_voltage=round(voltage, 4),
            calibration_method=calibrator.method,
            biofouling_factor=fouling.correction_factor,
            cleaning_alert=fouling.cleaning_alert,
            confidence=round(
                min(regime_result.confidence, fouling.reliability), 2
            ),
        )
