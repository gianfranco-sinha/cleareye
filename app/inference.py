"""InferenceEngine — three-stage prediction pipeline.

Stage 1: Regime classification (solution / colloid / suspension)
Stage 2: Regime-specific calibration (ADC → voltage → NTU)
Stage 3: Biofouling correction (drift detection, correction factor)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

try:
    from opentelemetry import trace
    _tracer = trace.get_tracer("cleareye.inference")
except ImportError:  # pragma: no cover
    trace = None  # type: ignore[assignment]
    _tracer = None  # type: ignore[assignment]

from contextlib import contextmanager

from app.biofouling import BiofoulingMonitor
from app.calibration import CalibratorBank
from app.config import settings
from app.regime import RegimeClassifier, RuleBasedRegimeClassifier, TurbidityRegime
from app.sensor_physics import adc_to_voltage


@contextmanager
def _span(name: str, **attributes: str | float | bool):
    """Start a trace span if OTel is available, otherwise no-op."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name, attributes=attributes):
        yield


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Reading(BaseModel):
    """Raw sensor reading from a deployed rig.

    Represents a single measurement from a turbidity monitoring station.
    The required fields (turbidity_adc, tds, water_temperature) drive the
    three-stage inference pipeline.  Optional fields are stored for
    downstream analysis but do not affect the current prediction.
    """

    timestamp: datetime = Field(description="ISO-8601 timestamp of the reading")
    rig_id: str = Field(description="Unique identifier of the sensor rig")
    turbidity_adc: int = Field(
        ge=0, le=1023,
        description="Raw 10-bit ADC value from the turbidity sensor (0–1023)",
    )
    tds: float = Field(
        ge=0,
        description="Total dissolved solids in ppm",
    )
    water_temperature: float = Field(
        description="Water temperature in degrees Celsius",
    )
    depth: float | None = Field(default=None, description="Water depth in metres")
    flow_rate: float | None = Field(default=None, description="Flow rate in m/s")
    ph: float | None = Field(default=None, description="pH value (0–14)")
    dissolved_oxygen: float | None = Field(
        default=None, description="Dissolved oxygen in mg/L",
    )
    seq: int | None = Field(
        default=None, description="Sequence number for event ordering",
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "timestamp": "2026-03-26T10:30:00Z",
            "rig_id": "thames-01",
            "turbidity_adc": 620,
            "tds": 310.5,
            "water_temperature": 12.4,
        }
    ]}}


class CalibratedReading(BaseModel):
    """Output of the three-stage inference pipeline.

    Contains the calibrated turbidity in NTU, the detected turbidity
    regime, and metadata about biofouling correction and overall
    prediction confidence.
    """

    timestamp: datetime = Field(description="Timestamp from the original reading")
    rig_id: str = Field(description="Rig that produced this reading")
    regime: TurbidityRegime = Field(
        description="Detected turbidity regime: solution, colloid, or suspension",
    )
    turbidity_ntu: float = Field(
        description="Calibrated turbidity in Nephelometric Turbidity Units",
    )
    turbidity_voltage: float = Field(
        description="Intermediate voltage derived from the ADC reading (V)",
    )
    calibration_method: str = Field(
        default="datasheet",
        description="Calibration method used (datasheet or ml_residual)",
    )
    biofouling_factor: float = Field(
        default=1.0,
        description="Multiplicative biofouling correction (1.0 = no correction)",
    )
    cleaning_alert: bool = Field(
        default=False,
        description="True if sensor cleaning is recommended",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall prediction confidence (0–1), min of regime and fouling reliability",
    )


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
        with _span("inference.predict", rig_id=reading.rig_id):
            # Stage 1: classify regime
            with _span("stage1.regime_classification"):
                regime_result = self.regime_classifier.classify(
                    turbidity_adc=reading.turbidity_adc,
                    tds=reading.tds,
                    temperature=reading.water_temperature,
                )
                if trace is not None:
                    span = trace.get_current_span()
                    span.set_attribute("regime", regime_result.regime.value)
                    span.set_attribute("regime.confidence", regime_result.confidence)

            # Stage 2: calibrate to NTU
            with _span("stage2.calibration"):
                voltage = adc_to_voltage(reading.turbidity_adc, v_ref=self.v_ref)
                calibrator = self.calibrator_bank.get(regime_result.regime)
                ntu = calibrator.calibrate(
                    voltage=voltage,
                    temperature=reading.water_temperature,
                    tds=reading.tds,
                )
                if trace is not None:
                    span = trace.get_current_span()
                    span.set_attribute("calibration.method", calibrator.method)
                    span.set_attribute("turbidity_ntu", ntu)

            # Stage 3: biofouling correction
            with _span("stage3.biofouling"):
                fouling = self.biofouling_monitor.assess(
                    rig_id=reading.rig_id,
                    calibrated_ntu=ntu,
                )
                corrected_ntu = ntu * fouling.correction_factor
                if trace is not None:
                    span = trace.get_current_span()
                    span.set_attribute("biofouling.factor", fouling.correction_factor)
                    span.set_attribute("biofouling.cleaning_alert", fouling.cleaning_alert)

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
