"""Calibration stage — datasheet transfer functions and ML residual correction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.regime import TurbidityRegime
from app.sensor_physics import (
    adc_to_voltage,
    sen0189_voltage_to_ntu,
    temperature_compensate,
)


class Calibrator(ABC):
    """Abstract calibrator: voltage + context → calibrated NTU."""

    @property
    @abstractmethod
    def method(self) -> str:
        """Identifier for the calibration method (e.g. 'datasheet', 'ml_residual')."""

    @abstractmethod
    def calibrate(
        self, voltage: float, temperature: float, tds: float
    ) -> float:
        """Return calibrated NTU from sensor voltage and environmental context."""


class DatasheetCalibrator(Calibrator):
    """Calibrator using the SEN0189 datasheet piecewise transfer function.

    Applies temperature compensation. No ML component.
    """

    @property
    def method(self) -> str:
        return "datasheet"

    def calibrate(
        self, voltage: float, temperature: float, tds: float
    ) -> float:
        ntu = sen0189_voltage_to_ntu(voltage)
        return temperature_compensate(ntu, temperature)


class CalibratorBank:
    """Holds one calibrator per regime and dispatches accordingly."""

    def __init__(
        self, calibrators: dict[TurbidityRegime, Calibrator] | None = None
    ) -> None:
        if calibrators is None:
            # Default: datasheet calibrator for all regimes
            default = DatasheetCalibrator()
            calibrators = {regime: default for regime in TurbidityRegime}
        self._calibrators = calibrators

    def get(self, regime: TurbidityRegime) -> Calibrator:
        return self._calibrators[regime]

    def set(self, regime: TurbidityRegime, calibrator: Calibrator) -> None:
        self._calibrators[regime] = calibrator
