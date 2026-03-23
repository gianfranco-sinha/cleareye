"""Probe hardware constants and transfer functions.

SEN0189 datasheet piecewise curve: voltage → NTU.
ADC → voltage conversion for different reference voltages.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# ADC conversion
# ---------------------------------------------------------------------------

def adc_to_voltage(adc: int, v_ref: float = 5.0, resolution: int = 1024) -> float:
    """Convert raw ADC count to voltage.

    Args:
        adc: Raw ADC reading (0 to resolution-1).
        v_ref: Reference voltage (5.0V for Arduino, 3.3V for ESP32).
        resolution: ADC resolution (1024 for 10-bit).
    """
    return adc * (v_ref / resolution)


# ---------------------------------------------------------------------------
# SEN0189 datasheet transfer function
# ---------------------------------------------------------------------------

# Piecewise linear approximation from DFRobot SEN0189 datasheet.
# The sensor output voltage *decreases* as turbidity *increases*.
#
# Region boundaries (voltage, NTU):
#   V > 4.0V  → ~0 NTU (clear water)
#   V 3.0–4.0 → 0–500 NTU (low turbidity, roughly linear)
#   V 2.5–3.0 → 500–2000 NTU (medium, steeper)
#   V < 2.5   → 2000–4000 NTU (high, exponential region)

_SEN0189_BREAKPOINTS: list[tuple[float, float]] = [
    # (voltage, NTU) — ordered high-voltage to low-voltage
    (4.2, 0.0),
    (4.0, 0.0),
    (3.0, 500.0),
    (2.5, 2000.0),
    (2.0, 4000.0),
]


def sen0189_voltage_to_ntu(voltage: float) -> float:
    """Convert SEN0189 output voltage to NTU using datasheet piecewise curve.

    Clamps output to [0, 4000] NTU.
    """
    if voltage >= _SEN0189_BREAKPOINTS[0][0]:
        return 0.0
    if voltage <= _SEN0189_BREAKPOINTS[-1][0]:
        return 4000.0

    for i in range(len(_SEN0189_BREAKPOINTS) - 1):
        v_hi, ntu_hi = _SEN0189_BREAKPOINTS[i]
        v_lo, ntu_lo = _SEN0189_BREAKPOINTS[i + 1]
        if v_lo <= voltage <= v_hi:
            # Linear interpolation within this segment
            t = (v_hi - voltage) / (v_hi - v_lo)
            return ntu_hi + t * (ntu_lo - ntu_hi)

    return 0.0


def sen0189_adc_to_ntu(
    adc: int, v_ref: float = 5.0, resolution: int = 1024
) -> float:
    """Convert SEN0189 raw ADC to NTU (convenience wrapper)."""
    voltage = adc_to_voltage(adc, v_ref, resolution)
    return sen0189_voltage_to_ntu(voltage)


# ---------------------------------------------------------------------------
# Temperature compensation
# ---------------------------------------------------------------------------

# Simple linear temperature compensation for turbidity sensors.
# Turbidity readings increase ~0.5–1% per °C above 25°C due to viscosity changes.
_TEMP_COEFFICIENT = 0.01  # 1% per °C
_TEMP_REFERENCE = 25.0  # °C


def temperature_compensate(ntu: float, temperature: float) -> float:
    """Apply temperature compensation to a turbidity reading.

    Adjusts for the effect of water temperature on optical scattering.
    Reference temperature is 25°C.
    """
    correction = 1.0 - _TEMP_COEFFICIENT * (temperature - _TEMP_REFERENCE)
    return ntu * correction
