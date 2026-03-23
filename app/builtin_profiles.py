"""Built-in sensor profiles — SEN0189, TDS meter, DS18B20.

Importing this module registers all built-in profiles in the sensor registry.
"""

from __future__ import annotations

from typing import Any

from app.profiles import SensorProfile, sensor_registry
from app.sensor_physics import adc_to_voltage, sen0189_voltage_to_ntu


class SEN0189TurbidityProfile(SensorProfile):
    """DFRobot SEN0189 analog turbidity sensor."""

    @property
    def name(self) -> str:
        return "sen0189"

    @property
    def quantity(self) -> str:
        return "turbidity"

    @property
    def raw_features(self) -> list[str]:
        return ["turbidity_adc"]

    @property
    def valid_range(self) -> tuple[float, float]:
        return (0.0, 1023.0)

    def transfer(self, raw: float, **kwargs: Any) -> float:
        v_ref = kwargs.get("v_ref", 5.0)
        voltage = adc_to_voltage(int(raw), v_ref=v_ref)
        return sen0189_voltage_to_ntu(voltage)


class TDSMeterProfile(SensorProfile):
    """Generic TDS meter (analog, ppm output)."""

    @property
    def name(self) -> str:
        return "tds_meter"

    @property
    def quantity(self) -> str:
        return "tds"

    @property
    def raw_features(self) -> list[str]:
        return ["tds"]

    @property
    def valid_range(self) -> tuple[float, float]:
        return (0.0, 1000.0)

    def transfer(self, raw: float, **kwargs: Any) -> float:
        # TDS meters typically output directly in ppm
        return raw


class DS18B20TemperatureProfile(SensorProfile):
    """Dallas DS18B20 digital temperature sensor."""

    @property
    def name(self) -> str:
        return "ds18b20"

    @property
    def quantity(self) -> str:
        return "water_temperature"

    @property
    def raw_features(self) -> list[str]:
        return ["water_temperature"]

    @property
    def valid_range(self) -> tuple[float, float]:
        return (-55.0, 125.0)

    def transfer(self, raw: float, **kwargs: Any) -> float:
        # DS18B20 outputs directly in °C
        return raw


# Register built-in profiles on import
sensor_registry.register(SEN0189TurbidityProfile())
sensor_registry.register(TDSMeterProfile())
sensor_registry.register(DS18B20TemperatureProfile())
