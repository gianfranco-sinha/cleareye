"""SensorProfile ABC, CalibrationStandard ABC, and registries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.exceptions import ConfigError, UnknownSensorProfile


# ---------------------------------------------------------------------------
# SensorProfile ABC
# ---------------------------------------------------------------------------

class SensorProfile(ABC):
    """Abstract base class for a sensor probe's characteristics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this sensor type."""

    @property
    @abstractmethod
    def quantity(self) -> str:
        """Primary physical quantity measured (key in quantities.yaml)."""

    @property
    @abstractmethod
    def raw_features(self) -> list[str]:
        """Names of raw features this sensor produces."""

    @property
    @abstractmethod
    def valid_range(self) -> tuple[float, float]:
        """Valid range for the primary raw reading."""

    @abstractmethod
    def transfer(self, raw: float, **kwargs: Any) -> float:
        """Convert raw sensor value to calibrated quantity in canonical units."""


# ---------------------------------------------------------------------------
# SensorProfile Registry
# ---------------------------------------------------------------------------

class SensorProfileRegistry:
    """Registry of available sensor profiles."""

    def __init__(self) -> None:
        self._profiles: dict[str, SensorProfile] = {}

    def register(self, profile: SensorProfile) -> None:
        self._profiles[profile.name] = profile

    def get(self, name: str) -> SensorProfile:
        if name not in self._profiles:
            raise UnknownSensorProfile(name)
        return self._profiles[name]

    def all(self) -> list[SensorProfile]:
        return list(self._profiles.values())


sensor_registry = SensorProfileRegistry()


# ---------------------------------------------------------------------------
# CalibrationStandard ABC
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityCategory:
    name: str
    range: tuple[float, float]
    description: str = ""


class CalibrationStandard(ABC):
    """Abstract base class for a turbidity reference scale."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def unit(self) -> str: ...

    @property
    @abstractmethod
    def scale(self) -> tuple[float, float]: ...

    @property
    @abstractmethod
    def categories(self) -> list[QualityCategory]: ...

    def classify(self, ntu: float) -> QualityCategory | None:
        for cat in self.categories:
            lo, hi = cat.range
            if lo <= ntu < hi:
                return cat
        return None


# ---------------------------------------------------------------------------
# YAML-driven CalibrationStandard
# ---------------------------------------------------------------------------

class YAMLCalibrationStandard(CalibrationStandard):
    """CalibrationStandard loaded from a YAML file."""

    def __init__(self, path: Path) -> None:
        try:
            raw: dict[str, Any] = yaml.safe_load(path.read_text())
        except Exception as exc:
            raise ConfigError(f"Failed to load standard from {path}: {exc}") from exc

        self._name: str = raw["name"]
        self._description: str = raw.get("description", "")
        self._unit: str = raw["unit"]
        self._scale: tuple[float, float] = tuple(raw["scale"])
        self._categories: list[QualityCategory] = [
            QualityCategory(
                name=c["name"],
                range=tuple(c["range"]),
                description=c.get("description", ""),
            )
            for c in raw["categories"]
        ]

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def scale(self) -> tuple[float, float]:
        return self._scale

    @property
    def categories(self) -> list[QualityCategory]:
        return self._categories


# ---------------------------------------------------------------------------
# CalibrationStandard Registry
# ---------------------------------------------------------------------------

class CalibrationStandardRegistry:
    """Registry of available calibration standards."""

    def __init__(self) -> None:
        self._standards: dict[str, CalibrationStandard] = {}

    def register(self, standard: CalibrationStandard) -> None:
        self._standards[standard.name] = standard

    def get(self, name: str) -> CalibrationStandard:
        if name not in self._standards:
            raise ConfigError(f"Unknown calibration standard: {name!r}")
        return self._standards[name]

    def all(self) -> list[CalibrationStandard]:
        return list(self._standards.values())

    def load_directory(self, directory: Path) -> None:
        for path in sorted(directory.glob("*.yaml")):
            standard = YAMLCalibrationStandard(path)
            self.register(standard)


standards_registry = CalibrationStandardRegistry()
standards_registry.load_directory(Path(__file__).resolve().parent.parent / "standards")
