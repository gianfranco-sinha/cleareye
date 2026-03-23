"""Quantity registry loader — reads quantities.yaml and provides lookup/validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.exceptions import ConfigError, ReadingOutOfRange

_QUANTITIES_PATH = Path(__file__).resolve().parent.parent / "quantities.yaml"


@dataclass(frozen=True)
class Quantity:
    name: str
    canonical_unit: str
    description: str
    valid_range: tuple[float, float]
    kind: str
    aliases: list[str] = field(default_factory=list)
    alternate_units: dict[str, str] = field(default_factory=dict)

    def validate(self, value: float) -> None:
        lo, hi = self.valid_range
        if not (lo <= value <= hi):
            raise ReadingOutOfRange(self.name, value, self.valid_range)


class QuantityRegistry:
    """In-memory registry of physical quantities loaded from YAML."""

    def __init__(self) -> None:
        self._quantities: dict[str, Quantity] = {}
        self._alias_map: dict[str, str] = {}

    def load(self, path: Path | None = None) -> None:
        path = path or _QUANTITIES_PATH
        try:
            raw: dict[str, Any] = yaml.safe_load(path.read_text())
        except Exception as exc:
            raise ConfigError(f"Failed to load quantities from {path}: {exc}") from exc

        for name, defn in raw.items():
            alt_units: dict[str, str] = {}
            for unit_name, unit_defn in defn.get("alternate_units", {}).items():
                alt_units[unit_name] = unit_defn["convert"]

            q = Quantity(
                name=name,
                canonical_unit=defn["canonical_unit"],
                description=defn.get("description", ""),
                valid_range=tuple(defn["valid_range"]),
                kind=defn.get("kind", "unknown"),
                aliases=defn.get("aliases", []),
                alternate_units=alt_units,
            )
            self._quantities[name] = q
            for alias in q.aliases:
                self._alias_map[alias] = name

    def get(self, name: str) -> Quantity:
        canonical = self._alias_map.get(name, name)
        if canonical not in self._quantities:
            raise ConfigError(f"Unknown quantity: {name!r}")
        return self._quantities[canonical]

    def validate(self, name: str, value: float) -> None:
        self.get(name).validate(value)

    def all(self) -> list[Quantity]:
        return list(self._quantities.values())


# Module-level singleton
registry = QuantityRegistry()
registry.load()
