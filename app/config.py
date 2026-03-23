"""Application configuration — loads settings from model_config.yaml and env."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.exceptions import ConfigError

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "model_config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file. Returns empty dict if file doesn't exist."""
    path = path or _CONFIG_PATH
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        raise ConfigError(f"Failed to load config from {path}: {exc}") from exc


class Settings:
    """Application settings with defaults."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        raw = load_config()
        if overrides:
            raw.update(overrides)
        self._raw = raw

    # Server
    @property
    def host(self) -> str:
        return self._raw.get("server", {}).get("host", "0.0.0.0")

    @property
    def port(self) -> int:
        return self._raw.get("server", {}).get("port", 8000)

    # Sensor defaults
    @property
    def default_v_ref(self) -> float:
        return self._raw.get("sensor", {}).get("v_ref", 5.0)

    @property
    def default_adc_resolution(self) -> int:
        return self._raw.get("sensor", {}).get("adc_resolution", 1024)

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)


settings = Settings()
