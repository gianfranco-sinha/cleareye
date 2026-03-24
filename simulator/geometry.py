"""Pipe geometry configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.exceptions import ConfigError

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "pipe_geometry.yaml"


@dataclass(frozen=True)
class PipeGeometry:
    """Pipe geometry and sensor configuration.

    All lengths stored in metres (converted from mm in YAML).
    """

    pipe_length: float
    inner_radius: float
    mesh_opening: float
    upstream_position: float
    downstream_position: float
    velocity_min: float
    velocity_max: float
    suspension_pe: float
    solution_pe: float
    perturbation_zones: list[dict] = field(default_factory=list)

    @property
    def sensor_spacing(self) -> float:
        return self.downstream_position - self.upstream_position


def load_geometry(path: Path | None = None) -> PipeGeometry:
    """Load pipe geometry from YAML file.

    Args:
        path: Path to YAML file. Defaults to project root pipe_geometry.yaml.

    Raises:
        ConfigError: If the file is missing or malformed.
    """
    path = path or _DEFAULT_PATH
    if not path.exists():
        raise ConfigError(f"Pipe geometry file not found: {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid pipe geometry YAML: {e}") from e

    try:
        pipe = data["pipe"]
        sensors = {s["name"]: s["position_mm"] for s in data["sensors"]}
        flow = data["flow"]["velocity_range"]
        thresholds = data.get("regime_thresholds", {})

        return PipeGeometry(
            pipe_length=pipe["length_mm"] / 1000.0,
            inner_radius=pipe["inner_radius_mm"] / 1000.0,
            mesh_opening=pipe["mesh_opening_mm"] / 1000.0,
            upstream_position=sensors["upstream"] / 1000.0,
            downstream_position=sensors["downstream"] / 1000.0,
            velocity_min=flow["min_m_s"],
            velocity_max=flow["max_m_s"],
            suspension_pe=thresholds.get("suspension_pe", 1000),
            solution_pe=thresholds.get("solution_pe", 10),
            perturbation_zones=data.get("perturbation_zones", []),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"Malformed pipe geometry: {e}") from e
