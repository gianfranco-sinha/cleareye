"""SensorRig — object model for a deployed sensor rig.

A ``SensorRig`` represents a physical monitoring station with one or more
sensors attached.  It can be constructed from a CSV file (or DataFrame)
by inspecting column headers and matching them against registered
:class:`SensorProfile` instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

import app.builtin_profiles  # noqa: F401 — ensure profiles are registered
from app.profiles import SensorProfile, sensor_registry

logger = logging.getLogger(__name__)

# Maps common alternate column names → canonical feature names so that
# raw CSVs with non-standard headers can still be matched to profiles.
_COLUMN_ALIASES: dict[str, str] = {
    "turbidity": "turbidity_adc",
    "temperature": "water_temperature",
    "temp": "water_temperature",
    "water_temp": "water_temperature",
}


@dataclass
class DetectedSensor:
    """A sensor that was detected on a rig from file column headers."""

    profile: SensorProfile
    matched_columns: list[str]
    sample_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def quantity(self) -> str:
        return self.profile.quantity

    def in_range(self) -> bool:
        """Check whether sampled data falls within the profile's valid range."""
        lo, hi = self.profile.valid_range
        stats_min = self.sample_stats.get("min")
        stats_max = self.sample_stats.get("max")
        if stats_min is None or stats_max is None:
            return True
        return lo <= stats_min and stats_max <= hi


@dataclass
class SensorRig:
    """A deployed sensor rig with detected sensors.

    Attributes:
        rig_id: Identifier for this rig (derived from filename or explicit).
        sensors: Sensors detected from the data source.
        unmatched_columns: Columns that could not be matched to any profile.
        row_count: Number of data rows in the source.
        time_range: (earliest, latest) timestamps if a time column exists.
    """

    rig_id: str
    sensors: list[DetectedSensor]
    unmatched_columns: list[str] = field(default_factory=list)
    row_count: int = 0
    time_range: tuple[str, str] | None = None

    @property
    def sensor_names(self) -> list[str]:
        return [s.name for s in self.sensors]

    @property
    def quantities(self) -> list[str]:
        return [s.quantity for s in self.sensors]

    def has_sensor(self, name: str) -> bool:
        return name in self.sensor_names

    def has_quantity(self, quantity: str) -> bool:
        return quantity in self.quantities

    def get_sensor(self, name: str) -> DetectedSensor | None:
        for s in self.sensors:
            if s.name == name:
                return s
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "rig_id": self.rig_id,
            "sensors": [
                {
                    "name": s.name,
                    "quantity": s.quantity,
                    "columns": s.matched_columns,
                    "in_range": s.in_range(),
                    "stats": s.sample_stats,
                }
                for s in self.sensors
            ],
            "unmatched_columns": self.unmatched_columns,
            "row_count": self.row_count,
            "time_range": self.time_range,
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: Path,
        rig_id: str | None = None,
    ) -> SensorRig:
        """Infer a SensorRig from a CSV file by matching column headers
        against registered sensor profiles.

        Args:
            path: Path to the CSV file.
            rig_id: Explicit rig identifier.  Defaults to the filename stem.
        """
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        # Handle the Date+Time header shift (same as CSVDataSource)
        if (
            df.columns.size >= 2
            and df.iloc[:, -1].isna().all()
            and "Date" in df.columns
            and "Time" in df.columns
        ):
            df = df.drop(columns=[df.columns[-1]])
            df.columns = ["Date", "TDS", "Turbidity", "Temperature"][: len(df.columns)]

        return cls.from_dataframe(
            df,
            rig_id=rig_id or path.stem,
            source_name=path.name,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        rig_id: str = "unknown",
        source_name: str = "dataframe",
    ) -> SensorRig:
        """Infer a SensorRig from a DataFrame by matching column headers."""
        # Normalise columns to canonical names
        canonical_cols: dict[str, str] = {}  # original → canonical
        for col in df.columns:
            lower = col.lower().strip()
            alias = _COLUMN_ALIASES.get(lower, lower)
            canonical_cols[col] = alias

        canonical_set = set(canonical_cols.values())

        # Match each profile against the available columns
        detected: list[DetectedSensor] = []
        claimed_columns: set[str] = set()

        for profile in sensor_registry.all():
            raw_features = profile.raw_features
            matched = [f for f in raw_features if f in canonical_set]

            if matched:
                # Find original column names that mapped to these features
                original_cols = [
                    orig for orig, canon in canonical_cols.items()
                    if canon in matched
                ]
                claimed_columns.update(original_cols)

                # Compute sample statistics for matched columns
                stats = _compute_stats(df, original_cols, canonical_cols)

                detected.append(DetectedSensor(
                    profile=profile,
                    matched_columns=original_cols,
                    sample_stats=stats,
                ))
                logger.info(
                    "Detected sensor '%s' (%s) from columns %s in %s",
                    profile.name, profile.quantity, original_cols, source_name,
                )

        # Identify time column
        time_range = None
        time_cols = {"date", "timestamp", "time", "datetime"}
        for orig, canon in canonical_cols.items():
            if canon in time_cols or orig.lower() in time_cols:
                claimed_columns.add(orig)
                try:
                    ts = pd.to_datetime(df[orig], errors="coerce").dropna()
                    if not ts.empty:
                        time_range = (str(ts.min()), str(ts.max()))
                except Exception:
                    pass

        # Everything else is unmatched
        unmatched = [
            col for col in df.columns
            if col not in claimed_columns
        ]

        rig = cls(
            rig_id=rig_id,
            sensors=detected,
            unmatched_columns=unmatched,
            row_count=len(df),
            time_range=time_range,
        )

        logger.info(
            "SensorRig '%s': %d sensors detected, %d unmatched columns, %d rows",
            rig_id, len(detected), len(unmatched), len(df),
        )
        return rig


def _compute_stats(
    df: pd.DataFrame,
    original_cols: list[str],
    canonical_map: dict[str, str],
) -> dict[str, Any]:
    """Compute basic statistics for matched columns."""
    stats: dict[str, Any] = {}
    for col in original_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        canon = canonical_map.get(col, col)
        stats["min"] = float(series.min())
        stats["max"] = float(series.max())
        stats["mean"] = round(float(series.mean()), 2)
        stats["std"] = round(float(series.std()), 2)
        stats["count"] = int(series.count())
        stats["column"] = canon
    return stats
