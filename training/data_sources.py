"""Data source abstractions — CSV, InfluxDB, Synthetic."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for training data sources."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data and return a DataFrame with standard column names."""


class CSVDataSource(DataSource):
    """Load training data from a CSV file.

    Handles the known data quirks:
    - Repeated header rows (filtered out)
    - Non-numeric turbidity_adc values (dropped)
    - Alternate column names (e.g. ``Turbidity`` → ``turbidity_adc``)
    """

    # Maps common alternate names → canonical column names
    _COLUMN_MAP: dict[str, str] = {
        "turbidity": "turbidity_adc",
        "tds": "tds",
        "temperature": "water_temperature",
        "date": "timestamp",
    }

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Handle DATA.CSV quirk: header has "Date,Time, TDS, Turbidity,Temperature"
        # (5 columns) but data rows only have 4 values because the ISO
        # timestamp in Date already includes the time.  pandas parses 4
        # values into the first 4 header slots, leaving the 5th (Temperature)
        # entirely NaN.  Fix: drop the NaN column and rename the remaining
        # 4 columns to their correct meaning.
        if (
            df.columns.size >= 2
            and df.iloc[:, -1].isna().all()
            and "Date" in df.columns
            and "Time" in df.columns
        ):
            df = df.drop(columns=[df.columns[-1]])  # drop all-NaN column
            df.columns = ["Date", "TDS", "Turbidity", "Temperature"]
            logger.info("Detected Date+Time header shift — corrected columns")

        # Normalise column names to canonical form
        rename = {}
        for col in df.columns:
            canonical = self._COLUMN_MAP.get(col.lower())
            if canonical and canonical not in df.columns:
                rename[col] = canonical
        if rename:
            df = df.rename(columns=rename)
            logger.info("Mapped CSV columns: %s", rename)

        # Drop repeated header rows (where turbidity_adc is non-numeric)
        if "turbidity_adc" in df.columns:
            df = df[pd.to_numeric(df["turbidity_adc"], errors="coerce").notna()]

        # Convert types
        numeric_cols = ["turbidity_adc", "tds", "water_temperature"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        logger.info("Loaded %d rows from %s", len(df), self.path.name)
        return df.dropna(subset=["turbidity_adc"]).reset_index(drop=True)


class SyntheticDataSource(DataSource):
    """Generate synthetic water quality readings across all three regimes.

    Useful for testing the training pipeline before real labelled data
    is available.
    """

    def __init__(self, n_samples: int = 1000, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed

    def load(self) -> pd.DataFrame:
        import numpy as np

        rng = np.random.default_rng(self.seed)
        n = self.n_samples
        n_per_regime = n // 3

        records = []
        # Solution regime: high ADC, low turbidity
        for _ in range(n_per_regime):
            records.append({
                "turbidity_adc": rng.integers(800, 1024),
                "tds": rng.uniform(100, 400),
                "water_temperature": rng.uniform(10, 25),
                "regime": "solution",
            })
        # Colloid regime: mid-range ADC
        for _ in range(n_per_regime):
            records.append({
                "turbidity_adc": rng.integers(400, 800),
                "tds": rng.uniform(200, 600),
                "water_temperature": rng.uniform(10, 25),
                "regime": "colloid",
            })
        # Suspension regime: low ADC
        for _ in range(n - 2 * n_per_regime):
            records.append({
                "turbidity_adc": rng.integers(0, 400),
                "tds": rng.uniform(50, 500),
                "water_temperature": rng.uniform(10, 25),
                "regime": "suspension",
            })

        df = pd.DataFrame(records)
        return df.sample(frac=1, random_state=self.seed).reset_index(drop=True)


class InfluxDBDataSource(DataSource):
    """Load training data from InfluxDB time series database.

    Queries the ``turbidity_readings`` measurement (configurable) and returns
    a DataFrame with columns matching the ``Reading`` schema.
    """

    def __init__(
        self,
        start: str,
        stop: str,
        rig_id: str | None = None,
        measurement: str | None = None,
    ) -> None:
        self.start = start
        self.stop = stop
        self.rig_id = rig_id
        self.measurement = measurement

    def load(self) -> pd.DataFrame:
        from app.database import influx_manager
        from app.exceptions import InsufficientDataError

        if not influx_manager.connected and not influx_manager._ensure_connected():
            raise InsufficientDataError(
                "InfluxDB is not connected — cannot load training data"
            )

        measurement = self.measurement or influx_manager.db_config.get(
            "readings_measurement", "turbidity_readings"
        )

        where = f"time >= '{self.start}' AND time < '{self.stop}'"
        if self.rig_id:
            where += f" AND \"rig_id\" = '{self.rig_id}'"

        query = f'SELECT * FROM "{measurement}" WHERE {where}'
        try:
            rows = influx_manager._query_v1(query)
        except Exception as e:
            raise InsufficientDataError(f"InfluxDB query failed: {e}") from e

        if not rows:
            raise InsufficientDataError(
                f"No data found in '{measurement}' for range {self.start} — {self.stop}"
            )

        df = pd.DataFrame(rows)

        # Convert types for known columns
        numeric_cols = ["turbidity_adc", "tds", "water_temperature"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.drop(columns=["time"])

        logger.info(
            "Loaded %d rows from InfluxDB (%s — %s)", len(df), self.start, self.stop
        )
        return df.dropna(subset=["turbidity_adc"]).reset_index(drop=True)
