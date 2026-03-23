"""Data source abstractions — CSV, InfluxDB, Synthetic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


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
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)

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
