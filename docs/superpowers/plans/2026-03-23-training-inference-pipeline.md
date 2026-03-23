# Training & Inference Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the FSM training pipeline and ML-aware inference engine so ClearEye can train a regime classifier on 50K field readings and load trained models at inference time with graceful fallback.

**Architecture:** FSM stage pipeline (ingest, clean, feature_engineer, split, scale, train, evaluate, save) producing versioned model artifacts. Inference engine loads artifacts at startup, falls back to datasheet/rule-based when unavailable. Three model types supported: classification, regression, autoencoder.

**Tech Stack:** Python 3.12, PyTorch, pandas, scikit-learn (StandardScaler), FastAPI, pydantic, PyYAML, OpenTelemetry

**Spec:** `docs/superpowers/specs/2026-03-23-training-inference-pipeline-design.md`

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `training/pipeline/context.py` | `PipelineContext` dataclass |
| `training/pipeline/stages.py` | Split, scale, evaluate, save stage functions |
| `training/pipeline/clean.py` | Clean stage: validation, staleness, sensor profile checks |
| `training/pipeline/features.py` | Feature engineering + label generation |
| `training/pipeline/trainer.py` | Training loop: classification, regression, autoencoder |
| `training/artifacts.py` | Artifact save/load, semver resolution, scaler JSON serialization |
| `training/label_studio.py` | `LabelStudioSource` stub |
| `app/ml_regime.py` | `MLRegimeClassifier` wraps RegimeClassifierNet + scaler |
| `app/ml_calibration.py` | `MLResidualCalibrator` wraps ResidualCorrectionNet + scaler |
| `app/model_loader.py` | Model discovery, loading, fallback logic |
| `app/telemetry.py` | OpenTelemetry tracer/meter initialization |
| `tests/unit/test_pipeline_context.py` | PipelineContext tests |
| `tests/unit/test_clean_stage.py` | Clean stage tests |
| `tests/unit/test_features_stage.py` | Feature engineering tests |
| `tests/unit/test_split_scale.py` | Split + scale stage tests |
| `tests/unit/test_trainer.py` | Training loop tests |
| `tests/unit/test_artifacts.py` | Artifact save/load tests |
| `tests/unit/test_ml_regime.py` | MLRegimeClassifier tests |
| `tests/unit/test_model_loader.py` | Model loader + fallback tests |
| `tests/integration/test_pipeline_e2e.py` | End-to-end pipeline on synthetic data |

### Modified files

| File | Change |
|------|--------|
| `training/pipeline/orchestrator.py` | Replace stub with real FSM using PipelineContext + stage functions |
| `training/data_sources.py` | Add `LabelStudioSource` import/re-export |
| `training/train.py` | Wire CLI entry point to orchestrator |
| `app/config.py` | Parse new `models:`, `artifacts:`, `label_studio:` config blocks |
| `app/prediction_service.py` | Use model_loader to build InferenceEngine with ML components when available |
| `model_config.yaml` | Update to new structure with `models:` block |
| `cleareye/__main__.py` | Wire `train` command to real pipeline |

---

## Task 1: PipelineContext dataclass

**Files:**
- Create: `training/pipeline/context.py`
- Test: `tests/unit/test_pipeline_context.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pipeline_context.py
"""Tests for PipelineContext dataclass."""

from training.pipeline.context import PipelineContext


class TestPipelineContext:
    def test_create_empty(self):
        ctx = PipelineContext(config={"training": {"batch_size": 64}})
        assert ctx.config["training"]["batch_size"] == 64
        assert ctx.raw_df is None
        assert ctx.clean_df is None
        assert ctx.models == {}

    def test_segments_default_empty(self):
        ctx = PipelineContext(config={})
        assert ctx.segments == []
        assert ctx.warnings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_pipeline_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.pipeline.context'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/pipeline/context.py
"""PipelineContext -- carries state between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch.nn as nn


@dataclass
class PipelineContext:
    """Mutable context passed through pipeline stages."""

    config: dict[str, Any]

    # Set by ingest
    raw_df: pd.DataFrame | None = None

    # Set by clean
    clean_df: pd.DataFrame | None = None
    segments: list[tuple[int, int]] = field(default_factory=list)

    # Set by feature_engineer
    features_df: pd.DataFrame | None = None
    labels: pd.Series | None = None

    # Set by split
    train_df: pd.DataFrame | None = None
    val_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None
    train_labels: pd.Series | None = None
    val_labels: pd.Series | None = None
    test_labels: pd.Series | None = None

    # Set by scale
    scalers: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    # Set by train
    models: dict[str, nn.Module] = field(default_factory=dict)

    # Set by evaluate
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Accumulated warnings
    warnings: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_pipeline_context.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/pipeline/context.py tests/unit/test_pipeline_context.py
git commit -m "feat: add PipelineContext dataclass for training pipeline"
```

---

## Task 2: Update model_config.yaml and app/config.py

**Files:**
- Modify: `model_config.yaml`
- Modify: `app/config.py`
- Test: `tests/unit/test_config.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config.py
"""Tests for updated config parsing."""

from app.config import Settings


class TestSettings:
    def test_models_block_parsed(self):
        s = Settings()
        models = s.models
        assert "regime_classifier" in models
        assert models["regime_classifier"]["type"] == "classification"

    def test_model_enabled_default_true(self):
        s = Settings()
        assert s.is_model_enabled("regime_classifier") is True

    def test_model_enabled_false(self):
        s = Settings()
        assert s.is_model_enabled("anomaly_detector") is False

    def test_artifacts_config(self):
        s = Settings()
        assert s.artifacts_base_dir == "models"

    def test_training_config(self):
        s = Settings()
        assert s.training["batch_size"] == 64
        assert s.training["device"] == "auto"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_config.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'models'`

- [ ] **Step 3: Update model_config.yaml to new structure**

Replace `model_config.yaml` with the structure from the spec: `models:` block with per-model entries, `artifacts:`, `label_studio:` sections. Keep `server:`, `sensor:`, `training:` sections.

- [ ] **Step 4: Update app/config.py with new properties**

Add to the `Settings` class:

```python
@property
def models(self) -> dict[str, Any]:
    return self._raw.get("models", {})

def is_model_enabled(self, model_name: str) -> bool:
    model = self.models.get(model_name, {})
    return model.get("enabled", True)

@property
def training(self) -> dict[str, Any]:
    defaults = {
        "batch_size": 64, "learning_rate": 0.001, "epochs": 100,
        "early_stopping_patience": 10, "train_split": 0.7,
        "val_split": 0.15, "test_split": 0.15, "device": "auto",
    }
    defaults.update(self._raw.get("training", {}))
    return defaults

@property
def artifacts_base_dir(self) -> str:
    return self._raw.get("artifacts", {}).get("base_dir", "models")
```

- [ ] **Step 5: Run tests to verify everything passes**

Run: `python3 -m pytest tests/unit/ -v`
Expected: all pass including new config tests and existing tests

- [ ] **Step 6: Commit**

```bash
git add model_config.yaml app/config.py tests/unit/test_config.py
git commit -m "feat: update config to support models block and artifact settings"
```

---

## Task 3: Clean stage with validation and staleness checks

**Files:**
- Create: `training/pipeline/clean.py`
- Test: `tests/unit/test_clean_stage.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_clean_stage.py
"""Tests for the clean pipeline stage."""

import pandas as pd
import pytest

from app.exceptions import InsufficientDataError
from training.pipeline.clean import clean
from training.pipeline.context import PipelineContext


def _make_ctx(df: pd.DataFrame) -> PipelineContext:
    return PipelineContext(config={}, raw_df=df)


class TestCleanStage:
    def test_drops_nan_rows(self):
        df = pd.DataFrame({
            "turbidity_adc": [500, None, 600],
            "tds": [300.0, 200.0, None],
            "water_temperature": [18.0, 18.0, 18.0],
            "timestamp": pd.to_datetime(["2023-10-07", "2023-10-08", "2023-10-09"]),
        })
        ctx = clean(_make_ctx(df))
        assert len(ctx.clean_df) == 1

    def test_drops_out_of_range(self):
        df = pd.DataFrame({
            "turbidity_adc": [500, 9999],
            "tds": [300.0, 300.0],
            "water_temperature": [18.0, 18.0],
            "timestamp": pd.to_datetime(["2023-10-07", "2023-10-08"]),
        })
        ctx = clean(_make_ctx(df))
        assert len(ctx.clean_df) == 1

    def test_sorts_by_timestamp(self):
        df = pd.DataFrame({
            "turbidity_adc": [500, 600],
            "tds": [300.0, 300.0],
            "water_temperature": [18.0, 18.0],
            "timestamp": pd.to_datetime(["2023-10-09", "2023-10-07"]),
        })
        ctx = clean(_make_ctx(df))
        assert ctx.clean_df.iloc[0]["turbidity_adc"] == 600

    def test_deduplicates_timestamps(self):
        df = pd.DataFrame({
            "turbidity_adc": [500, 500],
            "tds": [300.0, 300.0],
            "water_temperature": [18.0, 18.0],
            "timestamp": pd.to_datetime(["2023-10-07", "2023-10-07"]),
        })
        ctx = clean(_make_ctx(df))
        assert len(ctx.clean_df) == 1

    def test_clean_does_not_crash_on_small_data(self):
        df = pd.DataFrame({
            "turbidity_adc": [500],
            "tds": [300.0],
            "water_temperature": [18.0],
            "timestamp": pd.to_datetime(["2023-10-07"]),
        })
        ctx = clean(_make_ctx(df))
        assert len(ctx.clean_df) == 1


class TestStalenessChecks:
    def test_timestamp_gap_detection(self):
        timestamps = pd.to_datetime([
            "2023-10-07T00:00:00",
            "2023-10-07T00:00:06",
            "2023-10-07T00:00:12",
            "2023-10-07T00:05:12",
            "2023-10-07T00:05:18",
        ])
        df = pd.DataFrame({
            "turbidity_adc": [500] * 5,
            "tds": [300.0] * 5,
            "water_temperature": [18.0] * 5,
            "timestamp": timestamps,
        })
        ctx = clean(_make_ctx(df))
        assert len(ctx.segments) == 2

    def test_sequence_gap_detection(self):
        df = pd.DataFrame({
            "turbidity_adc": [500, 500, 500],
            "tds": [300.0, 300.0, 300.0],
            "water_temperature": [18.0, 18.0, 18.0],
            "timestamp": pd.to_datetime(["2023-10-07", "2023-10-08", "2023-10-09"]),
            "seq": [1, 2, 5],
        })
        ctx = clean(_make_ctx(df))
        assert any("seq" in w.lower() or "sequence" in w.lower() for w in ctx.warnings)

    def test_data_age_warning(self):
        df = pd.DataFrame({
            "turbidity_adc": [500],
            "tds": [300.0],
            "water_temperature": [18.0],
            "timestamp": pd.to_datetime(["2020-01-01"]),
        })
        ctx = clean(_make_ctx(df))
        assert any("age" in w.lower() or "stale" in w.lower() or "old" in w.lower() for w in ctx.warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_clean_stage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement clean stage**

```python
# training/pipeline/clean.py
"""Clean stage -- validation, deduplication, staleness checks."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from training.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

_SENSOR_RANGES = {
    "turbidity_adc": (0, 1023),
    "tds": (0, 1000),
    "water_temperature": (-55, 125),
}

_REQUIRED_COLS = ["turbidity_adc", "tds", "water_temperature"]
_GAP_MULTIPLIER = 5
_DATA_AGE_DAYS = 90


def clean(ctx: PipelineContext) -> PipelineContext:
    df = ctx.raw_df.copy()

    df = df.dropna(subset=_REQUIRED_COLS)

    for col, (lo, hi) in _SENSOR_RANGES.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    ctx.clean_df = df

    if "timestamp" in df.columns and len(df) > 1:
        _detect_timestamp_gaps(ctx)
        _check_data_age(ctx)
    else:
        ctx.segments = [(0, len(df))]

    if "seq" in df.columns:
        _detect_sequence_gaps(ctx)

    return ctx


def _detect_timestamp_gaps(ctx: PipelineContext) -> None:
    df = ctx.clean_df
    deltas = df["timestamp"].diff().dt.total_seconds().iloc[1:]
    median_interval = deltas.median()
    gap_threshold = median_interval * _GAP_MULTIPLIER

    segments = []
    seg_start = 0
    for i, delta in enumerate(deltas, start=1):
        if delta > gap_threshold:
            segments.append((seg_start, i))
            logger.warning(
                "Gap detected: %s to %s (%.0fs)",
                df.iloc[i - 1]["timestamp"], df.iloc[i]["timestamp"], delta,
            )
            seg_start = i
    segments.append((seg_start, len(df)))
    ctx.segments = segments


def _detect_sequence_gaps(ctx: PipelineContext) -> None:
    df = ctx.clean_df
    if "seq" not in df.columns:
        return
    seq = df["seq"].dropna().astype(int).sort_values()
    expected = set(range(seq.min(), seq.max() + 1))
    actual = set(seq)
    missing = sorted(expected - actual)
    if missing:
        ctx.warnings.append(
            f"Sequence gaps: {len(missing)} missing sequence numbers "
            f"(first missing: {missing[0]}, last missing: {missing[-1]})"
        )


def _check_data_age(ctx: PipelineContext) -> None:
    df = ctx.clean_df
    newest = df["timestamp"].max()
    if hasattr(newest, "to_pydatetime"):
        newest = newest.to_pydatetime()
    now = datetime.now()
    age_days = (now - newest).days
    if age_days > _DATA_AGE_DAYS:
        ctx.warnings.append(
            f"Data age warning: newest reading is {age_days} days old "
            f"(threshold: {_DATA_AGE_DAYS} days). Training data may not reflect current conditions."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_clean_stage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/pipeline/clean.py tests/unit/test_clean_stage.py
git commit -m "feat: implement clean stage with validation and staleness checks"
```

---

## Task 4: Feature engineering and label generation

**Files:**
- Create: `training/pipeline/features.py`
- Test: `tests/unit/test_features_stage.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_features_stage.py
"""Tests for feature engineering stage."""

import math

import numpy as np
import pandas as pd

from training.pipeline.context import PipelineContext
from training.pipeline.features import feature_engineer


def _make_clean_ctx(n: int = 100) -> PipelineContext:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "turbidity_adc": rng.integers(0, 1024, n),
        "tds": rng.uniform(0, 1000, n),
        "water_temperature": rng.uniform(10, 25, n),
        "timestamp": pd.date_range("2023-10-07", periods=n, freq="6s"),
    })
    config = {
        "models": {
            "regime_classifier": {
                "type": "classification",
                "features": ["turbidity_adc", "tds", "water_temperature"],
                "label_source": "rule_derived",
            }
        }
    }
    ctx = PipelineContext(config=config, clean_df=df)
    ctx.segments = [(0, n)]
    return ctx


class TestFeatureEngineer:
    def test_extracts_configured_features(self):
        ctx = feature_engineer(_make_clean_ctx())
        assert list(ctx.features_df.columns) == ["turbidity_adc", "tds", "water_temperature"]

    def test_rule_derived_labels(self):
        ctx = feature_engineer(_make_clean_ctx())
        assert ctx.labels is not None
        assert set(ctx.labels.unique()).issubset({"solution", "colloid", "suspension"})
        assert len(ctx.labels) == len(ctx.features_df)

    def test_self_supervised_no_labels(self):
        ctx = _make_clean_ctx()
        ctx.config["models"]["regime_classifier"]["label_source"] = "self_supervised"
        ctx = feature_engineer(ctx)
        assert ctx.labels is None

    def test_voltage_derived_feature(self):
        ctx = _make_clean_ctx()
        ctx.config["models"]["regime_classifier"]["features"] = [
            "voltage", "water_temperature", "tds"
        ]
        ctx = feature_engineer(ctx)
        assert "voltage" in ctx.features_df.columns
        assert ctx.features_df["voltage"].min() >= 0

    def test_cyclical_hour_features(self):
        ctx = _make_clean_ctx()
        ctx.config["models"]["regime_classifier"]["features"] = [
            "turbidity_adc", "hour_sin", "hour_cos"
        ]
        ctx = feature_engineer(ctx)
        assert "hour_sin" in ctx.features_df.columns
        assert "hour_cos" in ctx.features_df.columns
        assert ctx.features_df["hour_sin"].between(-1, 1).all()

    def test_d_adc_dt_respects_segments(self):
        ctx = _make_clean_ctx(10)
        ctx.segments = [(0, 5), (5, 10)]
        ctx.config["models"]["regime_classifier"]["features"] = [
            "turbidity_adc", "d_adc_dt"
        ]
        ctx = feature_engineer(ctx)
        assert "d_adc_dt" in ctx.features_df.columns
        assert ctx.features_df.iloc[0]["d_adc_dt"] == 0.0
        assert ctx.features_df.iloc[5]["d_adc_dt"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_features_stage.py -v`
Expected: FAIL

- [ ] **Step 3: Implement feature engineering**

```python
# training/pipeline/features.py
"""Feature engineering stage -- derives features, generates labels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.regime import RuleBasedRegimeClassifier
from app.sensor_physics import adc_to_voltage
from training.pipeline.context import PipelineContext

_classifier = RuleBasedRegimeClassifier()


def feature_engineer(ctx: PipelineContext) -> PipelineContext:
    df = ctx.clean_df.copy()
    model_config = _get_active_model_config(ctx.config)
    feature_names: list[str] = model_config["features"]
    label_source: str = model_config.get("label_source", "rule_derived")

    derived = _derive_features(df, ctx.segments)
    ctx.features_df = derived[feature_names].copy()

    if label_source == "rule_derived":
        ctx.labels = _rule_derived_labels(df)
    elif label_source == "self_supervised":
        ctx.labels = None
    elif label_source == "reference":
        ctx.labels = None
    elif label_source == "label_studio":
        raise NotImplementedError("Label Studio integration not yet implemented")
    else:
        raise ValueError(f"Unknown label_source: {label_source!r}")

    return ctx


def _get_active_model_config(config: dict[str, Any]) -> dict[str, Any]:
    models = config.get("models", {})
    for name, model_cfg in models.items():
        if model_cfg.get("enabled", True):
            return model_cfg
    raise ValueError("No enabled model found in config")


def _derive_features(df: pd.DataFrame, segments: list[tuple[int, int]]) -> pd.DataFrame:
    derived = df[["turbidity_adc", "tds", "water_temperature"]].copy()

    derived["voltage"] = df["turbidity_adc"].apply(lambda adc: adc_to_voltage(int(adc)))

    d_adc = np.zeros(len(df))
    for seg_start, seg_end in segments:
        seg_adc = df.iloc[seg_start:seg_end]["turbidity_adc"].values
        d = np.diff(seg_adc, prepend=seg_adc[0])
        d[0] = 0.0
        d_adc[seg_start:seg_end] = d
    derived["d_adc_dt"] = d_adc

    if "timestamp" in df.columns:
        hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        derived["hour_sin"] = np.sin(2 * math.pi * hours / 24.0)
        derived["hour_cos"] = np.cos(2 * math.pi * hours / 24.0)
    else:
        derived["hour_sin"] = 0.0
        derived["hour_cos"] = 0.0

    for col in ["ph", "dissolved_oxygen", "depth", "flow_rate"]:
        if col in df.columns:
            derived[col] = df[col]

    return derived


def _rule_derived_labels(df: pd.DataFrame) -> pd.Series:
    labels = []
    for _, row in df.iterrows():
        result = _classifier.classify(
            turbidity_adc=int(row["turbidity_adc"]),
            tds=float(row["tds"]),
            temperature=float(row["water_temperature"]),
        )
        labels.append(result.regime.value)
    return pd.Series(labels, index=df.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_features_stage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/pipeline/features.py tests/unit/test_features_stage.py
git commit -m "feat: implement feature engineering with derived features and label generation"
```

---

## Task 5: Split and scale stages

**Files:**
- Create: `training/pipeline/stages.py`
- Test: `tests/unit/test_split_scale.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_split_scale.py
"""Tests for split and scale pipeline stages."""

import numpy as np
import pandas as pd

from training.pipeline.context import PipelineContext
from training.pipeline.stages import split, scale


def _make_feature_ctx(n: int = 300) -> PipelineContext:
    rng = np.random.default_rng(42)
    features = pd.DataFrame({
        "turbidity_adc": rng.integers(0, 1024, n).astype(float),
        "tds": rng.uniform(0, 1000, n),
        "water_temperature": rng.uniform(10, 25, n),
    })
    labels = pd.Series(
        rng.choice(["solution", "colloid", "suspension"], n),
        index=features.index,
    )
    config = {
        "training": {"train_split": 0.7, "val_split": 0.15, "test_split": 0.15},
        "models": {"regime_classifier": {"type": "classification", "enabled": True}},
    }
    return PipelineContext(config=config, features_df=features, labels=labels)


class TestSplitStage:
    def test_split_sizes(self):
        ctx = split(_make_feature_ctx())
        total = len(ctx.train_df) + len(ctx.val_df) + len(ctx.test_df)
        assert total == 300
        assert len(ctx.train_df) == 210
        assert len(ctx.val_df) == 45

    def test_labels_split_alongside(self):
        ctx = split(_make_feature_ctx())
        assert len(ctx.train_labels) == len(ctx.train_df)
        assert len(ctx.val_labels) == len(ctx.val_df)
        assert len(ctx.test_labels) == len(ctx.test_df)

    def test_stratified_for_classification(self):
        ctx = split(_make_feature_ctx())
        for labels in [ctx.train_labels, ctx.val_labels, ctx.test_labels]:
            assert len(labels.unique()) == 3


class TestScaleStage:
    def test_scale_normalizes(self):
        ctx = split(_make_feature_ctx())
        ctx = scale(ctx)
        for col in ctx.train_df.columns:
            assert abs(ctx.train_df[col].mean()) < 0.1
            assert abs(ctx.train_df[col].std() - 1.0) < 0.1

    def test_scaler_saved(self):
        ctx = split(_make_feature_ctx())
        ctx = scale(ctx)
        assert "features" in ctx.scalers
        assert "mean" in ctx.scalers["features"]
        assert "std" in ctx.scalers["features"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_split_scale.py -v`
Expected: FAIL

- [ ] **Step 3: Implement split and scale**

```python
# training/pipeline/stages.py
"""Pipeline stage functions -- split, scale, evaluate, save."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from training.pipeline.context import PipelineContext


def split(ctx: PipelineContext) -> PipelineContext:
    training_cfg = ctx.config.get("training", {})
    train_ratio = training_cfg.get("train_split", 0.7)
    val_ratio = training_cfg.get("val_split", 0.15)

    features = ctx.features_df
    labels = ctx.labels
    stratify = labels if _is_classification(ctx) and labels is not None else None
    val_test_ratio = 1.0 - train_ratio

    train_df, val_test_df, train_labels, val_test_labels = train_test_split(
        features, labels, test_size=val_test_ratio, stratify=stratify, random_state=42,
    )

    test_fraction = 1.0 - (val_ratio / val_test_ratio) if val_test_ratio > 0 else 0.5
    stratify_vt = val_test_labels if stratify is not None else None
    val_df, test_df, val_labels, test_labels = train_test_split(
        val_test_df, val_test_labels, test_size=test_fraction, stratify=stratify_vt, random_state=42,
    )

    ctx.train_df = train_df.reset_index(drop=True)
    ctx.val_df = val_df.reset_index(drop=True)
    ctx.test_df = test_df.reset_index(drop=True)
    ctx.train_labels = train_labels.reset_index(drop=True)
    ctx.val_labels = val_labels.reset_index(drop=True)
    ctx.test_labels = test_labels.reset_index(drop=True)
    return ctx


def scale(ctx: PipelineContext) -> PipelineContext:
    train = ctx.train_df
    mean = train.mean().values
    std = train.std().values
    std[std == 0] = 1.0

    ctx.train_df = pd.DataFrame((train.values - mean) / std, columns=train.columns)
    ctx.val_df = pd.DataFrame((ctx.val_df.values - mean) / std, columns=train.columns)
    ctx.test_df = pd.DataFrame((ctx.test_df.values - mean) / std, columns=train.columns)

    ctx.scalers["features"] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "columns": list(train.columns),
    }
    return ctx


def _is_classification(ctx: PipelineContext) -> bool:
    models = ctx.config.get("models", {})
    for cfg in models.values():
        if cfg.get("enabled", True):
            return cfg.get("type") == "classification"
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_split_scale.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/pipeline/stages.py tests/unit/test_split_scale.py
git commit -m "feat: implement split and scale pipeline stages"
```

---

## Task 6: Training loop (classification, regression, autoencoder)

**Files:**
- Create: `training/pipeline/trainer.py`
- Test: `tests/unit/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_trainer.py
"""Tests for the training loop."""

import numpy as np
import pandas as pd
import torch

from training.pipeline.context import PipelineContext
from training.pipeline.trainer import train_models


def _make_scaled_ctx(model_type: str = "classification") -> PipelineContext:
    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    train_df = pd.DataFrame(rng.standard_normal((n_train, 3)), columns=["a", "b", "c"])
    val_df = pd.DataFrame(rng.standard_normal((n_val, 3)), columns=["a", "b", "c"])
    test_df = pd.DataFrame(rng.standard_normal((n_val, 3)), columns=["a", "b", "c"])

    if model_type == "classification":
        train_labels = pd.Series(rng.choice(["solution", "colloid", "suspension"], n_train))
        val_labels = pd.Series(rng.choice(["solution", "colloid", "suspension"], n_val))
        test_labels = pd.Series(rng.choice(["solution", "colloid", "suspension"], n_val))
    else:
        train_labels = pd.Series(rng.standard_normal(n_train))
        val_labels = pd.Series(rng.standard_normal(n_val))
        test_labels = pd.Series(rng.standard_normal(n_val))

    config = {
        "training": {
            "batch_size": 32, "learning_rate": 0.01, "epochs": 5,
            "early_stopping_patience": 3, "device": "cpu",
        },
        "models": {
            "test_model": {
                "type": model_type,
                "architecture": "mlp" if model_type != "autoencoder" else "autoencoder",
                "features": ["a", "b", "c"],
                "hidden_dim": 16,
                "encoding_dim": 2,
                "enabled": True,
            }
        },
    }
    return PipelineContext(
        config=config,
        train_df=train_df, val_df=val_df, test_df=test_df,
        train_labels=train_labels, val_labels=val_labels, test_labels=test_labels,
    )


class TestTrainModels:
    def test_classification_produces_model(self):
        ctx = train_models(_make_scaled_ctx("classification"))
        assert "test_model" in ctx.models
        assert isinstance(ctx.models["test_model"], torch.nn.Module)

    def test_regression_produces_model(self):
        ctx = train_models(_make_scaled_ctx("regression"))
        assert "test_model" in ctx.models

    def test_autoencoder_produces_model(self):
        ctx = train_models(_make_scaled_ctx("autoencoder"))
        assert "test_model" in ctx.models
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement training loop**

```python
# training/pipeline/trainer.py
"""Training loop -- supports classification, regression, autoencoder."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.models import AnomalyAutoencoder, RegimeClassifierNet, ResidualCorrectionNet
from training.pipeline.context import PipelineContext
from training.utils import detect_device

logger = logging.getLogger(__name__)

_REGIME_TO_IDX = {"solution": 0, "colloid": 1, "suspension": 2}


def train_models(ctx: PipelineContext) -> PipelineContext:
    training_cfg = ctx.config.get("training", {})
    device = detect_device(training_cfg.get("device", "cpu"))

    for model_name, model_cfg in ctx.config.get("models", {}).items():
        if not model_cfg.get("enabled", True):
            continue
        logger.info("Training model: %s", model_name)
        model = _train_single(ctx, model_name, model_cfg, training_cfg, device)
        ctx.models[model_name] = model

    return ctx


def _train_single(
    ctx: PipelineContext, name: str, model_cfg: dict[str, Any],
    training_cfg: dict[str, Any], device: torch.device,
) -> nn.Module:
    model_type = model_cfg["type"]
    input_dim = len(model_cfg["features"])
    hidden_dim = model_cfg.get("hidden_dim", 32)

    if model_type == "classification":
        model = RegimeClassifierNet(input_dim=input_dim, hidden_dim=hidden_dim)
        criterion = nn.CrossEntropyLoss()
    elif model_type == "regression":
        model = ResidualCorrectionNet(input_dim=input_dim, hidden_dim=hidden_dim)
        criterion = nn.MSELoss()
    elif model_type == "autoencoder":
        encoding_dim = model_cfg.get("encoding_dim", 4)
        model = AnomalyAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, encoding_dim=encoding_dim)
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.get("learning_rate", 0.001))

    train_X = torch.tensor(ctx.train_df.values, dtype=torch.float32)
    val_X = torch.tensor(ctx.val_df.values, dtype=torch.float32)

    if model_type == "classification":
        train_y = torch.tensor([_REGIME_TO_IDX[l] for l in ctx.train_labels], dtype=torch.long)
        val_y = torch.tensor([_REGIME_TO_IDX[l] for l in ctx.val_labels], dtype=torch.long)
    elif model_type == "regression":
        train_y = torch.tensor(ctx.train_labels.values, dtype=torch.float32)
        val_y = torch.tensor(ctx.val_labels.values, dtype=torch.float32)
    else:
        train_y = train_X.clone()
        val_y = val_X.clone()

    batch_size = training_cfg.get("batch_size", 64)
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)

    epochs = training_cfg.get("epochs", 100)
    patience = training_cfg.get("early_stopping_patience", 10)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_X.to(device))
            val_loss = criterion(val_out, val_y.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/pipeline/trainer.py tests/unit/test_trainer.py
git commit -m "feat: implement training loop for classification, regression, autoencoder"
```

---

## Task 7: Artifact save/load with semver and scaler JSON

**Files:**
- Create: `training/artifacts.py`
- Test: `tests/unit/test_artifacts.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_artifacts.py
"""Tests for artifact save/load."""

import json

import torch

from app.models import RegimeClassifierNet
from training.artifacts import save_artifact, load_artifact, resolve_latest_version


class TestArtifacts:
    def test_save_and_load_model(self, tmp_path):
        model = RegimeClassifierNet(input_dim=3, hidden_dim=16)
        scaler = {"mean": [1.0, 2.0, 3.0], "std": [0.5, 0.5, 0.5], "columns": ["a", "b", "c"]}
        metadata = {"version": "0.1.0", "model_type": "classification"}

        save_artifact(
            model=model, scaler=scaler, metadata=metadata,
            base_dir=tmp_path, model_name="test_model", version="0.1.0",
        )

        artifact_dir = tmp_path / "test_model" / "v0.1.0"
        assert (artifact_dir / "model.pt").exists()
        assert (artifact_dir / "scaler.json").exists()
        assert (artifact_dir / "metadata.json").exists()

        loaded_model, loaded_scaler, loaded_meta = load_artifact(
            base_dir=tmp_path, model_name="test_model", version="0.1.0",
            model_class=RegimeClassifierNet, model_kwargs={"input_dim": 3, "hidden_dim": 16},
        )
        assert isinstance(loaded_model, RegimeClassifierNet)
        assert loaded_scaler["mean"] == [1.0, 2.0, 3.0]
        assert loaded_meta["version"] == "0.1.0"

    def test_resolve_latest_version(self, tmp_path):
        (tmp_path / "my_model" / "v0.1.0").mkdir(parents=True)
        (tmp_path / "my_model" / "v0.2.0").mkdir(parents=True)
        (tmp_path / "my_model" / "v0.1.1").mkdir(parents=True)
        assert resolve_latest_version(tmp_path, "my_model") == "0.2.0"

    def test_resolve_latest_none_when_empty(self, tmp_path):
        assert resolve_latest_version(tmp_path, "missing_model") is None

    def test_scaler_saved_as_json(self, tmp_path):
        model = RegimeClassifierNet(input_dim=3, hidden_dim=16)
        scaler = {"mean": [1.0], "std": [1.0], "columns": ["x"]}
        save_artifact(
            model=model, scaler=scaler, metadata={},
            base_dir=tmp_path, model_name="m", version="0.1.0",
        )
        scaler_path = tmp_path / "m" / "v0.1.0" / "scaler.json"
        loaded = json.loads(scaler_path.read_text())
        assert loaded["mean"] == [1.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_artifacts.py -v`
Expected: FAIL

- [ ] **Step 3: Implement artifacts module**

```python
# training/artifacts.py
"""Artifact save/load -- model weights, scaler JSON, metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_artifact(
    model: nn.Module, scaler: dict[str, Any], metadata: dict[str, Any],
    base_dir: Path, model_name: str, version: str,
) -> Path:
    artifact_dir = base_dir / model_name / f"v{version}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "scaler.json").write_text(json.dumps(scaler, indent=2))
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return artifact_dir


def load_artifact(
    base_dir: Path, model_name: str, version: str,
    model_class: type[nn.Module], model_kwargs: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    artifact_dir = base_dir / model_name / f"v{version}"
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(artifact_dir / "model.pt", weights_only=True))
    model.eval()
    scaler = json.loads((artifact_dir / "scaler.json").read_text())
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    return model, scaler, metadata


def resolve_latest_version(base_dir: Path, model_name: str) -> str | None:
    model_dir = base_dir / model_name
    if not model_dir.exists():
        return None
    versions = []
    for d in model_dir.iterdir():
        if d.is_dir() and d.name.startswith("v"):
            match = re.match(r"^v(\d+)\.(\d+)\.(\d+)$", d.name)
            if match:
                versions.append(tuple(int(x) for x in match.groups()))
    if not versions:
        return None
    best = max(versions)
    return f"{best[0]}.{best[1]}.{best[2]}"


def next_version(base_dir: Path, model_name: str) -> str:
    current = resolve_latest_version(base_dir, model_name)
    if current is None:
        return "0.1.0"
    parts = [int(x) for x in current.split(".")]
    parts[2] += 1
    return ".".join(str(x) for x in parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_artifacts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/artifacts.py tests/unit/test_artifacts.py
git commit -m "feat: implement artifact save/load with semver and JSON scaler"
```

---

## Task 8: Evaluate and save stages + orchestrator

**Files:**
- Modify: `training/pipeline/stages.py` (add evaluate, save)
- Rewrite: `training/pipeline/orchestrator.py`
- Rewrite: `training/train.py`
- Test: `tests/integration/test_pipeline_e2e.py`

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/integration/test_pipeline_e2e.py
"""End-to-end pipeline test on synthetic data."""

from training.pipeline.orchestrator import PipelineOrchestrator


class TestPipelineE2E:
    def test_full_pipeline_synthetic(self, tmp_path):
        config = {
            "training": {
                "batch_size": 32, "learning_rate": 0.01, "epochs": 3,
                "early_stopping_patience": 2, "device": "cpu",
                "train_split": 0.7, "val_split": 0.15, "test_split": 0.15,
            },
            "models": {
                "regime_classifier": {
                    "type": "classification",
                    "architecture": "mlp",
                    "label_source": "rule_derived",
                    "features": ["turbidity_adc", "tds", "water_temperature"],
                    "hidden_dim": 16,
                    "enabled": True,
                }
            },
            "artifacts": {"base_dir": str(tmp_path / "models")},
        }

        orchestrator = PipelineOrchestrator(config=config, data_source="synthetic")
        ctx = orchestrator.run()

        assert "regime_classifier" in ctx.models
        assert "regime_classifier" in ctx.metrics
        assert "accuracy" in ctx.metrics["regime_classifier"]

        model_dir = tmp_path / "models" / "regime_classifier"
        assert model_dir.exists()
        versions = list(model_dir.iterdir())
        assert len(versions) == 1
        assert (versions[0] / "model.pt").exists()
        assert (versions[0] / "scaler.json").exists()
        assert (versions[0] / "metadata.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/integration/test_pipeline_e2e.py -v`
Expected: FAIL

- [ ] **Step 3: Add evaluate and save to stages.py**

Append to `training/pipeline/stages.py`:

```python
import torch
from training.artifacts import save_artifact, next_version


_REGIME_TO_IDX = {"solution": 0, "colloid": 1, "suspension": 2}


def evaluate(ctx: PipelineContext) -> PipelineContext:
    device = torch.device("cpu")

    for model_name, model in ctx.models.items():
        model_cfg = ctx.config["models"][model_name]
        model_type = model_cfg["type"]
        model.eval()
        test_X = torch.tensor(ctx.test_df.values, dtype=torch.float32).to(device)

        if model_type == "classification":
            with torch.no_grad():
                logits = model(test_X)
                preds = logits.argmax(dim=1).numpy()
            true = np.array([_REGIME_TO_IDX[l] for l in ctx.test_labels])
            accuracy = (preds == true).mean()
            ctx.metrics[model_name] = {"accuracy": float(accuracy)}

        elif model_type == "regression":
            test_y = torch.tensor(ctx.test_labels.values, dtype=torch.float32)
            with torch.no_grad():
                preds = model(test_X).numpy()
            true = test_y.numpy()
            ctx.metrics[model_name] = {
                "mae": float(np.abs(preds - true).mean()),
                "rmse": float(np.sqrt(((preds - true) ** 2).mean())),
            }

        elif model_type == "autoencoder":
            with torch.no_grad():
                recon = model(test_X)
                errors = ((test_X - recon) ** 2).mean(dim=1).numpy()
            ctx.metrics[model_name] = {
                "mean_recon_error": float(errors.mean()),
                "std_recon_error": float(errors.std()),
                "threshold_95": float(np.percentile(errors, 95)),
            }

    return ctx


def save(ctx: PipelineContext) -> PipelineContext:
    from datetime import datetime, timezone

    base_dir = Path(ctx.config.get("artifacts", {}).get("base_dir", "models"))

    for model_name, model in ctx.models.items():
        version = next_version(base_dir, model_name)
        model_cfg = ctx.config["models"][model_name]

        metadata = {
            "version": version,
            "model_type": model_cfg["type"],
            "features": model_cfg["features"],
            "label_source": model_cfg.get("label_source", "unknown"),
            "metrics": ctx.metrics.get(model_name, {}),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "config": ctx.config.get("training", {}),
            "warnings": ctx.warnings,
        }

        save_artifact(
            model=model, scaler=ctx.scalers.get("features", {}),
            metadata=metadata, base_dir=base_dir,
            model_name=model_name, version=version,
        )

    return ctx
```

- [ ] **Step 4: Rewrite orchestrator**

```python
# training/pipeline/orchestrator.py
"""FSM pipeline orchestrator."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from training.data_sources import CSVDataSource, SyntheticDataSource
from training.pipeline.clean import clean
from training.pipeline.context import PipelineContext
from training.pipeline.features import feature_engineer
from training.pipeline.stages import evaluate, save, scale, split
from training.pipeline.trainer import train_models

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    INGEST = "ingest"
    CLEAN = "clean"
    FEATURE_ENGINEER = "feature_engineer"
    SPLIT = "split"
    SCALE = "scale"
    TRAIN = "train"
    EVALUATE = "evaluate"
    SAVE = "save"
    DONE = "done"


class PipelineOrchestrator:
    def __init__(
        self, config: dict[str, Any],
        data_source: str = "synthetic", data_path: str | None = None,
    ) -> None:
        self.config = config
        self.data_source = data_source
        self.data_path = data_path

    def run(self) -> PipelineContext:
        ctx = PipelineContext(config=self.config)

        logger.info("Stage: %s", PipelineStage.INGEST.value)
        if self.data_source == "synthetic":
            source = SyntheticDataSource()
        else:
            from pathlib import Path
            source = CSVDataSource(Path(self.data_path))
        ctx.raw_df = source.load()

        logger.info("Stage: %s", PipelineStage.CLEAN.value)
        ctx = clean(ctx)

        logger.info("Stage: %s", PipelineStage.FEATURE_ENGINEER.value)
        ctx = feature_engineer(ctx)

        logger.info("Stage: %s", PipelineStage.SPLIT.value)
        ctx = split(ctx)

        logger.info("Stage: %s", PipelineStage.SCALE.value)
        ctx = scale(ctx)

        logger.info("Stage: %s", PipelineStage.TRAIN.value)
        ctx = train_models(ctx)

        logger.info("Stage: %s", PipelineStage.EVALUATE.value)
        ctx = evaluate(ctx)

        logger.info("Stage: %s", PipelineStage.SAVE.value)
        ctx = save(ctx)

        logger.info("Pipeline complete")
        return ctx
```

- [ ] **Step 5: Rewrite train.py**

```python
# training/train.py
"""Training entry point."""

from __future__ import annotations

from pathlib import Path

from app.config import load_config
from training.pipeline.orchestrator import PipelineOrchestrator


def train(
    config_path: Path | None = None,
    data_source: str = "synthetic",
    data_path: str | None = None,
) -> None:
    config = load_config(config_path)
    orchestrator = PipelineOrchestrator(
        config=config, data_source=data_source, data_path=data_path
    )
    ctx = orchestrator.run()

    for model_name, metrics in ctx.metrics.items():
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    if ctx.warnings:
        print("\nWarnings:")
        for w in ctx.warnings:
            print(f"  - {w}")
```

- [ ] **Step 6: Run end-to-end test**

Run: `python3 -m pytest tests/integration/test_pipeline_e2e.py -v`
Expected: PASS

- [ ] **Step 7: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add training/pipeline/orchestrator.py training/pipeline/stages.py training/train.py tests/integration/test_pipeline_e2e.py
git commit -m "feat: implement evaluate/save stages and pipeline orchestrator"
```

---

## Task 9: Label Studio stub

**Files:**
- Create: `training/label_studio.py`
- Test: `tests/unit/test_label_studio.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_label_studio.py
"""Tests for Label Studio stub."""

import pytest

from training.label_studio import LabelStudioSource


class TestLabelStudioSource:
    def test_stub_raises(self):
        source = LabelStudioSource(project_id="test")
        with pytest.raises(NotImplementedError, match="Label Studio"):
            source.load()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_label_studio.py -v`
Expected: FAIL

- [ ] **Step 3: Implement stub**

```python
# training/label_studio.py
"""Label Studio integration -- stub for future implementation."""

from __future__ import annotations

import pandas as pd

from training.data_sources import DataSource


class LabelStudioSource(DataSource):
    """Import annotated data from Label Studio. Stub -- not implemented."""

    def __init__(
        self, project_id: str, url: str = "http://localhost:8080",
        api_key: str | None = None, export_format: str = "JSON",
    ) -> None:
        self.project_id = project_id
        self.url = url
        self.api_key = api_key
        self.export_format = export_format

    def load(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Label Studio integration not yet implemented. "
            f"Would connect to {self.url}, project {self.project_id}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_label_studio.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/label_studio.py tests/unit/test_label_studio.py
git commit -m "feat: add Label Studio data source stub"
```

---

## Task 10: ML regime classifier (inference side)

**Files:**
- Create: `app/ml_regime.py`
- Test: `tests/unit/test_ml_regime.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_ml_regime.py
"""Tests for MLRegimeClassifier."""

from app.ml_regime import MLRegimeClassifier
from app.models import RegimeClassifierNet
from app.regime import TurbidityRegime
from training.artifacts import save_artifact


class TestMLRegimeClassifier:
    def test_classify_returns_regime_result(self):
        model = RegimeClassifierNet(input_dim=3, hidden_dim=16)
        scaler = {
            "mean": [500.0, 300.0, 18.0],
            "std": [200.0, 150.0, 5.0],
            "columns": ["turbidity_adc", "tds", "water_temperature"],
        }
        clf = MLRegimeClassifier(model=model, scaler=scaler)
        result = clf.classify(turbidity_adc=500, tds=300.0, temperature=18.0)
        assert result.regime in TurbidityRegime
        assert 0.0 <= result.confidence <= 1.0

    def test_load_from_artifacts(self, tmp_path):
        model = RegimeClassifierNet(input_dim=3, hidden_dim=16)
        scaler = {
            "mean": [500.0, 300.0, 18.0],
            "std": [200.0, 150.0, 5.0],
            "columns": ["turbidity_adc", "tds", "water_temperature"],
        }
        save_artifact(
            model=model, scaler=scaler,
            metadata={"version": "0.1.0"},
            base_dir=tmp_path, model_name="regime_classifier", version="0.1.0",
        )
        clf = MLRegimeClassifier.from_artifact(tmp_path, "regime_classifier", "0.1.0")
        result = clf.classify(turbidity_adc=500, tds=300.0, temperature=18.0)
        assert result.regime in TurbidityRegime
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_ml_regime.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MLRegimeClassifier**

```python
# app/ml_regime.py
"""ML-based regime classifier -- wraps RegimeClassifierNet + scaler."""

from __future__ import annotations

from pathlib import Path

import torch

from app.models import RegimeClassifierNet
from app.regime import RegimeResult, TurbidityRegime
from training.artifacts import load_artifact

_IDX_TO_REGIME = [TurbidityRegime.SOLUTION, TurbidityRegime.COLLOID, TurbidityRegime.SUSPENSION]


class MLRegimeClassifier:
    """ML regime classifier implementing the RegimeClassifier protocol."""

    def __init__(self, model: RegimeClassifierNet, scaler: dict) -> None:
        self._model = model
        self._model.eval()
        self._mean = scaler["mean"]
        self._std = scaler["std"]

    @classmethod
    def from_artifact(
        cls, base_dir: Path, model_name: str, version: str,
        hidden_dim: int = 32,
    ) -> MLRegimeClassifier:
        model, scaler, _ = load_artifact(
            base_dir=base_dir, model_name=model_name, version=version,
            model_class=RegimeClassifierNet,
            model_kwargs={"input_dim": 3, "hidden_dim": hidden_dim},
        )
        return cls(model=model, scaler=scaler)

    def classify(self, turbidity_adc: int, tds: float, temperature: float) -> RegimeResult:
        features = torch.tensor([float(turbidity_adc), tds, temperature], dtype=torch.float32)
        mean = torch.tensor(self._mean, dtype=torch.float32)
        std = torch.tensor(self._std, dtype=torch.float32)
        scaled = (features - mean) / std

        with torch.no_grad():
            logits = self._model(scaled.unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze()

        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        return RegimeResult(regime=_IDX_TO_REGIME[pred_idx], confidence=confidence)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_ml_regime.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ml_regime.py tests/unit/test_ml_regime.py
git commit -m "feat: implement MLRegimeClassifier for inference"
```

---

## Task 11: Model loader with fallback

**Files:**
- Create: `app/model_loader.py`
- Modify: `app/prediction_service.py`
- Test: `tests/unit/test_model_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_model_loader.py
"""Tests for model loader with graceful fallback."""

from app.model_loader import build_engine
from app.inference import InferenceEngine
from app.regime import RuleBasedRegimeClassifier


class TestBuildEngine:
    def test_fallback_to_rule_based_when_no_models(self, tmp_path):
        config = {
            "models": {"regime_classifier": {"type": "classification", "architecture": "mlp"}},
            "artifacts": {"base_dir": str(tmp_path / "empty_models")},
            "sensor": {"v_ref": 5.0},
        }
        engine = build_engine(config)
        assert isinstance(engine, InferenceEngine)
        assert isinstance(engine.regime_classifier, RuleBasedRegimeClassifier)

    def test_loads_ml_model_when_available(self, tmp_path):
        from app.models import RegimeClassifierNet
        from training.artifacts import save_artifact

        model = RegimeClassifierNet(input_dim=3, hidden_dim=32)
        scaler = {
            "mean": [500.0, 300.0, 18.0],
            "std": [200.0, 150.0, 5.0],
            "columns": ["turbidity_adc", "tds", "water_temperature"],
        }
        save_artifact(
            model=model, scaler=scaler,
            metadata={"version": "0.1.0"},
            base_dir=tmp_path, model_name="regime_classifier", version="0.1.0",
        )

        config = {
            "models": {"regime_classifier": {"type": "classification", "architecture": "mlp", "hidden_dim": 32}},
            "artifacts": {"base_dir": str(tmp_path)},
            "sensor": {"v_ref": 5.0},
        }
        engine = build_engine(config)
        assert isinstance(engine, InferenceEngine)
        from app.ml_regime import MLRegimeClassifier
        assert isinstance(engine.regime_classifier, MLRegimeClassifier)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_model_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement model loader**

```python
# app/model_loader.py
"""Model discovery, loading, and fallback logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.biofouling import BiofoulingMonitor
from app.calibration import CalibratorBank
from app.inference import InferenceEngine
from app.regime import RuleBasedRegimeClassifier

logger = logging.getLogger(__name__)


def build_engine(config: dict[str, Any]) -> InferenceEngine:
    base_dir = Path(config.get("artifacts", {}).get("base_dir", "models"))
    v_ref = config.get("sensor", {}).get("v_ref", 5.0)

    regime_classifier = _load_regime_classifier(config, base_dir)
    calibrator_bank = CalibratorBank()
    biofouling_monitor = BiofoulingMonitor()

    return InferenceEngine(
        regime_classifier=regime_classifier,
        calibrator_bank=calibrator_bank,
        biofouling_monitor=biofouling_monitor,
        v_ref=v_ref,
    )


def _load_regime_classifier(config: dict[str, Any], base_dir: Path):
    from training.artifacts import resolve_latest_version

    model_name = "regime_classifier"
    version = resolve_latest_version(base_dir, model_name)

    if version is None:
        logger.warning("No trained regime classifier found -- using rule-based fallback")
        return RuleBasedRegimeClassifier()

    try:
        from app.ml_regime import MLRegimeClassifier
        hidden_dim = config.get("models", {}).get(model_name, {}).get("hidden_dim", 32)
        clf = MLRegimeClassifier.from_artifact(base_dir, model_name, version, hidden_dim=hidden_dim)
        logger.info("Loaded ML regime classifier v%s", version)
        return clf
    except Exception as exc:
        logger.warning("Failed to load ML regime classifier: %s -- using rule-based fallback", exc)
        return RuleBasedRegimeClassifier()
```

- [ ] **Step 4: Update prediction_service.py**

Replace `get_engine` body to use model_loader:

```python
def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        from app.config import load_config
        from app.model_loader import build_engine
        config = load_config()
        _engine = build_engine(config)
    return _engine
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_model_loader.py -v`
Expected: PASS

- [ ] **Step 6: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add app/model_loader.py app/prediction_service.py tests/unit/test_model_loader.py
git commit -m "feat: implement model loader with graceful fallback to rule-based"
```

---

## Task 12: Wire CLI train command

**Files:**
- Modify: `cleareye/__main__.py`

- [ ] **Step 1: Update CLI**

Add `--source` and `--data` args to train subparser, wire to real pipeline:

```python
train_parser.add_argument("--source", choices=["csv", "synthetic"], default="synthetic")
train_parser.add_argument("--data", type=str, default=None, help="Path to CSV data file")
```

Replace the train handler:

```python
elif args.command == "train":
    from pathlib import Path
    from training.train import train
    train(config_path=Path(args.config), data_source=args.source, data_path=args.data)
```

- [ ] **Step 2: Smoke test CLI**

Run: `python3 -m cleareye train --source synthetic`
Expected: prints metrics for regime_classifier, exits cleanly

- [ ] **Step 3: Commit**

```bash
git add cleareye/__main__.py
git commit -m "feat: wire CLI train command to real pipeline"
```

---

## Task 13: Final integration test and cleanup

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 2: Run CLI verify**

Run: `python3 -m cleareye verify`
Expected: passes

- [ ] **Step 3: Run CLI train with synthetic data**

Run: `python3 -m cleareye train --source synthetic`
Expected: trains regime classifier, prints metrics, saves artifacts to `models/`

- [ ] **Step 4: Run CLI verify again (now with trained model)**

Run: `python3 -m cleareye verify`
Expected: uses ML regime classifier (loaded from artifacts)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: complete training and inference pipeline implementation"
```

---

## Task 14: OpenTelemetry instrumentation

**Files:**
- Create: `app/telemetry.py`
- Modify: `app/inference.py` (add spans)
- Modify: `app/main.py` (init telemetry, add FastAPI instrumentation)
- Modify: `training/pipeline/orchestrator.py` (add spans)
- Modify: `requirements.txt` (add otel deps)
- Test: `tests/unit/test_telemetry.py`

- [ ] **Step 1: Add OpenTelemetry dependencies**

Add to `requirements.txt`:

```
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-instrumentation-fastapi>=0.41b0
opentelemetry-exporter-otlp>=1.20.0
```

Run: `pip install -r requirements.txt`

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_telemetry.py
"""Tests for OpenTelemetry setup."""

from app.telemetry import get_tracer, get_meter


class TestTelemetry:
    def test_tracer_available(self):
        tracer = get_tracer()
        assert tracer is not None

    def test_meter_available(self):
        meter = get_meter()
        assert meter is not None

    def test_tracer_creates_span(self):
        tracer = get_tracer()
        with tracer.start_as_current_span("test-span") as span:
            assert span is not None
            span.set_attribute("test.key", "value")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_telemetry.py -v`
Expected: FAIL

- [ ] **Step 4: Implement telemetry module**

```python
# app/telemetry.py
"""OpenTelemetry initialization -- tracer and meter singletons.

Configured via standard OTEL environment variables:
  OTEL_SERVICE_NAME=cleareye
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
  OTEL_TRACES_EXPORTER=otlp|console|none
"""

from __future__ import annotations

import os

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "cleareye")
_resource = Resource.create({"service.name": _SERVICE_NAME})
_initialized = False


def _init() -> None:
    global _initialized
    if _initialized:
        return

    tracer_provider = TracerProvider(resource=_resource)
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(resource=_resource)
    metrics.set_meter_provider(meter_provider)

    exporter = os.environ.get("OTEL_TRACES_EXPORTER", "none")
    if exporter == "console":
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    _initialized = True


def get_tracer(name: str = "cleareye") -> trace.Tracer:
    _init()
    return trace.get_tracer(name)


def get_meter(name: str = "cleareye") -> metrics.Meter:
    _init()
    return metrics.get_meter(name)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_telemetry.py -v`
Expected: PASS

- [ ] **Step 6: Add inference spans**

In `app/inference.py`, update `InferenceEngine.predict`:

```python
from app.telemetry import get_tracer

_tracer = get_tracer()

def predict(self, reading: Reading) -> CalibratedReading:
    with _tracer.start_as_current_span("inference.predict") as root_span:
        root_span.set_attribute("rig_id", reading.rig_id)

        with _tracer.start_as_current_span("regime_classify"):
            regime_result = self.regime_classifier.classify(...)
            # set attributes: regime, confidence

        with _tracer.start_as_current_span("calibrate"):
            # existing calibration code
            # set attributes: voltage, ntu, method

        with _tracer.start_as_current_span("biofouling_assess"):
            # existing biofouling code
            # set attributes: correction_factor, cleaning_alert

        return CalibratedReading(...)
```

- [ ] **Step 7: Add training spans**

In `training/pipeline/orchestrator.py`, wrap each stage call with a span:

```python
from app.telemetry import get_tracer

_tracer = get_tracer()

# In PipelineOrchestrator.run():
with _tracer.start_as_current_span("training.pipeline"):
    with _tracer.start_as_current_span("training.ingest"):
        ctx.raw_df = source.load()
    with _tracer.start_as_current_span("training.clean"):
        ctx = clean(ctx)
    # ... etc for each stage
```

- [ ] **Step 8: Add FastAPI auto-instrumentation**

In `app/main.py`, after creating the app:

```python
from app.telemetry import get_tracer
get_tracer()  # ensure initialized

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)
```

- [ ] **Step 9: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 10: Commit**

```bash
git add app/telemetry.py app/inference.py app/main.py training/pipeline/orchestrator.py requirements.txt tests/unit/test_telemetry.py
git commit -m "feat: add OpenTelemetry instrumentation for inference and training"
```
