D# Training & Inference Pipeline Design

**Date:** 2026-03-23
**Status:** Approved
**Author:** John Sinha + Claude

## Overview

Combined training and inference pipeline for ClearEye. The training pipeline produces model artifacts that the inference engine loads at startup, with graceful fallback to datasheet/rule-based methods when models are unavailable.

**Immediate scope:** Regime classifier trained on rule-derived labels from 50K field readings (Oct 2023). Residual correction, autoencoder, and Label Studio integration are designed but disabled until data arrives.

## Training Pipeline

### Architecture

FSM stage pipeline — each stage is a standalone function with signature:

```python
def stage_name(ctx: PipelineContext) -> PipelineContext
```

Stages execute sequentially. Stateless — failures restart from scratch (pipeline completes in minutes at current data scale).

**Stages:**

```
ingest → clean → feature_engineer → split → scale → train → evaluate → save
```

Note: The design doc and existing `PipelineStage` enum include a `window` stage between `feature_engineer` and `split`. Windowing (e.g., rolling statistics, `d_adc_dt` computation) is folded into `feature_engineer` for simplicity — there is no separate windowing pass. The `PipelineStage` enum will be updated to match.

**Trigger:** CLI only — `python -m cleareye train --config model_config.yaml`.

### PipelineContext

Dataclass carrying state between stages:

| Field | Type | Set by stage |
|-------|------|-------------|
| `raw_df` | DataFrame | ingest |
| `clean_df` | DataFrame | clean |
| `features_df` | DataFrame | feature_engineer |
| `labels` | Series / None | feature_engineer |
| `train_df`, `val_df`, `test_df` | DataFrame | split |
| `scalers` | dict[str, StandardScaler] | scale |
| `models` | dict[str, nn.Module] | train |
| `metrics` | dict[str, dict] | evaluate |
| `config` | dict | ingest (loaded from YAML) |

### Ingest Stage

- Loads data from the configured `DataSource` (CSV, Synthetic, or Label Studio stub)
- `CSVDataSource`: drops repeated header rows, coerces types, parses timestamps
- `SyntheticDataSource`: generates plausible readings across all three regimes (for testing)
- `LabelStudioSource`: stub — raises `NotImplementedError`
- InfluxDB data source deferred to milestone 4

### Clean Stage

- Drop rows with NaN in required columns (`turbidity_adc`, `tds`, `water_temperature`)
- Validate ranges against `quantities.yaml` (calibrated-unit ranges)
- Validate raw values against `SensorProfile.valid_range` per sensor (e.g., DS18B20 -55°C to 125°C, SEN0189 ADC 0–1023). Readings outside the sensor's operating envelope are flagged as suspect and dropped — catches hardware faults (e.g., DS18B20 returning 85°C = wiring error)
- Sort by timestamp
- Deduplicate exact-timestamp rows
- Drop rows where `turbidity_adc` is non-numeric (header row contamination)
- Minimum data guard: abort with `InsufficientDataError` if fewer than 30 samples per regime after cleaning

**Staleness checks:**

- **Timestamp gap detection** — identify gaps between consecutive readings exceeding a configurable threshold (default: 5× expected sampling interval, e.g., >30s for a 6s interval). Gaps are logged with start/end timestamps and duration. The dataset is split into contiguous segments at gap boundaries; each segment is tagged so downstream stages (e.g., `d_adc_dt` computation) don't compute derivatives across gaps.
- **Sequence number gap detection** — if an optional `seq` field is present in the data, detect missing sequence numbers independently of timestamps. This catches dropped readings even when the clock is unreliable or absent. Missing sequence ranges are logged.
- **Data age warning** — if the most recent reading in the dataset is older than a configurable threshold (default: 90 days), log a warning that training data may not reflect current conditions (seasonal drift, sensor degradation, site changes). Training proceeds but the warning is recorded in `metadata.json`.

### Feature Engineering Stage

**Regime classifier features** (direct from raw readings):

| Feature | Source |
|---------|--------|
| `turbidity_adc` | Raw sensor |
| `tds` | Raw sensor |
| `water_temperature` | DS18B20 |

**Residual correction features** (future, when reference NTU data arrives):

| Feature | Source |
|---------|--------|
| `voltage` | Derived: `ADC * (V_ref / 1024)` |
| `water_temperature` | DS18B20 |
| `tds` | Raw sensor |
| `d_adc_dt` | Rate of change of ADC between consecutive readings |
| `hour_sin` | `sin(2π * hour / 24)` — cyclical encoding |
| `hour_cos` | `cos(2π * hour / 24)` — cyclical encoding |
| `ph` | Optional — pH sensor (future) |
| `dissolved_oxygen` | Optional — DO sensor (future) |

Optional fields (`ph`, `dissolved_oxygen`, `depth`, `flow_rate`) are excluded from feature vectors when `None`. Models trained with these features will only be usable on rigs that provide them. The `Reading` model carries all optional fields as nullable; feature engineering selects only those present in the configured feature list AND non-null in the data.

**Label generation** depends on `label_source` config:

| Source | Behaviour |
|--------|-----------|
| `rule_derived` | Run `RuleBasedRegimeClassifier` on each reading |
| `reference` | Join reference NTU by nearest timestamp |
| `label_studio` | Stub — not implemented |
| `self_supervised` | No labels; autoencoder uses reconstruction loss |

### Split Stage

- 70/15/15 train/val/test split
- Stratified by regime label for classification models
- Random split for regression and autoencoder

### Scale Stage

- `StandardScaler` fitted on training split only
- Applied to val and test splits
- Scaler object saved as part of model artifacts

### Train Stage

**Supported model types:**

| Type | Architecture | Loss | Output |
|------|-------------|------|--------|
| Classification | `RegimeClassifierNet` (MLP, softmax) | CrossEntropyLoss | 3-class regime label |
| Regression | `ResidualCorrectionNet` (MLP, linear) | MSELoss | Scalar NTU correction |
| Autoencoder | `AnomalyAutoencoder` (encoder-decoder MLP) | MSELoss (reconstruction) | Reconstruction error |

**Training loop (common):**
- Optimizer: Adam, lr from config (default 0.001)
- Early stopping: patience from config (default 10 epochs) on validation loss
- Device: auto-detected (CUDA > MPS > CPU)
- Batch size from config (default 64)

**Regime classifier specifics:**
- Trained on rule-derived labels from 50K readings
- Expected to achieve 95%+ agreement with rule-based classifier
- Value is generalisation to edge cases near decision boundaries

**Residual correction specifics** (future):
- One model per regime, trained independently
- Target: `reference_NTU - datasheet_NTU` (additive correction)
- Requires minimum 50 reference points per regime

**Autoencoder specifics** (future):
- Trained on "normal" readings (no fouling)
- High reconstruction error at inference = anomaly/drift signal
- `encoding_dim` controls bottleneck size

### Evaluate Stage

**Classification metrics:** accuracy, per-class precision/recall/F1, confusion matrix.

**Regression metrics:** MAE, RMSE, R² per regime.

**Autoencoder metrics:** mean/std reconstruction error on train vs test, anomaly threshold at 95th percentile.

All metrics saved to `metadata.json`.

### Save Stage

**Artifact layout:**

```
models/
├── regime_classifier/
│   └── v0.1.0/
│       ├── model.pt          # PyTorch state dict
│       ├── scaler.pkl         # Fitted StandardScaler
│       └── metadata.json      # Features, metrics, config, data hash, timestamp
├── calibration/
│   ├── solution/v0.1.0/...
│   ├── colloid/v0.1.0/...
│   └── suspension/v0.1.0/...
└── anomaly_detector/
    └── v0.1.0/...
```

**Versioning:** Semver, auto-increment patch on each training run. `metadata.json` includes:

```json
{
  "version": "0.1.0",
  "model_type": "classification",
  "features": ["turbidity_adc", "tds", "water_temperature"],
  "label_source": "rule_derived",
  "metrics": { "accuracy": 0.97, "f1_macro": 0.96 },
  "training_data_hash": "sha256:abc123...",
  "trained_at": "2026-03-23T14:30:00Z",
  "config": { "batch_size": 64, "lr": 0.001, "epochs": 100 }
}
```

## Inference Pipeline — ML Model Loading & Fallback

### Startup Behaviour

`InferenceEngine` reads `model_config.yaml` to determine which method to use per component. It attempts to load corresponding model artifacts.

**Fallback chain:**

```
Config says ml? → Model file exists? → Load ML model
                → Model file missing? → Log warning, use datasheet/rule-based
Config says rule_based/datasheet? → Use current implementations directly
```

### Model Loading Flow

1. Scan `models/{name}/` for highest semver directory
2. Load `model.pt` weights into the appropriate `nn.Module`
3. Load `scaler.pkl` — apply same scaling at inference time
4. Read `metadata.json` — validate feature list matches config

### New Components

**`MLRegimeClassifier`** — wraps `RegimeClassifierNet` + scaler. Implements the `RegimeClassifier` protocol (same `classify()` interface). Returns `RegimeResult` with softmax confidence. `InferenceEngine` accepts `RegimeClassifier` protocol, not the concrete `RuleBasedRegimeClassifier`.

**`MLResidualCalibrator`** — wraps `ResidualCorrectionNet` + scaler. Implements `Calibrator` ABC (including `method` property returning `"ml_residual"`). Applies datasheet transfer function first, then adds ML residual correction. `calibration_method` in `CalibratedReading` is set dynamically from the active calibrator's `method` property.

**`AnomalyDetectorEngine`** — wraps autoencoder. Computes reconstruction error, flags anomalies above threshold. Integrates with `BiofoulingMonitor`.

### No External API Changes

The `/predict` endpoint, `Reading`/`CalibratedReading` data models, and all FastAPI routes remain unchanged. Only the internal engine components swap out.

### Implementation Notes

- The existing `PipelineOrchestrator` stub (with `self.context: dict`) will be replaced by the `PipelineContext` dataclass and rewritten stage functions — not extended.
- `model_config.yaml` is restructured: the current flat `regime_classifier`/`calibration`/`biofouling` keys are replaced with a `models:` block containing per-model entries. `app/config.py` `Settings` must be updated to parse the new structure.
- `SyntheticDataSource` generates data with boundaries matching rule-based thresholds — it is for pipeline testing only, not meaningful ML evaluation. Real training uses the 50K field readings.
- Scaler artifacts are saved as JSON (mean/std arrays) rather than pickle, for safety and reproducibility.

## Configuration

### `model_config.yaml` Structure

```yaml
server:
  host: "0.0.0.0"
  port: 8000

sensor:
  v_ref: 5.0
  adc_resolution: 1024

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  device: auto

models:
  regime_classifier:
    type: classification
    architecture: mlp
    label_source: rule_derived
    features: [turbidity_adc, tds, water_temperature]
    hidden_dim: 32

  residual_solution:
    type: regression
    architecture: mlp
    label_source: reference
    features: [voltage, water_temperature, tds, d_adc_dt, hour_sin, hour_cos]
    hidden_dim: 32
    enabled: false

  residual_colloid:
    type: regression
    architecture: mlp
    label_source: reference
    features: [voltage, water_temperature, tds, d_adc_dt, hour_sin, hour_cos]
    hidden_dim: 32
    enabled: false

  residual_suspension:
    type: regression
    architecture: mlp
    label_source: reference
    features: [voltage, water_temperature, tds, d_adc_dt, hour_sin, hour_cos]
    hidden_dim: 32
    enabled: false

  anomaly_detector:
    type: autoencoder
    architecture: autoencoder
    label_source: self_supervised
    features: [turbidity_adc, tds, water_temperature]
    hidden_dim: 16
    encoding_dim: 4
    enabled: false

artifacts:
  base_dir: models
  version_strategy: semver

label_studio:
  enabled: false
  # url: http://localhost:8080
  # api_key: null
  # project_id: null
```

## Decisions

- **FSM stage pipeline** over monolithic script or DAG — right balance of structure and simplicity
- **Restart from scratch on failure** — pipeline is fast, no need for stage checkpointing
- **CLI-only training trigger** — training is deliberate, not automated at this stage
- **Graceful fallback** — inference engine always starts, logs warnings for missing models
- **Simple filesystem artifacts** — no MLflow overhead, metadata JSON captures what matters
- **Label Studio as stub** — interface defined, implementation deferred
- **Autoencoder for anomaly detection** — fits biofouling/drift use case, unsupervised
- **Rule-derived labels for initial training** — bootstrap regime classifier from rule-based labels
