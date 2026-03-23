# ClearEye — ML Platform for Water Quality Prediction

**Date:** 2026-03-23
**Status:** Approved
**Author:** John Sinha + Claude

## Overview

ClearEye is an ML platform for water quality prediction, focused on turbidity measurement science for open water bodies (UK rivers). It provides sensor characterisation, regime-aware calibration, and biofouling compensation for low-cost turbidity sensors deployed in the field.

Built by Smarts Sensors, Analytics and Instrumentation Ltd (water-sensors.co.uk). Target users: landowners, farmers, environmental researchers, conservation groups, local authorities, citizen scientists.

Architecture is templated from the iaq4j project (pipeline FSM, profile/standard registries, quantity registry, inference engine) but with its own domain language, standards, and sensor physics. No shared code between projects.

## Domain Model

### Core Concepts

- **WaterBody** — a monitored location (river, lake, stream). GPS coordinates, name, flow characteristics.
- **SensorRig** — a deployed hardware unit with a sensor inventory, deployment location, and deployment date.
- **SensorProfile** — ABC defining a probe's raw features, valid ranges, and transfer function. Built-in: `SEN0189TurbidityProfile`, `TDSMeterProfile`, `DS18B20TemperatureProfile`. Future: pH, DO, depth, flow.
- **TurbidityRegime** — enum: `SOLUTION`, `COLLOID`, `SUSPENSION`. Determines which calibration model applies.
- **CalibrationStandard** — ABC defining the NTU reference scale and quality categories. Built-in: `ISO7027Standard`, `EnvironmentAgencyStandard`.
- **Reading** — a timestamped set of raw sensor values from a rig.
- **CalibratedReading** — a Reading after the three-stage pipeline: regime-classified, calibrated to NTU, biofouling-assessed.

### Quantity Registry

Physical quantities defined in `quantities.yaml`:

| Quantity | Canonical Unit | Valid Range |
|----------|---------------|-------------|
| turbidity | NTU | 0–4000 |
| conductivity | µS/cm | 0–5000 |
| tds | ppm | 0–1000 |
| water_temperature | °C | 0–40 |
| depth | m | 0–20 |
| flow_rate | m/s | 0–10 |

## Sensor Physics

### Turbidity as a Transport Problem

Turbidity is a proxy for particle concentration. Inside the sensor pipe, two transport mechanisms govern how particles reach and pass the optical sensors:

- **Advection** — bulk flow carrying particles downstream
- **Diffusion** — random Brownian spreading, dominant for small colloids

The particle concentration field is governed by the **advection-diffusion equation**:

```
∂C/∂t + v·∂C/∂x = D·∂²C/∂x²
```

Where C = particle concentration (turbidity proxy), v = flow velocity in pipe, D = diffusion coefficient.

### Regime Discrimination via Transport Physics

The key insight: **different particle sizes produce different transport signatures**, and these are measurable with dual sensors.

**Large particles (suspension regime — e.g., soil, intact algae cells):**
- Low diffusion coefficient (D small)
- Transport dominated by advection
- Signal travels as a "plug" between sensors
- Sensor A spike → clear delayed spike at Sensor B
- Strong cross-correlation with defined time lag

**Colloids (colloid regime — e.g., EPS, degraded algae, fine clay):**
- High diffusion coefficient
- Transport = advection + strong spreading
- Sensor A spike → smeared/damped signal at Sensor B
- Lower peak, broader curve
- Weaker time correlation

**Dissolved substances (solution regime):**
- Molecular diffusion, no optical scattering
- Both sensors see uniform background
- No cross-correlation signal

### Péclet Number — The Physics-Based Regime Boundary

The **Péclet number** quantifies the ratio of advective to diffusive transport:

```
Pe = v·L / D
```

Where L = sensor spacing (200 mm), v = flow velocity, D = diffusion coefficient.

- **High Pe** → advection dominates → particles (suspension regime)
- **Low Pe** → diffusion dominates → colloids
- **Pe ≈ 1** → transition zone, mixed transport

This provides a physics-based regime boundary rather than empirical ADC thresholds.

### Dual-Sensor Pipe Geometry

**Hardware design:**
- Pipe length: 300 mm
- Sensor spacing: 200 mm (two SEN0189 sensors)
- Mesh-filtered ends (copper mesh)
- Slow-moving water (low Reynolds number, laminar flow)

**Three discriminating measurements from dual sensors:**

1. **Cross-correlation** — correlation between sensor A and sensor B time series:
   - Strong peak with clear time lag → particles
   - Weak/broad correlation → colloids

2. **Peak attenuation** — ratio of peak amplitudes:
   - `attenuation = peak_B / peak_A`
   - ≈ 1 → particles (minimal diffusion loss)
   - << 1 → colloids (significant diffusion spreading)

3. **Signal broadening** — comparison of peak widths:
   - Narrow peaks → particles
   - Wide/smeared peaks → colloids (diffusion broadening)

### Design Considerations

**Increasing sensitivity to colloids:**
- Reduce flow velocity inside pipe
- Increase sensor spacing (250–300 mm optimal)

**Increasing sensitivity to particles:**
- Add slight flow acceleration (funnel ends)
- Maintain laminar flow (avoid turbulence noise)

### Copper Mesh Behaviour

The mesh filters at pipe ends:
- **Block** large particles and debris
- **Do not stop** colloids — they pass through or diffuse through mesh openings
- **Allow** diffusion-driven ingress even with zero flow

Over long deployment periods, baseline drift = colloidal accumulation signal. This is useful for biofouling detection if modelled correctly.

### Combined Feature Set for Regime Classification

When dual-sensor data is available, the regime classifier gains physics-based features:

| Feature | Source | Physics |
|---------|--------|---------|
| `cross_correlation_peak` | Dual sensor | Transport mode indicator |
| `cross_correlation_lag` | Dual sensor | Flow velocity estimate |
| `peak_attenuation` | Dual sensor | Diffusion loss (particle size proxy) |
| `signal_broadening` | Dual sensor | Diffusion coefficient estimate |
| `peclet_number` | Derived | Pe = v·L/D, regime boundary |
| `effective_diffusion` | Derived | From attenuation + temperature + flow |

Combined with temperature (affects viscosity → diffusion coefficient) and a rough flow estimate, the system can estimate the effective diffusion coefficient D — making it a **low-cost particle size / transport classifier**, not just a turbidity meter.

### Single-Sensor vs Dual-Sensor Paths

- **Single-sensor (current, 50K dataset):** ADC/TDS thresholds for regime classification (rule-based or ML). No transport physics features available.
- **Dual-sensor (future rig):** Physics-based discrimination using cross-correlation, peak attenuation, signal broadening, and Péclet number. Falls back to single-sensor features if second sensor data is missing.

## Three-Stage ML Pipeline

### Architecture

```
Reading → [Stage 1: Regime Classifier] → regime label
       → [Stage 2: Calibration(regime)] → calibrated NTU
       → [Stage 3: Biofouling Monitor] → correction + alert
       → CalibratedReading
```

### Stage Composition (InferenceEngine.predict)

```python
def predict(self, reading: Reading) -> CalibratedReading:
    # Stage 1: classify regime from raw features
    regime = self.regime_classifier.classify(
        turbidity_adc=reading.turbidity_adc,
        tds=reading.tds,
        temperature=reading.water_temperature,
    )

    # Stage 2: dispatch to regime-specific calibration
    # Regime label selects which calibration model to use (not an input feature)
    voltage = self.transfer_fn.adc_to_voltage(reading.turbidity_adc)
    ntu = self.calibrators[regime].calibrate(
        voltage=voltage,
        temperature=reading.water_temperature,
        tds=reading.tds,
    )

    # Stage 3: biofouling correction (queries historical readings for this rig)
    fouling = self.biofouling_monitor.assess(
        rig_id=reading.rig_id,
        calibrated_ntu=ntu,
    )
    corrected_ntu = ntu * fouling.correction_factor

    return CalibratedReading(
        regime=regime,
        turbidity_ntu=corrected_ntu,
        turbidity_voltage=voltage,
        biofouling_factor=fouling.correction_factor,
        cleaning_alert=fouling.cleaning_alert,
        confidence=min(regime.confidence, fouling.reliability),
    )
```

### Stage 1 — Regime Classification

Input: raw ADC turbidity, TDS, temperature, (future: depth, flow rate).

Three regimes with distinct optical signatures:
- **Solution** — dissolved substances, low scattering, high ADC values (clear water), turbidity dominated by colour/absorption.
- **Colloid** — fine suspended particles (1nm–1µm), Rayleigh scattering, moderate ADC, TDS and turbidity partially correlated.
- **Suspension** — coarse particles (>1µm), Mie scattering, low ADC values (turbid water), TDS and turbidity decouple after rain events.

Initial implementation: rule-based thresholds from sensor characteristics (ADC ranges, TDS-turbidity correlation). Later replaced/augmented by a trained classifier when labelled regime data is available.

### Stage 2 — Calibration

Per-regime transfer function: ADC → voltage → NTU.

- **Baseline**: SEN0189 datasheet piecewise curve (voltage–NTU lookup). The SEN0189 output voltage decreases as turbidity increases. ADC reference voltage is configurable per rig (default 5.0V for Arduino, 3.3V for ESP32): `voltage = ADC * (V_ref / 1024.0)`.
- **ML residual** (milestone 2+): learns correction from reference NTU measurements, accounting for temperature cross-sensitivity, TDS interference, and regime-specific nonlinearities.
- Three separate models, one per regime — keeps each model simple and interpretable.

#### Feature Engineering

Input features for the ML residual models (milestone 2+):

| Feature | Source | Notes |
|---------|--------|-------|
| turbidity_adc | Raw sensor | 10-bit ADC |
| voltage | Derived | `ADC * (V_ref / 1024.0)` |
| water_temperature | DS18B20 | Temperature cross-sensitivity |
| tds | TDS sensor | Dissolved solids interference |
| d_adc_dt | Derived | Rate of change — event detection |
| hour_sin, hour_cos | Timestamp | Diurnal sediment patterns |
| depth | Future sensor | Pressure-turbidity relationship |
| flow_rate | Future sensor | Sediment resuspension correlation |

Regime label is used to **select** the model, not as an input feature.

Datasheet piecewise regions:
- V > 4.0V: ~0 NTU (clear)
- V 3.0–4.0V: ~0–500 NTU (low turbidity, roughly linear)
- V 2.5–3.0V: ~500–2000 NTU (medium, steeper)
- V < 2.5V: ~2000–4000 NTU (high, exponential region)

### Stage 3 — Biofouling Monitor

Runs in parallel, not inline. Detects sensor drift by monitoring:
- Baseline shift in clear-water readings over time (if the sensor's "clean water" ADC value drops, the optical window is fouling).
- Rate of drift (fast = algal bloom on sensor, slow = mineral deposit).
- Outputs: correction factor (applied to calibrated NTU), confidence score, cleaning alert threshold.

Requires historical deployment data — days/weeks of readings where drift accumulates between cleaning events. Later milestone.

**Data access pattern:** The biofouling monitor queries the last N days of calibrated readings from InfluxDB per rig. It runs on a scheduled interval (e.g., hourly), not per-reading. This avoids the cost of long-range queries on every inference call.

## Data Model

### Sensor Reading (input)

```json
{
    "timestamp": "2023-10-07T19:12:05",
    "rig_id": "rig-001",
    "turbidity_adc": 371,
    "tds": 554,
    "water_temperature": 18.25,
    "depth": null,
    "flow_rate": null
}
```

### Calibrated Reading (output)

```json
{
    "timestamp": "2023-10-07T19:12:05",
    "rig_id": "rig-001",
    "regime": "colloid",
    "turbidity_ntu": 142.3,
    "turbidity_voltage": 3.62,
    "calibration_method": "datasheet",
    "biofouling_factor": 1.0,
    "cleaning_alert": false,
    "confidence": 0.92           // min(regime_classifier_confidence, biofouling_reliability), range 0.0-1.0
}
```

### Calibration Standards (YAML-driven)

```yaml
# standards/environment_agency.yaml
name: environment_agency
description: UK Environment Agency river classification
scale: [0, 4000]
unit: NTU
categories:
  - name: High
    range: [0, 10]
    description: Near-natural conditions
  - name: Good
    range: [10, 25]
  - name: Moderate
    range: [25, 100]
  - name: Poor
    range: [100, 300]
  - name: Bad
    range: [300, 4000]
```

### Existing Field Data

50,383 readings from a DFRobot SEN0189 + TDS + DS18B20 rig deployed Oct 7–28, 2023:

| Field | Range | Notes |
|-------|-------|-------|
| timestamp | Oct 7–28, 2023 | ~6s sampling interval |
| tds | 0–956 | ppm |
| turbidity_adc | 0–1023 | 10-bit ADC raw |
| water_temperature | 10.8–27.6°C | |

4 contiguous segments (3 gaps, largest 14 days). Data has repeated header rows that need filtering on ingest.

## Project Structure

```
cleareye/
├── app/
│   ├── main.py                  # FastAPI service
│   ├── config.py                # Settings from model_config.yaml
│   ├── sensor_physics.py        # Probe hardware constants, SEN0189 transfer curve
│   ├── profiles.py              # SensorProfile ABC, CalibrationStandard ABC, registries
│   ├── builtin_profiles.py      # SEN0189TurbidityProfile, TDSMeterProfile, DS18B20Profile
│   ├── quantities.py            # Quantity registry loader
│   ├── regime.py                # TurbidityRegime enum, regime classifier
│   ├── calibration.py           # Datasheet transfer functions, ML residual correction
│   ├── biofouling.py            # Drift detection, correction factor, cleaning alerts
│   ├── inference.py             # InferenceEngine (three-stage pipeline)
│   ├── models.py                # PyTorch model architectures
│   ├── exceptions.py            # Domain exception hierarchy
│   └── prediction_service.py    # Service layer for routes
├── training/
│   ├── pipeline/                # FSM pipeline
│   │   └── orchestrator.py
│   ├── data_sources.py          # DataSource ABC: CSV, InfluxDB, Synthetic
│   ├── train.py                 # Training entry point
│   └── utils.py                 # Training loop, device detection, checkpointing
├── cleareye/
│   └── __main__.py              # CLI: train, version, verify
├── quantities.yaml              # Physical quantity definitions
├── model_config.yaml            # Model architecture + training hyperparams
├── standards/                   # YAML-driven calibration standards
│   ├── iso7027.yaml
│   └── environment_agency.yaml
├── tests/
│   ├── unit/
│   └── integration/
├── CLAUDE.md
└── requirements.txt
```

**Tech stack:** Python 3.12, FastAPI, PyTorch, pandas, numpy, InfluxDB, MQTT.

**Key difference from iaq4j:** The inference pipeline is three stages (regime classification → calibration → biofouling correction) rather than one (feature engineering → model forward pass → clamp/categorize).

## Milestones

### Milestone 1 — Sensor Characterisation (MVP)

- Datasheet-based ADC → voltage → NTU transfer function for SEN0189
- Temperature compensation (DS18B20 reading adjusts calibration)
- Rule-based regime classification from ADC/TDS thresholds
- CSV data ingestion, basic FastAPI endpoints
- CSV ingest with header-row deduplication and type validation (reject rows where turbidity_adc is non-numeric)
- Ingest the existing 50k-reading dataset, produce calibrated NTU output
- Tests for transfer function accuracy against datasheet reference points

### Milestone 2 — Training Pipeline

- FSM pipeline (templated from iaq4j): ingest → clean → feature engineer → window → split → scale → train → evaluate → save
- Synthetic data source (generates plausible water quality readings across all three regimes)
- ML regime classifier (replaces rule-based)
- ML residual correction per regime (requires reference NTU data — see below)
- Artifact semver, Merkle provenance

**Reference NTU data collection strategy:** To train the ML residual models, calibrated NTU ground truth is needed alongside continuous sensor readings. Planned approach: periodic grab samples analysed with a benchtop nephelometer (e.g., Hach 2100Q), timestamped and joined to the nearest sensor reading. Minimum target: 50+ reference points per regime, spanning the full NTU range. Formazin calibration standards in the lab provide additional controlled-condition reference points for the low-turbidity (solution) regime.

### Milestone 3 — Biofouling Detection

- Drift model trained on multi-deployment data (clean→fouled sensor trajectories)
- Correction factor output
- Cleaning alert thresholds
- Historical baseline tracking per deployed rig

### Milestone 4 — Field Deployment

- MQTT ingestion from sensor rigs via Node-RED
- InfluxDB persistence
- Real-time inference endpoint
- Rig registration and management API
- Dashboard/alerting integration

**MQTT topic structure:**
- `cleareye/{rig_id}/readings` — raw sensor readings from rigs (QoS 1)
- `cleareye/{rig_id}/calibrated` — calibrated output published back (QoS 0)
- `cleareye/{rig_id}/alerts` — cleaning alerts, anomalies (QoS 1)

Payload format matches the Reading JSON schema. Node-RED on the Pi bridges MQTT to InfluxDB (same pattern as iaq4j).

## License

Apache License 2.0 — open source, permissive, patent grant included. Compatible with citizen science mission and commercial use.

## Decisions

- **Separate project from iaq4j** — different domains, different industries, no value in shared abstractions
- **Regime-aware hybrid approach** — physics-first with ML residual correction, not end-to-end ML
- **Three separate calibration models** — one per regime, keeps each simple and interpretable
- **Biofouling as parallel monitor** — not inline, produces correction factor and cleaning alerts
- **Same tech stack as iaq4j** — reduces learning curve, proven patterns
- **DFRobot SEN0189** as the initial turbidity sensor — well-documented, datasheet transfer function available
