"""FastAPI application — ClearEye water quality prediction service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

import app.builtin_profiles  # noqa: F401 — registers sensor profiles
from app.exceptions import ClearEyeError
from app.inference import CalibratedReading, Reading
from app.prediction_service import predict, predict_batch
from app.profiles import sensor_registry, standards_registry

_DESCRIPTION = """\
ClearEye is an ML platform for **water quality prediction** built for open
water bodies (UK rivers and waterways).

It converts raw ADC readings from optical turbidity sensors into calibrated
**NTU** (Nephelometric Turbidity Units) values via a three-stage pipeline:

1. **Regime Classification** — solution / colloid / suspension
2. **Calibration** — datasheet transfer function (+ ML residual correction)
3. **Biofouling Correction** — drift detection and cleaning alerts

Default sensor: DFRobot SEN0189 (10-bit ADC, 0–1023).
"""

_TAGS_METADATA = [
    {
        "name": "Prediction",
        "description": "Run the three-stage inference pipeline on sensor readings.",
    },
    {
        "name": "Registry",
        "description": "Look up registered sensor profiles and calibration standards.",
    },
    {
        "name": "System",
        "description": "Health checks and operational status.",
    },
]

app = FastAPI(
    title="ClearEye",
    description=_DESCRIPTION,
    version="0.1.0",
    openapi_tags=_TAGS_METADATA,
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
)

# OpenTelemetry — auto-instruments all routes; no-op if no exporter is configured
from app.telemetry import setup_telemetry
setup_telemetry(app)


@app.get("/health", tags=["System"])
def health() -> dict:
    """Return service health status including InfluxDB connectivity."""
    from app.database import influx_manager

    return {
        "status": "ok",
        "influxdb": influx_manager.health_check(),
    }


@app.post(
    "/predict",
    response_model=CalibratedReading,
    tags=["Prediction"],
    summary="Predict turbidity for a single reading",
)
def predict_endpoint(reading: Reading) -> CalibratedReading:
    """Run the three-stage inference pipeline on a single sensor reading.

    **Stages:**
    1. Classify the turbidity regime (solution / colloid / suspension)
    2. Calibrate the raw ADC value to NTU via the regime-specific calibrator
    3. Apply biofouling correction factor

    Returns a `CalibratedReading` with the calibrated NTU, detected regime,
    and confidence score.
    """
    try:
        return predict(reading)
    except ClearEyeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post(
    "/predict/batch",
    response_model=list[CalibratedReading],
    tags=["Prediction"],
    summary="Predict turbidity for multiple readings",
)
def predict_batch_endpoint(readings: list[Reading]) -> list[CalibratedReading]:
    """Run the inference pipeline on a batch of sensor readings.

    Each reading is processed independently through all three stages.
    Useful for bulk-uploading historical data or processing multiple rigs
    in a single request.
    """
    if not readings:
        raise HTTPException(status_code=400, detail="Empty readings list")
    try:
        return predict_batch(readings)
    except ClearEyeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get(
    "/sensors",
    tags=["Registry"],
    summary="List registered sensor profiles",
)
def list_sensors() -> list[dict[str, str]]:
    """Return all registered sensor profiles.

    Each profile defines a sensor's raw features, valid ranges, and
    transfer function.  The default profile is the DFRobot SEN0189
    turbidity sensor.
    """
    return [
        {"name": p.name, "quantity": p.quantity}
        for p in sensor_registry.all()
    ]


@app.get(
    "/standards",
    tags=["Registry"],
    summary="List calibration standards",
)
def list_standards() -> list[dict[str, str]]:
    """Return all loaded calibration standards.

    Standards are YAML-driven and define NTU scales with quality
    categories (e.g. ISO 7027, UK Environment Agency).
    """
    return [
        {"name": s.name, "description": s.description, "unit": s.unit}
        for s in standards_registry.all()
    ]


@app.get(
    "/standards/{name}/classify",
    tags=["Registry"],
    summary="Classify an NTU value against a standard",
)
def classify_quality(name: str, ntu: float) -> dict:
    """Look up which quality category a given NTU value falls into
    for the specified calibration standard.

    For example, querying the Environment Agency standard with
    `ntu=5.0` might return category *"Good"*.
    """
    try:
        standard = standards_registry.get(name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Standard {name!r} not found")
    category = standard.classify(ntu)
    if category is None:
        return {"standard": name, "ntu": ntu, "category": None}
    return {
        "standard": name,
        "ntu": ntu,
        "category": category.name,
        "description": category.description,
    }
