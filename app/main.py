"""FastAPI application — ClearEye water quality prediction service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

import app.builtin_profiles  # noqa: F401 — registers sensor profiles
from app.exceptions import ClearEyeError
from app.inference import CalibratedReading, Reading
from app.prediction_service import predict, predict_batch
from app.profiles import sensor_registry, standards_registry

app = FastAPI(
    title="ClearEye",
    description="ML platform for water quality prediction — turbidity measurement for open water bodies",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=CalibratedReading)
def predict_endpoint(reading: Reading) -> CalibratedReading:
    try:
        return predict(reading)
    except ClearEyeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/batch", response_model=list[CalibratedReading])
def predict_batch_endpoint(readings: list[Reading]) -> list[CalibratedReading]:
    if not readings:
        raise HTTPException(status_code=400, detail="Empty readings list")
    try:
        return predict_batch(readings)
    except ClearEyeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/sensors")
def list_sensors() -> list[dict[str, str]]:
    return [
        {"name": p.name, "quantity": p.quantity}
        for p in sensor_registry.all()
    ]


@app.get("/standards")
def list_standards() -> list[dict[str, str]]:
    return [
        {"name": s.name, "description": s.description, "unit": s.unit}
        for s in standards_registry.all()
    ]


@app.get("/standards/{name}/classify")
def classify_quality(name: str, ntu: float) -> dict:
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
