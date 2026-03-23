"""Service layer — bridges FastAPI routes to the inference engine."""

from __future__ import annotations

from app.inference import CalibratedReading, InferenceEngine, Reading

# Module-level engine instance (initialised on first import of builtin_profiles)
_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def predict(reading: Reading) -> CalibratedReading:
    return get_engine().predict(reading)


def predict_batch(readings: list[Reading]) -> list[CalibratedReading]:
    engine = get_engine()
    return [engine.predict(r) for r in readings]
