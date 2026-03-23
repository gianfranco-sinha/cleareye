# CLAUDE.md

## Project

ClearEye — ML platform for water quality prediction. Turbidity measurement science for open water bodies (UK rivers). Regime-aware calibration, biofouling compensation. DFRobot SEN0189 default sensor. Python 3.12, FastAPI, PyTorch.

## Architecture

Three-stage ML pipeline:
1. **Regime Classifier** — solution / colloid / suspension (rule-based initially, ML later)
2. **Calibration** — datasheet transfer function baseline + ML residual correction per regime
3. **Biofouling Monitor** — drift detection, correction factor, cleaning alerts (parallel, not inline)

Key abstractions:
- `SensorProfile` ABC — raw features, valid ranges, transfer function per probe
- `CalibrationStandard` ABC — NTU scale and quality categories
- `TurbidityRegime` enum — SOLUTION, COLLOID, SUSPENSION
- `quantities.yaml` — physical quantity registry with units and valid ranges

## Commands

```bash
# Dev server
uvicorn app.main:app --reload --port 8000

# CLI
python -m cleareye --help

# Tests
python -m pytest
python -m pytest tests/unit/ -v
```

## Conventions

- Import order: stdlib, third-party, local
- Absolute imports: `from app.profiles import SensorProfile`
- Python 3.12+ type hints (use `X | None`, `list[str]`, etc.)
- Config source of truth: `model_config.yaml`
- Sensor profiles registered via `import app.builtin_profiles` side effect
