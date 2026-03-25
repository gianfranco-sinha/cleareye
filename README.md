# ClearEye

ML platform for water quality prediction — turbidity measurement science for open water bodies.

Built by [Smarts Sensors,Instrumentation and Analytics Ltd](https://www.water-sensors.co.uk).

## What it does

ClearEye provides regime-aware turbidity calibration for low-cost sensors deployed in UK rivers and waterways. It converts raw ADC readings from optical turbidity sensors into calibrated NTU values using a three-stage ML pipeline:

1. **Regime Classification** — identifies whether the water sample is solution, colloidal, or suspension-dominated
2. **Calibration** — applies regime-specific transfer functions (datasheet baseline + ML residual correction)
3. **Biofouling Monitor** — detects sensor drift from biofilm accumulation, applies correction factors, alerts when cleaning is needed

## Simulator

ClearEye includes a 1D advection-diffusion simulator for modelling turbidity transport in pipe geometries. It serves as the physics engine behind synthetic training data and feature extraction.

- **Forward solver** — Crank-Nicolson semi-implicit finite differences with upwind advection, supporting bidirectional flow (signed velocity)
- **Taylor dispersion** — computes effective diffusion coefficient D_eff = D_mol + R²v²/(48·D_mol) with Stokes-Einstein temperature correction
- **Inverse solver** — recovers physical parameters (D_molecular, velocity, concentration) from observed dual-sensor signals via L-BFGS-B optimisation
- **Synthetic data generation** — samples from physical parameter ranges, runs forward simulations, and extracts transport features for ML training
- **Boundary conditions** — Pulse, Step, Ramp, and Arbitrary inlet profiles

```bash
# Generate synthetic training data
from simulator import generate_synthetic_dataset
df = generate_synthetic_dataset(n_samples=1000, seed=42)
```

## Sensors

- DFRobot SEN0189 (turbidity, analog)
- TDS/conductivity meter
- DS18B20 (water temperature)
- Future: pH, dissolved oxygen, depth, flow rate

## Quick start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run dev server
uvicorn app.main:app --reload --port 8000

# CLI
python -m cleareye --help
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
