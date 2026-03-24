"""Synthetic dataset generation for ML training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.regime import TurbidityRegime
from app.transport_physics import dual_sensor_features
from simulator.conditions import Pulse
from simulator.geometry import PipeGeometry, load_geometry
from simulator.solver import AdvectionDiffusionSolver, SimulationParams


def generate_synthetic_dataset(
    n_samples: int = 1000,
    seed: int | None = None,
    geometry: PipeGeometry | None = None,
    nx: int = 60,
    duration: float = 30.0,
) -> pd.DataFrame:
    """Generate a synthetic dataset of dual-sensor signals with ground truth.

    Samples D_molecular, velocity, C0, and temperature from physical ranges,
    runs the forward solver for each, and extracts features.

    Args:
        n_samples: Number of synthetic samples to generate.
        seed: Random seed for reproducibility.
        geometry: Pipe geometry. Defaults to loading from YAML.
        nx: Spatial grid points for solver.
        duration: Simulation duration in seconds.

    Returns:
        DataFrame with columns: upstream_signal, downstream_signal,
        d_molecular, d_effective, velocity, c0, temperature,
        peclet_number_true, regime, and transport physics features.
    """
    rng = np.random.default_rng(seed)
    geo = geometry or load_geometry()
    solver = AdvectionDiffusionSolver(geo)

    records: list[dict] = []

    for _ in range(n_samples):
        # Sample parameters (log-uniform for D, v, c0; uniform for temp)
        d_mol = float(np.exp(rng.uniform(np.log(1e-12), np.log(1e-5))))
        velocity = float(np.exp(rng.uniform(np.log(0.001), np.log(0.5))))
        c0 = float(np.exp(rng.uniform(np.log(1.0), np.log(5000.0))))
        temperature = float(rng.uniform(1.0, 35.0))

        params = SimulationParams(
            d_molecular=d_mol, velocity=velocity, c0=c0,
            temperature=temperature, nx=nx, dt=None,
            duration=duration, boundary_type="pulse",
            geometry=geo,
        )
        bc = Pulse(c0=c0, t0=duration * 0.15, sigma=duration * 0.05)

        try:
            result = solver.solve(params, bc)
        except Exception:
            continue  # skip failed simulations

        if np.any(np.isnan(result.downstream)):
            continue

        # Ground-truth regime from Pe
        pe = result.peclet_number
        if pe > geo.suspension_pe:
            regime = TurbidityRegime.SUSPENSION
        elif pe < geo.solution_pe:
            regime = TurbidityRegime.SOLUTION
        else:
            regime = TurbidityRegime.COLLOID

        # Extract transport features using existing analytical functions
        features = dual_sensor_features(
            signal_a=result.upstream,
            signal_b=result.downstream,
            velocity=velocity,
            temperature=temperature,
            sensor_spacing=geo.sensor_spacing,
        )

        records.append({
            "upstream_signal": result.upstream,
            "downstream_signal": result.downstream,
            "d_molecular": d_mol,
            "d_effective": result.d_effective,
            "velocity": velocity,
            "c0": c0,
            "temperature": temperature,
            "peclet_number_true": pe,
            "regime": regime.value,
            "cross_correlation_peak": features["cross_correlation_peak"],
            "cross_correlation_lag": features["cross_correlation_lag"],
            "peak_attenuation": features["peak_attenuation"],
            "signal_broadening": features["signal_broadening"],
        })

    return pd.DataFrame(records)
