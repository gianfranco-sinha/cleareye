"""Inverse solver -- fit advection-diffusion parameters to observed data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from app.regime import TurbidityRegime
from simulator.conditions import Pulse
from simulator.geometry import PipeGeometry
from simulator.solver import AdvectionDiffusionSolver, SimulationParams, SimulationResult


@dataclass
class FitResult:
    """Result of parameter fitting."""

    d_molecular: float
    d_effective: float
    velocity: float
    c0: float
    peclet_number: float
    regime: TurbidityRegime
    residual: float
    converged: bool
    at_bound: bool
    simulation: SimulationResult


# Parameter bounds (physical ranges)
_D_BOUNDS = (1e-12, 1e-5)
_V_BOUNDS = (0.001, 0.5)
_C0_BOUNDS = (1.0, 5000.0)


class InverseSolver:
    """Fit advection-diffusion parameters to observed sensor data.

    Uses L-BFGS-B to minimise the squared difference between
    simulated and observed sensor traces.
    """

    def __init__(
        self,
        geometry: PipeGeometry,
        nx: int = 60,
        w_upstream: float = 1.0,
        w_downstream: float = 1.0,
    ) -> None:
        self.geometry = geometry
        self.nx = nx
        self.w_upstream = w_upstream
        self.w_downstream = w_downstream
        self._solver = AdvectionDiffusionSolver(geometry)

    def _make_bc(self, c0: float, duration: float) -> Pulse:
        """Create the boundary condition used during fitting."""
        return Pulse(c0=c0, t0=duration * 0.15, sigma=duration * 0.05)

    def fit(
        self,
        observed_upstream: np.ndarray,
        observed_downstream: np.ndarray,
        time: np.ndarray,
        temperature: float = 20.0,
    ) -> FitResult:
        """Fit D, v, C0 to observed sensor traces.

        Args:
            observed_upstream: Observed upstream concentration time series.
            observed_downstream: Observed downstream concentration time series.
            time: Time array matching the observations (seconds).
            temperature: Water temperature in deg C.

        Returns:
            FitResult with best-fit parameters and diagnostics.
        """
        duration = float(time[-1] - time[0])

        # Normalise observations for numerical stability
        obs_scale = max(
            np.max(np.abs(observed_upstream)),
            np.max(np.abs(observed_downstream)),
            1.0,
        )
        obs_up_norm = observed_upstream / obs_scale
        obs_down_norm = observed_downstream / obs_scale

        def cost(params_vec: np.ndarray) -> float:
            log_d, log_v, log_c0 = params_vec
            d_mol = np.exp(log_d)
            vel = np.exp(log_v)
            c0 = np.exp(log_c0)

            sim_params = SimulationParams(
                d_molecular=d_mol, velocity=vel, c0=c0,
                temperature=temperature, nx=self.nx, dt=None,
                duration=duration, boundary_type="pulse",
                geometry=self.geometry,
            )
            bc = self._make_bc(c0, duration)

            try:
                result = self._solver.solve(sim_params, bc)
            except Exception:
                return 1e10

            # Interpolate simulation to observation times, normalise
            sim_up = np.interp(time, result.time, result.upstream) / obs_scale
            sim_down = np.interp(time, result.time, result.downstream) / obs_scale

            if np.any(np.isnan(sim_up)) or np.any(np.isnan(sim_down)):
                return 1e10

            cost_val = (
                self.w_upstream * np.mean((sim_up - obs_up_norm) ** 2)
                + self.w_downstream * np.mean((sim_down - obs_down_norm) ** 2)
            )
            return float(cost_val)

        # Initial guess (log-space): mid-range
        x0 = np.array([
            np.log(1e-8),   # D
            np.log(0.05),   # v
            np.log(100.0),  # C0
        ])
        bounds = [
            (np.log(_D_BOUNDS[0]), np.log(_D_BOUNDS[1])),
            (np.log(_V_BOUNDS[0]), np.log(_V_BOUNDS[1])),
            (np.log(_C0_BOUNDS[0]), np.log(_C0_BOUNDS[1])),
        ]

        opt_result = minimize(
            cost, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-8},
        )

        # Extract best-fit parameters
        best_d = np.exp(opt_result.x[0])
        best_v = np.exp(opt_result.x[1])
        best_c0 = np.exp(opt_result.x[2])

        # Check if any parameter hit its bound
        at_bound = any(
            abs(opt_result.x[i] - bounds[i][0]) < 1e-6
            or abs(opt_result.x[i] - bounds[i][1]) < 1e-6
            for i in range(3)
        )

        # Run final forward simulation with best-fit params
        final_params = SimulationParams(
            d_molecular=best_d, velocity=best_v, c0=best_c0,
            temperature=temperature, nx=self.nx, dt=None,
            duration=duration, boundary_type="pulse",
            geometry=self.geometry,
        )
        bc = self._make_bc(best_c0, duration)
        final_sim = self._solver.solve(final_params, bc)

        # Determine regime from Pe
        pe = final_sim.peclet_number
        geo = self.geometry
        if pe > geo.suspension_pe:
            regime = TurbidityRegime.SUSPENSION
        elif pe < geo.solution_pe:
            regime = TurbidityRegime.SOLUTION
        else:
            regime = TurbidityRegime.COLLOID

        return FitResult(
            d_molecular=best_d,
            d_effective=final_sim.d_effective,
            velocity=best_v,
            c0=best_c0,
            peclet_number=pe,
            regime=regime,
            residual=float(opt_result.fun),
            converged=opt_result.success,
            at_bound=at_bound,
            simulation=final_sim,
        )
