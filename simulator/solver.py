"""1D advection-diffusion solver using Crank-Nicolson finite differences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from simulator.conditions import BoundaryCondition, InitialCondition
from simulator.geometry import PipeGeometry
from simulator.taylor_dispersion import effective_diffusion


@dataclass
class SimulationParams:
    """Input parameters for a single simulation run."""

    d_molecular: float
    velocity: float          # signed: +ve = inlet->outlet, -ve = reverse
    c0: float
    temperature: float
    nx: int
    dt: float | None
    duration: float
    boundary_type: str
    geometry: PipeGeometry


@dataclass
class SimulationResult:
    """Output of a forward simulation run."""

    concentration: np.ndarray
    upstream: np.ndarray
    downstream: np.ndarray
    time: np.ndarray
    x: np.ndarray
    params: SimulationParams
    peclet_number: float
    d_effective: float
    transit_time: float
    mass_conservation_error: float


class AdvectionDiffusionSolver:
    """1D advection-diffusion PDE solver.

    Uses Crank-Nicolson (semi-implicit) finite differences with
    first-order upwind advection. Tridiagonal system solved via
    scipy.linalg.solve_banded.

    Velocity is signed: +ve flows inlet->outlet, -ve reverses.
    The solver is stateless -- safe for concurrent use.
    """

    def __init__(self, geometry: PipeGeometry) -> None:
        self.geometry = geometry

    def solve(
        self,
        params: SimulationParams,
        boundary_condition: BoundaryCondition,
        initial_condition: InitialCondition | None = None,
    ) -> SimulationResult:
        """Run the forward simulation.

        Args:
            params: Simulation parameters.
            boundary_condition: Callable(t) returning inlet concentration.
            initial_condition: C(x, t=0). Defaults to zeros.

        Returns:
            SimulationResult with full concentration field and sensor traces.
        """
        geo = self.geometry
        nx = params.nx
        L = geo.pipe_length

        # Spatial grid
        x = np.linspace(0, L, nx)
        dx = x[1] - x[0]

        # Effective diffusion (with Taylor dispersion)
        d_eff = effective_diffusion(
            params.d_molecular, params.velocity,
            geo.inner_radius, params.temperature,
        )

        # Timestep selection
        v = params.velocity
        speed = abs(v)
        if params.dt is not None:
            dt = params.dt
        else:
            # Auto: consider both Courant (advection) and diffusion accuracy
            dt_candidates = []
            if speed > 0:
                dt_candidates.append(dx / speed)  # Courant: r_a = 1
            if d_eff > 0:
                dt_candidates.append(dx**2 / d_eff)  # Diffusion: r_d = 1
            dt = min(dt_candidates) if dt_candidates else 0.01

        # Cap timesteps to prevent memory explosion
        max_nt = 200_000
        nt = min(int(params.duration / dt) + 1, max_nt)
        time = np.linspace(0, params.duration, nt)
        dt = time[1] - time[0] if nt > 1 else dt

        # Initial condition
        if initial_condition is not None:
            c = initial_condition.values.copy()
            if len(c) != nx:
                c = np.interp(x, np.linspace(0, L, len(c)), c)
        else:
            c = np.zeros(nx)

        # Store full field
        concentration = np.zeros((nt, nx))
        concentration[0, :] = c

        # Sensor indices (nearest grid point)
        upstream_idx = int(round(geo.upstream_position / dx))
        downstream_idx = int(round(geo.downstream_position / dx))
        upstream_idx = min(max(upstream_idx, 0), nx - 1)
        downstream_idx = min(max(downstream_idx, 0), nx - 1)

        # Crank-Nicolson coefficients
        r_d = d_eff * dt / dx**2
        r_a = v * dt / dx  # signed Courant number

        # Build tridiagonal bands for implicit side (interior points only)
        n_interior = nx - 2  # exclude boundary points

        if v >= 0:
            # Upwind from left
            a_sub = -r_d / 2 - r_a / 2
            b_diag = 1 + r_d + r_a / 2
            c_sup = -r_d / 2
            d_sub = r_d / 2 + r_a / 2
            e_diag = 1 - r_d - r_a / 2
            f_sup = r_d / 2
        else:
            # Upwind from right (v < 0)
            a_sub = -r_d / 2
            b_diag = 1 + r_d + abs(r_a) / 2
            c_sup = -r_d / 2 - abs(r_a) / 2
            d_sub = r_d / 2
            e_diag = 1 - r_d - abs(r_a) / 2
            f_sup = r_d / 2 + abs(r_a) / 2

        # Banded matrix for solve_banded (shape: 3 x n_interior)
        ab = np.zeros((3, n_interior))
        ab[0, 1:] = c_sup       # super-diagonal (shifted)
        ab[1, :] = b_diag       # main diagonal
        ab[2, :-1] = a_sub      # sub-diagonal (shifted)

        # Neumann outlet BC: ghost point C[nx] = C[nx-1]
        # The implicit c_sup term for the last interior point folds into diagonal
        ab[1, -1] += c_sup

        # Track mass for conservation check
        initial_mass = np.trapz(c, x)
        mass_in = 0.0
        mass_out = 0.0

        # Time stepping
        nan_detected = False
        for n in range(nt - 1):
            t_now = time[n + 1]

            # Inlet boundary (Dirichlet)
            c_inlet = float(boundary_condition(t_now))
            mass_in += c_inlet * speed * dt

            # Build RHS -- vectorised, no Python loops
            rhs = d_sub * c[:-2] + e_diag * c[1:-1] + f_sup * c[2:]

            # Left boundary: implicit side needs c_inlet at n+1
            rhs[0] -= a_sub * c_inlet

            # Neumann outlet: c[nx-1] = c[nx-2] is already set from previous
            # step, so the vectorised f_sup * c[2:] already includes the
            # correct boundary value. No ghost point addition needed here
            # (the implicit side handles it via ab[1, -1] += c_sup).

            # Check RHS for overflow before solving
            if np.any(np.isnan(rhs)) or np.any(np.isinf(rhs)):
                nan_detected = True
                break

            # Solve tridiagonal system
            try:
                c_new_interior = solve_banded((1, 1), ab, rhs)
            except (np.linalg.LinAlgError, ValueError):
                nan_detected = True
                break

            if np.any(np.isnan(c_new_interior)) or np.any(np.isinf(c_new_interior)):
                nan_detected = True
                break

            # Assemble full solution
            c[0] = c_inlet
            c[1:-1] = c_new_interior
            c[-1] = c[-2]  # Neumann BC

            # Track outlet mass flux
            mass_out += c[-1] * speed * dt

            concentration[n + 1, :] = c

        # Sensor traces
        upstream_trace = concentration[:, upstream_idx]
        downstream_trace = concentration[:, downstream_idx]

        # Peclet number and transit time
        sensor_spacing = geo.sensor_spacing
        pe = (speed * sensor_spacing / d_eff) if d_eff > 0 else float("inf")
        transit_time = sensor_spacing / v if v != 0 else float("inf")

        # Mass conservation error
        final_mass = np.trapz(c, x)
        total_expected = initial_mass + mass_in - mass_out
        if abs(initial_mass + mass_in) > 0:
            mass_error = (final_mass - total_expected) / (initial_mass + mass_in)
        else:
            mass_error = 0.0

        return SimulationResult(
            concentration=concentration,
            upstream=upstream_trace,
            downstream=downstream_trace,
            time=time,
            x=x,
            params=params,
            peclet_number=pe,
            d_effective=d_eff,
            transit_time=transit_time,
            mass_conservation_error=mass_error,
        )
