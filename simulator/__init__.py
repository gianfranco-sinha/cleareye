"""Advection-diffusion simulator for ClearEye dual-sensor pipe geometry.

Public API:
    - AdvectionDiffusionSolver, SimulationParams, SimulationResult
    - InverseSolver, FitResult
    - generate_synthetic_dataset
    - PipeGeometry, load_geometry
    - Pulse, Step, Ramp, Arbitrary, InitialCondition
    - effective_diffusion, temperature_correct_diffusion
"""

from simulator.conditions import Arbitrary, BoundaryCondition, InitialCondition, Pulse, Ramp, Step
from simulator.fitting import FitResult, InverseSolver
from simulator.geometry import PipeGeometry, load_geometry
from simulator.solver import AdvectionDiffusionSolver, SimulationParams, SimulationResult
from simulator.synthetic import generate_synthetic_dataset
from simulator.taylor_dispersion import effective_diffusion, temperature_correct_diffusion

__all__ = [
    "AdvectionDiffusionSolver",
    "Arbitrary",
    "BoundaryCondition",
    "FitResult",
    "InitialCondition",
    "InverseSolver",
    "PipeGeometry",
    "Pulse",
    "Ramp",
    "SimulationParams",
    "SimulationResult",
    "Step",
    "effective_diffusion",
    "generate_synthetic_dataset",
    "load_geometry",
    "temperature_correct_diffusion",
]
