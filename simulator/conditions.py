"""Boundary and initial conditions for the advection-diffusion solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class BoundaryCondition(Protocol):
    """Protocol for inlet boundary conditions. Callable(t) -> concentration."""

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray: ...


class Pulse:
    """Gaussian pulse inlet boundary condition.

    C(t) = c0 * exp(-(t - t0)^2 / (2 * sigma^2))
    """

    def __init__(self, c0: float, t0: float = 5.0, sigma: float = 1.0) -> None:
        self.c0 = c0
        self.t0 = t0
        self.sigma = sigma

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.c0 * np.exp(-((t - self.t0) ** 2) / (2.0 * self.sigma**2))


class Step:
    """Step function inlet boundary condition.

    C(t) = 0 for t < t0, c0 for t >= t0.
    """

    def __init__(self, c0: float, t0: float = 0.0) -> None:
        self.c0 = c0
        self.t0 = t0

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.c0 * np.heaviside(np.asarray(t) - self.t0, 1.0)


class Ramp:
    """Linear ramp inlet boundary condition.

    C(t) = 0 for t < t0, linear 0->c0 over duration tau, c0 after.
    """

    def __init__(self, c0: float, t0: float = 0.0, tau: float = 5.0) -> None:
        self.c0 = c0
        self.t0 = t0
        self.tau = tau

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        elapsed = (t_arr - self.t0) / self.tau
        return self.c0 * np.clip(elapsed, 0.0, 1.0)


class Arbitrary:
    """Arbitrary inlet boundary condition from user-supplied data.

    Linearly interpolated to solver timestep grid. Values outside
    the supplied time range are clamped to the nearest endpoint.
    """

    def __init__(self, t_data: np.ndarray, c_data: np.ndarray) -> None:
        self.t_data = t_data
        self.c_data = c_data

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.interp(
            t, self.t_data, self.c_data,
            left=self.c_data[0], right=self.c_data[-1],
        )


@dataclass
class InitialCondition:
    """Initial concentration field C(x, t=0)."""

    values: np.ndarray

    @classmethod
    def zeros(cls, nx: int) -> InitialCondition:
        return cls(values=np.zeros(nx))

    @classmethod
    def uniform(cls, c0: float, nx: int) -> InitialCondition:
        return cls(values=np.full(nx, c0))

    @classmethod
    def from_array(cls, arr: np.ndarray) -> InitialCondition:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(arr).__name__}")
        return cls(values=arr.copy())
