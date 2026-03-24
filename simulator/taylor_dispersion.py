"""Taylor dispersion and temperature-corrected diffusion."""

from __future__ import annotations


def temperature_correct_diffusion(
    d_molecular: float,
    temperature: float = 20.0,
) -> float:
    """Apply Stokes-Einstein temperature correction to diffusion coefficient.

    Uses a linear viscosity approximation relative to 20 deg C.
    Accurate within ~5% for 10-30 deg C.

    Args:
        d_molecular: Molecular diffusion coefficient at 20 deg C (m^2/s).
        temperature: Water temperature in deg C.

    Returns:
        Temperature-corrected diffusion coefficient (m^2/s).
    """
    viscosity_ratio = 1.0 + 0.02 * (20.0 - temperature)
    return d_molecular / max(viscosity_ratio, 0.1)


def effective_diffusion(
    d_molecular: float,
    velocity: float,
    pipe_radius: float,
    temperature: float = 20.0,
    perturbation_multiplier: float = 1.0,
) -> float:
    """Compute effective dispersion coefficient with Taylor dispersion.

    D_eff = D_corrected + (R^2 * v^2) / (48 * D_corrected)

    Args:
        d_molecular: Molecular diffusion coefficient at 20 deg C (m^2/s).
        velocity: Mean flow velocity (m/s).
        pipe_radius: Pipe inner radius (m).
        temperature: Water temperature in deg C.
        perturbation_multiplier: Local dispersion multiplier. Default 1.0.

    Returns:
        Effective dispersion coefficient (m^2/s).
    """
    d_corrected = temperature_correct_diffusion(d_molecular, temperature)

    if velocity == 0.0 or d_corrected == 0.0:
        return d_corrected * perturbation_multiplier

    taylor_term = (pipe_radius**2 * velocity**2) / (48.0 * d_corrected)
    return (d_corrected + taylor_term) * perturbation_multiplier
