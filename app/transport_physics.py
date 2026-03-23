"""Transport physics — advection-diffusion model for dual-sensor pipe geometry.

Provides cross-correlation, peak attenuation, signal broadening, and
Péclet number calculations for physics-based regime discrimination.

Dual-sensor geometry:
  - Pipe length: 300 mm
  - Sensor spacing: 200 mm (default)
  - Mesh-filtered ends
  - Laminar flow assumed
"""

from __future__ import annotations

import numpy as np

# Default pipe geometry (metres)
DEFAULT_SENSOR_SPACING = 0.200  # 200 mm
DEFAULT_PIPE_LENGTH = 0.300  # 300 mm


# ---------------------------------------------------------------------------
# Péclet number
# ---------------------------------------------------------------------------

def peclet_number(
    velocity: float,
    diffusion_coeff: float,
    sensor_spacing: float = DEFAULT_SENSOR_SPACING,
) -> float:
    """Compute the Péclet number: Pe = v·L / D.

    Args:
        velocity: Flow velocity in m/s.
        diffusion_coeff: Diffusion coefficient in m²/s.
        sensor_spacing: Distance between sensors in metres.

    Returns:
        Pe — ratio of advective to diffusive transport.
        High Pe → particles (advection dominates).
        Low Pe → colloids (diffusion dominates).
    """
    if diffusion_coeff <= 0:
        return float("inf")
    return velocity * sensor_spacing / diffusion_coeff


# ---------------------------------------------------------------------------
# Cross-correlation between dual sensors
# ---------------------------------------------------------------------------

def cross_correlation(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation between two sensor time series.

    Args:
        signal_a: Upstream sensor readings (1D array).
        signal_b: Downstream sensor readings (1D array).
        max_lag: Maximum lag to compute (samples). Defaults to len//4.

    Returns:
        (lags, correlation) — arrays of lag indices and normalised correlation values.
    """
    a = signal_a - signal_a.mean()
    b = signal_b - signal_b.mean()

    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if norm == 0:
        n = len(a)
        max_lag = max_lag or n // 4
        return np.arange(-max_lag, max_lag + 1), np.zeros(2 * max_lag + 1)

    # np.correlate(a, b, "full")[n-1+k] = sum(a[i]*b[i-k])
    # Positive lag k means b is delayed relative to a (downstream sensor)
    full_corr = np.correlate(b, a, mode="full")
    full_corr = full_corr / norm

    n = len(a)
    max_lag = max_lag or n // 4
    center = n - 1
    lags = np.arange(-max_lag, max_lag + 1)
    correlation = full_corr[center - max_lag : center + max_lag + 1]

    return lags, correlation


def correlation_peak(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    max_lag: int | None = None,
) -> tuple[float, int]:
    """Find the peak cross-correlation and its lag.

    Returns:
        (peak_value, lag_at_peak) — peak correlation strength and
        the lag (in samples) where it occurs.
        Strong peak with positive lag → particles transported downstream.
        Weak/absent peak → colloids (diffusion dominates).
    """
    lags, corr = cross_correlation(signal_a, signal_b, max_lag)
    peak_idx = np.argmax(corr)
    return float(corr[peak_idx]), int(lags[peak_idx])


# ---------------------------------------------------------------------------
# Peak attenuation
# ---------------------------------------------------------------------------

def peak_attenuation(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    baseline_a: float | None = None,
    baseline_b: float | None = None,
) -> float:
    """Compute peak attenuation ratio between sensors.

    attenuation = (peak_B - baseline_B) / (peak_A - baseline_A)

    Values:
        ≈ 1 → particles (minimal diffusion loss)
        << 1 → colloids (significant diffusion spreading)

    Args:
        signal_a: Upstream sensor readings.
        signal_b: Downstream sensor readings.
        baseline_a: Baseline (clear water) for sensor A. Defaults to median.
        baseline_b: Baseline (clear water) for sensor B. Defaults to median.
    """
    if baseline_a is None:
        baseline_a = float(np.median(signal_a))
    if baseline_b is None:
        baseline_b = float(np.median(signal_b))

    peak_a = float(np.max(np.abs(signal_a - baseline_a)))
    peak_b = float(np.max(np.abs(signal_b - baseline_b)))

    if peak_a == 0:
        return 0.0
    return peak_b / peak_a


# ---------------------------------------------------------------------------
# Signal broadening
# ---------------------------------------------------------------------------

def signal_broadening(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    threshold_fraction: float = 0.5,
) -> float:
    """Compute signal broadening ratio between sensors.

    Measures peak width at a fraction of peak height (default: half-maximum).

    broadening = width_B / width_A

    Values:
        ≈ 1 → particles (no diffusion spreading)
        > 1 → colloids (diffusion causes peak broadening)

    Args:
        signal_a: Upstream sensor readings.
        signal_b: Downstream sensor readings.
        threshold_fraction: Fraction of peak height for width measurement.
    """
    width_a = _peak_width(signal_a, threshold_fraction)
    width_b = _peak_width(signal_b, threshold_fraction)

    if width_a == 0:
        return 0.0
    return width_b / width_a


def _peak_width(signal: np.ndarray, threshold_fraction: float) -> int:
    """Measure peak width at a fraction of peak height above baseline."""
    baseline = np.median(signal)
    centered = np.abs(signal - baseline)
    peak_height = centered.max()
    if peak_height == 0:
        return 0

    threshold = peak_height * threshold_fraction
    above = centered >= threshold
    if not above.any():
        return 0

    # Width = last index above threshold - first index above threshold
    indices = np.where(above)[0]
    return int(indices[-1] - indices[0])


# ---------------------------------------------------------------------------
# Effective diffusion coefficient estimation
# ---------------------------------------------------------------------------

def estimate_diffusion_coefficient(
    attenuation: float,
    velocity: float,
    sensor_spacing: float = DEFAULT_SENSOR_SPACING,
    temperature: float = 20.0,
) -> float:
    """Rough estimate of effective diffusion coefficient from attenuation.

    Uses a simplified 1D advection-diffusion model where the peak
    attenuation is related to D via:

        attenuation ≈ exp(-4·D / (L·v))

    Rearranging: D ≈ -L·v·ln(attenuation) / 4

    Temperature correction: D scales with T/viscosity (Stokes-Einstein).
    Reference at 20°C; viscosity ratio approximated linearly.

    Args:
        attenuation: Peak attenuation ratio (0 to 1).
        velocity: Flow velocity in m/s.
        sensor_spacing: Distance between sensors in metres.
        temperature: Water temperature in °C.

    Returns:
        Estimated diffusion coefficient in m²/s. Returns 0.0 if
        attenuation is 0 or 1 (no useful information).
    """
    if attenuation <= 0 or attenuation >= 1 or velocity <= 0:
        return 0.0

    import math

    # Base estimate from attenuation
    d_est = -sensor_spacing * velocity * math.log(attenuation) / 4.0

    # Temperature correction (Stokes-Einstein scaling)
    # Water viscosity ratio relative to 20°C, linear approximation
    viscosity_ratio = 1.0 + 0.02 * (20.0 - temperature)
    d_est = d_est / max(viscosity_ratio, 0.1)

    return max(d_est, 0.0)


# ---------------------------------------------------------------------------
# Dual-sensor regime features (composite)
# ---------------------------------------------------------------------------

def dual_sensor_features(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    velocity: float,
    temperature: float = 20.0,
    sensor_spacing: float = DEFAULT_SENSOR_SPACING,
    sampling_interval: float = 6.0,
) -> dict[str, float]:
    """Extract all dual-sensor transport features for regime classification.

    Args:
        signal_a: Upstream sensor time series (ADC values).
        signal_b: Downstream sensor time series (ADC values).
        velocity: Estimated flow velocity in m/s.
        temperature: Water temperature in °C.
        sensor_spacing: Distance between sensors in metres.
        sampling_interval: Time between readings in seconds.

    Returns:
        Dict of physics-based features:
        - cross_correlation_peak: peak correlation strength
        - cross_correlation_lag: lag at peak (samples)
        - peak_attenuation: amplitude ratio
        - signal_broadening: width ratio
        - peclet_number: Pe = v·L/D
        - effective_diffusion: estimated D in m²/s
    """
    corr_peak, corr_lag = correlation_peak(signal_a, signal_b)
    atten = peak_attenuation(signal_a, signal_b)
    broadening = signal_broadening(signal_a, signal_b)
    d_eff = estimate_diffusion_coefficient(
        atten, velocity, sensor_spacing, temperature
    )
    pe = peclet_number(velocity, d_eff, sensor_spacing) if d_eff > 0 else float("inf")

    return {
        "cross_correlation_peak": corr_peak,
        "cross_correlation_lag": corr_lag,
        "peak_attenuation": atten,
        "signal_broadening": broadening,
        "peclet_number": pe,
        "effective_diffusion": d_eff,
    }
