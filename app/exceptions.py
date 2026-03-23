"""Domain exception hierarchy for ClearEye."""


class ClearEyeError(Exception):
    """Base exception for all ClearEye errors."""


class SensorError(ClearEyeError):
    """Error related to sensor readings or profiles."""


class ReadingOutOfRange(SensorError):
    """A sensor reading is outside the valid range for its quantity."""

    def __init__(self, quantity: str, value: float, valid_range: tuple[float, float]):
        self.quantity = quantity
        self.value = value
        self.valid_range = valid_range
        super().__init__(
            f"{quantity}={value} outside valid range {valid_range}"
        )


class UnknownSensorProfile(SensorError):
    """Requested sensor profile is not registered."""

    def __init__(self, profile_name: str):
        self.profile_name = profile_name
        super().__init__(f"Unknown sensor profile: {profile_name!r}")


class CalibrationError(ClearEyeError):
    """Error during calibration stage."""


class RegimeClassificationError(ClearEyeError):
    """Error during regime classification."""


class BiofoulingError(ClearEyeError):
    """Error during biofouling assessment."""


class ConfigError(ClearEyeError):
    """Error loading or validating configuration."""


class TrainingError(ClearEyeError):
    """Error during model training."""


class InsufficientDataError(TrainingError):
    """Not enough data to train a model."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class DataValidationError(TrainingError):
    """Training data failed validation checks."""
