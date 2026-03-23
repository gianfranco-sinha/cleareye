"""Biofouling monitor — drift detection, correction factor, cleaning alerts.

Runs in parallel (not inline). Milestone 3 feature; this module provides
the interface and a no-op stub for milestone 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoulingAssessment:
    """Result of a biofouling assessment for a rig."""

    correction_factor: float  # Multiplicative correction (1.0 = no fouling)
    reliability: float        # 0.0–1.0, confidence in the correction
    cleaning_alert: bool      # True if sensor needs cleaning


class BiofoulingMonitor:
    """Stub biofouling monitor — returns neutral assessment.

    Will be replaced with a trained drift model in milestone 3 that
    queries historical calibrated readings per rig.
    """

    def assess(self, rig_id: str, calibrated_ntu: float) -> FoulingAssessment:
        """Assess biofouling state for a given rig.

        Args:
            rig_id: Identifier for the deployed sensor rig.
            calibrated_ntu: Current calibrated turbidity reading.
        """
        return FoulingAssessment(
            correction_factor=1.0,
            reliability=1.0,
            cleaning_alert=False,
        )
