"""FSM pipeline orchestrator — placeholder for milestone 2.

Pipeline stages: ingest → clean → feature_engineer → window → split → scale → train → evaluate → save
"""

from __future__ import annotations

from enum import Enum


class PipelineStage(str, Enum):
    """Training pipeline stages."""

    INGEST = "ingest"
    CLEAN = "clean"
    FEATURE_ENGINEER = "feature_engineer"
    WINDOW = "window"
    SPLIT = "split"
    SCALE = "scale"
    TRAIN = "train"
    EVALUATE = "evaluate"
    SAVE = "save"
    DONE = "done"


class PipelineOrchestrator:
    """FSM-based training pipeline orchestrator.

    Manages state transitions through the training pipeline stages.
    Will be fully implemented in milestone 2.
    """

    def __init__(self) -> None:
        self.stage = PipelineStage.INGEST
        self.context: dict = {}

    def advance(self) -> PipelineStage:
        """Advance to the next pipeline stage."""
        stages = list(PipelineStage)
        current_idx = stages.index(self.stage)
        if current_idx < len(stages) - 1:
            self.stage = stages[current_idx + 1]
        return self.stage

    def run(self) -> None:
        """Run the full pipeline. Placeholder."""
        raise NotImplementedError("Pipeline orchestrator not yet implemented")
