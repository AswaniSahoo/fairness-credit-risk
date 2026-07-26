"""Configuration for the fairness-aware credit risk pipeline.

Only pipeline-level hyperparameters live here. Per-dataset constants (protected
attributes, privileged/unprivileged encodings, favorable label, file paths) live
in ``src.data.registry`` because the disparity direction reverses between the
German Credit and Taiwan Credit datasets, so a single global value is wrong for
one of them.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Top-level configuration aggregate."""

    RANDOM_STATE: int = 42

    # Three-way split: the calibration share is carved from the training set to
    # fit post-processing without reusing training rows (finding B4).
    TEST_SIZE: float = 0.2
    CALIBRATION_SIZE: float = 0.2

    # Defaults reproduce the numbers in reports/track_comparison.json. Lowering them
    # changes the selected model, so a reader running the pipeline unmodified should get
    # what is published rather than something close to it.
    N_TRIALS: int = 60
    CV_FOLDS: int = 5


config = Config()
