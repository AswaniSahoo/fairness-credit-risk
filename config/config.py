"""Configuration for the fairness-aware credit risk pipeline.

Field defaults use ``default_factory`` because a dataclass instance is unhashable, and
from Python 3.11 the dataclass machinery rejects such values as bare defaults. The
previous version raised ``ValueError`` at import time on any interpreter newer than 3.10.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


@dataclass
class DataConfig:
    """Dataset locations, target column, and protected attribute names."""

    RAW_DATA_PATH: Path = RAW_DATA_DIR / "german.data"
    PROCESSED_DATA_PATH: Path = (
        PROCESSED_DATA_DIR / "german_credit_numerical_final.csv"
    )

    TARGET_COLUMN: str = "credit_risk"

    # Held out of the feature matrix and used for fairness measurement only.
    # See finding B5: `personal_status_sex` also encodes sex and is still a feature,
    # so excluding `gender` alone does not remove sex from the model.
    PROTECTED_ATTRIBUTES: list[str] = field(
        default_factory=lambda: ["gender", "age", "foreign_worker"]
    )

    APPLY_FAIRNESS_PREPROCESSING: bool = True


@dataclass
class ModelConfig:
    """Model families to search and the weight given to fairness in the objective."""

    MODELS_TO_TRY: list[str] = field(
        default_factory=lambda: [
            "random_forest",
            "xgboost",
            "logistic_regression",
            "lightgbm",
        ]
    )

    FAIRNESS_WEIGHT: float = 0.3


@dataclass
class FairnessConfig:
    """Protected attribute under analysis and the thresholds it is judged against."""

    PRIMARY_PROTECTED_ATTRIBUTE: str = "gender"

    # Privileged group encoding in the processed dataset: 1 = male, 0 = female.
    PRIVILEGED_VALUE: int = 1
    UNPRIVILEGED_VALUE: int = 0

    # The favorable outcome is "good credit", encoded as 0 in `credit_risk`.
    FAVORABLE_LABEL: int = 0

    # Four-fifths rule: a selection-rate ratio below 0.8 is evidence of adverse impact.
    DISPARATE_IMPACT_THRESHOLD: float = 0.8
    STATISTICAL_PARITY_THRESHOLD: float = 0.1


@dataclass
class Config:
    """Top-level configuration aggregate."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)

    RANDOM_STATE: int = 42

    # Three-way split: the calibration share is carved out to fit post-processing
    # without reusing training rows. See finding B4.
    TEST_SIZE: float = 0.2
    CALIBRATION_SIZE: float = 0.2

    N_TRIALS: int = 50
    CV_FOLDS: int = 3


config = Config()
