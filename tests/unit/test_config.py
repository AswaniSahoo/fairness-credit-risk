"""Regression tests for configuration loading.

Covers finding R2: `config/config.py` previously used dataclass instances as bare field
defaults, which Python 3.11+ rejects with
``ValueError: mutable default ... use default_factory``. The module could not be imported
at all on any interpreter newer than 3.10.

Also covers finding R1: absolute `/home/aswani/automl` paths made the project unrunnable
on any other machine.
"""

import pytest

pytestmark = pytest.mark.unit


def test_config_imports_on_current_interpreter():
    """R2: importing the config module must not raise."""
    from config.config import config

    assert config.RANDOM_STATE == 42


def test_aggregate_fields_are_independent_instances():
    """R2: default_factory must give each Config its own sub-config objects.

    A shared class-level instance would let one Config mutate another. Two separate
    Config() objects must therefore hold distinct DataConfig instances.
    """
    from config.config import Config

    first = Config()
    second = Config()

    assert first.data is not second.data
    assert first.model is not second.model
    assert first.fairness is not second.fairness

    first.data.PROTECTED_ATTRIBUTES.append("injected")
    assert "injected" not in second.data.PROTECTED_ATTRIBUTES


def test_data_paths_resolve_under_project_root_and_exist():
    """R1: paths must be derived from the repo, not hardcoded to one machine."""
    from config.config import config
    from src.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR

    assert config.data.PROCESSED_DATA_PATH.is_relative_to(PROJECT_ROOT)
    assert config.data.RAW_DATA_PATH.is_relative_to(PROJECT_ROOT)
    assert config.data.PROCESSED_DATA_PATH.parent == PROCESSED_DATA_DIR
    assert config.data.RAW_DATA_PATH.parent == RAW_DATA_DIR

    assert config.data.PROCESSED_DATA_PATH.exists()
    assert config.data.RAW_DATA_PATH.exists()


def test_protected_and_target_columns_match_the_dataset_header():
    """The configured column names must exist in the processed dataset."""
    import csv

    from config.config import config

    with config.data.PROCESSED_DATA_PATH.open(newline="") as handle:
        header = next(csv.reader(handle))

    assert config.data.TARGET_COLUMN in header
    for column in config.data.PROTECTED_ATTRIBUTES:
        assert column in header, f"{column} missing from dataset header"
    assert config.fairness.PRIMARY_PROTECTED_ATTRIBUTE in header


def test_split_fractions_leave_a_majority_for_training():
    """Test plus calibration shares must not consume the training set."""
    from config.config import config

    assert config.TEST_SIZE == pytest.approx(0.2)
    assert config.CALIBRATION_SIZE == pytest.approx(0.2)
    assert config.TEST_SIZE + config.CALIBRATION_SIZE < 0.5
