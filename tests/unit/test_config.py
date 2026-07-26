"""Regression tests for configuration loading.

Covers finding R2: the config module previously used dataclass instances as bare
field defaults, which Python 3.11+ rejects. The import must succeed on any
interpreter.

Per-dataset constants (paths, protected attributes, encodings) moved to
``src.data.registry`` because the disparity direction reverses between
datasets. This test file validates only the pipeline-level hyperparameters
that remain in ``config.config``.
"""

import pytest

pytestmark = pytest.mark.unit


def test_config_imports_on_current_interpreter():
    """R2: importing the config module must not raise."""
    from config.config import config

    assert config.RANDOM_STATE == 42


def test_aggregate_fields_are_independent_instances():
    """R2: two Config() objects must be independent."""
    from config.config import Config

    first = Config()
    second = Config()

    # Scalar fields are value-equal but the instances are distinct objects.
    assert first is not second
    assert first.RANDOM_STATE == second.RANDOM_STATE


def test_split_fractions_leave_a_majority_for_training():
    """Both shares are fractions of the whole dataset, not of a remainder."""
    from config.config import config

    assert config.TEST_SIZE == pytest.approx(0.2)
    assert config.CALIBRATION_SIZE == pytest.approx(0.2)
    # 0.2 and 0.2 of the whole leaves 0.6 for training, which is the 600/200/200 split
    # recorded in reports/track_comparison.json for the 1000-row German Credit file.
    assert 1.0 - config.TEST_SIZE - config.CALIBRATION_SIZE == pytest.approx(0.6)


def test_search_defaults_reproduce_the_published_run():
    """The recorded comparison used 60 trials and 5 folds.

    Defaults below those would make an unmodified pipeline run select a different model
    from the one whose numbers are published.
    """
    from config.config import config

    assert config.N_TRIALS == 60
    assert config.CV_FOLDS == 5
