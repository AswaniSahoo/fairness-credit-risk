"""Integration tests for T4, the externally scored track.

T4 cannot re-run the model that produced its numbers, so the guarantee it offers is
narrower and has to be enforced rather than described: the recorded predictions are for
exactly the applicants in this split's test block, and everything downstream of the scores
is the same code every other track uses.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.registry import GERMAN_CREDIT
from src.paths import PROJECT_ROOT
from src.pipelines.tracks import (
    load_inputs,
    load_offline_predictions,
    run_offline_predictions_track,
)

pytestmark = pytest.mark.integration

EXCHANGE = PROJECT_ROOT / "notebooks" / "exchange"
PREDICTIONS = EXCHANGE / "tabfm_predictions_german.csv"
METADATA = EXCHANGE / "tabfm_meta_german.json"


@pytest.fixture(scope="module")
def inputs():
    return load_inputs(GERMAN_CREDIT, seed=42, test_size=0.2, calibration_size=0.2)


def test_recorded_predictions_cover_the_split_test_block(inputs):
    """The handoff file must describe this split's test rows, in order."""
    scores = load_offline_predictions(PREDICTIONS, inputs.split)

    assert len(scores) == len(inputs.split.test) == 200
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_predictions_for_the_wrong_rows_are_refused(tmp_path, inputs):
    """A file with the right shape but the wrong applicants must not be accepted.

    This is the failure the check exists for. Shifting every row identifier by one keeps the
    row count, the column names and the value range intact, so nothing but the identifiers
    reveals that the scores belong to different people.
    """
    original = pd.read_csv(PREDICTIONS)
    shifted = original.copy()
    shifted["__row_id"] = shifted["__row_id"] + 1
    path = tmp_path / "shifted.csv"
    shifted.to_csv(path, index=False)

    with pytest.raises(ValueError, match="do not match the test block"):
        load_offline_predictions(path, inputs.split)


def test_a_truncated_prediction_file_is_refused(tmp_path, inputs):
    original = pd.read_csv(PREDICTIONS)
    path = tmp_path / "short.csv"
    original.head(150).to_csv(path, index=False)

    with pytest.raises(ValueError, match="holds 150 rows"):
        load_offline_predictions(path, inputs.split)


def test_out_of_range_probabilities_are_refused(tmp_path, inputs):
    original = pd.read_csv(PREDICTIONS)
    broken = original.copy()
    broken.loc[0, "proba_default"] = 1.4
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        load_offline_predictions(path, inputs.split)


def test_a_missing_prediction_file_names_how_to_produce_it(tmp_path, inputs):
    with pytest.raises(FileNotFoundError, match="Colab staging notebook"):
        load_offline_predictions(tmp_path / "absent.csv", inputs.split)


def test_track_reproduces_the_recorded_prototype_result(inputs):
    """T4's metrics must match the session 003 prototype, which measured the same scores.

    The prototype computed these numbers in a notebook with its own metric code. Routing the
    same recorded probabilities through the production path has to land on the same values,
    or one of the two implementations is wrong.
    """
    run = run_offline_predictions_track(
        GERMAN_CREDIT,
        inputs,
        predictions_path=PREDICTIONS,
        metadata_path=METADATA,
        seed=42,
        n_bootstrap=200,
    )

    assert run.track == "T4"
    assert run.deployable is False
    assert run.performance["roc_auc"] == pytest.approx(0.8435, abs=5e-4)
    assert run.performance["brier"] == pytest.approx(0.1466, abs=5e-4)
    assert run.fairness["disparate_impact"] == pytest.approx(0.8321, abs=5e-4)


def test_track_records_the_checkpoint_that_produced_the_scores(inputs):
    """Provenance is the whole defence of an externally scored track."""
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    run = run_offline_predictions_track(
        GERMAN_CREDIT,
        inputs,
        predictions_path=PREDICTIONS,
        metadata_path=METADATA,
        seed=42,
        n_bootstrap=100,
    )

    assert run.model["model_type"] == metadata["checkpoint"]
    assert run.model["scored_offline"] is True
    assert run.model["params"]["tabfm_commit"] == metadata["tabfm_commit"]
    assert run.model["params"]["device"] == "cuda"
    # No calibration-block predictions exist, so no threshold may be selected from them.
    assert run.operating_point is None


def test_track_carries_a_threshold_sweep_for_the_tradeoff_curve(inputs):
    run = run_offline_predictions_track(
        GERMAN_CREDIT,
        inputs,
        predictions_path=PREDICTIONS,
        metadata_path=METADATA,
        seed=42,
        n_bootstrap=100,
    )

    sweep = run.threshold_sweep
    assert sweep is not None and len(sweep) == 50
    assert sweep[0]["selection_rate"] == pytest.approx(0.0)
    assert sweep[-1]["selection_rate"] == pytest.approx(1.0)
    for row in sweep:
        assert "disparate_impact" in row
