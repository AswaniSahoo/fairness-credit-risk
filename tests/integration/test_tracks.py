"""Integration tests for the comparison tracks.

Runs the real T0 track on German Credit at a reduced trial count. The point is not the
metric values, which depend on the search, but the invariants a comparison depends on: the
shared split, the encoded width, the reported group sizes, and the artifact key.
"""

import json

import pytest

from src.data.registry import GERMAN_CREDIT
from src.pipelines.tracks import load_inputs, record_run, run_baseline_track

pytestmark = pytest.mark.integration

SEED = 42


@pytest.fixture(scope="module")
def baseline():
    inputs = load_inputs(
        GERMAN_CREDIT, seed=SEED, test_size=0.2, calibration_size=0.2
    )
    run, estimator, search, background = run_baseline_track(
        GERMAN_CREDIT,
        inputs,
        seed=SEED,
        n_trials=2,
        cv_folds=2,
        n_bootstrap=50,
    )
    return run, estimator, inputs.split


def test_track_uses_the_shared_three_way_split(baseline):
    run, _, split = baseline

    assert run.split_sizes == {"train": 600, "calibration": 200, "test": 200}
    assert run.split_fingerprint == split.fingerprint
    # The calibration block must be left untouched by the baseline; it exists for the
    # post-processing track, which may not see test rows.
    assert split.calibration.size == 200


def test_track_reports_the_encoded_feature_width(baseline):
    run, estimator, _ = baseline

    assert run.n_encoded_features == 47
    assert estimator.named_steps["classifier"].n_features_in_ == 47


def test_track_reports_the_group_sizes_behind_every_rate(baseline):
    """Rates on 62 women are not interchangeable with rates on 138 men, so the counts are
    part of the record rather than a footnote."""
    run, _, _ = baseline

    assert run.fairness["n_privileged"] == pytest.approx(138.0)
    assert run.fairness["n_unprivileged"] == pytest.approx(62.0)
    assert run.protected_attribute == "gender"


def test_every_reported_metric_carries_an_interval_containing_its_point(baseline):
    run, _, _ = baseline

    expected = {
        "roc_auc", "balanced_accuracy", "f1", "recall", "brier",
        "disparate_impact", "statistical_parity_difference",
        "equal_opportunity_difference", "equalized_odds_difference",
    }
    assert set(run.intervals) == expected

    for name, interval in run.intervals.items():
        assert interval["ci_low"] <= interval["point"] <= interval["ci_high"], name


def test_fairness_is_computed_from_predictions_not_labels(baseline):
    """Finding B1 at the track level.

    The label-only disparate impact of this dataset is 0.8966. A model-derived value that
    happened to equal it to four decimals would mean the metric had regressed to reading
    labels.
    """
    run, _, _ = baseline

    assert run.fairness["disparate_impact"] != pytest.approx(0.8966, abs=5e-5)


def test_the_search_record_names_the_objective_and_trial_budget(baseline):
    run, _, _ = baseline

    assert run.model["scoring"] == "roc_auc"
    assert run.model["n_trials"] == 2
    assert run.model["cv_folds"] == 2
    assert run.model["model_type"] in {
        "random_forest", "xgboost", "lightgbm", "logistic_regression"
    }


def test_record_run_keys_by_dataset_and_track_without_dropping_other_runs(baseline, tmp_path):
    run, _, _ = baseline
    path = tmp_path / "track_comparison.json"

    record_run(run, path)
    run.track = "T1"
    record_run(run, path)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert set(written["runs"]) == {"german_credit|T0", "german_credit|T1"}
    assert written["schema_version"] == 2
    assert written["runs"]["german_credit|T0"]["provenance"] == GERMAN_CREDIT.provenance
