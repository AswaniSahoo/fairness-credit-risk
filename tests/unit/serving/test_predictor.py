"""Unit tests for the serving predictor."""

from __future__ import annotations

import pytest

from src.serving.predictor import Predictor

pytestmark = pytest.mark.unit

# A valid applicant in the codebook domain, used across multiple tests.
SAMPLE_APPLICANT: dict = {
    "duration": 24,
    "amount": 5951,
    "age": 35,
    "number_credits": 1,
    "people_liable": 1,
    "employment_duration": 2,
    "installment_rate": 2,
    "present_residence": 3,
    "job": 2,
    "status": 1,
    "credit_history": 2,
    "purpose": 3,
    "savings": 0,
    "other_debtors": 0,
    "property": 1,
    "other_installment_plans": 2,
    "housing": 1,
    "telephone": 0,
}


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor()


def test_predict_returns_required_fields(predictor: Predictor):
    result = predictor.predict(SAMPLE_APPLICANT)

    assert "probability_of_default" in result
    assert "decision" in result
    assert "threshold" in result
    assert "track" in result
    assert "dataset" in result


def test_probability_is_bounded(predictor: Predictor):
    result = predictor.predict(SAMPLE_APPLICANT)
    p = result["probability_of_default"]

    assert 0.0 <= p <= 1.0


def test_decision_is_consistent_with_threshold(predictor: Predictor):
    result = predictor.predict(SAMPLE_APPLICANT)

    if result["probability_of_default"] >= result["threshold"]:
        assert result["decision"] == "decline"
    else:
        assert result["decision"] == "approve"


def test_threshold_matches_run_record(predictor: Predictor):
    result = predictor.predict(SAMPLE_APPLICANT)

    assert result["threshold"] == predictor.run["threshold"]


def test_track_and_dataset_match_run(predictor: Predictor):
    result = predictor.predict(SAMPLE_APPLICANT)

    assert result["track"] == predictor.run["track"]
    assert result["dataset"] == predictor.run["dataset"]


def test_missing_feature_raises_key_error(predictor: Predictor):
    incomplete = {k: v for k, v in SAMPLE_APPLICANT.items() if k != "duration"}

    with pytest.raises(KeyError, match="duration"):
        predictor.predict(incomplete)


def test_missing_artifact_raises_file_not_found():
    from pathlib import Path

    with pytest.raises(FileNotFoundError, match="not found"):
        Predictor(artifact_path=Path("nonexistent/model.joblib"))


def test_no_group_keyed_threshold(predictor: Predictor):
    """Finding B7: the run record's threshold must be a scalar, not a dict."""
    assert isinstance(predictor.threshold, float)


def test_predictor_spec_excludes_prohibited_bases(predictor: Predictor):
    """No prohibited basis may appear in the feature columns that reach the model."""
    prohibited = {
        attr.column for attr in predictor.spec.protected if attr.prohibited_basis
    }
    features = set(predictor.spec.feature_columns)
    assert prohibited.isdisjoint(features)



@pytest.mark.parametrize("attribute", ["gender", "foreign_worker"])
def test_prohibited_basis_input_is_refused(predictor: Predictor, attribute: str):
    """Refusing is not the same as ignoring.

    `extract_features` selects only the registry's feature columns, so a supplied `gender`
    would be dropped without complaint and the caller could reasonably believe the model
    had considered it. The contract is that this service does not accept a prohibited
    basis at all.
    """
    with pytest.raises(ValueError, match="prohibited-basis attributes are not accepted"):
        predictor.predict({**SAMPLE_APPLICANT, attribute: 1})


def test_the_sex_proxy_is_also_refused(predictor: Predictor):
    """`personal_status_sex` code 1 is the only female code in the dataset, so the column
    identifies sex exactly. Finding B5."""
    with pytest.raises(ValueError, match="personal_status_sex"):
        predictor.predict({**SAMPLE_APPLICANT, "personal_status_sex": 1})
