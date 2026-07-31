"""Integration tests for the FastAPI credit risk API.

Uses TestClient (no live server, no network). Asserts response fields, not just status
codes.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas.prediction import CreditApplicationRequest
from src.paths import REPORTS_DIR
from src.serving.predictor import Predictor

pytestmark = pytest.mark.integration

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
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor()


# --- /health ---


def test_health_reports_loaded_track(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["artifact_loaded"] is True
    assert body["track"] == "T0"
    assert body["dataset"] == "german_credit"


# --- /predict ---


def test_predict_returns_decision_fields(client: TestClient):
    response = client.post("/predict", json=SAMPLE_APPLICANT)
    assert response.status_code == 200
    body = response.json()

    assert "probability_of_default" in body
    assert "decision" in body
    assert body["decision"] in ("approve", "decline")
    assert "threshold" in body
    assert "track" in body
    assert "dataset" in body
    assert 0.0 <= body["probability_of_default"] <= 1.0


def test_predict_decision_consistent_with_threshold(client: TestClient):
    response = client.post("/predict", json=SAMPLE_APPLICANT)
    body = response.json()

    if body["probability_of_default"] >= body["threshold"]:
        assert body["decision"] == "decline"
    else:
        assert body["decision"] == "approve"


def test_predict_422_names_offending_field(client: TestClient):
    """A missing required field produces a 422 that identifies it."""
    bad_payload = {k: v for k, v in SAMPLE_APPLICANT.items() if k != "amount"}
    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422
    body = response.json()
    # Pydantic v2 returns the field path in detail[*].loc
    field_names = [
        err["loc"][-1] for err in body["detail"] if "loc" in err
    ]
    assert "amount" in field_names


def test_predict_422_on_out_of_range(client: TestClient):
    bad = {**SAMPLE_APPLICANT, "age": 10}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# --- /metrics ---


def test_metrics_matches_on_disk_artifact(client: TestClient):
    """Published numbers come from the comparison artifact, never hardcoded."""
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()

    comparison = json.loads(
        (REPORTS_DIR / "track_comparison.json").read_text(encoding="utf-8")
    )
    run = comparison["runs"]["german_credit|T0"]

    assert body["track"] == "T0"
    assert body["performance"] == run["performance"]
    assert body["fairness"] == run["fairness"]
    assert body["intervals"] == run["intervals"]


# --- /model-info ---


def test_model_info_fields(client: TestClient):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()

    assert body["model_type"] == "xgboost"
    assert isinstance(body["params"], dict)
    assert body["n_encoded_features"] == 47
    assert "split_fingerprint" in body
    assert body["track"] == "T0"


# --- schema prohibition ---


def test_request_schema_excludes_prohibited_bases():
    """Finding B7: no protected attribute may appear in the request schema."""
    prohibited = {"gender", "personal_status_sex", "foreign_worker"}
    schema_fields = set(CreditApplicationRequest.model_fields.keys())

    assert prohibited.isdisjoint(schema_fields), (
        f"Prohibited bases in request schema: {prohibited & schema_fields}"
    )


# --- single inference path (Finding B6) ---


def test_api_and_predictor_return_identical_probability(
    client: TestClient, predictor: Predictor
):
    """The API and the direct Predictor call must produce identical results.

    This is the structural guarantee that app.py (Streamlit) and the API cannot diverge:
    both use Predictor.predict, and no other code path touches the model.
    """
    api_response = client.post("/predict", json=SAMPLE_APPLICANT)
    api_prob = api_response.json()["probability_of_default"]

    direct = predictor.predict(SAMPLE_APPLICANT)
    direct_prob = direct["probability_of_default"]

    assert abs(api_prob - direct_prob) < 1e-9


# --- SHAP reason codes ---


def test_predict_with_reasons_endpoint(client: TestClient):
    """POST /predict?include_reasons=true returns reason codes on decline."""
    # Build a risky applicant likely to be declined.
    risky = {
        **SAMPLE_APPLICANT,
        "status": 3,
        "savings": 4,
        "employment_duration": 0,
        "amount": 18000,
        "duration": 60,
        "credit_history": 0,
    }
    response = client.post("/predict?include_reasons=true", json=risky)
    assert response.status_code == 200
    body = response.json()

    if body["decision"] == "decline":
        assert body["reason_codes"] is not None
        assert len(body["reason_codes"]) == 4
        for reason in body["reason_codes"]:
            assert "feature" in reason
            assert "shap_value" in reason
            assert "direction" in reason
            assert "contribution" in reason
    else:
        assert body["reason_codes"] is None


def test_predict_without_reasons_has_no_reason_codes(client: TestClient):
    """Default /predict does not include reason_codes field."""
    response = client.post("/predict", json=SAMPLE_APPLICANT)
    body = response.json()

    assert body.get("reason_codes") is None


def test_explain_endpoint(client: TestClient):
    """POST /explain always returns reason codes regardless of decision."""
    response = client.post("/explain", json=SAMPLE_APPLICANT)
    assert response.status_code == 200
    body = response.json()

    assert "reason_codes" in body
    assert body["reason_codes"] is not None
    assert len(body["reason_codes"]) == 4
    assert body["decision"] in ("approve", "decline")
    for reason in body["reason_codes"]:
        assert reason["direction"] in ("increases risk", "decreases risk", "neutral")

