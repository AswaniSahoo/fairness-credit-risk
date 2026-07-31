"""Unit tests for SHAP-based explanation reason codes."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.explanations import (
    DEFAULT_N_REASONS,
    build_explainer,
    explain_batch,
    explain_single,
    global_feature_importance,
    sample_background,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def tree_pipeline() -> Pipeline:
    """A small tree pipeline fitted on synthetic data."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=20, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="module")
def lr_pipeline() -> Pipeline:
    """A logistic regression pipeline fitted on synthetic data."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="module")
def background_data() -> np.ndarray:
    X, _ = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X)


@pytest.fixture(scope="module")
def feature_names() -> list[str]:
    return [f"feature_{i}" for i in range(10)]


def test_tree_explainer_returns_correct_count(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    explainer = build_explainer(tree_pipeline, background_data)
    row = background_data[0]
    reasons = explain_single(explainer, row, feature_names, n_reasons=4)

    assert len(reasons) == 4


def test_reason_codes_sorted_by_absolute_shap_value(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    explainer = build_explainer(tree_pipeline, background_data)
    row = background_data[0]
    reasons = explain_single(explainer, row, feature_names, n_reasons=10)

    abs_values = [abs(r.shap_value) for r in reasons]
    assert abs_values == sorted(abs_values, reverse=True)


def test_direction_matches_sign(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    explainer = build_explainer(tree_pipeline, background_data)
    row = background_data[0]
    reasons = explain_single(explainer, row, feature_names, n_reasons=10)

    for reason in reasons:
        if reason.shap_value > 0:
            assert reason.direction == "increases risk"
        elif reason.shap_value < 0:
            assert reason.direction == "decreases risk"
        else:
            assert reason.direction == "neutral"


def test_feature_names_match_encoded_columns(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    explainer = build_explainer(tree_pipeline, background_data)
    row = background_data[0]
    reasons = explain_single(explainer, row, feature_names)

    for reason in reasons:
        assert reason.feature in feature_names


def test_logistic_regression_uses_kernel_explainer(
    lr_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    """Non-tree models should still produce valid reason codes."""
    explainer = build_explainer(lr_pipeline, background_data)
    row = background_data[0]
    reasons = explain_single(explainer, row, feature_names, n_reasons=4)

    assert len(reasons) == 4
    for reason in reasons:
        assert reason.feature in feature_names
        assert isinstance(reason.shap_value, float)


def test_explain_single_on_approved_applicant(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    """Reasons are computable for any row, regardless of decision."""
    explainer = build_explainer(tree_pipeline, background_data)
    # Try all rows to find one with low positive-class probability.
    for i in range(len(background_data)):
        reasons = explain_single(explainer, background_data[i], feature_names)
        assert len(reasons) == DEFAULT_N_REASONS


def test_reason_code_as_dict_roundtrip(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    explainer = build_explainer(tree_pipeline, background_data)
    reasons = explain_single(explainer, background_data[0], feature_names)

    for reason in reasons:
        d = reason.as_dict()
        assert set(d) == {"feature", "shap_value", "direction", "contribution"}
        assert isinstance(d["shap_value"], float)
        assert isinstance(d["contribution"], str)


def test_explain_batch_reduces_the_class_axis_to_the_positive_class(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    """The class axis must be dropped, not flattened or carried through.

    shap 0.46 returns (n_rows, n_features, n_classes) for a binary classifier. A
    width check on shape[1] passes for that array because shape[1] is already
    n_features, so an unreduced 3-D array reaches global_feature_importance and is
    ranked as though its last axis were features. This asserts the reduction happened
    and that it selected class 1, the default class.
    """
    explainer = build_explainer(tree_pipeline, background_data)
    rows = background_data[:5]

    values = explain_batch(explainer, rows, feature_names)

    assert values.ndim == 2
    assert values.shape == (5, len(feature_names))

    raw = np.asarray(explainer.shap_values(rows, check_additivity=False))
    assert raw.shape == (5, len(feature_names), 2)
    np.testing.assert_allclose(values, raw[:, :, 1])


def test_explain_single_matches_the_batch_row(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    """One row explained alone must equal that row explained in a batch."""
    explainer = build_explainer(tree_pipeline, background_data)

    reasons = explain_single(explainer, background_data[0], feature_names, n_reasons=10)
    batch = explain_batch(explainer, background_data[:1], feature_names)

    by_name = {name: batch[0, i] for i, name in enumerate(feature_names)}
    for reason in reasons:
        assert reason.shap_value == pytest.approx(by_name[reason.feature], abs=1e-12)


def test_explain_single_rejects_a_feature_name_count_mismatch(
    tree_pipeline: Pipeline, background_data: np.ndarray
):
    """A width disagreement must raise rather than rank the wrong names."""
    explainer = build_explainer(tree_pipeline, background_data)

    with pytest.raises(ValueError, match="expected"):
        explain_single(explainer, background_data[0], ["only_one_name"])


def test_indicator_reasons_state_whether_the_category_applied(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    """A one-hot column must never be described as though an absent category applied.

    Regulation B requires the principal reasons for the adverse action. An attribution on
    a category the applicant does not hold is legitimate and common, but reporting it as
    "status_3 pushed the risk score higher" reads as though the applicant held status 3.
    """
    explainer = build_explainer(tree_pipeline, background_data)

    row = background_data[0].copy()
    indicator = feature_names[0]
    # Feature 0 is treated as a one-hot column for this assertion; its value decides the
    # phrasing, so both settings are exercised against the same explainer.
    row[0] = 1.0
    present = explain_single(
        explainer, row, feature_names, n_reasons=10, indicator_features={indicator}
    )
    row[0] = 0.0
    absent = explain_single(
        explainer, row, feature_names, n_reasons=10, indicator_features={indicator}
    )

    present_reason = next(r for r in present if r.feature == indicator)
    absent_reason = next(r for r in absent if r.feature == indicator)

    assert "applied to this application" in present_reason.contribution
    assert "did not apply" not in present_reason.contribution
    assert "did not apply to this application" in absent_reason.contribution


def test_continuous_reasons_are_unaffected_by_the_indicator_phrasing(
    tree_pipeline: Pipeline, background_data: np.ndarray, feature_names: list[str]
):
    """A column that is not one-hot keeps the plain wording."""
    explainer = build_explainer(tree_pipeline, background_data)

    reasons = explain_single(
        explainer, background_data[0], feature_names, n_reasons=10, indicator_features=set()
    )

    for reason in reasons:
        assert "applied to this application" not in reason.contribution
        if reason.direction != "neutral":
            assert reason.contribution.endswith(("higher", "lower"))


def test_global_feature_importance_sorted():
    rng = np.random.default_rng(42)
    shap_values = rng.standard_normal((100, 10))
    names = [f"f_{i}" for i in range(10)]

    importance = global_feature_importance(shap_values, names, top_n=5)

    assert len(importance) == 5
    mean_abs = [entry["mean_abs_shap"] for entry in importance]
    assert mean_abs == sorted(mean_abs, reverse=True)


def test_global_feature_importance_top_n_capped():
    rng = np.random.default_rng(42)
    shap_values = rng.standard_normal((100, 3))
    names = ["a", "b", "c"]

    importance = global_feature_importance(shap_values, names, top_n=10)

    assert len(importance) == 3  # capped at actual number of features


def test_sample_background_respects_size():
    data = np.random.default_rng(42).standard_normal((500, 10))

    bg = sample_background(data, n_samples=100, seed=42)

    assert bg.shape == (100, 10)


def test_sample_background_returns_copy_when_small():
    data = np.random.default_rng(42).standard_normal((50, 10))

    bg = sample_background(data, n_samples=200, seed=42)

    assert bg.shape == (50, 10)
    assert bg is not data  # must be a copy
