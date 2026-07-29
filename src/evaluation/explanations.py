"""Per-decision feature attributions for adverse-action reason codes.

When a credit application is declined, Regulation B (12 CFR 1002.9) requires the lender
to state the principal reasons. SHAP values decompose the model's score into per-feature
contributions, so the top contributors become the adverse-action reasons.

The explainer operates on the *encoded* feature matrix (47 columns for German Credit),
not on the raw input. This matches the model's actual decision surface, so the explanation
cannot disagree with the prediction. Feature names come from
``src.preprocessing.features.encoded_feature_names``, which was built for this purpose.

Two explainer types are used:

- ``TreeExplainer`` for tree-based models (RandomForest, XGBoost, LightGBM). Fast,
  exact, and deterministic.
- ``KernelExplainer`` for logistic regression. Uses a kmeans summary of the background
  data to keep computation tractable.

The explainer is built against the **classifier step only** of the fitted pipeline, after
the encoder has already transformed the data. This avoids recomputing the encoding inside
SHAP's perturbation loop, which would be both slow and incorrect (the encoder is not
differentiable, and SHAP's masking assumptions do not hold for one-hot blocks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ECOA guidance: "no more than four" principal reasons for adverse action.
DEFAULT_N_REASONS = 4

# Number of background rows kept for inference-time explanations. Enough for a stable
# expected value without shipping the full training block.
BACKGROUND_SAMPLE_SIZE = 200

_TREE_TYPES = frozenset({
    "RandomForestClassifier",
    "XGBClassifier",
    "LGBMClassifier",
})


@dataclass(frozen=True)
class ReasonCode:
    """One line of an adverse-action notice.

    ``feature`` is the encoded column name, ``shap_value`` is its signed contribution to
    the score, and ``direction`` states whether it pushed the risk higher or lower.
    ``contribution`` is the human-readable sentence a notice would carry.
    """

    feature: str
    shap_value: float
    direction: str
    contribution: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "shap_value": self.shap_value,
            "direction": self.direction,
            "contribution": self.contribution,
        }


def _classifier_type_name(estimator: Pipeline) -> str:
    """Name of the classifier class inside a pipeline."""
    return type(estimator.named_steps["classifier"]).__name__


def _is_tree_model(estimator: Pipeline) -> bool:
    return _classifier_type_name(estimator) in _TREE_TYPES


def build_explainer(
    estimator: Pipeline,
    X_background_encoded: NDArray[np.float64],
) -> Any:
    """Create a SHAP explainer for the classifier step of a fitted pipeline.

    ``X_background_encoded`` must already be transformed by the pipeline's encoder step.
    For tree models, ``TreeExplainer`` is used and the background is the training data
    that sets the expected value. For linear models, ``KernelExplainer`` uses a kmeans
    summary to keep the perturbation count tractable.

    Raises:
        ValueError: If the pipeline has no ``classifier`` step.
    """
    import shap

    classifier = estimator.named_steps["classifier"]

    if _is_tree_model(estimator):
        return shap.TreeExplainer(
            classifier,
            data=X_background_encoded,
            model_output="probability",
        )

    # KernelExplainer for logistic regression or any non-tree model.
    n_summary = min(50, len(X_background_encoded))
    summary = shap.kmeans(X_background_encoded, n_summary)
    return shap.KernelExplainer(classifier.predict_proba, summary)


def _direction_label(shap_value: float) -> str:
    """Human-readable direction for a SHAP contribution to the default score."""
    if shap_value > 0:
        return "increases risk"
    if shap_value < 0:
        return "decreases risk"
    return "neutral"


def _contribution_sentence(feature: str, direction: str) -> str:
    """Plain-language reason for an adverse-action notice."""
    clean = feature.replace("_", " ")
    if direction == "increases risk":
        return f"{clean} pushed the risk score higher"
    if direction == "decreases risk":
        return f"{clean} pushed the risk score lower"
    return f"{clean} had no effect on the risk score"


def explain_single(
    explainer: Any,
    encoded_row: NDArray[np.float64],
    feature_names: list[str],
    *,
    n_reasons: int = DEFAULT_N_REASONS,
) -> list[ReasonCode]:
    """Top-N reason codes for a single applicant.

    ``encoded_row`` is a 1-D or 2-D array of the encoded features for one row.
    SHAP values are for the positive class (default), so a positive value means the
    feature pushed the model toward predicting default.

    Returns:
        A list of ``ReasonCode`` sorted by descending absolute SHAP value, length
        ``min(n_reasons, len(feature_names))``.
    """
    import shap

    row = np.atleast_2d(encoded_row)

    if isinstance(explainer, shap.TreeExplainer):
        shap_values = explainer.shap_values(row, check_additivity=False)
        # TreeExplainer with model_output="probability" returns an array per row.
        # For binary classification it may return a list of two arrays (one per class)
        # or a single array. We want the positive-class (index 1) contributions.
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1]).ravel()
        else:
            values = np.asarray(shap_values).ravel()
    else:
        # KernelExplainer returns [n_samples, n_features, n_classes] or
        # [n_samples, n_features] depending on the model.
        shap_values = explainer.shap_values(row, silent=True)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1]).ravel()
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                values = arr[0, :, 1]
            else:
                values = arr.ravel()

    if values.shape[0] != len(feature_names):
        raise ValueError(
            f"SHAP values length {values.shape[0]} does not match "
            f"feature_names length {len(feature_names)}"
        )

    # Sort by absolute contribution, descending.
    ranked = np.argsort(-np.abs(values))
    count = min(n_reasons, len(feature_names))

    reasons: list[ReasonCode] = []
    for idx in ranked[:count]:
        sv = float(values[idx])
        direction = _direction_label(sv)
        reasons.append(
            ReasonCode(
                feature=feature_names[idx],
                shap_value=sv,
                direction=direction,
                contribution=_contribution_sentence(feature_names[idx], direction),
            )
        )
    return reasons


def explain_batch(
    explainer: Any,
    X_encoded: NDArray[np.float64],
    feature_names: list[str],
) -> NDArray[np.float64]:
    """SHAP values for a batch of rows, for summary statistics and plots.

    Returns:
        Array of shape ``(n_samples, n_features)`` with positive-class SHAP values.
    """
    import shap

    if isinstance(explainer, shap.TreeExplainer):
        shap_values = explainer.shap_values(X_encoded, check_additivity=False)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1])
        else:
            values = np.asarray(shap_values)
    else:
        shap_values = explainer.shap_values(X_encoded, silent=True)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1])
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                values = arr[:, :, 1]
            else:
                values = arr

    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP values width {values.shape[1]} does not match "
            f"feature_names length {len(feature_names)}"
        )
    return values


def global_feature_importance(
    shap_values: NDArray[np.float64],
    feature_names: list[str],
    *,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Mean absolute SHAP value per feature, top N, for the comparison artifact.

    Returns:
        A list of dicts with ``feature`` and ``mean_abs_shap``, sorted descending.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranked = np.argsort(-mean_abs)
    count = min(top_n, len(feature_names))

    return [
        {
            "feature": feature_names[int(idx)],
            "mean_abs_shap": float(mean_abs[idx]),
        }
        for idx in ranked[:count]
    ]


def sample_background(
    X_encoded: NDArray[np.float64],
    *,
    n_samples: int = BACKGROUND_SAMPLE_SIZE,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Subsample encoded training rows for the explainer background.

    Stored in the track artifact so inference-time explanations do not need the full
    training block.
    """
    if len(X_encoded) <= n_samples:
        return X_encoded.copy()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X_encoded), size=n_samples, replace=False)
    return X_encoded[np.sort(indices)]
