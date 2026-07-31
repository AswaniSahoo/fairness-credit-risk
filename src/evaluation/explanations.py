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
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from src.training.search import POSITIVE_CLASS

logger = logging.getLogger(__name__)

# ECOA guidance: "no more than four" principal reasons for adverse action.
DEFAULT_N_REASONS = 4

# Number of background rows kept for inference-time explanations. Enough for a stable
# expected value without shipping the full training block.
BACKGROUND_SAMPLE_SIZE = 200

# Rows explained when estimating global feature importance. Interventional TreeExplainer
# costs O(rows x background rows x trees): measured at 0.019 s/row on the Taiwan LightGBM,
# so explaining all 18,000 training rows takes 350 s per track against 10 s for this sample.
# Mean absolute SHAP is an average over rows, so the extra precision buys nothing a ranking
# would show.
IMPORTANCE_SAMPLE_SIZE = 500

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


def _positive_class_shap(raw: Any, n_features: int) -> NDArray[np.float64]:
    """Reduce any SHAP return shape to positive-class values of shape (n_rows, n_features).

    shap 0.46 returns one ``(n_rows, n_features, n_classes)`` array for a binary classifier;
    earlier versions returned a list holding one array per class. Both shapes are handled
    here rather than at each call site, because the two failure modes differ and only one of
    them is loud: flattening the 3-D array raises on the width check, while passing it
    through unreduced satisfies a ``shape[1]`` check and yields a 3-D array that
    ``global_feature_importance`` then ranks as though it were per-feature.

    Column ``POSITIVE_CLASS`` is the default class. ``classes_`` is sorted ascending and is
    asserted against that constant in ``src.training.search``, so index and label agree.

    Raises:
        ValueError: If the reduced width does not match ``n_features``.
    """
    if isinstance(raw, list):
        values = np.asarray(raw[POSITIVE_CLASS], dtype=np.float64)
    else:
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim == 3:
            values = values[:, :, POSITIVE_CLASS]

    values = np.atleast_2d(values)

    if values.ndim != 2 or values.shape[1] != n_features:
        raise ValueError(
            f"SHAP values reduced to shape {values.shape}, expected (n_rows, {n_features}). "
            f"Raw SHAP output had shape {np.asarray(raw).shape}."
        )
    return values


def _direction_label(shap_value: float) -> str:
    """Human-readable direction for a SHAP contribution to the default score."""
    if shap_value > 0:
        return "increases risk"
    if shap_value < 0:
        return "decreases risk"
    return "neutral"


def _contribution_sentence(
    feature: str,
    direction: str,
    *,
    value: float | None = None,
    is_indicator: bool = False,
) -> str:
    """Plain-language reason for an adverse-action notice.

    One-hot columns are phrased by whether the category applied to this application. A
    positive attribution on an absent category is common and meaningful - not holding the
    best checking-account status raises the score - but stating it as though the applicant
    held that status would misstate a principal reason under Regulation B.
    """
    if direction == "neutral":
        return f"{feature} had no effect on the risk score"

    push = "higher" if direction == "increases risk" else "lower"

    if is_indicator:
        if value is not None and value >= 0.5:
            return f"{feature} applied to this application and pushed the risk score {push}"
        return (
            f"{feature} did not apply to this application, and its absence pushed the "
            f"risk score {push}"
        )

    return f"{feature.replace('_', ' ')} pushed the risk score {push}"


def explain_single(
    explainer: Any,
    encoded_row: NDArray[np.float64],
    feature_names: list[str],
    *,
    n_reasons: int = DEFAULT_N_REASONS,
    indicator_features: Collection[str] = (),
) -> list[ReasonCode]:
    """Top-N reason codes for a single applicant.

    ``encoded_row`` is a 1-D or 2-D array of the encoded features for one row.
    SHAP values are for the positive class (default), so a positive value means the
    feature pushed the model toward predicting default.

    ``indicator_features`` names the one-hot columns, from
    ``src.preprocessing.features.indicator_feature_names``. Reason codes for those columns
    are phrased by whether the category applied to this application, because an attribution
    on an absent category otherwise reads as though the applicant held it.

    Returns:
        A list of ``ReasonCode`` sorted by descending absolute SHAP value, length
        ``min(n_reasons, len(feature_names))``.
    """
    import shap

    row = np.atleast_2d(encoded_row)

    if isinstance(explainer, shap.TreeExplainer):
        raw = explainer.shap_values(row, check_additivity=False)
    else:
        raw = explainer.shap_values(row, silent=True)

    values = _positive_class_shap(raw, len(feature_names))[0]

    # Sort by absolute contribution, descending.
    ranked = np.argsort(-np.abs(values))
    count = min(n_reasons, len(feature_names))

    indicators = frozenset(indicator_features)

    reasons: list[ReasonCode] = []
    for idx in ranked[:count]:
        name = feature_names[idx]
        sv = float(values[idx])
        direction = _direction_label(sv)
        reasons.append(
            ReasonCode(
                feature=name,
                shap_value=sv,
                direction=direction,
                contribution=_contribution_sentence(
                    name,
                    direction,
                    value=float(row[0, idx]),
                    is_indicator=name in indicators,
                ),
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
        raw = explainer.shap_values(X_encoded, check_additivity=False)
    else:
        raw = explainer.shap_values(X_encoded, silent=True)

    values = _positive_class_shap(raw, len(feature_names))

    if values.shape[0] != len(X_encoded):
        raise ValueError(
            f"SHAP values cover {values.shape[0]} rows, expected {len(X_encoded)}"
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
