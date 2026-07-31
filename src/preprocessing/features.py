"""Feature extraction and encoding, driven by the dataset registry.

Replaces the previous approach, which fed label-encoded integers straight to the model.
That treated `purpose` code 9 as nine times code 1 and placed "no checking account"
(`status` 3) at the top of an ordered scale it does not belong to. Trees can carve such a
column at arbitrary split points and partially recover, but a linear model cannot, and
neither can express a non-monotone category effect cleanly. This was finding B8.

Three rules are enforced structurally rather than by convention:

1. The feature matrix is built from ``spec.feature_columns`` only. A prohibited-basis
   column cannot reach the model, because the registry refuses to validate a spec that
   lists one as a feature.
2. Nominal category domains come from the codebook, so the encoded width is identical for
   every track and every split. A comparison across tracks is otherwise not holding the
   feature space fixed.
3. All fitted statistics come from whatever rows the caller fits on. Nothing in this module
   reads the full dataset, so fitting on the training split cannot leak calibration or
   test statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.registry import DatasetSpec


class CategoryFolder(BaseEstimator, TransformerMixin):
    """Apply the registry's undocumented-code folds before encoding.

    Stateless: it learns nothing, so it behaves identically on train, calibration, test and
    a single inference row. It lives inside the pipeline rather than in a preparation
    script so that serving cannot forget to apply it.
    """

    def __init__(self, folds: dict[str, dict[int, int]] | None = None) -> None:
        # Stored exactly as passed. scikit-learn's clone contract requires that every
        # constructor argument round-trip unchanged through get_params, so normalising None
        # to {} here would make the estimator unclonable. fairlearn's ExponentiatedGradient
        # clones the pipeline on every iteration, so this is load-bearing, not pedantry.
        self.folds = folds

    def fit(self, X: pd.DataFrame, y: object = None) -> CategoryFolder:  # noqa: N803, ARG002
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        folded = X.copy()
        for column, mapping in (self.folds or {}).items():
            if column in folded.columns:
                folded[column] = folded[column].replace(mapping)
        return folded

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001, ANN201
        return np.asarray(input_features, dtype=object)


def build_encoder(spec: DatasetSpec) -> Pipeline:
    """Construct the unfitted encoder for a dataset.

    Numeric and ordinal columns are scaled. Ordinal columns are scaled rather than one-hot
    encoded because the codebook establishes their order, and discarding that order would
    throw away real information; see ``DatasetSpec`` for which columns qualify and why.
    Nominal columns are one-hot encoded over their declared domain, with binary columns
    reduced to a single indicator.

    ``handle_unknown='error'`` is deliberate. An out-of-domain category at inference time
    is a data problem that must surface, not something to silently encode as all-zeros.
    """
    scaled = spec.numeric + spec.ordinal
    categories = [list(spec.nominal_categories[column]) for column in spec.nominal]

    return Pipeline(
        [
            ("fold", CategoryFolder(spec.category_folds)),
            (
                "encode",
                ColumnTransformer(
                    [
                        ("scaled", StandardScaler(), list(scaled)),
                        (
                            "onehot",
                            OneHotEncoder(
                                categories=categories,
                                drop="if_binary",
                                handle_unknown="error",
                                sparse_output=False,
                                dtype=np.float64,
                            ),
                            list(spec.nominal),
                        ),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )


def encoded_feature_names(pipeline: Pipeline) -> list[str]:
    """Column names of the encoded matrix, for coefficient and SHAP reporting.

    Accepts either the encoder pipeline itself or the full fitted estimator that carries it
    as its ``encoder`` step. Both are passed by callers holding a fitted model, and requiring
    the caller to remember which one produced a bare ``KeyError: 'encode'`` from three
    separate call sites during the SHAP work. Descending one named step is unambiguous, so
    the function does it rather than making every caller do it.

    Raises:
        ValueError: If neither an ``encode`` step nor an ``encoder`` step is present.
        AttributeError: If the encoder has not been fitted.
    """
    return list(_encode_step(pipeline).get_feature_names_out())


def _encode_step(pipeline: Pipeline) -> ColumnTransformer:
    """The fitted ColumnTransformer, given the encoder pipeline or the full estimator."""
    steps = pipeline.named_steps
    if "encode" not in steps and "encoder" in steps:
        steps = pipeline.named_steps["encoder"].named_steps
    if "encode" not in steps:
        raise ValueError(
            f"no 'encode' step in {list(pipeline.named_steps)}; expected the encoder "
            "pipeline or an estimator with an 'encoder' step"
        )
    return steps["encode"]


def indicator_feature_names(pipeline: Pipeline) -> frozenset[str]:
    """Encoded columns produced by the one-hot branch, so 0 and 1 mean absent and present.

    A reason code for one of these columns must say whether the category applied to the
    application. Without that, a positive attribution on an absent category reads as though
    the applicant held it, which would misstate a principal reason under Regulation B.

    Raises:
        AttributeError: If the encoder has not been fitted.
    """
    onehot = _encode_step(pipeline).named_transformers_["onehot"]
    return frozenset(str(name) for name in onehot.get_feature_names_out())


def extract_features(spec: DatasetSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """Select the feature columns in registry order.

    Raises:
        KeyError: If a declared feature column is missing from ``frame``.
    """
    missing = [column for column in spec.feature_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{spec.name}: feature columns missing from frame: {missing}")
    return frame.loc[:, list(spec.feature_columns)]


def extract_target(spec: DatasetSpec, frame: pd.DataFrame) -> NDArray[np.int_]:
    """Return the target as an integer array."""
    return frame[spec.target].to_numpy(dtype=int)


def extract_groups(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    column: str | None = None,
) -> NDArray[np.int_]:
    """Return a protected attribute's values, for measurement only.

    Defaults to the dataset's primary protected attribute. Raises through
    ``spec.protected_attribute`` if ``column`` is not registered as protected, which stops
    an arbitrary feature being passed to a fairness metric by mistake.
    """
    attribute = spec.protected_attribute(column)
    return frame[attribute.column].to_numpy()
