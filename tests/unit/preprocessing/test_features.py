"""Tests for the registry-driven encoder.

Widths and means are hand-derived from the codebook domains declared in the registry, so a
change to either the roles or the domains fails here rather than silently altering every
downstream metric.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.registry import GERMAN_CREDIT, TAIWAN_CREDIT
from src.preprocessing.features import (
    build_encoder,
    encoded_feature_names,
    extract_features,
    extract_groups,
    extract_target,
)

pytestmark = pytest.mark.unit

# 4 status + 5 credit_history + 10 purpose + 5 savings + 3 other_debtors + 4 property
# + 3 other_installment_plans + 3 housing + 1 telephone (binary, dropped to one column)
GERMAN_ONEHOT_WIDTH = 4 + 5 + 10 + 5 + 3 + 4 + 3 + 3 + 1
GERMAN_SCALED_WIDTH = 5 + 4  # numeric + ordinal
GERMAN_ENCODED_WIDTH = GERMAN_SCALED_WIDTH + GERMAN_ONEHOT_WIDTH


@pytest.fixture(scope="module")
def german() -> pd.DataFrame:
    return GERMAN_CREDIT.load()


def test_raw_feature_frame_holds_only_registry_features(german):
    features = extract_features(GERMAN_CREDIT, german)

    assert list(features.columns) == list(GERMAN_CREDIT.feature_columns)
    assert features.shape == (1000, 18)


def test_encoding_expands_categoricals_instead_of_treating_them_as_continuous(german):
    """Finding B8 regression.

    18 raw columns become 47 encoded ones. If a future change reverts to label-encoded
    integers the width collapses back to 18 and this fails.
    """
    encoder = build_encoder(GERMAN_CREDIT)
    encoded = encoder.fit_transform(extract_features(GERMAN_CREDIT, german))

    assert encoded.shape == (1000, GERMAN_ENCODED_WIDTH)
    assert GERMAN_ENCODED_WIDTH == 47
    assert encoded.shape[1] > 18


def test_no_encoded_column_identifies_sex(german):
    """Finding B5 regression, stated as the property that actually matters.

    Rather than checking a column name is absent, this asserts that no encoded column is a
    deterministic function of sex. `personal_status_sex` code 1 held for exactly the 310
    female rows, so had it survived encoding, one one-hot column would match the female
    mask exactly.
    """
    encoder = build_encoder(GERMAN_CREDIT)
    encoded = encoder.fit_transform(extract_features(GERMAN_CREDIT, german))
    is_female = extract_groups(GERMAN_CREDIT, german) == 0

    for index, name in enumerate(encoded_feature_names(encoder)):
        column = encoded[:, index]
        for value in np.unique(column):
            matches_group = column == value
            assert not np.array_equal(matches_group, is_female), (
                f"{name} == {value} identifies the female group exactly"
            )
            assert not np.array_equal(matches_group, ~is_female), (
                f"{name} == {value} identifies the male group exactly"
            )


def test_one_hot_rows_sum_to_one_per_nominal_column(german):
    """A row must land in exactly one category of each nominal column.

    Catches a domain declared with a duplicate or a missing value, which would otherwise
    show up much later as a quietly degraded model.
    """
    encoder = build_encoder(GERMAN_CREDIT)
    encoded = encoder.fit_transform(extract_features(GERMAN_CREDIT, german))
    names = encoded_feature_names(encoder)

    # telephone is binary and reduced to a single indicator, so it is excluded here.
    for column in [c for c in GERMAN_CREDIT.nominal if c != "telephone"]:
        indices = [i for i, name in enumerate(names) if name.startswith(f"{column}_")]
        assert len(indices) == len(GERMAN_CREDIT.nominal_categories[column]), column
        assert np.allclose(encoded[:, indices].sum(axis=1), 1.0), column


def test_scaler_statistics_come_only_from_the_rows_it_was_fitted_on(german):
    """Leakage guard.

    The scaler's mean must equal the mean of the fitted subset, and must differ from the
    full-dataset mean. Fitting on everything and then splitting is the classic version of
    this bug.
    """
    train = german.iloc[:600]
    encoder = build_encoder(GERMAN_CREDIT)
    encoder.fit(extract_features(GERMAN_CREDIT, train))

    scaler = encoder.named_steps["encode"].named_transformers_["scaled"]
    scaled_columns = list(GERMAN_CREDIT.numeric + GERMAN_CREDIT.ordinal)
    amount_index = scaled_columns.index("amount")

    assert scaler.mean_[amount_index] == pytest.approx(train["amount"].mean())
    assert scaler.mean_[amount_index] != pytest.approx(german["amount"].mean())


def test_transform_width_is_independent_of_which_rows_were_fitted(german):
    """Track comparability requirement.

    Fitting on a subset that happens to miss a rare category must not change the encoded
    width, because the domain is declared rather than observed. `purpose` code 6 (repairs)
    is rare, so a small head slice is a realistic way to lose a category.
    """
    small = german.iloc[:40]
    encoder = build_encoder(GERMAN_CREDIT)
    encoder.fit(extract_features(GERMAN_CREDIT, small))

    assert small["purpose"].nunique() < german["purpose"].nunique()
    encoded_all = encoder.transform(extract_features(GERMAN_CREDIT, german))
    assert encoded_all.shape == (1000, GERMAN_ENCODED_WIDTH)


def test_out_of_domain_category_raises_instead_of_encoding_as_zeros(german):
    encoder = build_encoder(GERMAN_CREDIT)
    encoder.fit(extract_features(GERMAN_CREDIT, german))

    corrupted = german.copy()
    corrupted.loc[0, "purpose"] = 99

    with pytest.raises(ValueError, match="Found unknown categories"):
        encoder.transform(extract_features(GERMAN_CREDIT, corrupted))


def test_target_and_groups_are_extracted_with_the_registry_encoding(german):
    y = extract_target(GERMAN_CREDIT, german)
    groups = extract_groups(GERMAN_CREDIT, german)

    assert int((y == GERMAN_CREDIT.favorable_label).sum()) == 700
    assert int((groups == GERMAN_CREDIT.protected_attribute().privileged_value).sum()) == 690


def test_groups_cannot_be_read_from_a_non_protected_column(german):
    with pytest.raises(KeyError, match="not a protected attribute"):
        extract_groups(GERMAN_CREDIT, german, "amount")


def test_taiwan_folds_undocumented_codes_inside_the_pipeline():
    """The fold must be part of the fitted pipeline, not a preparation step.

    A raw EDUCATION code of 5 has to survive transform by being folded to 4, otherwise
    serving would raise on rows the training data contained.
    """
    frame = pd.DataFrame(
        {
            **{column: [0.0, 1.0] for column in TAIWAN_CREDIT.numeric},
            **{column: [0, 1] for column in TAIWAN_CREDIT.ordinal},
            "EDUCATION": [5, 2],
            "MARRIAGE": [0, 1],
        }
    )
    encoder = build_encoder(TAIWAN_CREDIT)
    encoded = encoder.fit_transform(extract_features(TAIWAN_CREDIT, frame))
    names = encoded_feature_names(encoder)

    education_4 = names.index("EDUCATION_4")
    marriage_3 = names.index("MARRIAGE_3")
    assert encoded[0, education_4] == 1.0
    assert encoded[0, marriage_3] == 1.0
    # 14 numeric + 6 ordinal scaled, 4 EDUCATION + 3 MARRIAGE one-hot.
    assert encoded.shape == (2, 14 + 6 + 4 + 3)
