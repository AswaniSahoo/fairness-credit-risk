"""Tests for the dataset registry.

The registry's job is to make two specific mistakes impossible: using a prohibited-basis
column as a feature, and applying one dataset's disparity direction to the other. Both are
asserted here against the real files, because a spec that agrees with itself but not with
the data on disk is worthless.
"""

import pandas as pd
import pytest

from src.data.registry import (
    DATASETS,
    GERMAN_CREDIT,
    TAIWAN_CREDIT,
    DatasetSpec,
    ProtectedAttribute,
    get_dataset,
)

pytestmark = pytest.mark.unit


def test_german_feature_columns_exclude_every_prohibited_basis():
    features = set(GERMAN_CREDIT.feature_columns)

    assert "gender" not in features
    assert "foreign_worker" not in features
    # Finding B5: the proxy must go too, or excluding `gender` achieves nothing.
    assert "personal_status_sex" not in features
    assert GERMAN_CREDIT.target not in features


def test_german_retains_age_as_a_feature():
    """Regulation B permits age in an empirically derived scoring system, unlike sex.

    The distinction is deliberate, so it is asserted rather than left to a comment that
    could drift away from the code.
    """
    assert "age" in GERMAN_CREDIT.numeric
    assert "age_group" in GERMAN_CREDIT.excluded


def test_german_feature_count_is_eighteen():
    # 5 numeric + 4 ordinal + 9 nominal. The old pipeline used 18 columns too, but
    # included the sex proxy and dropped `age`; this set trades one for the other.
    assert len(GERMAN_CREDIT.feature_columns) == 18
    assert len(set(GERMAN_CREDIT.feature_columns)) == 18


def test_german_favorable_label_is_good_credit():
    assert GERMAN_CREDIT.favorable_label == 0
    assert GERMAN_CREDIT.unfavorable_label == 1


def test_german_primary_protected_direction_is_male_privileged():
    attribute = GERMAN_CREDIT.protected_attribute()

    assert attribute.column == "gender"
    assert attribute.privileged_value == 1
    assert attribute.unprivileged_value == 0


def test_taiwan_direction_is_reversed_relative_to_german():
    """The case a global privileged-value constant gets wrong.

    Women are the majority and default less in the Taiwan data, so the privileged value is
    2 (female) there and 1 (male) in German Credit.
    """
    assert TAIWAN_CREDIT.protected_attribute().privileged_value == 2
    assert TAIWAN_CREDIT.protected_attribute().unprivileged_value == 1
    assert GERMAN_CREDIT.protected_attribute().privileged_value == 1


def test_taiwan_folds_the_undocumented_category_codes():
    # EDUCATION 0, 5 and 6 have no published meaning; they join the documented "other".
    assert TAIWAN_CREDIT.category_folds["EDUCATION"] == {0: 4, 5: 4, 6: 4}
    assert TAIWAN_CREDIT.category_folds["MARRIAGE"] == {0: 3}


def test_unknown_protected_attribute_raises():
    with pytest.raises(KeyError, match="not a protected attribute"):
        GERMAN_CREDIT.protected_attribute("nationality")


def test_unknown_dataset_lists_the_registered_names():
    with pytest.raises(KeyError, match="german_credit"):
        get_dataset("hmda")


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_every_registered_spec_matches_its_file_on_disk(name):
    """Catches role drift: a renamed or added column fails instead of silently changing
    the feature matrix width."""
    spec = DATASETS[name]
    frame = spec.load()

    assert set(spec.feature_columns) <= set(frame.columns)
    for attribute in spec.protected:
        assert attribute.privileged_value in set(frame[attribute.column])
        assert attribute.unprivileged_value in set(frame[attribute.column])


def _minimal_spec(**overrides) -> DatasetSpec:
    base = {
        "name": "toy",
        "path": GERMAN_CREDIT.path,
        "target": "y",
        "favorable_label": 0,
        "numeric": ("x",),
        "ordinal": (),
        "nominal": (),
        "excluded": ("g",),
        "protected": (
            ProtectedAttribute(
                column="g",
                privileged_value=1,
                unprivileged_value=0,
                prohibited_basis=True,
                rationale="toy",
            ),
        ),
        "primary_protected": "g",
        "provenance": "toy",
    }
    return DatasetSpec(**{**base, **overrides})


TOY_FRAME = pd.DataFrame({"x": [1, 2], "y": [0, 1], "g": [1, 0]})


def test_validate_accepts_a_consistent_spec():
    _minimal_spec().validate(TOY_FRAME)


def test_validate_rejects_an_undeclared_column():
    frame = TOY_FRAME.assign(surprise=[1, 2])
    with pytest.raises(ValueError, match="no declared role: \\['surprise'\\]"):
        _minimal_spec().validate(frame)


def test_validate_rejects_a_declared_column_absent_from_the_data():
    with pytest.raises(ValueError, match="absent from data"):
        _minimal_spec(numeric=("x", "ghost")).validate(TOY_FRAME)


def test_validate_rejects_a_column_in_two_roles():
    with pytest.raises(ValueError, match="two roles: \\['x'\\]"):
        _minimal_spec(nominal=("x",)).validate(TOY_FRAME)


def test_validate_rejects_a_prohibited_basis_used_as_a_feature():
    """The single most important guard in this module."""
    with pytest.raises(ValueError, match="prohibited basis used as a feature: \\['g'\\]"):
        _minimal_spec(numeric=("x", "g"), excluded=()).validate(TOY_FRAME)
