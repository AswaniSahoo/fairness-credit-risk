"""Provenance tests for the processed German Credit file.

`data/processed/german_credit_numerical_final.csv` holds integers with no record of which
UCI code produced each one. Every downstream decision depends on that mapping: whether a
column may be treated as ordinal, which value is the favorable label, and which value
identifies the unprivileged group. These tests assert the mapping instead of trusting it.

Expected values are derived from `data/raw/german.doc` and from the observation that the
processed file was produced by alphabetical label encoding, so `A11 < A12 < A13 < A14`
becomes `0 < 1 < 2 < 3`. `purpose` is the one place where alphabetical order diverges from
codebook order, because `A410` sorts between `A41` and `A42`; that column is therefore
nominal and its integers carry no order.
"""

import pandas as pd
import pytest

from src.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

pytestmark = pytest.mark.integration

RAW_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "amount", "savings",
    "employment_duration", "installment_rate", "personal_status_sex", "other_debtors",
    "present_residence", "property", "age", "other_installment_plans", "housing",
    "number_credits", "job", "people_liable", "telephone", "foreign_worker",
    "credit_risk",
]

# Hand-derived from german.doc. Ordinal claims later made in the registry rest on these.
EXPECTED_CODE_MAPS = {
    "status": {"A11": 0, "A12": 1, "A13": 2, "A14": 3},
    "credit_history": {"A30": 0, "A31": 1, "A32": 2, "A33": 3, "A34": 4},
    "purpose": {
        "A40": 0, "A41": 1, "A410": 2, "A42": 3, "A43": 4,
        "A44": 5, "A45": 6, "A46": 7, "A48": 8, "A49": 9,
    },
    "savings": {"A61": 0, "A62": 1, "A63": 2, "A64": 3, "A65": 4},
    "employment_duration": {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4},
    "personal_status_sex": {"A91": 0, "A92": 1, "A93": 2, "A94": 3},
    "other_debtors": {"A101": 0, "A102": 1, "A103": 2},
    "property": {"A121": 0, "A122": 1, "A123": 2, "A124": 3},
    "other_installment_plans": {"A141": 0, "A142": 1, "A143": 2},
    "housing": {"A151": 0, "A152": 1, "A153": 2},
    "job": {"A171": 0, "A172": 1, "A173": 2, "A174": 3},
    "telephone": {"A191": 0, "A192": 1},
    "foreign_worker": {"A201": 0, "A202": 1},
}

PASSTHROUGH_COLUMNS = [
    "duration", "amount", "installment_rate", "present_residence", "age",
    "number_credits", "people_liable",
]


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "german.data", sep=" ", header=None, names=RAW_COLUMNS)


@pytest.fixture(scope="module")
def processed() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_DIR / "german_credit_numerical_final.csv")


def test_processed_file_has_the_documented_shape(processed):
    # UCI publishes 1000 instances and 20 attributes plus the target. The processed file
    # adds `gender` and `age_group`, both derived, giving 23 columns.
    assert processed.shape == (1000, 23)


def test_row_order_is_preserved_against_the_raw_file(raw, processed):
    # Every mapping assertion below compares row-wise, so a reordered file would make
    # them meaningless rather than failing loudly on its own.
    assert len(raw) == len(processed)
    assert processed["duration"].tolist() == raw["duration"].tolist()
    assert processed["amount"].tolist() == raw["amount"].tolist()


@pytest.mark.parametrize(("column", "code_map"), sorted(EXPECTED_CODE_MAPS.items()))
def test_categorical_codes_map_as_documented(raw, processed, column, code_map):
    expected = raw[column].map(code_map)
    assert expected.isna().sum() == 0, f"unmapped codes in {column}"
    assert expected.tolist() == processed[column].tolist()


@pytest.mark.parametrize("column", PASSTHROUGH_COLUMNS)
def test_numeric_columns_pass_through_unchanged(raw, processed, column):
    assert processed[column].tolist() == raw[column].tolist()


def test_target_is_recoded_so_good_credit_is_zero(raw, processed):
    # german.doc: 1 = good, 2 = bad. The processed file uses 0 = good, so the favorable
    # label is 0 and the positive (default) class is 1.
    assert processed["credit_risk"].tolist() == (raw["credit_risk"] - 1).tolist()
    assert int((processed["credit_risk"] == 0).sum()) == 700
    assert int((processed["credit_risk"] == 1).sum()) == 300


def test_gender_is_derived_from_attribute_nine_with_male_as_one(raw, processed):
    # A92 and A95 are the two female codes; A95 does not occur in this dataset.
    female = raw["personal_status_sex"].isin(["A92", "A95"])
    assert (~female).astype(int).tolist() == processed["gender"].tolist()
    assert int((processed["gender"] == 1).sum()) == 690
    assert int((processed["gender"] == 0).sum()) == 310
    assert "A95" not in set(raw["personal_status_sex"])


def test_personal_status_sex_is_a_perfect_sex_proxy(processed):
    """Finding B5, stated as a property of the data rather than of the model.

    Code 1 is A92, the only female code present, so `personal_status_sex == 1` holds for
    every female row and no male row. Keeping the column as a feature therefore keeps sex
    in the model even when `gender` is dropped. This is the reason the registry marks it
    prohibited rather than merely redundant.
    """
    is_code_one = processed["personal_status_sex"] == 1
    is_female = processed["gender"] == 0

    assert is_code_one.equals(is_female)
    assert int(is_code_one.sum()) == 310


def test_foreign_worker_one_means_not_a_foreign_worker(raw, processed):
    # A201 = yes, A202 = no, encoded 0 and 1. The larger group is the foreign workers,
    # so a naive reading of "1 = yes" would invert the privileged group.
    assert processed.loc[raw["foreign_worker"] == "A201", "foreign_worker"].unique().tolist() == [0]
    assert int((processed["foreign_worker"] == 0).sum()) == 963


def test_label_only_disparity_reproduces_the_published_figures(processed):
    """Keeps the B1 evidence in the test suite.

    Disparate impact computed from the labels alone is 0.8966, and the previously
    published "model" fairness figures were 0.8903 and -0.0795. The old metric code was
    reporting the dataset's own disparity, not the model's.
    """
    favorable = processed["credit_risk"] == 0
    male_rate = favorable[processed["gender"] == 1].mean()
    female_rate = favorable[processed["gender"] == 0].mean()

    assert male_rate == pytest.approx(0.7232, abs=5e-5)
    assert female_rate == pytest.approx(0.6484, abs=5e-5)
    assert female_rate / male_rate == pytest.approx(0.8966, abs=5e-5)
    assert female_rate - male_rate == pytest.approx(-0.0748, abs=5e-5)


def test_age_group_is_consistent_with_age(processed):
    # age_group is a derived convenience column and must not disagree with `age`.
    bounds = {"18-25": (18, 25), "26-35": (26, 35), "36-45": (36, 45),
              "46-55": (46, 55), "55+": (56, 200)}
    for group, (low, high) in bounds.items():
        ages = processed.loc[processed["age_group"] == group, "age"]
        assert ages.min() >= low, group
        assert ages.max() <= high, group
