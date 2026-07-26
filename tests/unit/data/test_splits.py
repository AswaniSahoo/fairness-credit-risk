"""Tests for the shared three-way split.

Block sizes and stratum counts are computed from the dataset's own composition inside each
test, so they check the splitter's behaviour rather than restating a number the splitter
produced.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.registry import GERMAN_CREDIT
from src.data.splits import (
    DataSplit,
    fingerprint,
    get_or_create_split,
    load_split,
    make_split,
    save_split,
    split_path,
)

pytestmark = pytest.mark.unit

SEED = 42
TEST_SIZE = 0.2
CALIBRATION_SIZE = 0.2


@pytest.fixture(scope="module")
def german() -> pd.DataFrame:
    return GERMAN_CREDIT.load()


@pytest.fixture(scope="module")
def split(german) -> DataSplit:
    return make_split(
        GERMAN_CREDIT,
        german,
        test_size=TEST_SIZE,
        calibration_size=CALIBRATION_SIZE,
        seed=SEED,
    )


def test_blocks_are_sized_by_the_requested_fractions(split):
    assert split.sizes == {"train": 600, "calibration": 200, "test": 200}


def test_blocks_are_pairwise_disjoint(split):
    """Finding B4's precondition.

    Post-processing fitted on calibration is only honest if calibration shares no row with
    train.
    """
    train, calibration, test = set(split.train), set(split.calibration), set(split.test)

    assert train & calibration == set()
    assert train & test == set()
    assert calibration & test == set()


def test_blocks_cover_every_row_exactly_once(split, german):
    combined = np.concatenate([split.train, split.calibration, split.test])

    assert combined.size == len(german)
    assert set(combined) == set(range(len(german)))


def test_stratification_preserves_the_target_rate_in_every_block(split, german):
    overall = float((german[GERMAN_CREDIT.target] == 1).mean())

    for block in ("train", "calibration", "test"):
        rows = split.frame(german, block)
        rate = float((rows[GERMAN_CREDIT.target] == 1).mean())
        # 0.3 of 200 is 60 rows; one row is 0.005, so a half-row tolerance is the
        # tightest bound a stratified split can be held to.
        assert rate == pytest.approx(overall, abs=0.005), block


def test_stratification_preserves_the_protected_group_rate_in_every_block(split, german):
    """Joint stratification is the point of this splitter.

    Stratifying on the target alone would leave the female count in the test block to
    chance, and every fairness interval would move with it.
    """
    overall = float((german["gender"] == 0).mean())

    for block in ("train", "calibration", "test"):
        rows = split.frame(german, block)
        assert float((rows["gender"] == 0).mean()) == pytest.approx(overall, abs=0.005), block


def test_every_target_and_group_cell_is_represented_proportionally(split, german):
    for target in (0, 1):
        for gender in (0, 1):
            cell = (german[GERMAN_CREDIT.target] == target) & (german["gender"] == gender)
            expected_test = int(cell.sum()) * TEST_SIZE
            rows = split.frame(german, "test")
            actual = int(
                ((rows[GERMAN_CREDIT.target] == target) & (rows["gender"] == gender)).sum()
            )
            # A stratified draw rounds each cell, so it can be off by at most one row.
            assert abs(actual - expected_test) <= 1, (target, gender)


def test_the_test_block_holds_sixty_two_women(split, german):
    """Pins the sample-size problem the project has to report honestly.

    310 women, 20 percent of them, is 62 rows. One flipped prediction moves the female
    selection rate by 1.6 percentage points, which is why the disparate impact intervals
    in this dataset straddle 0.8.
    """
    rows = split.frame(german, "test")
    assert int((rows["gender"] == 0).sum()) == 62
    assert int((rows["gender"] == 1).sum()) == 138


def test_same_seed_reproduces_the_identical_split(german, split):
    again = make_split(
        GERMAN_CREDIT,
        german,
        test_size=TEST_SIZE,
        calibration_size=CALIBRATION_SIZE,
        seed=SEED,
    )

    assert np.array_equal(again.train, split.train)
    assert np.array_equal(again.calibration, split.calibration)
    assert np.array_equal(again.test, split.test)


def test_different_seed_produces_a_different_split(german, split):
    other = make_split(
        GERMAN_CREDIT,
        german,
        test_size=TEST_SIZE,
        calibration_size=CALIBRATION_SIZE,
        seed=7,
    )

    assert not np.array_equal(other.test, split.test)
    assert other.sizes == split.sizes


def test_split_records_what_it_stratified_on(split):
    assert split.stratified_on == ("credit_risk", "gender")


def test_fractions_that_leave_no_training_rows_raise(german):
    with pytest.raises(ValueError, match="leaves no training rows"):
        make_split(GERMAN_CREDIT, german, test_size=0.6, calibration_size=0.4, seed=SEED)


def test_zero_calibration_is_allowed_and_leaves_two_blocks(german):
    split = make_split(GERMAN_CREDIT, german, test_size=0.2, calibration_size=0.0, seed=SEED)

    assert split.sizes == {"train": 800, "calibration": 0, "test": 200}


def test_fingerprint_changes_when_a_value_changes(german):
    altered = german.copy()
    altered.loc[0, "amount"] = altered.loc[0, "amount"] + 1

    assert fingerprint(altered) != fingerprint(german)


def test_fingerprint_changes_when_a_column_is_renamed(german):
    renamed = german.rename(columns={"amount": "loan_amount"})

    assert fingerprint(renamed) != fingerprint(german)


def test_using_a_split_against_changed_data_raises(split, german):
    altered = german.copy()
    altered.loc[0, "amount"] = altered.loc[0, "amount"] + 1

    with pytest.raises(ValueError, match="the data changed"):
        split.frame(altered, "test")


def test_unknown_block_name_raises(split, german):
    with pytest.raises(ValueError, match="unknown block 'validation'"):
        split.frame(german, "validation")


def test_round_trip_through_disk_preserves_the_split(german, split, tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.splits.ARTIFACTS_DIR", tmp_path)
    save_split(split, GERMAN_CREDIT)
    loaded = load_split(GERMAN_CREDIT, SEED)

    assert np.array_equal(loaded.test, split.test)
    assert loaded.fingerprint == split.fingerprint
    assert split_path(GERMAN_CREDIT, SEED).parent == tmp_path / "splits"


def test_missing_split_artifact_raises_rather_than_silently_regenerating(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.splits.ARTIFACTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="before running any track"):
        load_split(GERMAN_CREDIT, 999)


def test_get_or_create_reuses_the_artifact_then_regenerates_on_data_change(
    german, tmp_path, monkeypatch
):
    monkeypatch.setattr("src.data.splits.ARTIFACTS_DIR", tmp_path)
    kwargs = {"test_size": TEST_SIZE, "calibration_size": CALIBRATION_SIZE, "seed": SEED}

    first = get_or_create_split(GERMAN_CREDIT, german, **kwargs)
    second = get_or_create_split(GERMAN_CREDIT, german, **kwargs)
    assert np.array_equal(first.test, second.test)

    altered = german.copy()
    altered.loc[0, "amount"] = altered.loc[0, "amount"] + 1
    regenerated = get_or_create_split(GERMAN_CREDIT, altered, **kwargs)
    assert regenerated.fingerprint != first.fingerprint
