"""Tests for the in-repo reweighing implementation.

Expected weights are computed by hand from w(a, y) = P(A=a) P(Y=y) / P(A=a, Y=y).
"""

import numpy as np
import pytest

from src.preprocessing.reweighing import reweighing_weights

pytestmark = pytest.mark.unit

# 8 rows. group 1 labels [0,0,0,1]; group 0 labels [0,1,1,1]
# Marginals: P(A=1)=0.5, P(A=0)=0.5, P(Y=0)=0.5, P(Y=1)=0.5
Y_TRUE = np.array([0, 0, 0, 1, 0, 1, 1, 1])
GROUPS = np.array([1, 1, 1, 1, 0, 0, 0, 0])


def test_weights_match_hand_derived_values():
    weights = reweighing_weights(Y_TRUE, GROUPS)

    # (A=1, Y=0): 3/8 observed vs 0.25 expected -> 0.5*0.5/0.375 = 0.6667 (over-represented)
    assert weights[0] == pytest.approx(2 / 3, abs=1e-12)
    # (A=1, Y=1): 1/8 observed vs 0.25 expected -> 0.5*0.5/0.125 = 2.0 (under-represented)
    assert weights[3] == pytest.approx(2.0, abs=1e-12)
    # (A=0, Y=0): 1/8 observed -> 2.0
    assert weights[4] == pytest.approx(2.0, abs=1e-12)
    # (A=0, Y=1): 3/8 observed -> 0.6667
    assert weights[5] == pytest.approx(2 / 3, abs=1e-12)


def test_total_weight_is_conserved():
    """Reweighing redistributes mass; it must not change the effective sample size."""
    weights = reweighing_weights(Y_TRUE, GROUPS)
    assert weights.sum() == pytest.approx(len(Y_TRUE), abs=1e-12)


def test_independent_groups_and_labels_give_unit_weights():
    # Each (group, label) cell holds exactly 1/4 of rows, so no reweighting is needed.
    y_true = np.array([0, 1, 0, 1])
    groups = np.array([1, 1, 0, 0])
    assert reweighing_weights(y_true, groups) == pytest.approx(np.ones(4))


def test_under_represented_cells_get_weight_above_one():
    weights = reweighing_weights(Y_TRUE, GROUPS)
    minority_cell = (GROUPS == 1) & (Y_TRUE == 1)
    majority_cell = (GROUPS == 1) & (Y_TRUE == 0)
    assert (weights[minority_cell] > 1.0).all()
    assert (weights[majority_cell] < 1.0).all()


def test_works_with_the_taiwan_one_two_encoding():
    groups = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    weights = reweighing_weights(Y_TRUE, groups)
    assert weights[0] == pytest.approx(2 / 3, abs=1e-12)
    assert weights.sum() == pytest.approx(8.0)


def test_empty_cell_does_not_produce_infinite_weights():
    # group 0 has no positive labels at all.
    y_true = np.array([0, 1, 0, 0])
    groups = np.array([1, 1, 0, 0])
    weights = reweighing_weights(y_true, groups)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        reweighing_weights(np.array([0, 1, 0]), np.array([1, 0]))
