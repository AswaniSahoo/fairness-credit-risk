"""Unit tests for cost-based threshold selection.

Expected values are derived by hand in the comments. The four-row fixture is built so that
the cost ratio changes the answer, which is the whole point of the module: at ratio 1 the
selected threshold is 0.525 and at ratio 5 it is 0.25.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.operating_point import (
    DEFAULT_COST_RATIO,
    SENSITIVITY_RATIOS,
    confusion_counts,
    cost_sensitivity,
    expected_cost,
    select_cost_minimising_threshold,
    threshold_sweep,
)

pytestmark = pytest.mark.unit

# Four applicants. 0 is good (favorable), 1 is default (positive class).
#   score  0.10  true 0
#   score  0.40  true 1
#   score  0.45  true 0
#   score  0.60  true 1
# Candidate thresholds are 0.0, the midpoints 0.25 / 0.425 / 0.525, and 1.0.
SCORES = np.array([0.10, 0.40, 0.45, 0.60])
LABELS = np.array([0, 1, 0, 1])


def test_confusion_counts_names_the_four_cells_correctly():
    # At threshold 0.5 predictions are [0, 0, 0, 1]:
    #   index 0 true 0 pred 0 -> true negative
    #   index 1 true 1 pred 0 -> false negative, an approved applicant who defaults
    #   index 2 true 0 pred 0 -> true negative
    #   index 3 true 1 pred 1 -> true positive
    y_pred = np.array([0, 0, 0, 1])

    tp, fp, tn, fn = confusion_counts(LABELS, y_pred)

    assert (tp, fp, tn, fn) == (1, 0, 2, 1)


def test_expected_cost_weights_the_false_negative_by_the_ratio():
    # 1 false negative, 0 false positives, 4 applicants.
    # ratio 1 -> (1*1 + 0)/4 = 0.25
    # ratio 5 -> (5*1 + 0)/4 = 1.25
    y_pred = np.array([0, 0, 0, 1])

    assert expected_cost(LABELS, y_pred, cost_ratio=1.0) == pytest.approx(0.25)
    assert expected_cost(LABELS, y_pred, cost_ratio=5.0) == pytest.approx(1.25)


def test_expected_cost_rejects_a_non_positive_ratio():
    with pytest.raises(ValueError, match="cost_ratio must be positive"):
        expected_cost(LABELS, np.array([0, 0, 0, 1]), cost_ratio=0.0)


def test_symmetric_cost_selects_the_higher_of_two_tied_thresholds():
    # At ratio 1, cost by candidate threshold:
    #   0.000 -> preds [1,1,1,1]: fn 0, fp 2 -> 2/4 = 0.50
    #   0.250 -> preds [0,1,1,1]: fn 0, fp 1 -> 1/4 = 0.25
    #   0.425 -> preds [0,0,1,1]: fn 1, fp 1 -> 2/4 = 0.50
    #   0.525 -> preds [0,0,0,1]: fn 1, fp 0 -> 1/4 = 0.25
    #   1.000 -> preds [0,0,0,0]: fn 2, fp 0 -> 2/4 = 0.50
    # Minimum 0.25 is tied between 0.25 and 0.525; the larger is chosen because it
    # declines fewer applicants.
    point = select_cost_minimising_threshold(LABELS, SCORES, cost_ratio=1.0)

    assert point.threshold == pytest.approx(0.525)
    assert point.expected_cost_per_applicant == pytest.approx(0.25)


def test_asymmetric_cost_moves_the_threshold_down():
    # At ratio 5:
    #   0.000 -> fn 0, fp 2 -> (0 + 2)/4 = 0.50
    #   0.250 -> fn 0, fp 1 -> (0 + 1)/4 = 0.25
    #   0.425 -> fn 1, fp 1 -> (5 + 1)/4 = 1.50
    #   0.525 -> fn 1, fp 0 -> (5 + 0)/4 = 1.25
    #   1.000 -> fn 2, fp 0 -> (10 + 0)/4 = 2.50
    # Unique minimum at 0.25.
    point = select_cost_minimising_threshold(LABELS, SCORES, cost_ratio=5.0)

    assert point.threshold == pytest.approx(0.25)
    assert point.expected_cost_per_applicant == pytest.approx(0.25)
    # Both defaults are now caught: recall is 1.0 where threshold 0.5 caught one of two.
    assert point.recall == pytest.approx(1.0)


def test_selected_point_reports_the_block_it_was_fitted_on():
    point = select_cost_minimising_threshold(
        LABELS, SCORES, cost_ratio=DEFAULT_COST_RATIO, fitted_on="calibration"
    )

    assert point.fitted_on == "calibration"
    assert point.n_fitted_rows == 4
    # Threshold 0.25 declines three of four applicants, so the approval rate is 0.25.
    assert point.selection_rate == pytest.approx(0.25)


def test_raising_the_cost_of_a_missed_default_never_raises_the_threshold():
    """The selected threshold must be non-increasing in the cost ratio.

    Making false negatives more expensive can only make the model decline more readily.
    A violation would mean the search is not finding the minimum.
    """
    rng = np.random.default_rng(42)
    scores = rng.uniform(size=200)
    # Labels correlated with the score, so the problem is learnable rather than noise.
    labels = (rng.uniform(size=200) < scores).astype(int)

    thresholds = [
        select_cost_minimising_threshold(labels, scores, cost_ratio=r).threshold
        for r in SENSITIVITY_RATIOS
    ]

    assert thresholds == sorted(thresholds, reverse=True)


def test_cost_sensitivity_reports_one_row_per_ratio():
    rows = cost_sensitivity(LABELS, SCORES)

    assert [row["cost_ratio"] for row in rows] == list(SENSITIVITY_RATIOS)
    for row in rows:
        assert set(row) >= {"cost_ratio", "threshold", "recall", "selection_rate"}


def test_select_rejects_an_empty_block():
    with pytest.raises(ValueError, match="empty block"):
        select_cost_minimising_threshold(np.array([]), np.array([]))


def test_select_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="rows"):
        select_cost_minimising_threshold(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_threshold_sweep_spans_approving_everyone_to_approving_nobody():
    groups = np.array([1, 1, 0, 0])

    rows = threshold_sweep(LABELS, SCORES, groups, n_points=11)

    assert len(rows) == 11
    # At threshold 0 every score clears the bar, so every applicant is declined.
    assert rows[0]["selection_rate"] == pytest.approx(0.0)
    assert rows[0]["recall"] == pytest.approx(1.0)
    # At threshold 1 nobody is declined, so no default is caught.
    assert rows[-1]["selection_rate"] == pytest.approx(1.0)
    assert rows[-1]["recall"] == pytest.approx(0.0)


def test_threshold_sweep_includes_fairness_when_a_function_is_supplied():
    groups = np.array([1, 1, 0, 0])

    class _Result:
        disparate_impact = 0.5
        statistical_parity_difference = -0.25

    rows = threshold_sweep(
        LABELS, SCORES, groups, n_points=5, fairness_fn=lambda *_: _Result()
    )

    for row in rows:
        assert row["disparate_impact"] == pytest.approx(0.5)
        assert row["statistical_parity_difference"] == pytest.approx(-0.25)
