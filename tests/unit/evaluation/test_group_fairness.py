"""Tests for group fairness metrics.

Every expected value is derived by hand in the test body. None is copied from the
implementation's output, which would only pin in whatever the code currently does.

The central test here is finding B1: the previous implementation computed disparate impact
from ground-truth labels, so the metrics were independent of the model. Any implementation
that regresses to that behaviour fails
``test_metrics_change_when_only_predictions_change``.
"""

import numpy as np
import pytest

from src.evaluation.group_fairness import (
    group_fairness,
    predict_at_threshold,
    threshold_for_selection_rate,
)

pytestmark = pytest.mark.unit

# Shared fixture, 8 rows. Favorable outcome is 0 ("good credit").
#   privileged (group 1): 4 rows, predictions 0,0,0,1 -> 3 favorable -> rate 0.75
#   unprivileged (group 0): 4 rows, predictions 0,0,1,1 -> 2 favorable -> rate 0.50
Y_TRUE = np.array([0, 0, 0, 0, 0, 0, 0, 0])
Y_PRED = np.array([0, 0, 0, 1, 0, 0, 1, 1])
GROUPS = np.array([1, 1, 1, 1, 0, 0, 0, 0])


def _result(y_true=Y_TRUE, y_pred=Y_PRED, groups=GROUPS):
    return group_fairness(
        y_true, y_pred, groups, privileged_value=1, unprivileged_value=0, favorable_label=0
    )


def test_selection_rates_match_hand_count():
    result = _result()
    assert result.privileged.selection_rate == pytest.approx(0.75)
    assert result.unprivileged.selection_rate == pytest.approx(0.50)
    assert result.privileged.n == 4
    assert result.unprivileged.n == 4


def test_disparate_impact_is_unprivileged_over_privileged():
    # 0.50 / 0.75 = 2/3
    assert _result().disparate_impact == pytest.approx(2 / 3, abs=1e-12)


def test_statistical_parity_difference_is_unprivileged_minus_privileged():
    # 0.50 - 0.75 = -0.25
    assert _result().statistical_parity_difference == pytest.approx(-0.25)


def test_metrics_change_when_only_predictions_change():
    """Finding B1 regression.

    Labels and groups are held fixed while predictions change. The old label-based
    implementation returned identical metrics for both, because it never read y_pred.
    """
    baseline = _result()

    all_favorable = group_fairness(
        Y_TRUE, np.zeros(8, dtype=int), GROUPS,
        privileged_value=1, unprivileged_value=0, favorable_label=0,
    )

    # Approving everyone gives both groups a selection rate of 1, hence perfect parity.
    assert all_favorable.disparate_impact == pytest.approx(1.0)
    assert all_favorable.statistical_parity_difference == pytest.approx(0.0)
    assert all_favorable.disparate_impact != pytest.approx(baseline.disparate_impact)


def test_labels_as_predictions_recovers_the_dataset_bias_not_a_model_metric():
    """Documents precisely what the old implementation was measuring.

    Base rates: privileged 3/4 favorable, unprivileged 2/4 favorable -> DI 2/3.
    Passing y_true as y_pred must return the dataset's own disparity, and a genuine
    model metric must be free to differ from it.
    """
    y_true = np.array([0, 0, 0, 1, 0, 0, 1, 1])
    label_di = group_fairness(
        y_true, y_true, GROUPS, privileged_value=1, unprivileged_value=0
    ).disparate_impact
    assert label_di == pytest.approx(2 / 3, abs=1e-12)

    model_di = group_fairness(
        y_true, np.zeros(8, dtype=int), GROUPS, privileged_value=1, unprivileged_value=0
    ).disparate_impact
    assert model_di == pytest.approx(1.0)


def test_equal_opportunity_uses_only_truly_favorable_rows():
    # privileged rows 0..3, true favorable at 0,1,2 -> predictions 0,0,0 -> tpr 1.0
    # unprivileged rows 4..7, true favorable at 4,5 -> predictions 0,1 -> tpr 0.5
    y_true = np.array([0, 0, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    result = group_fairness(y_true, y_pred, GROUPS, privileged_value=1, unprivileged_value=0)

    assert result.privileged.tpr == pytest.approx(1.0)
    assert result.unprivileged.tpr == pytest.approx(0.5)
    assert result.equal_opportunity_difference == pytest.approx(-0.5)


def test_equalized_odds_takes_the_worse_of_the_two_gaps():
    # tpr gap -0.5 (above); fpr: privileged row 3 true=1 pred=1 -> fpr 0.0
    #                            unprivileged rows 6,7 true=1 pred=1,1 -> fpr 0.0
    # worse gap is therefore |−0.5| = 0.5
    y_true = np.array([0, 0, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    result = group_fairness(y_true, y_pred, GROUPS, privileged_value=1, unprivileged_value=0)

    assert result.privileged.fpr == pytest.approx(0.0)
    assert result.unprivileged.fpr == pytest.approx(0.0)
    assert result.equalized_odds_difference == pytest.approx(0.5)


def test_direction_is_governed_by_the_caller_not_a_global_constant():
    """The Taiwan dataset reverses which group is disadvantaged, so swapping the
    privileged and unprivileged arguments must invert the orientation."""
    forward = group_fairness(Y_TRUE, Y_PRED, GROUPS, privileged_value=1, unprivileged_value=0)
    reversed_ = group_fairness(Y_TRUE, Y_PRED, GROUPS, privileged_value=0, unprivileged_value=1)

    assert forward.disparate_impact == pytest.approx(2 / 3)
    assert reversed_.disparate_impact == pytest.approx(1.5)
    assert reversed_.statistical_parity_difference == pytest.approx(+0.25)


def test_supports_the_taiwan_one_two_encoding():
    """SEX is encoded 1=male, 2=female in the Taiwan dataset, not 0/1."""
    groups = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    result = group_fairness(Y_TRUE, Y_PRED, groups, privileged_value=1, unprivileged_value=2)
    assert result.disparate_impact == pytest.approx(2 / 3)


def test_disparate_impact_is_infinite_rather_than_silently_zero():
    # Privileged group never approved -> the ratio is undefined; report inf.
    y_pred = np.array([1, 1, 1, 1, 0, 0, 1, 1])
    result = group_fairness(Y_TRUE, y_pred, GROUPS, privileged_value=1, unprivileged_value=0)
    assert result.privileged.selection_rate == pytest.approx(0.0)
    assert np.isinf(result.disparate_impact)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        group_fairness(
            np.array([0, 1]), np.array([0]), np.array([1, 0]),
            privileged_value=1, unprivileged_value=0,
        )


def test_absent_group_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="unprivileged_value=9"):
        group_fairness(
            Y_TRUE, Y_PRED, GROUPS, privileged_value=1, unprivileged_value=9
        )


def test_threshold_matching_hits_the_requested_selection_rate():
    # Ten evenly spaced scores; approving 30% means the threshold sits at the 0.3 quantile.
    scores = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    threshold = threshold_for_selection_rate(scores, 0.30)
    predictions = predict_at_threshold(scores, threshold, favorable_label=0)

    assert np.mean(predictions == 0) == pytest.approx(0.30)
    # Rows below the threshold are the three lowest scores.
    assert predictions.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_threshold_matching_rejects_out_of_range_targets():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        threshold_for_selection_rate(np.array([0.1, 0.9]), 1.5)
