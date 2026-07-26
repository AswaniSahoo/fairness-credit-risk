"""Tests for performance metrics and bootstrap intervals.

Expected values come from a hand-built confusion matrix written out in the fixture comment,
so a change in `pos_label` or an averaging default fails here instead of shifting every
published number by a few points.
"""

import numpy as np
import pytest

from src.evaluation.performance import (
    Interval,
    bootstrap_interval,
    performance_metrics,
    stratified_bootstrap_indices,
)

pytestmark = pytest.mark.unit

# 10 rows. Positive class is 1 (default).
#   true 1, pred 1 -> 3 true positives
#   true 1, pred 0 -> 1 false negative
#   true 0, pred 1 -> 2 false positives
#   true 0, pred 0 -> 4 true negatives
Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PRED = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
SCORES = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.55, 0.3, 0.2, 0.1, 0.05])


def test_precision_and_recall_are_reported_for_the_default_class():
    metrics = performance_metrics(Y_TRUE, Y_PRED, SCORES, positive_class=1)

    # precision 3/(3+2), recall 3/(3+1)
    assert metrics["precision"] == pytest.approx(0.6)
    assert metrics["recall"] == pytest.approx(0.75)
    # f1 = 2 * 0.6 * 0.75 / 1.35
    assert metrics["f1"] == pytest.approx(2 * 0.6 * 0.75 / 1.35)


def test_accuracy_and_balanced_accuracy_differ_under_imbalance():
    metrics = performance_metrics(Y_TRUE, Y_PRED, SCORES)

    # 3 TP + 4 TN out of 10
    assert metrics["accuracy"] == pytest.approx(0.7)
    # mean of recall 0.75 and specificity 4/6
    assert metrics["balanced_accuracy"] == pytest.approx((0.75 + 4 / 6) / 2)


def test_roc_auc_uses_the_scores_not_the_hard_predictions():
    metrics = performance_metrics(Y_TRUE, Y_PRED, SCORES)

    # 4 positives x 6 negatives = 24 pairs. The only misordered pairs are score 0.4
    # (a positive) against 0.6 and 0.55 (both negatives), so 22 of 24 are concordant.
    assert metrics["roc_auc"] == pytest.approx(22 / 24)


def test_brier_is_the_mean_squared_probability_error():
    metrics = performance_metrics(Y_TRUE, Y_PRED, SCORES)
    expected = float(np.mean((SCORES - Y_TRUE) ** 2))

    assert metrics["brier"] == pytest.approx(expected)


def test_bootstrap_preserves_every_group_and_label_cell_size():
    """Unstratified resampling of 62 women occasionally empties a label cell, which makes
    group rates undefined and turns the interval into a property of the resampling."""
    groups = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    replicates = stratified_bootstrap_indices(Y_TRUE, groups, n_replicates=50, seed=0)

    assert len(replicates) == 50
    for index in replicates:
        assert index.size == len(Y_TRUE)
        for group in (0, 1):
            for label in (0, 1):
                expected = int(((groups == group) & (Y_TRUE == label)).sum())
                actual = int(((groups[index] == group) & (Y_TRUE[index] == label)).sum())
                assert actual == expected, (group, label)


def test_bootstrap_is_reproducible_under_the_same_seed():
    groups = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    first = stratified_bootstrap_indices(Y_TRUE, groups, n_replicates=5, seed=3)
    second = stratified_bootstrap_indices(Y_TRUE, groups, n_replicates=5, seed=3)

    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_interval_brackets_the_point_estimate_for_a_stable_statistic():
    groups = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    replicates = stratified_bootstrap_indices(Y_TRUE, groups, n_replicates=200, seed=0)
    point = float(Y_TRUE.mean())

    interval = bootstrap_interval(lambda index: float(Y_TRUE[index].mean()), replicates, point)

    # Cell sizes are preserved, so the label mean is identical in every replicate.
    assert interval.point == pytest.approx(0.4)
    assert interval.low == pytest.approx(0.4)
    assert interval.high == pytest.approx(0.4)


def test_interval_widens_with_a_noisy_statistic():
    groups = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    replicates = stratified_bootstrap_indices(Y_TRUE, groups, n_replicates=500, seed=0)

    interval = bootstrap_interval(
        lambda index: float(SCORES[index].mean()), replicates, float(SCORES.mean())
    )

    assert interval.low < interval.point < interval.high


def test_non_finite_replicates_are_dropped_rather_than_poisoning_the_interval():
    replicates = [np.array([0, 1]), np.array([2, 3])]

    interval = bootstrap_interval(lambda index: float("inf"), replicates, 0.5)

    assert interval.point == pytest.approx(0.5)
    assert np.isnan(interval.low)
    assert np.isnan(interval.high)


def test_excludes_zero_reports_distinguishability_in_both_directions():
    assert Interval(0.05, 0.01, 0.09).excludes_zero
    assert Interval(-0.05, -0.09, -0.01).excludes_zero
    assert not Interval(0.01, -0.02, 0.04).excludes_zero
