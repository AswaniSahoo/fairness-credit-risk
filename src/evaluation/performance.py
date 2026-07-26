"""Performance metrics and bootstrap confidence intervals.

A 200-row test set cannot support the three-decimal claims the project used to publish. One
flipped prediction moves accuracy by half a percentage point, and the female selection rate
by 1.6 points. Every reported number therefore carries an interval, and comparisons between
tracks are paired on the same resampled rows so the shared test-set noise cancels.

Resampling is stratified within (group, label) cells. An unstratified bootstrap on 62 women
occasionally draws a replicate with almost no women in one label cell, which produces
undefined group rates and intervals that reflect the resampling scheme rather than the
model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval."""

    point: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "ci_low": self.low, "ci_high": self.high}

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is distinguishable from no effect."""
        return self.low > 0.0 or self.high < 0.0


def performance_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    scores: ArrayLike,
    *,
    positive_class: int = 1,
) -> dict[str, float]:
    """Threshold-dependent and threshold-free performance on one block.

    ``scores`` are probabilities of ``positive_class``, which is the default. Precision and
    recall are reported for that class, so recall is the share of true defaults caught:
    the quantity a lender actually cares about.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    scores_arr = np.asarray(scores).ravel()

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, pos_label=positive_class, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_arr, y_pred_arr, pos_label=positive_class, zero_division=0)
        ),
        "f1": float(f1_score(y_true_arr, y_pred_arr, pos_label=positive_class, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_arr, scores_arr)),
        "average_precision": float(average_precision_score(y_true_arr, scores_arr)),
        # Calibration, not just ranking. A model can rank well and still be systematically
        # over-confident, which matters when a probability drives a pricing decision.
        "brier": float(brier_score_loss(y_true_arr, scores_arr)),
    }


def stratified_bootstrap_indices(
    y_true: NDArray[np.int_],
    groups: NDArray[Any],
    *,
    n_replicates: int,
    seed: int,
) -> list[NDArray[np.int_]]:
    """Row indices for each replicate, resampled within (group, label) cells.

    Preserving cell sizes keeps every replicate's group and label composition equal to the
    observed one, so an interval reflects sampling variability in outcomes rather than in
    who happens to be drawn.
    """
    rng = np.random.default_rng(seed)
    cells = [
        np.flatnonzero((groups == group) & (y_true == label))
        for group in np.unique(groups)
        for label in np.unique(y_true)
    ]
    cells = [cell for cell in cells if cell.size > 0]

    return [
        np.concatenate([rng.choice(cell, size=cell.size, replace=True) for cell in cells])
        for _ in range(n_replicates)
    ]


def bootstrap_interval(
    statistic: Callable[[NDArray[np.int_]], float],
    replicates: list[NDArray[np.int_]],
    point: float,
    *,
    alpha: float = 0.05,
) -> Interval:
    """Percentile interval for a statistic evaluated over precomputed replicates.

    ``replicates`` is shared across statistics and across tracks, which is what makes
    differences paired: the same resampled rows are used everywhere, so test-set noise
    common to both tracks cancels instead of adding.

    Non-finite replicate values are dropped, which happens when a replicate leaves a
    denominator empty. The count that survived is not reported here; callers comparing
    intervals should keep the replicate count fixed.
    """
    values = np.array([statistic(index) for index in replicates], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Interval(point=point, low=float("nan"), high=float("nan"))

    low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return Interval(point=point, low=float(low), high=float(high))
