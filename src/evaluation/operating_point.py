"""Choosing the decision threshold, instead of inheriting 0.5 from nobody.

A probability model does not make decisions. A threshold does, and 0.5 is not a neutral
default: it is the point where the two kinds of mistake are treated as equally expensive.
In lending they are not. A missed default costs outstanding principal; a good applicant
wrongly declined costs the margin that loan would have earned. Treating those as equal is a
business assumption that nobody in this repository ever made deliberately, and on the Taiwan
baseline it produces a model that catches 36.47 percent of defaults.

This module selects the threshold that minimises expected misclassification cost, given an
explicit cost ratio. Two disciplines apply:

- **Fitted on calibration, reported on test.** The same rule as the T3 post-processing track
  and for the same reason (finding B4): a threshold chosen on the block it is scored on is
  chosen with knowledge of the answer.
- **The ratio is an assumption, so it is swept.** ``cost_sensitivity`` reports the threshold
  and its consequences across a range of ratios, so a reader sees how much of the result is
  the data and how much is the assumption. A single ratio reported alone invites the
  suspicion that it was picked to flatter the outcome.

The ratio is not measured from either dataset. Neither carries recovery rates or interest
margins, so any number claiming to be derived would be assumption wearing a measurement's
clothes. ``DEFAULT_COST_RATIO`` is documented as a choice in MODEL_CARD.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.evaluation.group_fairness import predict_at_threshold

logger = logging.getLogger(__name__)

# Cost of failing to catch a default, relative to declining a good applicant. Illustrative
# and conservative: lost principal exceeds foregone margin, but by how much depends on
# recovery rates neither dataset records. Swept by cost_sensitivity.
DEFAULT_COST_RATIO = 5.0

# Ratios reported in the sensitivity analysis. 1.0 is included because it is what threshold
# 0.5 implicitly assumes, which is the comparison that makes the default's cost visible.
SENSITIVITY_RATIOS = (1.0, 2.0, 5.0, 10.0, 20.0)


@dataclass(frozen=True)
class OperatingPoint:
    """A chosen threshold and the evidence for choosing it."""

    threshold: float
    cost_ratio: float
    expected_cost_per_applicant: float
    fitted_on: str
    n_fitted_rows: int
    selection_rate: float
    recall: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "cost_ratio": self.cost_ratio,
            "expected_cost_per_applicant": self.expected_cost_per_applicant,
            "fitted_on": self.fitted_on,
            "n_fitted_rows": self.n_fitted_rows,
            "selection_rate": self.selection_rate,
            "recall": self.recall,
        }


def confusion_counts(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    positive_class: int = 1,
) -> tuple[int, int, int, int]:
    """Return (true positives, false positives, true negatives, false negatives).

    The positive class is default. A false negative is an approved applicant who defaults;
    a false positive is a declined applicant who would have repaid.
    """
    truth = np.asarray(y_true).ravel() == positive_class
    predicted = np.asarray(y_pred).ravel() == positive_class

    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    tn = int(np.sum(~truth & ~predicted))
    fn = int(np.sum(truth & ~predicted))
    return tp, fp, tn, fn


def expected_cost(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    cost_ratio: float = DEFAULT_COST_RATIO,
    positive_class: int = 1,
) -> float:
    """Mean misclassification cost per applicant, in units of one wrongly declined applicant.

    ``cost_ratio`` is the cost of a false negative relative to a false positive. Correct
    decisions cost nothing here: this measures the cost of being wrong, not profit.
    """
    if cost_ratio <= 0:
        raise ValueError(f"cost_ratio must be positive, got {cost_ratio}")

    _, fp, _, fn = confusion_counts(y_true, y_pred, positive_class=positive_class)
    n = len(np.asarray(y_true).ravel())
    if n == 0:
        raise ValueError("cannot compute expected cost on an empty block")
    return (cost_ratio * fn + fp) / n


def _candidate_thresholds(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Every threshold that produces a distinct labelling, plus the two extremes.

    Midpoints between adjacent unique scores are used rather than the scores themselves, so
    the chosen value does not sit exactly on an observation and flip under a tie.
    """
    unique = np.unique(scores)
    if len(unique) == 1:
        return np.array([unique[0]], dtype=np.float64)
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.concatenate([[0.0], midpoints, [1.0]]).astype(np.float64)


def select_cost_minimising_threshold(
    y_true: ArrayLike,
    scores: ArrayLike,
    *,
    cost_ratio: float = DEFAULT_COST_RATIO,
    favorable_label: int = 0,
    positive_class: int = 1,
    fitted_on: str = "calibration",
) -> OperatingPoint:
    """Threshold minimising expected cost on the block supplied.

    Among thresholds with equal expected cost the largest is returned, which declines the
    fewest applicants. Ties are common on small blocks, and picking silently among them
    would make the result depend on floating-point ordering.

    Raises:
        ValueError: If the block is empty or ``cost_ratio`` is not positive.
    """
    truth = np.asarray(y_true).ravel()
    score_arr = np.asarray(scores, dtype=np.float64).ravel()
    if len(truth) == 0:
        raise ValueError("cannot fit an operating point on an empty block")
    if len(truth) != len(score_arr):
        raise ValueError(
            f"y_true has {len(truth)} rows and scores has {len(score_arr)}"
        )

    best_threshold = 0.5
    best_cost = np.inf
    for threshold in _candidate_thresholds(score_arr):
        y_pred = predict_at_threshold(score_arr, threshold, favorable_label=favorable_label)
        cost = expected_cost(
            truth, y_pred, cost_ratio=cost_ratio, positive_class=positive_class
        )
        # Strictly-less keeps the first minimum; the >= on equality keeps the largest
        # threshold among equals, because candidates are generated in ascending order.
        if cost < best_cost or (cost == best_cost and threshold > best_threshold):
            best_cost = cost
            best_threshold = float(threshold)

    y_pred = predict_at_threshold(score_arr, best_threshold, favorable_label=favorable_label)
    tp, _, _, fn = confusion_counts(truth, y_pred, positive_class=positive_class)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    point = OperatingPoint(
        threshold=best_threshold,
        cost_ratio=float(cost_ratio),
        expected_cost_per_applicant=float(best_cost),
        fitted_on=fitted_on,
        n_fitted_rows=int(len(truth)),
        selection_rate=float(np.mean(y_pred == favorable_label)),
        recall=recall,
    )
    logger.info(
        "operating point on %s at cost ratio %.1f: threshold %.4f, approves %.4f, "
        "recall %.4f, expected cost %.4f per applicant",
        fitted_on, cost_ratio, point.threshold, point.selection_rate, point.recall,
        point.expected_cost_per_applicant,
    )
    return point


def cost_sensitivity(
    y_true: ArrayLike,
    scores: ArrayLike,
    *,
    ratios: tuple[float, ...] = SENSITIVITY_RATIOS,
    favorable_label: int = 0,
    positive_class: int = 1,
) -> list[dict[str, float]]:
    """Selected threshold and its consequences across cost ratios.

    This is what stops the headline ratio being an unexamined choice: a reader can see
    the threshold move, and see that ratio 1.0 is roughly what 0.5 assumes.
    """
    return [
        {
            "cost_ratio": ratio,
            **select_cost_minimising_threshold(
                y_true,
                scores,
                cost_ratio=ratio,
                favorable_label=favorable_label,
                positive_class=positive_class,
            ).as_dict(),
        }
        for ratio in ratios
    ]


def threshold_sweep(
    y_true: ArrayLike,
    scores: ArrayLike,
    groups: ArrayLike,
    *,
    n_points: int = 50,
    favorable_label: int = 0,
    positive_class: int = 1,
    fairness_fn: Any = None,
) -> list[dict[str, float]]:
    """Metrics across the threshold range, for the fairness-accuracy tradeoff curve.

    A single operating point reduces each track to one dot on the tradeoff plot, which hides
    that a track can be moved along its own curve. ``fairness_fn`` takes
    ``(y_true, y_pred, groups)`` and returns an object with ``disparate_impact`` and
    ``statistical_parity_difference``; it is injected so this module does not depend on the
    dataset registry.
    """
    truth = np.asarray(y_true).ravel()
    score_arr = np.asarray(scores, dtype=np.float64).ravel()
    group_arr = np.asarray(groups).ravel()

    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.0, 1.0, n_points):
        y_pred = predict_at_threshold(score_arr, threshold, favorable_label=favorable_label)
        tp, fp, tn, fn = confusion_counts(truth, y_pred, positive_class=positive_class)

        row: dict[str, float] = {
            "threshold": float(threshold),
            "selection_rate": float(np.mean(y_pred == favorable_label)),
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "expected_cost_per_applicant": expected_cost(
                truth, y_pred, cost_ratio=DEFAULT_COST_RATIO, positive_class=positive_class
            ),
        }
        if fairness_fn is not None:
            result = fairness_fn(truth, y_pred, group_arr)
            row["disparate_impact"] = float(result.disparate_impact)
            row["statistical_parity_difference"] = float(
                result.statistical_parity_difference
            )
        rows.append(row)
    return rows
