"""Group fairness metrics computed from model predictions.

Replaces the previous implementation, which constructed an AIF360 ``StandardDataset``
with ``label_name='true_label'`` and put predictions in a separate, unused column. AIF360
then computed base rates over the dataset's *labels*, so disparate impact and statistical
parity were mathematically independent of the model. Confirmed empirically: feeding
ground-truth labels as predictions reproduced the previously published figures of 0.8903
and -0.0795 exactly. See finding B1.

Two design rules follow from that failure and from supporting two datasets:

1. ``y_true`` and ``y_pred`` are separate, explicitly named arguments, and the group rates
   are computed from ``y_pred``. Transposing them cannot silently produce a plausible
   number.
2. Privileged and unprivileged group values are always supplied by the caller. They are
   not global constants, because the disparity direction reverses between datasets: in
   German Credit women are disadvantaged and encoded 0/1, while in the Taiwan dataset
   women default less and the encoding is 1/2.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GroupRates:
    """Outcome rates for a single protected group.

    Rates are expressed with respect to the *favorable* outcome, which for credit risk is
    "not a default". ``tpr`` is therefore the rate at which genuinely creditworthy
    applicants are approved, and ``fpr`` the rate at which defaulters are approved.
    """

    n: int
    n_favorable_true: int
    selection_rate: float
    tpr: float
    fpr: float


@dataclass(frozen=True)
class FairnessResult:
    """Group fairness metrics for one privileged/unprivileged pair.

    ``disparate_impact`` and ``statistical_parity_difference`` are oriented as
    unprivileged relative to privileged, so a ratio below 1 and a negative difference both
    mean the unprivileged group is approved less often.
    """

    privileged: GroupRates
    unprivileged: GroupRates
    disparate_impact: float
    statistical_parity_difference: float
    equal_opportunity_difference: float
    equalized_odds_difference: float

    def as_dict(self) -> dict[str, float]:
        """Flatten to a JSON-serialisable mapping for the comparison artifact."""
        return {
            "disparate_impact": self.disparate_impact,
            "statistical_parity_difference": self.statistical_parity_difference,
            "equal_opportunity_difference": self.equal_opportunity_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "selection_rate_privileged": self.privileged.selection_rate,
            "selection_rate_unprivileged": self.unprivileged.selection_rate,
            "tpr_privileged": self.privileged.tpr,
            "tpr_unprivileged": self.unprivileged.tpr,
            "fpr_privileged": self.privileged.fpr,
            "fpr_unprivileged": self.unprivileged.fpr,
            "n_privileged": float(self.privileged.n),
            "n_unprivileged": float(self.unprivileged.n),
        }


def _rates_for_mask(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    mask: NDArray[np.bool_],
    favorable_label: int,
) -> GroupRates:
    true_group, pred_group = y_true[mask], y_pred[mask]
    is_favorable_true = true_group == favorable_label
    is_unfavorable_true = ~is_favorable_true

    return GroupRates(
        n=int(true_group.size),
        n_favorable_true=int(is_favorable_true.sum()),
        selection_rate=(
            float(np.mean(pred_group == favorable_label)) if pred_group.size else float("nan")
        ),
        tpr=(
            float(np.mean(pred_group[is_favorable_true] == favorable_label))
            if is_favorable_true.any()
            else float("nan")
        ),
        fpr=(
            float(np.mean(pred_group[is_unfavorable_true] == favorable_label))
            if is_unfavorable_true.any()
            else float("nan")
        ),
    )


def group_fairness(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: ArrayLike,
    *,
    privileged_value: int,
    unprivileged_value: int,
    favorable_label: int = 0,
) -> FairnessResult:
    """Compute group fairness metrics from predictions.

    Args:
        y_true: Ground-truth labels. Used only for the conditional rates (TPR, FPR),
            never for selection rates.
        y_pred: Predicted labels, in the same encoding as ``y_true``.
        groups: Protected attribute value per row.
        privileged_value: Group value treated as privileged, from the dataset registry.
        unprivileged_value: Group value treated as unprivileged.
        favorable_label: Label representing the favorable outcome. Defaults to 0,
            which is "good credit" in both supported datasets.

    Returns:
        A ``FairnessResult``. ``disparate_impact`` is ``inf`` when the privileged
        selection rate is zero, which is reported rather than silently clamped.

    Raises:
        ValueError: If input lengths differ, or a named group is absent from ``groups``.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    groups_arr = np.asarray(groups).ravel()

    if not (y_true_arr.size == y_pred_arr.size == groups_arr.size):
        raise ValueError(
            "y_true, y_pred and groups must be the same length; got "
            f"{y_true_arr.size}, {y_pred_arr.size}, {groups_arr.size}"
        )
    if privileged_value == unprivileged_value:
        raise ValueError("privileged_value and unprivileged_value must differ")

    privileged_mask = groups_arr == privileged_value
    unprivileged_mask = groups_arr == unprivileged_value
    if not privileged_mask.any():
        raise ValueError(f"no rows with privileged_value={privileged_value}")
    if not unprivileged_mask.any():
        raise ValueError(f"no rows with unprivileged_value={unprivileged_value}")

    privileged = _rates_for_mask(y_true_arr, y_pred_arr, privileged_mask, favorable_label)
    unprivileged = _rates_for_mask(y_true_arr, y_pred_arr, unprivileged_mask, favorable_label)

    disparate_impact = (
        unprivileged.selection_rate / privileged.selection_rate
        if privileged.selection_rate > 0
        else float("inf")
    )
    tpr_gap = unprivileged.tpr - privileged.tpr
    fpr_gap = unprivileged.fpr - privileged.fpr

    return FairnessResult(
        privileged=privileged,
        unprivileged=unprivileged,
        disparate_impact=disparate_impact,
        statistical_parity_difference=unprivileged.selection_rate - privileged.selection_rate,
        equal_opportunity_difference=tpr_gap,
        # Equalized odds requires both gaps to close, so the worse one governs.
        equalized_odds_difference=float(max(abs(tpr_gap), abs(fpr_gap))),
    )


def threshold_for_selection_rate(
    scores: ArrayLike,
    target_selection_rate: float,
) -> float:
    """Find the score threshold that approves ``target_selection_rate`` of applicants.

    ``scores`` are probabilities of the *unfavorable* outcome, and a row is predicted
    favorable when its score falls below the threshold. The threshold achieving a given
    approval rate is therefore that quantile of the score distribution.

    Comparing group fairness across models at a shared nominal threshold is invalid when
    the models are calibrated differently: a model that predicts lower probabilities
    approves more people, which mechanically compresses disparate impact toward 1.
    Matching the global selection rate first isolates fairness from calibration. On the
    German Credit prototype this accounted for roughly 28 percent of TabFM's apparent
    fairness advantage.
    """
    if not 0.0 <= target_selection_rate <= 1.0:
        raise ValueError(f"target_selection_rate must be in [0, 1], got {target_selection_rate}")
    return float(np.quantile(np.asarray(scores).ravel(), target_selection_rate))


def predict_at_threshold(
    scores: ArrayLike,
    threshold: float,
    favorable_label: int = 0,
) -> NDArray[np.int_]:
    """Convert unfavorable-outcome scores to labels at ``threshold``."""
    scores_arr = np.asarray(scores).ravel()
    unfavorable_label = 1 - favorable_label
    return np.where(scores_arr >= threshold, unfavorable_label, favorable_label).astype(int)
