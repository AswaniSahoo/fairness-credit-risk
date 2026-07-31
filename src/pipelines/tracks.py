"""The four comparison tracks.

Each track is one intervention measured under identical conditions: the same split artifact,
the same encoder, the same metric code, the same bootstrap replicates. Only the intervention
differs, so a difference between two tracks is attributable to the intervention rather than to
the setup.

| Track | Intervention                          | Stage           |
|-------|---------------------------------------|-----------------|
| T0    | none                                  | control         |
| T1    | Kamiran-Calders reweighing, per fold  | pre-processing  |
| T2    | fairlearn ExponentiatedGradient       | in-processing   |
| T3    | group-specific thresholds             | post-processing |

T0 exists so that every mitigation can be quoted with its actual cost rather than against a
straw man. That was not hypothetical: an earlier prototype compared a foundation model against
an untuned 300-tree forest with no gradient booster in the comparison at all.

T3 is retained for comparison only and is marked not deployable. Keying a credit decision on
the applicant's sex is disparate treatment under ECOA and Regulation B regardless of whether
it improves a fairness metric. Measuring it and saying so is more useful than omitting it.

Two deliberate deviations from "every track runs its own search", both recorded in each run's
`search_note` so a reader is not misled:

- T2 reuses T0's selected configuration as its base learner. Searching it would mean 60 trials
  times 5 folds times roughly 10 ExponentiatedGradient refits. Holding the base learner fixed
  also isolates the constraint as the only difference, which is the stronger comparison.
- T3 wraps T0's fitted model, because post-processing operates on a trained model by
  definition. Its thresholds are fitted on the calibration block, never on train or test.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from numpy.typing import NDArray
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

from src.data.registry import DatasetSpec
from src.data.splits import DataSplit, get_or_create_split
from src.evaluation.explanations import (
    IMPORTANCE_SAMPLE_SIZE,
    build_explainer,
    explain_batch,
    global_feature_importance,
    sample_background,
)
from src.evaluation.group_fairness import (
    FairnessResult,
    group_fairness,
    predict_at_threshold,
    threshold_for_selection_rate,
)
from src.evaluation.performance import (
    bootstrap_interval,
    performance_metrics,
    stratified_bootstrap_indices,
)
from src.paths import ARTIFACTS_DIR, REPORTS_DIR
from src.preprocessing.features import (
    encoded_feature_names,
    extract_features,
    extract_groups,
    extract_target,
)
from src.preprocessing.reweighing import reweighing_weights
from src.training.search import (
    POSITIVE_CLASS,
    SearchResult,
    build_pipeline,
    fit_best,
    positive_class_probabilities,
    run_search,
)

logger = logging.getLogger(__name__)

COMPARISON_PATH = REPORTS_DIR / "track_comparison.json"
DEFAULT_THRESHOLD = 0.5


@dataclass
class TrackRun:
    """One track's measured result, plus everything needed to reproduce and judge it."""

    dataset: str
    track: str
    intervention: str
    stage: str
    description: str
    deployable: bool
    deployability_note: str
    search_note: str
    seed: int
    threshold: float | None
    selection_rate: float
    split_sizes: dict[str, int]
    split_fingerprint: str
    n_encoded_features: int
    model: dict[str, Any]
    performance: dict[str, float]
    fairness: dict[str, float]
    intervals: dict[str, dict[str, float]]
    matched: dict[str, Any] | None
    protected_attribute: str
    n_bootstrap: int
    feature_importance: list[dict[str, Any]] | None
    provenance: str
    recorded_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "dataset", "track", "intervention", "stage", "description",
                "deployable", "deployability_note", "search_note", "seed", "threshold",
                "selection_rate", "split_sizes", "split_fingerprint",
                "n_encoded_features", "model", "performance", "fairness", "intervals",
                "matched", "protected_attribute", "n_bootstrap", "feature_importance",
                "provenance", "recorded_at",
            )
        }


def _fairness(
    spec: DatasetSpec,
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    groups: NDArray[Any],
) -> FairnessResult:
    attribute = spec.protected_attribute()
    return group_fairness(
        y_true,
        y_pred,
        groups,
        privileged_value=attribute.privileged_value,
        unprivileged_value=attribute.unprivileged_value,
        favorable_label=spec.favorable_label,
    )


def _interval_statistics(
    spec: DatasetSpec,
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    scores: NDArray[np.float64],
    groups: NDArray[Any],
) -> dict[str, Any]:
    """Statistics as functions of row indices, so every one sees the identical replicate."""
    return {
        "roc_auc": lambda i: float(roc_auc_score(y_true[i], scores[i])),
        "balanced_accuracy": lambda i: float(balanced_accuracy_score(y_true[i], y_pred[i])),
        "f1": lambda i: float(
            f1_score(y_true[i], y_pred[i], pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "recall": lambda i: float(
            recall_score(y_true[i], y_pred[i], pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "brier": lambda i: float(brier_score_loss(y_true[i], scores[i])),
        "disparate_impact": lambda i: _fairness(
            spec, y_true[i], y_pred[i], groups[i]
        ).disparate_impact,
        "statistical_parity_difference": lambda i: _fairness(
            spec, y_true[i], y_pred[i], groups[i]
        ).statistical_parity_difference,
        "equal_opportunity_difference": lambda i: _fairness(
            spec, y_true[i], y_pred[i], groups[i]
        ).equal_opportunity_difference,
        "equalized_odds_difference": lambda i: _fairness(
            spec, y_true[i], y_pred[i], groups[i]
        ).equalized_odds_difference,
    }


def _matched_block(
    spec: DatasetSpec,
    y_true: NDArray[np.int_],
    scores: NDArray[np.float64],
    groups: NDArray[Any],
    reference_selection_rate: float,
) -> dict[str, Any]:
    """Fairness recomputed at the threshold reproducing a reference approval rate.

    Comparing group fairness across models at a shared nominal threshold is invalid when they
    are calibrated differently: a model predicting lower probabilities approves more people,
    which mechanically compresses disparate impact toward 1. On the earlier prototype this
    accounted for roughly 28 percent of one model's apparent fairness advantage, so the
    matched-rate view is reported alongside the nominal one rather than instead of it.
    """
    threshold = threshold_for_selection_rate(scores, reference_selection_rate)
    matched_pred = predict_at_threshold(scores, threshold, favorable_label=spec.favorable_label)
    fairness = _fairness(spec, y_true, matched_pred, groups)

    return {
        "reference_selection_rate": reference_selection_rate,
        "threshold": threshold,
        "selection_rate": float(np.mean(matched_pred == spec.favorable_label)),
        "fairness": fairness.as_dict(),
        "performance": performance_metrics(
            y_true, matched_pred, scores, positive_class=POSITIVE_CLASS
        ),
    }


def _assemble_run(
    spec: DatasetSpec,
    split: DataSplit,
    *,
    track: str,
    intervention: str,
    stage: str,
    description: str,
    deployable: bool,
    deployability_note: str,
    search_note: str,
    model_summary: dict[str, Any],
    n_encoded_features: int,
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    scores: NDArray[np.float64],
    groups: NDArray[Any],
    seed: int,
    n_bootstrap: int,
    threshold: float | None,
    reference_selection_rate: float | None,
    feature_importance: list[dict[str, Any]] | None = None,
) -> TrackRun:
    """Build a run record. Every track goes through this function, and only this function."""
    performance = performance_metrics(y_true, y_pred, scores, positive_class=POSITIVE_CLASS)
    fairness = _fairness(spec, y_true, y_pred, groups)

    replicates = stratified_bootstrap_indices(
        y_true, groups, n_replicates=n_bootstrap, seed=seed
    )
    statistics = _interval_statistics(spec, y_true, y_pred, scores, groups)
    combined = {**performance, **fairness.as_dict()}
    intervals = {
        name: bootstrap_interval(statistic, replicates, combined[name]).as_dict()
        for name, statistic in statistics.items()
    }

    matched = None
    if reference_selection_rate is not None:
        matched = _matched_block(spec, y_true, scores, groups, reference_selection_rate)

    return TrackRun(
        dataset=spec.name,
        track=track,
        intervention=intervention,
        stage=stage,
        description=description,
        deployable=deployable,
        deployability_note=deployability_note,
        search_note=search_note,
        seed=seed,
        threshold=threshold,
        selection_rate=float(np.mean(y_pred == spec.favorable_label)),
        split_sizes=split.sizes,
        split_fingerprint=split.fingerprint,
        n_encoded_features=n_encoded_features,
        model=model_summary,
        performance=performance,
        fairness=fairness.as_dict(),
        intervals=intervals,
        matched=matched,
        protected_attribute=spec.protected_attribute().column,
        n_bootstrap=n_bootstrap,
        feature_importance=feature_importance,
        provenance=spec.provenance,
    )


@dataclass
class TrackInputs:
    """The shared data every track starts from."""

    frame: pd.DataFrame
    split: DataSplit
    X_train: pd.DataFrame
    y_train: NDArray[np.int_]
    groups_train: NDArray[Any]
    X_calibration: pd.DataFrame
    y_calibration: NDArray[np.int_]
    groups_calibration: NDArray[Any]
    X_test: pd.DataFrame
    y_test: NDArray[np.int_]
    groups_test: NDArray[Any]


def load_inputs(
    spec: DatasetSpec,
    *,
    seed: int,
    test_size: float,
    calibration_size: float,
) -> TrackInputs:
    """Load the dataset and the one shared split, and slice it into blocks."""
    frame = spec.load()
    split = get_or_create_split(
        spec, frame, test_size=test_size, calibration_size=calibration_size, seed=seed
    )
    blocks = {name: split.frame(frame, name) for name in ("train", "calibration", "test")}
    logger.info("%s split: %s", spec.name, split.sizes)

    return TrackInputs(
        frame=frame,
        split=split,
        X_train=extract_features(spec, blocks["train"]),
        y_train=extract_target(spec, blocks["train"]),
        groups_train=extract_groups(spec, blocks["train"]),
        X_calibration=extract_features(spec, blocks["calibration"]),
        y_calibration=extract_target(spec, blocks["calibration"]),
        groups_calibration=extract_groups(spec, blocks["calibration"]),
        X_test=extract_features(spec, blocks["test"]),
        y_test=extract_target(spec, blocks["test"]),
        groups_test=extract_groups(spec, blocks["test"]),
    )


def compute_shap_importance(
    estimator: Any,
    X_train: pd.DataFrame,
    seed: int,
) -> tuple[list[dict[str, Any]], NDArray[np.float64]]:
    """Compute global SHAP feature importance and a background sample.

    Public because ``scripts/backfill_shap_background.py`` fills this in for artifacts
    fitted before the field existed. Sharing one function is what makes the backfilled
    value equal to the value a fresh run would have written; a second implementation
    would only look equivalent.

    Importance is estimated from ``IMPORTANCE_SAMPLE_SIZE`` training rows rather than the
    whole block, because interventional SHAP cost is linear in rows and the quantity is a
    per-feature mean. The explained rows are drawn under a different seed from the
    background rows so the reference distribution is not the sample being explained.

    Returns:
        A tuple of (feature_importance list for the run record, background array for the
        track artifact).
    """
    encoder = estimator.named_steps["encoder"]
    feature_names = encoded_feature_names(encoder)
    X_train_encoded = np.asarray(encoder.transform(X_train), dtype=np.float64)

    background = sample_background(X_train_encoded, seed=seed)
    explained = sample_background(
        X_train_encoded, n_samples=IMPORTANCE_SAMPLE_SIZE, seed=seed + 1
    )

    explainer = build_explainer(estimator, background)
    shap_values = explain_batch(explainer, explained, feature_names)
    importance = global_feature_importance(shap_values, feature_names)
    logger.info(
        "SHAP importance over %d of %d training rows: top feature %s (mean |SHAP| %.4f)",
        len(explained),
        len(X_train_encoded),
        importance[0]["feature"] if importance else "n/a",
        importance[0]["mean_abs_shap"] if importance else 0.0,
    )
    return importance, background


def run_baseline_track(
    spec: DatasetSpec,
    inputs: TrackInputs,
    *,
    seed: int,
    n_trials: int,
    cv_folds: int,
    n_bootstrap: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[TrackRun, Any, SearchResult, NDArray[np.float64]]:
    """T0: tuned model, no fairness intervention. The control."""
    search = run_search(
        spec, inputs.X_train, inputs.y_train, n_trials=n_trials, cv_folds=cv_folds, seed=seed
    )
    estimator = fit_best(spec, search, inputs.X_train, inputs.y_train)

    scores = positive_class_probabilities(estimator, inputs.X_test)
    y_pred = predict_at_threshold(scores, threshold, favorable_label=spec.favorable_label)

    importance, background = compute_shap_importance(estimator, inputs.X_train, seed)

    run = _assemble_run(
        spec,
        inputs.split,
        track="T0",
        intervention="none",
        stage="control",
        description=(
            "Tuned baseline. Optuna search over random forest, XGBoost, LightGBM and "
            "regularised logistic regression, selected by cross-validated ROC-AUC on the "
            "training block only."
        ),
        deployable=True,
        deployability_note=(
            "Deployable. A single global threshold is applied to every applicant and no "
            "protected attribute reaches the model."
        ),
        search_note=f"Own search: {n_trials} trials, {cv_folds}-fold CV.",
        model_summary=search.summary(),
        n_encoded_features=int(estimator.named_steps["classifier"].n_features_in_),
        y_true=inputs.y_test,
        y_pred=y_pred,
        scores=scores,
        groups=inputs.groups_test,
        seed=seed,
        n_bootstrap=n_bootstrap,
        threshold=threshold,
        reference_selection_rate=None,
        feature_importance=importance,
    )
    return run, estimator, search, background


def run_reweighing_track(
    spec: DatasetSpec,
    inputs: TrackInputs,
    *,
    seed: int,
    n_trials: int,
    cv_folds: int,
    n_bootstrap: int,
    reference_selection_rate: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[TrackRun, Any, SearchResult, NDArray[np.float64]]:
    """T1: Kamiran-Calders reweighing, computed inside each training fold.

    The weights are recomputed from the rows of each training fold rather than once over the
    whole training block. Computing them once and slicing would let the validation fold's group
    composition influence the weights the model was fitted under, which is the leakage that
    finding B3 required removing.
    """
    search = run_search(
        spec,
        inputs.X_train,
        inputs.y_train,
        n_trials=n_trials,
        cv_folds=cv_folds,
        seed=seed,
        groups=inputs.groups_train,
        weight_fn=reweighing_weights,
    )
    estimator = fit_best(
        spec,
        search,
        inputs.X_train,
        inputs.y_train,
        groups=inputs.groups_train,
        weight_fn=reweighing_weights,
    )

    scores = positive_class_probabilities(estimator, inputs.X_test)
    y_pred = predict_at_threshold(scores, threshold, favorable_label=spec.favorable_label)

    importance, background = compute_shap_importance(estimator, inputs.X_train, seed)

    weights = reweighing_weights(inputs.y_train, inputs.groups_train)
    summary = {
        **search.summary(),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }

    run = _assemble_run(
        spec,
        inputs.split,
        track="T1",
        intervention="Kamiran-Calders reweighing",
        stage="pre-processing",
        description=(
            "Sample weights equalising the group-by-outcome cells, recomputed inside every "
            "cross-validation fold from that fold's training rows only."
        ),
        deployable=True,
        deployability_note=(
            "Deployable. Weights affect training only; inference is unchanged and sees no "
            "protected attribute."
        ),
        search_note=f"Own search under weighting: {n_trials} trials, {cv_folds}-fold CV.",
        model_summary=summary,
        n_encoded_features=int(estimator.named_steps["classifier"].n_features_in_),
        y_true=inputs.y_test,
        y_pred=y_pred,
        scores=scores,
        groups=inputs.groups_test,
        seed=seed,
        n_bootstrap=n_bootstrap,
        threshold=threshold,
        reference_selection_rate=reference_selection_rate,
        feature_importance=importance,
    )
    return run, estimator, search, background


EPSILON_GRID = (0.02, 0.05, 0.10)
# How much balanced accuracy the constrained model may give up against the control, judged on
# the calibration block. A demographic parity constraint can always be satisfied by approving
# everybody, so a sweep without an accuracy floor can select a classifier with a perfect
# fairness metric and no ability to distinguish applicants at all.
ACCURACY_FLOOR = 0.03


def _selection_gap(
    spec: DatasetSpec,
    estimator: Any,
    X: pd.DataFrame,  # noqa: N803
    y: NDArray[np.int_],
    groups: NDArray[Any],
) -> float:
    """Selection-rate gap of a thresholded estimator on one block."""
    scores = positive_class_probabilities(estimator, X)
    y_pred = predict_at_threshold(scores, DEFAULT_THRESHOLD, favorable_label=spec.favorable_label)
    return _fairness(spec, y, y_pred, groups).statistical_parity_difference


def run_constrained_track(
    spec: DatasetSpec,
    inputs: TrackInputs,
    *,
    baseline_search: SearchResult,
    baseline_estimator: Any,
    seed: int,
    n_bootstrap: int,
    reference_selection_rate: float,
    epsilon_grid: tuple[float, ...] = EPSILON_GRID,
    accuracy_floor: float = ACCURACY_FLOOR,
    max_iter: int = 50,
) -> tuple[TrackRun, Any]:
    """T2: fairlearn ExponentiatedGradient under a demographic parity constraint.

    Constrained optimisation rather than sample weighting: the reduction solves a saddle-point
    problem, repeatedly refitting the base learner under Lagrange-multiplier weights until the
    training-set selection-rate gap is within the bound. The result is a randomised ensemble,
    which is why its scores are the mixture probability rather than a single model's.

    Two things measured here are worth stating plainly, because both were arrived at by getting
    it wrong first.

    The constraint is enforced on the data it is measured on, which is the training block. A
    tuned booster fits those rows closely enough that its training-set parity gap is already
    inside a tight bound while its test-set gap is an order of magnitude larger. When that
    happens the reduction has nothing to correct, returns the base learner, and the track is a
    null result for a specific and reportable reason rather than because the method is useless.
    `constraint_binding` in the record says which case applies.

    The base learner keeps T0's configuration, including its class weighting. Removing the class
    weighting so that the reduction's instance weights would be the only reweighting was tried
    and rejected: with `scale_pos_weight` reset to 1.0 the booster predicted the positive class
    for 2 percent of training rows, and the reduction then mixed near-trivial classifiers into a
    model that approved 97.5 percent of applicants at a perfect fairness metric.

    The constraint strength is swept and chosen on the calibration block, never on test.
    """
    params = dict(baseline_search.params)
    train_gap = _selection_gap(
        spec, baseline_estimator, inputs.X_train, inputs.y_train, inputs.groups_train
    )
    calibration_pred_baseline = predict_at_threshold(
        positive_class_probabilities(baseline_estimator, inputs.X_calibration),
        DEFAULT_THRESHOLD,
        favorable_label=spec.favorable_label,
    )
    baseline_calibration = float(
        balanced_accuracy_score(inputs.y_calibration, calibration_pred_baseline)
    )

    frontier: list[dict[str, float]] = []
    fitted: dict[float, Any] = {}
    differs_from_baseline = False
    for epsilon in epsilon_grid:
        mitigator = ExponentiatedGradient(
            estimator=build_pipeline(spec, baseline_search.model_type, params, seed),
            constraints=DemographicParity(difference_bound=epsilon),
            eps=epsilon,
            max_iter=max_iter,
            sample_weight_name="classifier__sample_weight",
        )
        mitigator.fit(inputs.X_train, inputs.y_train, sensitive_features=inputs.groups_train)
        fitted[epsilon] = mitigator

        calibration_pred = np.asarray(
            mitigator.predict(inputs.X_calibration, random_state=seed)
        ).astype(int)
        differs_from_baseline |= not np.array_equal(
            calibration_pred, calibration_pred_baseline
        )
        calibration_fairness = _fairness(
            spec, inputs.y_calibration, calibration_pred, inputs.groups_calibration
        )
        candidate = {
            "epsilon": epsilon,
            "n_ensemble_predictors": len(mitigator.predictors_),
            "n_weights_nonzero": int(
                (np.asarray(mitigator.weights_, dtype=float) > 1e-6).sum()
            ),
            "calibration_balanced_accuracy": float(
                balanced_accuracy_score(inputs.y_calibration, calibration_pred)
            ),
            "calibration_statistical_parity_difference": (
                calibration_fairness.statistical_parity_difference
            ),
            "calibration_selection_rate": float(
                np.mean(calibration_pred == spec.favorable_label)
            ),
        }
        frontier.append(candidate)
        logger.info(
            "T2 epsilon %.2f: calibration balanced accuracy %.4f, SPD %+.4f, %d of %d "
            "ensemble weights nonzero",
            epsilon,
            candidate["calibration_balanced_accuracy"],
            candidate["calibration_statistical_parity_difference"],
            candidate["n_weights_nonzero"],
            candidate["n_ensemble_predictors"],
        )

    if not differs_from_baseline:
        chosen = frontier[0]
        constraint_binding = False
        selection_note = (
            f"The constraint did not bind at any epsilon in {list(epsilon_grid)}. The base "
            f"learner's training-block parity gap is {train_gap:+.4f}, already inside the "
            "tightest bound, so the reduction returned the base learner unchanged. The "
            "test-block disparity is a generalisation gap, which a constraint measured on the "
            "training distribution cannot see. The tightest epsilon is reported."
        )
    else:
        eligible = [
            candidate
            for candidate in frontier
            if candidate["calibration_balanced_accuracy"]
            >= baseline_calibration - accuracy_floor
        ]
        constraint_binding = True
        if eligible:
            chosen = min(
                eligible,
                key=lambda c: abs(c["calibration_statistical_parity_difference"]),
            )
            selection_note = (
                "Chosen on calibration: smallest parity gap among candidates within "
                f"{accuracy_floor} balanced accuracy of the control's "
                f"{baseline_calibration:.4f}."
            )
        else:
            chosen = max(frontier, key=lambda c: c["epsilon"])
            selection_note = (
                f"No candidate stayed within {accuracy_floor} balanced accuracy of the "
                f"control's {baseline_calibration:.4f} on calibration, so the loosest "
                "constraint is reported and the method is recorded as unable to pay for its "
                "parity gain here."
            )

    epsilon = float(chosen["epsilon"])
    mitigator = fitted[epsilon]
    logger.info("T2 selected epsilon %.2f. %s", epsilon, selection_note)

    # `_pmf_predict` is fairlearn's only access to the ensemble's mixture probabilities. The
    # public `predict` returns a randomised label draw, which cannot produce a ROC curve or a
    # Brier score. The private call is deliberate and pinned by the fairlearn version in
    # requirements.txt; a failure here should surface rather than be worked around silently.
    scores = np.asarray(mitigator._pmf_predict(inputs.X_test))[:, 1]

    # Labels come from the randomised draw, not from thresholding the mixture at 0.5. The
    # parity guarantee holds for the randomised classifier; a deterministic majority reading
    # of the same ensemble collapses back onto the base learner's decisions and reproduces the
    # control's disparity almost exactly, which is what a first implementation here did.
    y_pred = np.asarray(
        mitigator.predict(inputs.X_test, random_state=seed)
    ).astype(int)

    n_unique_scores = int(np.unique(np.round(scores, 6)).size)
    summary = {
        "model_type": f"exponentiated_gradient({baseline_search.model_type})",
        "params": params,
        "constraint": "demographic_parity",
        "epsilon": epsilon,
        "epsilon_grid": list(epsilon_grid),
        "epsilon_selection": selection_note,
        "constraint_binding": constraint_binding,
        "train_statistical_parity_difference": train_gap,
        "calibration_frontier": frontier,
        "baseline_calibration_balanced_accuracy": baseline_calibration,
        "max_iter": max_iter,
        "n_ensemble_predictors": len(mitigator.predictors_),
        "ensemble_weights_nonzero": int(
            (np.asarray(mitigator.weights_, dtype=float) > 1e-6).sum()
        ),
        # When the mixture concentrates on one predictor the score takes only two values, and
        # ROC-AUC then equals balanced accuracy by construction. Recorded so a reader does not
        # read the drop against T0's AUC as lost ranking ability: the constrained classifier
        # emits a decision, not a graded score.
        "n_unique_scores": n_unique_scores,
        "scores_are_graded": n_unique_scores > 2,
        "class_weighting": (
            "T0's class weighting is kept. Removing it so the reduction's instance weights "
            "would be the only reweighting produced a base learner predicting the positive "
            "class for 2 percent of training rows, and an ensemble approving 97.5 percent of "
            "applicants."
        ),
        "cv_score": baseline_search.cv_score,
        "scoring": baseline_search.scoring,
        "n_trials": baseline_search.n_trials,
        "cv_folds": baseline_search.cv_folds,
        "seed": seed,
    }

    run = _assemble_run(
        spec,
        inputs.split,
        track="T2",
        intervention="ExponentiatedGradient, demographic parity",
        stage="in-processing",
        description=(
            "Constrained optimisation. The reduction refits the base learner under "
            "Lagrange-multiplier weights until the selection-rate gap is within epsilon, "
            "producing a randomised ensemble."
        ),
        deployable=True,
        deployability_note=(
            "Deployable with a caveat: the classifier is randomised, so two identical "
            "applications can receive different decisions. That is defensible for a fairness "
            "constraint but has to be disclosed, and it complicates adverse-action reasoning."
        ),
        search_note=(
            "Reuses T0's selected configuration as the base learner. Searching it would cost "
            "trials times folds times ensemble refits, and holding the base learner fixed "
            f"isolates the constraint as the only difference from the control. {selection_note}"
        ),
        model_summary=summary,
        n_encoded_features=int(baseline_estimator.named_steps["classifier"].n_features_in_),
        y_true=inputs.y_test,
        y_pred=y_pred,
        scores=scores,
        groups=inputs.groups_test,
        seed=seed,
        n_bootstrap=n_bootstrap,
        # No global threshold: labels come from the randomised ensemble draw, so a cut-off
        # would misdescribe the decision rule. Metrics are from one draw at the run's seed.
        threshold=None,
        # Matching a target approval rate requires a graded score to take a quantile of. A
        # two-valued score cannot be re-thresholded, so the comparison is omitted rather than
        # reported as if it meant something.
        reference_selection_rate=reference_selection_rate if n_unique_scores > 2 else None,
        # T2 is a randomised ensemble; per-feature SHAP is not well-defined for the mixture.
        # T0's feature importance applies to the base learner, which is the same model.
        feature_importance=None,
    )
    return run, mitigator


def run_threshold_track(
    spec: DatasetSpec,
    inputs: TrackInputs,
    *,
    baseline_estimator: Any,
    baseline_search: SearchResult,
    seed: int,
    n_bootstrap: int,
) -> tuple[TrackRun, Any]:
    """T3: group-specific thresholds fitted on the calibration block.

    Retained for comparison and marked not deployable. The decision rule reads the applicant's
    sex and applies a different cut-off accordingly, which is disparate treatment under ECOA
    and Regulation B whatever it does to a fairness metric. This track documents why the
    project's original design, which shipped exactly this rule behind the API, was abandoned.

    Thresholds are fitted on calibration. The original implementation fitted them on the
    training block's own probabilities, where a tree ensemble is close to memorising its
    training set, so the cut-offs were tuned to noise the model had already absorbed. That was
    finding B4.
    """
    postprocessor = ThresholdOptimizer(
        estimator=baseline_estimator,
        constraints="true_positive_rate_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    postprocessor.fit(
        inputs.X_calibration,
        inputs.y_calibration,
        sensitive_features=inputs.groups_calibration,
    )

    y_pred = np.asarray(
        postprocessor.predict(
            inputs.X_test,
            sensitive_features=inputs.groups_test,
            random_state=seed,
        )
    ).astype(int)
    # The underlying model is unchanged, so ranking and calibration metrics come from its own
    # scores. Only the decision rule differs.
    scores = positive_class_probabilities(baseline_estimator, inputs.X_test)

    summary = {
        "model_type": f"threshold_optimizer({baseline_search.model_type})",
        "params": baseline_search.params,
        "constraint": "true_positive_rate_parity",
        "objective": "balanced_accuracy_score",
        "fitted_on": "calibration",
        "cv_score": baseline_search.cv_score,
        "scoring": baseline_search.scoring,
        "n_trials": baseline_search.n_trials,
        "cv_folds": baseline_search.cv_folds,
        "seed": seed,
    }

    run = _assemble_run(
        spec,
        inputs.split,
        track="T3",
        intervention="group-specific thresholds",
        stage="post-processing",
        description=(
            "A separate decision threshold per protected group, chosen on the calibration "
            "block to equalise true positive rates."
        ),
        deployable=False,
        deployability_note=(
            "NOT deployable. The rule keys the decision on the applicant's sex, which is "
            "disparate treatment under ECOA and Regulation B even when it improves a group "
            "fairness metric. Reported to document why this design was abandoned."
        ),
        search_note="Wraps T0's fitted model; thresholds fitted on the calibration block.",
        model_summary=summary,
        n_encoded_features=int(baseline_estimator.named_steps["classifier"].n_features_in_),
        y_true=inputs.y_test,
        y_pred=y_pred,
        scores=scores,
        groups=inputs.groups_test,
        seed=seed,
        n_bootstrap=n_bootstrap,
        # No single global threshold exists; the cut-off depends on the applicant's group.
        threshold=None,
        reference_selection_rate=None,
        # T3 wraps T0's model unchanged; T0's feature importance applies directly.
        feature_importance=None,
    )
    return run, postprocessor


def save_track_model(
    spec: DatasetSpec,
    run: TrackRun,
    model: Any,
    background: NDArray[np.float64] | None = None,
) -> Path:
    """Persist a fitted track model next to the run record that describes it.

    ``background`` is a subsampled encoded training block used by the SHAP explainer at
    inference time. Including it in the artifact means serving does not need the full
    training data.
    """
    path = ARTIFACTS_DIR / "tracks" / f"{spec.name}_{run.track}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, Any] = {"model": model, "run": run.to_dict()}
    if background is not None:
        artifact["background"] = background
    joblib.dump(artifact, path)
    logger.info("model written to %s", path)
    return path


def record_run(run: TrackRun, path: Path = COMPARISON_PATH) -> Path:
    """Merge a run into the comparison artifact, keyed by dataset and track.

    This file is the only permitted source for a published number. README tables and charts are
    generated from it; nothing is typed by hand. That rule exists because the comparison chart
    this project previously published contained values that appear in no artifact at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))

    runs = existing.get("runs", {})
    runs[f"{run.dataset}|{run.track}"] = run.to_dict()
    existing["runs"] = runs
    existing["schema_version"] = 2

    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    return path
