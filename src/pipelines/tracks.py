"""Track T0: tuned baseline with no fairness intervention.

This is the control. Its purpose is to establish what discrimination is achievable on the
dataset when nothing is done about disparity, so that every mitigation can be quoted with
its actual cost rather than compared against a straw man.

The straw man is not hypothetical here. The earlier prototype compared a foundation model
against an untuned 300-tree Random Forest with no gradient boosting track at all, and
concluded the foundation model led on ROC-AUC. That conclusion was not publishable, because
the comparison had never included a tuned gradient booster. This track searches Random
Forest, XGBoost, LightGBM and regularised logistic regression under the corrected objective
before any fairness claim is made.

Everything reported comes from the test block. The calibration block is untouched here and
exists for the post-processing track, which must not see test rows.
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
from src.evaluation.group_fairness import group_fairness, predict_at_threshold
from src.evaluation.performance import (
    bootstrap_interval,
    performance_metrics,
    stratified_bootstrap_indices,
)
from src.paths import ARTIFACTS_DIR, REPORTS_DIR
from src.preprocessing.features import extract_features, extract_groups, extract_target
from src.training.search import (
    POSITIVE_CLASS,
    fit_best,
    positive_class_probabilities,
    run_search,
)

logger = logging.getLogger(__name__)

COMPARISON_PATH = REPORTS_DIR / "track_comparison.json"
DEFAULT_THRESHOLD = 0.5


@dataclass
class TrackRun:
    """One track's measured result, plus everything needed to reproduce it."""

    dataset: str
    track: str
    description: str
    seed: int
    threshold: float
    split_sizes: dict[str, int]
    split_fingerprint: str
    n_encoded_features: int
    model: dict[str, Any]
    performance: dict[str, float]
    fairness: dict[str, float]
    intervals: dict[str, dict[str, float]]
    protected_attribute: str
    n_bootstrap: int
    provenance: str
    recorded_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "track": self.track,
            "description": self.description,
            "seed": self.seed,
            "threshold": self.threshold,
            "split_sizes": self.split_sizes,
            "split_fingerprint": self.split_fingerprint,
            "n_encoded_features": self.n_encoded_features,
            "model": self.model,
            "performance": self.performance,
            "fairness": self.fairness,
            "intervals": self.intervals,
            "protected_attribute": self.protected_attribute,
            "n_bootstrap": self.n_bootstrap,
            "provenance": self.provenance,
            "recorded_at": self.recorded_at,
        }


def _interval_statistics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    scores: NDArray[np.float64],
    groups: NDArray[Any],
    spec: DatasetSpec,
) -> dict[str, Any]:
    """Statistics evaluated on bootstrap replicates, keyed by metric name.

    Each is a function of row indices, so all of them see the identical replicate.
    """
    attribute = spec.protected_attribute()

    def fairness_on(index: NDArray[np.int_]):
        return group_fairness(
            y_true[index],
            y_pred[index],
            groups[index],
            privileged_value=attribute.privileged_value,
            unprivileged_value=attribute.unprivileged_value,
            favorable_label=spec.favorable_label,
        )

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
        "disparate_impact": lambda i: fairness_on(i).disparate_impact,
        "statistical_parity_difference": lambda i: fairness_on(
            i
        ).statistical_parity_difference,
        "equal_opportunity_difference": lambda i: fairness_on(i).equal_opportunity_difference,
        "equalized_odds_difference": lambda i: fairness_on(i).equalized_odds_difference,
    }


def run_baseline_track(
    spec: DatasetSpec,
    *,
    seed: int,
    n_trials: int,
    cv_folds: int,
    test_size: float,
    calibration_size: float,
    n_bootstrap: int = 2000,
    threshold: float = DEFAULT_THRESHOLD,
    save_model: bool = True,
) -> tuple[TrackRun, Any, DataSplit]:
    """Search, fit and evaluate the unmitigated baseline.

    Returns:
        The recorded run, the fitted estimator, and the split it used.
    """
    frame = spec.load()
    split = get_or_create_split(
        spec,
        frame,
        test_size=test_size,
        calibration_size=calibration_size,
        seed=seed,
    )
    logger.info("split %s: %s", spec.name, split.sizes)

    train = split.frame(frame, "train")
    test = split.frame(frame, "test")
    X_train, y_train = extract_features(spec, train), extract_target(spec, train)
    X_test, y_test = extract_features(spec, test), extract_target(spec, test)
    groups_test = extract_groups(spec, test)

    search = run_search(
        spec, X_train, y_train, n_trials=n_trials, cv_folds=cv_folds, seed=seed
    )
    estimator = fit_best(spec, search, X_train, y_train)

    scores = positive_class_probabilities(estimator, X_test)
    y_pred = predict_at_threshold(scores, threshold, favorable_label=spec.favorable_label)

    performance = performance_metrics(y_test, y_pred, scores, positive_class=POSITIVE_CLASS)
    attribute = spec.protected_attribute()
    fairness = group_fairness(
        y_test,
        y_pred,
        groups_test,
        privileged_value=attribute.privileged_value,
        unprivileged_value=attribute.unprivileged_value,
        favorable_label=spec.favorable_label,
    )

    replicates = stratified_bootstrap_indices(
        y_test, groups_test, n_replicates=n_bootstrap, seed=seed
    )
    statistics = _interval_statistics(y_test, y_pred, scores, groups_test, spec)
    combined = {**performance, **fairness.as_dict()}
    intervals = {
        name: bootstrap_interval(statistic, replicates, combined[name]).as_dict()
        for name, statistic in statistics.items()
    }

    encoded_width = int(estimator.named_steps["classifier"].n_features_in_)
    run = TrackRun(
        dataset=spec.name,
        track="T0",
        description=(
            "Tuned baseline, no fairness intervention. Optuna search over random forest, "
            "XGBoost, LightGBM and logistic regression, selected by cross-validated "
            "ROC-AUC on the training block only."
        ),
        seed=seed,
        threshold=threshold,
        split_sizes=split.sizes,
        split_fingerprint=split.fingerprint,
        n_encoded_features=encoded_width,
        model=search.summary(),
        performance=performance,
        fairness=fairness.as_dict(),
        intervals=intervals,
        protected_attribute=attribute.column,
        n_bootstrap=n_bootstrap,
        provenance=spec.provenance,
    )

    if save_model:
        path = ARTIFACTS_DIR / "tracks" / f"{spec.name}_T0.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": estimator, "run": run.to_dict()}, path)
        logger.info("model written to %s", path)

    return run, estimator, split


def record_run(run: TrackRun, path: Path = COMPARISON_PATH) -> Path:
    """Merge a run into the comparison artifact, keyed by dataset and track.

    This file is the only permitted source for a published number. README tables and charts
    read from it; nothing is typed by hand. That rule exists because the previously
    published comparison chart contained values that appear in no artifact at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))

    runs = existing.get("runs", {})
    runs[f"{run.dataset}|{run.track}"] = run.to_dict()
    existing["runs"] = runs
    existing["schema_version"] = 1

    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    return path
