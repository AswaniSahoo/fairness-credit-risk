"""Hyperparameter search for the model tracks.

Replaces `automl_tuner.py`, which had three defects that made its model selection
meaningless.

Finding B2, the reason a rewrite was needed rather than a patch: the objective read
``predict_proba(...)[:, 0]``, the probability of *good* credit, and passed it to
``roc_auc_score`` against a target where 1 means default. That returns ``1 - AUC``, so the
search maximised the inverse of discrimination and every ranking it produced was upside
down. A model scoring 0.84 was being credited with 0.16 and discarded. Here the positive
class is named once, in ``POSITIVE_CLASS``, and asserted against the fitted estimator's
``classes_`` so the two cannot drift apart.

Second, the encoder is inside the cross-validated pipeline, so scaler means and category
handling are refitted on each training fold. Fitting a preprocessor once on all of the
training data before cross-validation leaks fold-level statistics into the validation
score, which inflates it.

Third, the previous objective mixed a fairness term into model selection by default. The
control track needs to establish the accuracy ceiling without that, otherwise there is no
honest reference point to measure a mitigation's cost against. Fairness weighting is opt-in
through ``fairness_penalty``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.registry import DatasetSpec
from src.preprocessing.features import build_encoder

logger = logging.getLogger(__name__)

# The label whose probability the AUC is computed against. In both supported datasets the
# target is 1 for default and 0 for good credit, so the positive class is the default and
# `predict_proba` column 1 is the correct one. Finding B2 was using column 0 here.
POSITIVE_CLASS = 1

MODEL_TYPES = ("random_forest", "xgboost", "lightgbm", "logistic_regression")

WeightFn = Callable[[NDArray[np.int_], NDArray[Any]], NDArray[np.float64]]


@dataclass
class TrialRecord:
    """One evaluated configuration, kept for the optimisation history report."""

    number: int
    model_type: str
    score: float
    params: dict[str, Any]


@dataclass
class SearchResult:
    """Outcome of a search, with everything needed to rebuild the winning model."""

    model_type: str
    params: dict[str, Any]
    cv_score: float
    scoring: str
    n_trials: int
    cv_folds: int
    seed: int
    trials: list[TrialRecord] = field(default_factory=list)

    def history(self) -> pd.DataFrame:
        return pd.DataFrame([vars(record) for record in self.trials])

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable record for the comparison artifact."""
        return {
            "model_type": self.model_type,
            "params": self.params,
            "cv_score": self.cv_score,
            "scoring": self.scoring,
            "n_trials": self.n_trials,
            "cv_folds": self.cv_folds,
            "seed": self.seed,
        }


def build_classifier(model_type: str, params: dict[str, Any], seed: int) -> Any:
    """Instantiate a classifier from a parameter dict.

    Kept separate from parameter suggestion so the winning configuration is rebuilt by the
    same code path that evaluated it. In the previous implementation the rebuild duplicated
    the construction logic and used ``.get`` with silent defaults, so a reconstructed model
    could differ from the one that won.

    Raises:
        ValueError: On an unknown model type.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model_type == "xgboost":
        return XGBClassifier(
            random_state=seed, n_jobs=-1, eval_metric="logloss", **params
        )
    if model_type == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)
    if model_type == "logistic_regression":
        return LogisticRegression(random_state=seed, solver="saga", max_iter=5000, **params)
    raise ValueError(f"unknown model type {model_type!r}; known: {MODEL_TYPES}")


def suggest_params(trial: optuna.Trial, model_type: str) -> dict[str, Any]:
    """Sample a configuration for one model family.

    `class_weight` and `scale_pos_weight` are searched rather than fixed. The dataset is
    imbalanced, but whether rebalancing helps a threshold-free ranking metric is an
    empirical question, and the old pipeline hardcoded 'balanced' on the assumption that it
    does.

    Raises:
        ValueError: On an unknown model type.
    """
    if model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 24),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.5]),
            "class_weight": trial.suggest_categorical("rf_class_weight", [None, "balanced"]),
        }
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("xgb_max_depth", 2, 8),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("xgb_scale_pos_weight", 0.5, 4.0),
        }
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 800, step=50),
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 4, 64),
            "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.005, 0.3, log=True),
            "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 5, 60),
            "subsample": trial.suggest_float("lgbm_subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("lgbm_subsample_freq", 0, 5),
            "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-3, 20.0, log=True),
            "class_weight": trial.suggest_categorical("lgbm_class_weight", [None, "balanced"]),
        }
    if model_type == "logistic_regression":
        penalty = trial.suggest_categorical("lr_penalty", ["l1", "l2", "elasticnet"])
        params: dict[str, Any] = {
            "C": trial.suggest_float("lr_C", 1e-3, 20.0, log=True),
            "penalty": penalty,
            "class_weight": trial.suggest_categorical("lr_class_weight", [None, "balanced"]),
        }
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("lr_l1_ratio", 0.0, 1.0)
        return params
    raise ValueError(f"unknown model type {model_type!r}; known: {MODEL_TYPES}")


def build_pipeline(
    spec: DatasetSpec,
    model_type: str,
    params: dict[str, Any],
    seed: int,
) -> Pipeline:
    """Encoder plus classifier as one estimator.

    Bundling them is what makes per-fold refitting automatic and makes serving skew
    impossible: there is no way to transform with one encoder and predict with a model
    fitted under another.
    """
    return Pipeline(
        [
            ("encoder", build_encoder(spec)),
            ("classifier", build_classifier(model_type, params, seed)),
        ]
    )


def positive_class_probabilities(estimator: Pipeline, X: pd.DataFrame) -> NDArray[np.float64]:
    """Probability of the unfavorable outcome.

    Asserts the fitted class order instead of trusting it. ``predict_proba`` columns follow
    ``classes_``, which is sorted ascending, so column 1 is ``POSITIVE_CLASS`` for a 0/1
    target. Reading the wrong column silently returns ``1 - AUC``, which is finding B2 and
    is invisible in any output that does not compare against an independent computation.

    Raises:
        ValueError: If the fitted estimator's classes are not the expected 0/1 pair.
    """
    # .tolist() converts numpy scalars to Python ints, so the message reads [0, 2] rather
    # than [np.int64(0), np.int64(2)].
    classes = np.asarray(estimator.named_steps["classifier"].classes_).tolist()
    if classes != [1 - POSITIVE_CLASS, POSITIVE_CLASS]:
        raise ValueError(
            f"expected classes [{1 - POSITIVE_CLASS}, {POSITIVE_CLASS}], got {classes}"
        )
    return estimator.predict_proba(X)[:, 1]


def cross_validated_auc(
    spec: DatasetSpec,
    model_type: str,
    params: dict[str, Any],
    X: pd.DataFrame,  # noqa: N803
    y: NDArray[np.int_],
    *,
    cv_folds: int,
    seed: int,
    groups: NDArray[Any] | None = None,
    weight_fn: WeightFn | None = None,
) -> tuple[float, list[float]]:
    """Mean out-of-fold ROC-AUC for one configuration.

    ``weight_fn`` is called with the training fold's labels and protected values only, so a
    reweighting scheme computes its weights from training rows and never sees the validation
    fold. That is finding B3's requirement; passing precomputed whole-dataset weights would
    leak.

    Returns:
        The mean AUC and the per-fold values.

    Raises:
        ValueError: If ``weight_fn`` is given without ``groups``.
    """
    if weight_fn is not None and groups is None:
        raise ValueError("weight_fn requires groups")

    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores: list[float] = []

    for train_idx, validation_idx in folds.split(X, y):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]
        y_train, y_validation = y[train_idx], y[validation_idx]

        estimator = build_pipeline(spec, model_type, params, seed)
        if weight_fn is not None:
            assert groups is not None  # narrowed by the guard above
            weights = weight_fn(y_train, groups[train_idx])
            estimator.fit(X_train, y_train, classifier__sample_weight=weights)
        else:
            estimator.fit(X_train, y_train)

        scores.append(
            float(
                roc_auc_score(
                    y_validation, positive_class_probabilities(estimator, X_validation)
                )
            )
        )

    return float(np.mean(scores)), scores


def run_search(
    spec: DatasetSpec,
    X: pd.DataFrame,  # noqa: N803
    y: NDArray[np.int_],
    *,
    n_trials: int,
    cv_folds: int,
    seed: int,
    model_types: Sequence[str] = MODEL_TYPES,
    groups: NDArray[Any] | None = None,
    weight_fn: WeightFn | None = None,
) -> SearchResult:
    """Search hyperparameters by cross-validated ROC-AUC on the training block.

    The objective is discrimination alone. Fairness is measured afterwards on the test
    block, and mitigation tracks change the training procedure rather than the selection
    criterion, so the tracks stay comparable on the axis they are being compared along.
    """
    records: list[TrialRecord] = []

    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical("model_type", list(model_types))
        params = suggest_params(trial, model_type)
        mean_auc, _ = cross_validated_auc(
            spec,
            model_type,
            params,
            X,
            y,
            cv_folds=cv_folds,
            seed=seed,
            groups=groups,
            weight_fn=weight_fn,
        )
        records.append(
            TrialRecord(
                number=trial.number, model_type=model_type, score=mean_auc, params=params
            )
        )
        return mean_auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    # Exceptions are deliberately not swallowed. The previous implementation returned 0.0
    # from a bare `except`, so a systematically failing model family looked merely bad and
    # the failure never surfaced. See finding S5.
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = max(records, key=lambda record: record.score)
    logger.info(
        "search complete: %s trials, best %s at ROC-AUC %.4f",
        len(records),
        best.model_type,
        best.score,
    )
    return SearchResult(
        model_type=best.model_type,
        params=best.params,
        cv_score=best.score,
        scoring="roc_auc",
        n_trials=n_trials,
        cv_folds=cv_folds,
        seed=seed,
        trials=records,
    )


def fit_best(
    spec: DatasetSpec,
    result: SearchResult,
    X: pd.DataFrame,  # noqa: N803
    y: NDArray[np.int_],
    *,
    groups: NDArray[Any] | None = None,
    weight_fn: WeightFn | None = None,
) -> Pipeline:
    """Refit the winning configuration on the whole training block."""
    estimator = build_pipeline(spec, result.model_type, result.params, result.seed)
    if weight_fn is not None:
        if groups is None:
            raise ValueError("weight_fn requires groups")
        estimator.fit(X, y, classifier__sample_weight=weight_fn(y, groups))
    else:
        estimator.fit(X, y)
    return estimator
