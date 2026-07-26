"""Tests for the model search.

The central test is finding B2. The old objective scored ``predict_proba[:, 0]`` against a
target where 1 means default, so it computed ``1 - AUC`` and selected models by inverted
discrimination. A test that merely asserts "AUC is a float between 0 and 1" would have
passed against that bug, so the assertions here compare against an independently computed
value and against the 0.5 no-skill line.

These tests fit models, but on 200 synthetic rows with tiny estimators, so they stay in the
default `unit` run. The B2 regression must execute on every CI run to be worth having.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data.registry import DatasetSpec, ProtectedAttribute
from src.paths import PROCESSED_DATA_DIR
from src.training.search import (
    POSITIVE_CLASS,
    build_classifier,
    build_pipeline,
    cross_validated_auc,
    fit_best,
    positive_class_probabilities,
    run_search,
    suggest_params,
)

pytestmark = pytest.mark.unit

SEED = 0
N_ROWS = 200

TOY_SPEC = DatasetSpec(
    name="toy",
    path=PROCESSED_DATA_DIR / "does_not_need_to_exist.csv",
    target="y",
    favorable_label=0,
    numeric=("x1", "x2"),
    ordinal=(),
    nominal=("c",),
    excluded=("g",),
    protected=(
        ProtectedAttribute(
            column="g",
            privileged_value=1,
            unprivileged_value=0,
            prohibited_basis=True,
            rationale="synthetic",
        ),
    ),
    primary_protected="g",
    provenance="synthetic fixture",
    nominal_categories={"c": (0, 1, 2)},
)


@pytest.fixture(scope="module")
def separable():
    """A target that is strongly but not perfectly predictable from x1.

    Deliberately learnable: a correct AUC lands near 0.95, and the inverted computation
    lands near 0.05, so the two are impossible to confuse.
    """
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N_ROWS)
    signal = x1 + 0.35 * rng.normal(size=N_ROWS)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": rng.normal(size=N_ROWS),
            "c": rng.integers(0, 3, size=N_ROWS),
            "g": rng.integers(0, 2, size=N_ROWS),
            "y": (signal > 0).astype(int),
        }
    )


@pytest.fixture(scope="module")
def X(separable):  # noqa: N802
    return separable[["x1", "x2", "c"]]


@pytest.fixture(scope="module")
def y(separable):
    return separable["y"].to_numpy()


def test_positive_class_is_the_default_label():
    # Both registered datasets encode default as 1 and good credit as 0.
    assert POSITIVE_CLASS == 1


def test_probabilities_are_read_from_the_default_class_column(X, y):
    """Finding B2, at the narrowest point.

    A model fitted on a learnable target must give the default class a higher probability
    for rows that defaulted. Reading column 0 instead returns the complement, and its AUC
    is 1 minus the correct one.
    """
    estimator = build_pipeline(TOY_SPEC, "logistic_regression", {"C": 1.0}, SEED).fit(X, y)

    correct = positive_class_probabilities(estimator, X)
    inverted = estimator.predict_proba(X)[:, 0]

    assert np.allclose(correct + inverted, 1.0)
    assert roc_auc_score(y, correct) > 0.9
    assert roc_auc_score(y, inverted) == pytest.approx(1.0 - roc_auc_score(y, correct))
    # The mean probability of default must be higher among the rows that defaulted.
    assert correct[y == 1].mean() > correct[y == 0].mean()


def test_probabilities_raise_when_the_class_order_is_not_the_expected_pair(X):
    y_shifted = np.where(np.arange(len(X)) % 2 == 0, 0, 2)
    estimator = build_pipeline(TOY_SPEC, "logistic_regression", {"C": 1.0}, SEED)
    estimator.fit(X, y_shifted)

    with pytest.raises(ValueError, match=r"expected classes \[0, 1\], got \[0, 2\]"):
        positive_class_probabilities(estimator, X)


@pytest.mark.parametrize(
    "model_type", ["random_forest", "xgboost", "lightgbm", "logistic_regression"]
)
def test_cross_validated_auc_beats_no_skill_for_every_model_family(X, y, model_type):
    """Finding B2 regression across the whole search space.

    Under the old objective each of these would score near 0.05 and the search would rank
    them by which model was worst.
    """
    params = {
        "random_forest": {"n_estimators": 40, "max_depth": 4},
        "xgboost": {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.2},
        "lightgbm": {"n_estimators": 40, "num_leaves": 8, "learning_rate": 0.2},
        "logistic_regression": {"C": 1.0},
    }[model_type]

    mean_auc, folds = cross_validated_auc(
        TOY_SPEC, model_type, params, X, y, cv_folds=3, seed=SEED
    )

    assert len(folds) == 3
    assert mean_auc > 0.85, model_type
    assert mean_auc == pytest.approx(float(np.mean(folds)))


def test_cross_validated_auc_equals_an_independently_computed_fold_score(X, y):
    """Reproduces the first fold by hand, outside the function under test."""
    from sklearn.model_selection import StratifiedKFold

    params = {"C": 1.0}
    _, folds = cross_validated_auc(
        TOY_SPEC, "logistic_regression", params, X, y, cv_folds=3, seed=SEED
    )

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    train_idx, validation_idx = next(iter(splitter.split(X, y)))
    estimator = build_pipeline(TOY_SPEC, "logistic_regression", params, SEED)
    estimator.fit(X.iloc[train_idx], y[train_idx])
    expected = roc_auc_score(
        y[validation_idx], estimator.predict_proba(X.iloc[validation_idx])[:, 1]
    )

    assert folds[0] == pytest.approx(expected, abs=1e-12)


def test_the_encoder_is_fitted_inside_the_pipeline(X, y):
    """Leakage guard.

    Because the encoder is a pipeline step, cross-validation refits it per fold. Fitting it
    once on the full training block would leak fold statistics into validation scores.
    """
    estimator = build_pipeline(TOY_SPEC, "logistic_regression", {"C": 1.0}, SEED)
    assert list(estimator.named_steps) == ["encoder", "classifier"]

    subset = X.iloc[:50]
    estimator.fit(subset, y[:50])
    scaler = estimator.named_steps["encoder"].named_steps["encode"].named_transformers_["scaled"]

    assert scaler.mean_[0] == pytest.approx(subset["x1"].mean())
    assert scaler.mean_[0] != pytest.approx(X["x1"].mean())


def test_weights_are_computed_from_the_training_fold_only(X, y, separable):
    """Finding B3's requirement.

    A weighting scheme must see 2/3 of the rows on each of 3 folds, never all of them.
    Precomputing weights on the whole block and slicing them would pass a size check but
    leak the validation fold's composition into the weights.
    """
    seen_sizes = []
    groups = separable["g"].to_numpy()

    def weight_fn(y_fold, groups_fold):
        seen_sizes.append((len(y_fold), len(groups_fold)))
        return np.ones(len(y_fold))

    cross_validated_auc(
        TOY_SPEC,
        "logistic_regression",
        {"C": 1.0},
        X,
        y,
        cv_folds=3,
        seed=SEED,
        groups=groups,
        weight_fn=weight_fn,
    )

    assert len(seen_sizes) == 3
    for n_labels, n_groups in seen_sizes:
        assert n_labels == n_groups
        assert n_labels < len(y)
        assert n_labels == pytest.approx(len(y) * 2 / 3, abs=1)


def test_weight_function_without_groups_raises(X, y):
    with pytest.raises(ValueError, match="weight_fn requires groups"):
        cross_validated_auc(
            TOY_SPEC,
            "logistic_regression",
            {"C": 1.0},
            X,
            y,
            cv_folds=3,
            seed=SEED,
            weight_fn=lambda labels, groups: np.ones(len(labels)),  # noqa: ARG005
        )


def test_unknown_model_type_raises_in_both_construction_and_suggestion():
    with pytest.raises(ValueError, match="unknown model type 'catboost'"):
        build_classifier("catboost", {}, SEED)

    with pytest.raises(ValueError, match="unknown model type 'catboost'"):
        suggest_params(pytest.importorskip("optuna").trial.FixedTrial({}), "catboost")


def test_search_selects_a_configuration_that_ranks_well(X, y):
    result = run_search(
        TOY_SPEC,
        X,
        y,
        n_trials=6,
        cv_folds=3,
        seed=SEED,
        model_types=("logistic_regression",),
    )

    assert result.scoring == "roc_auc"
    assert result.model_type == "logistic_regression"
    assert result.cv_score > 0.85
    assert len(result.trials) == 6
    # The reported best must be the best trial actually evaluated, not the last one.
    assert result.cv_score == max(record.score for record in result.trials)
    assert "C" in result.params


def test_search_is_reproducible_under_the_same_seed(X, y):
    kwargs = {
        "n_trials": 4,
        "cv_folds": 3,
        "seed": SEED,
        "model_types": ("logistic_regression",),
    }
    first = run_search(TOY_SPEC, X, y, **kwargs)
    second = run_search(TOY_SPEC, X, y, **kwargs)

    assert first.params == second.params
    assert first.cv_score == pytest.approx(second.cv_score, abs=1e-12)


def test_fit_best_rebuilds_the_winning_configuration(X, y):
    result = run_search(
        TOY_SPEC, X, y, n_trials=3, cv_folds=3, seed=SEED,
        model_types=("logistic_regression",),
    )
    estimator = fit_best(TOY_SPEC, result, X, y)

    assert estimator.named_steps["classifier"].C == pytest.approx(result.params["C"])
    assert roc_auc_score(y, positive_class_probabilities(estimator, X)) > 0.9
