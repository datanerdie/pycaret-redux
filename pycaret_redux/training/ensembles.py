"""Ensemble methods: blend_models, stack_models, ensemble_model."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from pycaret_redux.config import ExperimentConfig
from pycaret_redux.metrics.registry import MetricRegistry
from pycaret_redux.training.cross_validation import (
    run_cross_validation,
)


def blend_models(
    estimator_list: list[Any],
    config: ExperimentConfig,
    metric_registry: MetricRegistry,
    fold: int | Any | None = None,
    round_to: int = 4,
    method: str = "auto",
    weights: list[float] | None = None,
    optimize: str = "Accuracy",
    choose_better: bool = False,
    verbose: bool = True,
) -> Any:
    """Create a VotingClassifier from a list of estimators.

    Parameters
    ----------
    estimator_list : list
        List of fitted estimators.
    method : str
        "soft", "hard", or "auto". Auto uses soft if all support predict_proba.
    weights : list, optional
        Voting weights per estimator.

    Returns
    -------
    Fitted VotingClassifier.
    """
    # Determine voting method
    if method == "auto":
        all_proba = all(hasattr(e, "predict_proba") for e in estimator_list)
        voting = "soft" if all_proba else "hard"
    else:
        voting = method

    # Build named estimators
    named_estimators = [(f"model_{i}", clone(e)) for i, e in enumerate(estimator_list)]

    ensemble = VotingClassifier(
        estimators=named_estimators,
        voting=voting,
        weights=weights,
        n_jobs=config.setup_config.n_jobs if config.setup_config else -1,
    )

    # Train with CV
    fitted, fold_scores, mean_scores, _ = run_cross_validation(
        estimator=ensemble,
        config=config,
        metric_registry=metric_registry,
        fold=fold,
        round_to=round_to,
    )

    if verbose:
        print(f"Voting Classifier ({voting}) created.")
        print(fold_scores.to_string())

    return fitted


def stack_models(
    estimator_list: list[Any],
    config: ExperimentConfig,
    metric_registry: MetricRegistry,
    meta_model: Any | None = None,
    meta_model_fold: int = 5,
    fold: int | Any | None = None,
    round_to: int = 4,
    method: str = "auto",
    restack: bool = False,
    optimize: str = "Accuracy",
    choose_better: bool = False,
    verbose: bool = True,
) -> Any:
    """Create a StackingClassifier.

    Parameters
    ----------
    estimator_list : list
        Base estimators.
    meta_model : estimator, optional
        Second-level learner. Default: LogisticRegression.
    restack : bool
        Include original features in meta-model input.

    Returns
    -------
    Fitted StackingClassifier.
    """
    if meta_model is None:
        meta_model = LogisticRegression(max_iter=1000, random_state=config.seed)

    # Determine stack method
    if method == "auto":
        all_proba = all(hasattr(e, "predict_proba") for e in estimator_list)
        stack_method = "predict_proba" if all_proba else "predict"
    else:
        stack_method = method

    named_estimators = [(f"model_{i}", clone(e)) for i, e in enumerate(estimator_list)]

    stacker = StackingClassifier(
        estimators=named_estimators,
        final_estimator=clone(meta_model),
        cv=meta_model_fold,
        stack_method=stack_method,
        passthrough=restack,
        n_jobs=config.setup_config.n_jobs if config.setup_config else -1,
    )

    # Train with CV
    fitted, fold_scores, mean_scores, _ = run_cross_validation(
        estimator=stacker,
        config=config,
        metric_registry=metric_registry,
        fold=fold,
        round_to=round_to,
    )

    if verbose:
        print("Stacking Classifier created.")
        print(fold_scores.to_string())

    return fitted


def ensemble_model(
    estimator: Any,
    config: ExperimentConfig,
    metric_registry: MetricRegistry,
    method: str = "bagging",
    fold: int | Any | None = None,
    round_to: int = 4,
    n_estimators: int = 10,
    optimize: str = "Accuracy",
    choose_better: bool = False,
    verbose: bool = True,
) -> Any:
    """Create a BaggingClassifier ensemble.

    Parameters
    ----------
    estimator : estimator
        Base estimator to bag.
    n_estimators : int
        Number of base estimators.

    Returns
    -------
    Fitted BaggingClassifier.
    """
    bagger = BaggingClassifier(
        estimator=clone(estimator),
        n_estimators=n_estimators,
        random_state=config.seed,
        n_jobs=config.setup_config.n_jobs if config.setup_config else -1,
    )

    fitted, fold_scores, mean_scores, _ = run_cross_validation(
        estimator=bagger,
        config=config,
        metric_registry=metric_registry,
        fold=fold,
        round_to=round_to,
    )

    if verbose:
        print(f"Bagging Classifier (n={n_estimators}) created.")
        print(fold_scores.to_string())

    return fitted
