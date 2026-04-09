"""tune_model logic: hyperparameter tuning with RandomizedSearchCV or Optuna."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

from pycaret_redux.config import ExperimentConfig
from pycaret_redux.metrics.registry import MetricRegistry
from pycaret_redux.metrics.scoring import build_sklearn_scorer
from pycaret_redux.models.registry import ModelRegistry
from pycaret_redux.training.cross_validation import _build_full_pipeline, _extract_estimator


def tune_model(
    estimator: Any,
    config: ExperimentConfig,
    model_registry: ModelRegistry,
    metric_registry: MetricRegistry,
    fold: int | Any | None = None,
    round_to: int = 4,
    n_iter: int = 10,
    custom_grid: dict[str, list] | None = None,
    optimize: str = "Accuracy",
    choose_better: bool = True,
    search_library: str = "sklearn",
    verbose: bool = True,
    **kwargs,
) -> tuple[Any, pd.DataFrame | None]:
    """Tune model hyperparameters.

    Parameters
    ----------
    estimator : fitted estimator
        Model to tune (from create_model).
    config : ExperimentConfig
        Experiment config.
    model_registry : ModelRegistry
        Model registry for looking up tuning grids.
    metric_registry : MetricRegistry
        Metric registry.
    n_iter : int
        Number of random search iterations.
    custom_grid : dict, optional
        Custom parameter grid. Overrides default.
    optimize : str
        Metric to optimize.
    choose_better : bool
        If True, return tuned model only if it's better.
    search_library : str
        "sklearn" for RandomizedSearchCV, "optuna" for Optuna.

    Returns
    -------
    (tuned_model, results_df)
    """
    cv = fold if fold is not None else config.fold_generator

    # Resolve tuning grid
    param_grid = custom_grid
    if param_grid is None:
        param_grid = _get_tuning_grid(estimator, model_registry)

    if not param_grid:
        if verbose:
            print("No tuning grid available for this model. Returning as-is.")
        return estimator, None

    # Resolve scoring metric
    optimize_id = _resolve_optimize_metric(optimize, metric_registry)
    scorer = build_sklearn_scorer(metric_registry.get(optimize_id))

    # Build pipeline
    pipeline = _build_full_pipeline(config.pipeline, clone(estimator))

    # Prefix params with "estimator__" for pipeline
    prefixed_grid = {f"estimator__{k}": v for k, v in param_grid.items()}

    if search_library == "optuna":
        tuned_pipeline = _tune_with_optuna(
            pipeline, prefixed_grid, config, cv, scorer, n_iter, optimize_id, verbose
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=prefixed_grid,
            n_iter=min(n_iter, _grid_size(prefixed_grid)),
            scoring=scorer,
            cv=cv,
            random_state=config.seed,
            n_jobs=config.setup_config.n_jobs if config.setup_config else -1,
            refit=True,
            error_score=0.0,
        )
        search.fit(config.X_train, config.y_train)
        tuned_pipeline = search.best_estimator_

        if verbose:
            print(f"Best score ({optimize}): {search.best_score_:.4f}")
            best_params = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
            print(f"Best params: {best_params}")

    tuned_model = _extract_estimator(tuned_pipeline)

    if choose_better:
        # Compare with original on the same metric
        original_score = _evaluate_model(estimator, config, scorer)
        tuned_score = _evaluate_model(tuned_model, config, scorer)
        if original_score >= tuned_score:
            if verbose:
                print("Original model is better or equal. Returning original.")
            return estimator, None

    return tuned_model, None


def _get_tuning_grid(estimator: Any, registry: ModelRegistry) -> dict[str, list]:
    """Look up the tuning grid for an estimator."""
    est_type = type(estimator)
    for entry in registry._models.values():
        if entry.class_def is est_type:
            return entry.tuning.grid
    return {}


def _resolve_optimize_metric(optimize: str, registry: MetricRegistry) -> str:
    """Resolve optimize string to metric ID."""
    try:
        registry.get(optimize)
        return optimize
    except KeyError:
        pass
    for entry in registry._metrics.values():
        name_match = entry.name.lower() == optimize.lower()
        display_match = entry.display_name.lower() == optimize.lower()
        if name_match or display_match:
            return entry.id
    raise ValueError(f"Unknown optimize metric: '{optimize}'")


def _grid_size(grid: dict) -> int:
    """Estimate the total number of parameter combinations."""
    size = 1
    for values in grid.values():
        if hasattr(values, "__len__"):
            size *= len(values)
        else:
            size *= 10  # for distributions
    return size


def _evaluate_model(estimator: Any, config: ExperimentConfig, scorer: Any) -> float:
    """Quick evaluation of a model on test data."""
    pipeline = _build_full_pipeline(config.pipeline, estimator)
    pipeline.fit(config.X_train, config.y_train)
    return scorer(pipeline, config.X_test, config.y_test)


def _tune_with_optuna(pipeline, param_grid, config, cv, scorer, n_iter, optimize_id, verbose):
    """Tune using Optuna if available."""
    try:
        from optuna.distributions import CategoricalDistribution  # noqa: F401
        from optuna.integration import OptunaSearchCV  # noqa: F401
    except ImportError:
        raise ImportError(
            "Optuna is required for search_library='optuna'. Install with: uv add optuna"
        )

    # Convert grid to optuna distributions
    optuna_dists = {}
    for key, values in param_grid.items():
        if isinstance(values, list):
            optuna_dists[key] = CategoricalDistribution(values)
        else:
            optuna_dists[key] = values

    from optuna.integration import OptunaSearchCV

    search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=optuna_dists,
        n_trials=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=config.seed,
        refit=True,
        error_score=0.0,
        verbose=0 if not verbose else 1,
    )
    search.fit(config.X_train, config.y_train)

    if verbose:
        print(f"Best score ({optimize_id}): {search.best_score_:.4f}")
    return search.best_estimator_
