"""Cross-validation execution, fold generation, and score aggregation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from pycaret_redux.config import ExperimentConfig
from pycaret_redux.metrics.registry import MetricRegistry
from pycaret_redux.metrics.scoring import build_sklearn_scorer


def run_cross_validation(
    estimator: Any,
    config: ExperimentConfig,
    metric_registry: MetricRegistry,
    fold: int | Any | None = None,
    round_to: int = 4,
    fit_kwargs: dict | None = None,
    return_train_score: bool = False,
    n_jobs: int | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, float], float]:
    """Run cross-validation for a single estimator.

    Parameters
    ----------
    estimator : estimator
        The model to cross-validate.
    config : ExperimentConfig
        Experiment config with data and pipeline.
    metric_registry : MetricRegistry
        Metrics to evaluate.
    fold : int or CV splitter, optional
        Number of folds or CV splitter. Defaults to config fold_generator.
    round_to : int
        Decimal places.
    fit_kwargs : dict, optional
        Extra arguments to pass to fit().
    return_train_score : bool
        Whether to return train scores.
    n_jobs : int, optional
        Parallelism for CV.

    Returns
    -------
    (fitted_model, fold_scores_df, mean_scores, fit_time)
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    cv = fold if fold is not None else config.fold_generator
    if n_jobs is not None:
        n_jobs_cv = n_jobs
    elif config.setup_config:
        n_jobs_cv = config.setup_config.n_jobs
    else:
        n_jobs_cv = -1

    # Build pipeline: preprocessing + estimator
    pipeline = _build_full_pipeline(config.pipeline, estimator)

    # Build scorers (skip proba-based metrics if estimator doesn't support them)
    has_proba = hasattr(estimator, "predict_proba")
    scoring = {}
    for metric_id, entry in metric_registry.get_active().items():
        if entry.needs_proba and not has_proba:
            continue
        try:
            scoring[metric_id] = build_sklearn_scorer(entry)
        except Exception:
            pass

    # Handle imbalance resampling per fold
    X_train = config.X_train
    y_train = config.y_train

    start_time = time.time()

    if config.setup_config and config.setup_config.fix_imbalance:
        # Manual CV loop with per-fold resampling
        fold_results = _cv_with_resampling(
            pipeline, X_train, y_train, cv, scoring,
            config.setup_config.fix_imbalance_method,
            config.seed, fit_kwargs, return_train_score,
        )
    else:
        # Use sklearn cross_validate
        fold_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs_cv,
            return_train_score=return_train_score,
            error_score=0.0,
            params=fit_kwargs if fit_kwargs else None,
        )

    fit_time = round(time.time() - start_time, 2)

    # Aggregate scores
    fold_scores, mean_scores = _aggregate_scores(
        fold_results, metric_registry, round_to, return_train_score
    )

    # Refit on full training data
    final_pipeline = _build_full_pipeline(config.pipeline, clone(estimator))
    final_pipeline.fit(X_train, y_train, **fit_kwargs)

    # Extract the fitted estimator from pipeline
    fitted_model = _extract_estimator(final_pipeline)

    return fitted_model, fold_scores, mean_scores, fit_time


def _build_full_pipeline(
    preprocessing_pipeline: Pipeline | None,
    estimator: Any,
) -> Pipeline:
    """Combine preprocessing pipeline with estimator."""
    if preprocessing_pipeline is not None:
        steps = list(preprocessing_pipeline.steps) + [("estimator", estimator)]
        return Pipeline(steps)
    return Pipeline([("estimator", estimator)])


def _extract_estimator(pipeline: Pipeline) -> Any:
    """Extract the fitted estimator from a pipeline."""
    return pipeline.named_steps["estimator"]


def _cv_with_resampling(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: dict,
    imbalance_method: str | Any,
    seed: int,
    fit_kwargs: dict,
    return_train_score: bool,
) -> dict[str, np.ndarray]:
    """Manual CV loop with per-fold resampling."""
    from pycaret_redux.preprocessing.imbalance import resample

    results: dict[str, list] = {f"test_{k}": [] for k in scoring}
    if return_train_score:
        results.update({f"train_{k}": [] for k in scoring})

    for train_idx, val_idx in cv.split(X, y):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Resample training fold
        X_train_resampled, y_train_resampled = resample(
            X_train_fold, y_train_fold, method=imbalance_method, seed=seed
        )

        # Fit and score
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train_resampled, y_train_resampled, **fit_kwargs)

        for metric_id, scorer in scoring.items():
            test_score = scorer(fold_pipeline, X_val_fold, y_val_fold)
            results[f"test_{metric_id}"].append(test_score)

            if return_train_score:
                train_score = scorer(fold_pipeline, X_train_fold, y_train_fold)
                results[f"train_{metric_id}"].append(train_score)

    return {k: np.array(v) for k, v in results.items()}


def _aggregate_scores(
    fold_results: dict[str, np.ndarray],
    metric_registry: MetricRegistry,
    round_to: int,
    return_train_score: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Aggregate per-fold CV scores into a DataFrame and mean dict."""
    active_metrics = metric_registry.get_active()
    rows = []

    # Determine number of folds
    n_folds = 0
    for key in fold_results:
        if key.startswith("test_"):
            n_folds = len(fold_results[key])
            break

    for i in range(n_folds):
        row: dict[str, Any] = {"Fold": i}
        for metric_id in active_metrics:
            key = f"test_{metric_id}"
            if key in fold_results:
                row[active_metrics[metric_id].display_name] = round(
                    float(fold_results[key][i]), round_to
                )
        rows.append(row)

    # Add mean and std rows
    mean_row: dict[str, Any] = {"Fold": "Mean"}
    std_row: dict[str, Any] = {"Fold": "SD"}
    mean_scores: dict[str, float] = {}

    for metric_id in active_metrics:
        key = f"test_{metric_id}"
        if key in fold_results:
            display = active_metrics[metric_id].display_name
            mean_val = round(float(np.mean(fold_results[key])), round_to)
            std_val = round(float(np.std(fold_results[key])), round_to)
            mean_row[display] = mean_val
            std_row[display] = std_val
            mean_scores[metric_id] = mean_val

    rows.append(mean_row)
    rows.append(std_row)

    fold_scores = pd.DataFrame(rows).set_index("Fold")
    return fold_scores, mean_scores
