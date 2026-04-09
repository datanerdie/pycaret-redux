"""Cross-validation execution, fold generation, and score aggregation."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
    logger.info("Starting cross-validation for %s", type(estimator).__name__)
    logger.debug("CV splitter: %s, n_splits=%s", type(cv).__name__, getattr(cv, "n_splits", "?"))
    if n_jobs is not None:
        n_jobs_cv = n_jobs
    elif config.setup_config:
        n_jobs_cv = config.setup_config.n_jobs
    else:
        n_jobs_cv = -1

    # Build pipeline: preprocessing + estimator
    pipeline = _build_full_pipeline(config.pipeline, estimator)

    # Build scorers (skip proba-based metrics only if model has no probability output)
    has_proba = hasattr(estimator, "predict_proba")
    has_decision = hasattr(estimator, "decision_function")
    scoring = {}
    for metric_id, entry in metric_registry.get_active().items():
        if entry.needs_proba and not has_proba and not has_decision:
            continue
        try:
            scoring[metric_id] = build_sklearn_scorer(entry)
        except Exception:
            pass

    # Handle imbalance
    X_train = config.X_train
    y_train = config.y_train
    use_resampling = False

    if config.setup_config and config.setup_config.fix_imbalance:
        method = config.setup_config.fix_imbalance_method
        if isinstance(method, str) and method.lower() == "class_weight":
            # Set class_weight='balanced' on the estimator if it supports it
            est = pipeline.named_steps.get("estimator", estimator)
            if hasattr(est, "class_weight"):
                est.set_params(class_weight="balanced")
                logger.info("Set class_weight='balanced' on %s", type(est).__name__)
            else:
                logger.warning(
                    "%s does not support class_weight. Falling back to SMOTE.",
                    type(est).__name__,
                )
                use_resampling = True
        else:
            use_resampling = True

    start_time = time.time()

    if use_resampling:
        logger.info("Using manual CV loop with per-fold imbalance resampling")
        # Manual CV loop with per-fold resampling
        fold_results = _cv_with_resampling(
            pipeline,
            X_train,
            y_train,
            cv,
            scoring,
            config.setup_config.fix_imbalance_method,
            config.seed,
            fit_kwargs,
            return_train_score,
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
    logger.info("Cross-validation completed in %.2fs", fit_time)

    # Aggregate scores
    fold_scores, mean_scores = _aggregate_scores(
        fold_results, metric_registry, round_to, return_train_score
    )

    # Refit on full training data
    logger.info("Refitting on full training data")
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

    # Add mean, std, and confidence interval rows
    mean_row: dict[str, Any] = {"Fold": "Mean"}
    std_row: dict[str, Any] = {"Fold": "SD"}
    ci_row: dict[str, Any] = {"Fold": "95% CI"}
    mean_scores: dict[str, float] = {}

    for metric_id in active_metrics:
        key = f"test_{metric_id}"
        if key in fold_results:
            scores = fold_results[key]
            display = active_metrics[metric_id].display_name
            mean_val = round(float(np.mean(scores)), round_to)
            std_val = round(float(np.std(scores)), round_to)

            # 95% confidence interval (t-distribution)
            from scipy import stats as scipy_stats

            n = len(scores)
            se = std_val / np.sqrt(n) if n > 1 else 0
            t_crit = scipy_stats.t.ppf(0.975, df=max(n - 1, 1))
            ci_lower = round(mean_val - t_crit * se, round_to)
            ci_upper = round(mean_val + t_crit * se, round_to)

            mean_row[display] = mean_val
            std_row[display] = std_val
            ci_row[display] = f"[{ci_lower}, {ci_upper}]"
            mean_scores[metric_id] = mean_val

    rows.append(mean_row)
    rows.append(std_row)
    rows.append(ci_row)

    fold_scores = pd.DataFrame(rows).set_index("Fold")
    return fold_scores, mean_scores


def run_nested_cross_validation(
    estimator: Any,
    config: ExperimentConfig,
    metric_registry: MetricRegistry,
    param_grid: dict[str, list] | None = None,
    outer_cv: Any | None = None,
    inner_cv: int = 5,
    n_iter: int = 10,
    optimize: str = "acc",
    round_to: int = 4,
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run nested cross-validation for unbiased model evaluation.

    Outer loop estimates generalization performance. Inner loop performs
    hyperparameter tuning. This prevents optimistic bias from using the
    same data for tuning and evaluation.

    Parameters
    ----------
    estimator : estimator
        The model to evaluate.
    config : ExperimentConfig
        Experiment config.
    metric_registry : MetricRegistry
        Metrics to evaluate.
    param_grid : dict, optional
        Hyperparameter search space for inner loop.
    outer_cv : CV splitter, optional
        Outer CV splitter. Defaults to config fold_generator.
    inner_cv : int
        Number of inner CV folds.
    n_iter : int
        Number of random search iterations in inner loop.
    optimize : str
        Metric ID to optimize in inner loop.
    round_to : int
        Decimal places.
    n_jobs : int, optional
        Parallelism.

    Returns
    -------
    (outer_fold_scores_df, mean_scores)
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    from pycaret_redux.metrics.scoring import build_sklearn_scorer

    outer = outer_cv if outer_cv is not None else config.fold_generator
    if n_jobs is None:
        n_jobs = config.setup_config.n_jobs if config.setup_config else -1

    # Build scorers
    has_proba = hasattr(estimator, "predict_proba")
    scoring = {}
    for metric_id, entry in metric_registry.get_active().items():
        if entry.needs_proba and not has_proba:
            continue
        try:
            scoring[metric_id] = build_sklearn_scorer(entry)
        except Exception:
            pass

    # Inner CV scorer
    optimize_entry = metric_registry.get(optimize)
    inner_scorer = build_sklearn_scorer(optimize_entry)
    inner_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=config.seed)

    pipeline = _build_full_pipeline(config.pipeline, estimator)

    # Prefix params for pipeline
    prefixed_grid = {}
    if param_grid:
        prefixed_grid = {f"estimator__{k}": v for k, v in param_grid.items()}

    X_train = config.X_train
    y_train = config.y_train

    outer_results: dict[str, list] = {f"test_{k}": [] for k in scoring}

    logger.info("Starting nested CV: outer=%s, inner=%d folds", type(outer).__name__, inner_cv)

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X_train, y_train)):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

        if prefixed_grid:
            inner_search = RandomizedSearchCV(
                clone(pipeline),
                param_distributions=prefixed_grid,
                n_iter=min(n_iter, 100),
                scoring=inner_scorer,
                cv=inner_splitter,
                random_state=config.seed,
                n_jobs=n_jobs,
                refit=True,
                error_score=0.0,
            )
            inner_search.fit(X_tr, y_tr)
            best_pipeline = inner_search.best_estimator_
        else:
            best_pipeline = clone(pipeline)
            best_pipeline.fit(X_tr, y_tr)

        for metric_id, scorer in scoring.items():
            score = scorer(best_pipeline, X_te, y_te)
            outer_results[f"test_{metric_id}"].append(score)

    fold_results = {k: np.array(v) for k, v in outer_results.items()}
    fold_scores, mean_scores = _aggregate_scores(fold_results, metric_registry, round_to, False)
    return fold_scores, mean_scores
