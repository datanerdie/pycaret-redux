"""create_model logic: train a single model with CV."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from pycaret_redux.config import ExperimentConfig
from pycaret_redux.metrics.registry import MetricRegistry
from pycaret_redux.models.factory import create_estimator
from pycaret_redux.models.registry import ModelRegistry
from pycaret_redux.training.cross_validation import run_cross_validation

logger = logging.getLogger(__name__)


def create_model(
    estimator: str | Any,
    config: ExperimentConfig,
    model_registry: ModelRegistry,
    metric_registry: MetricRegistry,
    fold: int | Any | None = None,
    round_to: int = 4,
    cross_validation: bool = True,
    fit_kwargs: dict | None = None,
    return_train_score: bool = False,
    verbose: bool = True,
    **kwargs,
) -> tuple[Any, pd.DataFrame | None, dict[str, float], float]:
    """Train a single model, optionally with cross-validation.

    Parameters
    ----------
    estimator : str or estimator
        Model ID (e.g. "lr") or sklearn-compatible estimator.
    config : ExperimentConfig
        Experiment configuration.
    model_registry : ModelRegistry
        Model registry for resolving string IDs.
    metric_registry : MetricRegistry
        Metric registry.
    fold : int or CV splitter, optional
        Override fold configuration.
    round_to : int
        Decimal places.
    cross_validation : bool
        Whether to run CV.
    fit_kwargs : dict, optional
        Extra fit arguments.
    return_train_score : bool
        Include train scores.
    verbose : bool
        Print results.
    **kwargs
        Extra estimator parameters.

    Returns
    -------
    (fitted_model, fold_scores_df_or_None, mean_scores, fit_time_seconds)
    """
    # Create fresh estimator
    model = create_estimator(estimator, model_registry, **kwargs)
    logger.info("Created estimator: %s", type(model).__name__)
    logger.debug(
        "Estimator params: %s",
        model.get_params() if hasattr(model, "get_params") else "?",
    )

    if cross_validation:
        logger.info("Training with cross-validation")
        fitted_model, fold_scores, mean_scores, fit_time = run_cross_validation(
            estimator=model,
            config=config,
            metric_registry=metric_registry,
            fold=fold,
            round_to=round_to,
            fit_kwargs=fit_kwargs,
            return_train_score=return_train_score,
        )

        if verbose:
            from pycaret_redux.utils.display import display_fold_scores

            model_name = type(model).__name__
            display_fold_scores(fold_scores, model_name)

        return fitted_model, fold_scores, mean_scores, fit_time
    else:
        logger.info("Training without cross-validation (direct fit)")
        # Train without CV
        import time

        from pycaret_redux.training.cross_validation import (
            _build_full_pipeline,
            _extract_estimator,
        )

        start = time.time()
        pipeline = _build_full_pipeline(config.pipeline, model)
        pipeline.fit(config.X_train, config.y_train, **(fit_kwargs or {}))
        fitted_model = _extract_estimator(pipeline)
        fit_time = round(time.time() - start, 2)

        return fitted_model, None, {}, fit_time
