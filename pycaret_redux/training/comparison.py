"""compare_models logic: train and evaluate all models."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from pycaret_redux.config import ExperimentConfig
from pycaret_redux.metrics.registry import MetricRegistry
from pycaret_redux.models.registry import ModelRegistry
from pycaret_redux.training.creation import create_model


def compare_models(
    config: ExperimentConfig,
    model_registry: ModelRegistry,
    metric_registry: MetricRegistry,
    include: list[str | Any] | None = None,
    exclude: list[str] | None = None,
    fold: int | Any | None = None,
    round_to: int = 4,
    cross_validation: bool = True,
    sort: str = "Accuracy",
    n_select: int = 1,
    budget_time: float | None = None,
    turbo: bool = True,
    errors: str = "ignore",
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> tuple[Any | list[Any], pd.DataFrame]:
    """Compare all available models using cross-validation.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    model_registry : ModelRegistry
        Model registry.
    metric_registry : MetricRegistry
        Metric registry.
    include : list, optional
        Model IDs to include. If None, use all.
    exclude : list, optional
        Model IDs to exclude.
    sort : str
        Metric name or display name to sort by.
    n_select : int
        Number of top models to return.
    budget_time : float, optional
        Time budget in minutes. Stop when exceeded.
    turbo : bool
        Only use turbo-enabled models.
    errors : str
        "ignore" to skip failed models, "raise" to propagate errors.
    verbose : bool
        Print progress.

    Returns
    -------
    (top_model_or_list, comparison_dataframe)
    """
    # Determine model list
    if include is not None:
        model_ids = [
            m if isinstance(m, str) else _get_model_id(m, model_registry) for m in include
        ]
    else:
        model_ids = model_registry.get_ids(turbo_only=turbo)

    if exclude:
        model_ids = [m for m in model_ids if m not in exclude]

    # Resolve sort metric
    sort_metric_id = _resolve_sort_metric(sort, metric_registry)

    results_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    total_start = time.time()

    # Progress bar
    progress = None
    if verbose:
        from pycaret_redux.utils.display import create_progress

        progress = create_progress(len(model_ids), "Comparing models")

    for model_id in model_ids:
        # Check time budget
        if budget_time is not None:
            elapsed_minutes = (time.time() - total_start) / 60
            if elapsed_minutes >= budget_time:
                break

        if progress is not None:
            entry = model_registry.get(model_id)
            progress.set_postfix_str(entry.name)

        try:
            model, fold_scores, mean_scores = create_model(
                estimator=model_id,
                config=config,
                model_registry=model_registry,
                metric_registry=metric_registry,
                fold=fold,
                round_to=round_to,
                cross_validation=cross_validation,
                fit_kwargs=fit_kwargs,
                verbose=False,
            )

            entry = model_registry.get(model_id)
            row = {"Model": entry.name}
            for metric_id, score in mean_scores.items():
                me = metric_registry.get(metric_id)
                row[me.display_name] = score
            results_rows.append(row)
            fitted_models[model_id] = model

        except Exception:
            if errors == "raise":
                raise
            continue
        finally:
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    if not results_rows:
        raise RuntimeError("No models were successfully trained.")

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(results_rows)

    # Sort by target metric (descending if greater_is_better)
    sort_entry = metric_registry.get(sort_metric_id)
    sort_col = sort_entry.display_name
    if sort_col in comparison_df.columns:
        ascending = not sort_entry.greater_is_better
        comparison_df = comparison_df.sort_values(
            sort_col, ascending=ascending
        ).reset_index(drop=True)

    if verbose:
        from pycaret_redux.utils.display import display_comparison

        display_comparison(comparison_df, sort_col=sort_col)

    # Select top N models
    top_model_ids = []
    for _, row in comparison_df.head(n_select).iterrows():
        for mid, entry in model_registry._models.items():
            if entry.name == row["Model"] and mid in fitted_models:
                top_model_ids.append(mid)
                break

    top_models = [fitted_models[mid] for mid in top_model_ids if mid in fitted_models]

    if n_select == 1 and top_models:
        return top_models[0], comparison_df
    return top_models, comparison_df


def _resolve_sort_metric(sort: str, registry: MetricRegistry) -> str:
    """Resolve a sort string to a metric ID."""
    # Try direct ID match
    try:
        registry.get(sort)
        return sort
    except KeyError:
        pass
    # Try name/display_name match
    for entry in registry._metrics.values():
        if entry.name.lower() == sort.lower() or entry.display_name.lower() == sort.lower():
            return entry.id
    raise ValueError(
        f"Unknown sort metric: '{sort}'. "
        f"Available: {', '.join(registry._metrics.keys())}"
    )


def _get_model_id(estimator: Any, registry: ModelRegistry) -> str:
    """Try to find the model ID for a given estimator instance."""
    est_type = type(estimator)
    for entry in registry._models.values():
        if entry.class_def is est_type:
            return entry.id
    raise ValueError(f"Could not find model ID for estimator type: {est_type}")
