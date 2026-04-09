"""Scorer creation and metric calculation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

from pycaret_redux.metrics.registry import MetricEntry, MetricRegistry


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | None,
    registry: MetricRegistry,
    round_to: int = 4,
) -> dict[str, float]:
    """Calculate all active metrics for given predictions.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_pred_proba : array-like or None
        Predicted probabilities (for AUC etc.).
    registry : MetricRegistry
        Metric registry with active metrics.
    round_to : int
        Decimal places for rounding.

    Returns
    -------
    dict[str, float]
        Metric ID -> score mapping.
    """
    scores: dict[str, float] = {}
    for metric_id, entry in registry.get_active().items():
        try:
            if entry.needs_proba:
                if y_pred_proba is None:
                    scores[metric_id] = 0.0
                    continue
                score = entry.score_func(y_true, y_pred_proba, **entry.scorer_kwargs)
            else:
                score = entry.score_func(y_true, y_pred, **entry.scorer_kwargs)
            scores[metric_id] = round(float(score), round_to)
        except Exception:
            scores[metric_id] = 0.0

    return scores


def build_sklearn_scorer(entry: MetricEntry) -> Any:
    """Build a sklearn scorer from a MetricEntry."""
    if entry.needs_proba:
        return make_scorer(
            entry.score_func,
            response_method="predict_proba",
            greater_is_better=entry.greater_is_better,
            **entry.scorer_kwargs,
        )
    return make_scorer(
        entry.score_func,
        greater_is_better=entry.greater_is_better,
        **entry.scorer_kwargs,
    )
