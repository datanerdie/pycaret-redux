"""Scorer creation and metric calculation helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score

from pycaret_redux.metrics.registry import MetricEntry, MetricRegistry

logger = logging.getLogger(__name__)


def _safe_roc_auc(y_true, y_proba, **kwargs):
    """Compute ROC AUC handling both binary and multiclass cases.

    For binary: uses positive class probability (column 1).
    For multiclass: uses full probability matrix with multi_class='ovr'.
    Returns 0.0 on any error instead of NaN.
    """
    try:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        n_classes = len(np.unique(y_true))

        if n_classes <= 2:
            # Binary: roc_auc_score expects 1D probability for positive class
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            return roc_auc_score(y_true, y_proba)
        else:
            # Multiclass: needs full probability matrix
            return roc_auc_score(y_true, y_proba, average="weighted", multi_class="ovr")
    except Exception:
        return 0.0


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
                if metric_id == "auc":
                    score = _safe_roc_auc(y_true, y_pred_proba)
                else:
                    score = entry.score_func(y_true, y_pred_proba, **entry.scorer_kwargs)
            else:
                score = entry.score_func(y_true, y_pred, **entry.scorer_kwargs)
            scores[metric_id] = round(float(score), round_to)
        except Exception:
            scores[metric_id] = 0.0

    return scores


def _auc_scorer(estimator, X, y):
    """Custom AUC scorer that handles binary/multiclass correctly."""
    try:
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X)
        elif hasattr(estimator, "decision_function"):
            y_proba = estimator.decision_function(X)
        else:
            return 0.0
        return _safe_roc_auc(y, y_proba)
    except Exception:
        return 0.0


def build_sklearn_scorer(entry: MetricEntry) -> Any:
    """Build a sklearn scorer from a MetricEntry."""
    if entry.id == "auc":
        # Custom AUC scorer handles binary/multiclass + error recovery
        return _auc_scorer

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
