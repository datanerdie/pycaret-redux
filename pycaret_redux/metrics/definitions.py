"""Default classification metric definitions."""

from __future__ import annotations

from sklearn import metrics

from pycaret_redux.metrics.registry import MetricEntry


def get_default_metrics() -> list[MetricEntry]:
    """Return all default classification metric entries."""
    return [
        MetricEntry(
            id="acc",
            name="Accuracy",
            display_name="Accuracy",
            score_func=metrics.accuracy_score,
        ),
        MetricEntry(
            id="auc",
            name="AUC",
            display_name="AUC",
            score_func=metrics.roc_auc_score,
            needs_proba=True,
            scorer_kwargs={"average": "weighted", "multi_class": "ovr"},
        ),
        MetricEntry(
            id="recall",
            name="Recall",
            display_name="Recall",
            score_func=metrics.recall_score,
            scorer_kwargs={"average": "weighted", "zero_division": 0},
        ),
        MetricEntry(
            id="precision",
            name="Precision",
            display_name="Prec.",
            score_func=metrics.precision_score,
            scorer_kwargs={"average": "weighted", "zero_division": 0},
        ),
        MetricEntry(
            id="f1",
            name="F1",
            display_name="F1",
            score_func=metrics.f1_score,
            scorer_kwargs={"average": "weighted", "zero_division": 0},
        ),
        MetricEntry(
            id="kappa",
            name="Kappa",
            display_name="Kappa",
            score_func=metrics.cohen_kappa_score,
        ),
        MetricEntry(
            id="mcc",
            name="MCC",
            display_name="MCC",
            score_func=metrics.matthews_corrcoef,
        ),
    ]
