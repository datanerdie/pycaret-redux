"""Core classification plots: AUC, confusion matrix, PR, threshold, etc."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)


def plot_auc(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot ROC/AUC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(estimator, X, y, ax=ax)
    ax.set_title("ROC Curve")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(estimator, X, y, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_threshold(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot metric vs threshold for binary classification."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    if not hasattr(estimator, "predict_proba"):
        raise ValueError("Estimator must support predict_proba for threshold plot.")

    probas = estimator.predict_proba(X)[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)

    f1s, precisions, recalls = [], [], []
    for t in thresholds:
        preds = (probas >= t).astype(int)
        f1s.append(f1_score(y, preds, zero_division=0))
        precisions.append(precision_score(y, preds, zero_division=0))
        recalls.append(recall_score(y, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, f1s, label="F1")
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Threshold")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_precision_recall(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot precision-recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(estimator, X, y, ax=ax)
    ax.set_title("Precision-Recall Curve")
    plt.tight_layout()
    return fig


def plot_prediction_error(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot prediction error (predicted vs actual class distribution)."""
    preds = estimator.predict(X)
    classes = np.unique(y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Actual distribution
    actual_counts = [np.sum(y == c) for c in classes]
    axes[0].bar(classes, actual_counts, color="steelblue", alpha=0.7)
    axes[0].set_title("Actual")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")

    # Predicted distribution
    pred_counts = [np.sum(preds == c) for c in classes]
    axes[1].bar(classes, pred_counts, color="coral", alpha=0.7)
    axes[1].set_title("Predicted")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")

    plt.suptitle("Prediction Error")
    plt.tight_layout()
    return fig


def plot_class_report(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot classification report as a heatmap."""
    preds = estimator.predict(X)
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    # Filter to class rows only
    class_keys = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
    metrics = ["precision", "recall", "f1-score"]

    data = np.array([[report[c][m] for m in metrics] for c in class_keys])

    fig, ax = plt.subplots(figsize=(8, max(3, len(class_keys) * 0.6)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(class_keys)))
    ax.set_yticklabels(class_keys)

    for i in range(len(class_keys)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center")

    ax.set_title("Classification Report")
    fig.colorbar(im)
    plt.tight_layout()
    return fig
