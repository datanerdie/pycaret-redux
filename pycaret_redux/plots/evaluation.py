"""Learning and validation curve plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve


def plot_learning_curve(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot learning curve showing train/validation score vs training size."""
    cv = kwargs.get("cv", 5)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue"
    )
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
    ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation Score")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_validation_curve(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot validation curve for a single hyperparameter."""
    param_name = kwargs.get("param_name")
    param_range = kwargs.get("param_range")

    if param_name is None:
        # Try to find a reasonable default parameter
        if hasattr(estimator, "max_depth"):
            param_name = "max_depth"
            param_range = list(range(1, 12))
        elif hasattr(estimator, "C"):
            param_name = "C"
            param_range = np.logspace(-3, 3, 10)
        elif hasattr(estimator, "n_estimators"):
            param_name = "n_estimators"
            param_range = [10, 50, 100, 200, 300]
        else:
            raise ValueError("Cannot auto-detect parameter. Pass param_name and param_range.")

    train_scores, val_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=kwargs.get("cv", 5),
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(
        param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue"
    )
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
    ax.plot(param_range, train_mean, "o-", color="blue", label="Training Score")
    ax.plot(param_range, val_mean, "o-", color="orange", label="Validation Score")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Validation Curve ({param_name})")
    ax.legend()
    plt.tight_layout()
    return fig
