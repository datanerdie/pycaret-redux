"""Feature importance and SHAP plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot feature importance (from model or permutation-based)."""
    n_features = kwargs.get("n_features", 20)

    # Try to get feature importance from model
    importances = None
    feature_names = None

    if hasattr(X, "columns"):
        feature_names = list(X.columns)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).flatten()
        if len(importances) != X.shape[1]:
            # Multiclass: average across classes
            importances = np.abs(estimator.coef_).mean(axis=0)

    if importances is None:
        # Fallback to permutation importance
        from sklearn.inspection import permutation_importance

        result = permutation_importance(estimator, X, y, n_repeats=5, random_state=0, n_jobs=-1)
        importances = result.importances_mean

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Sort and take top N
    indices = np.argsort(importances)[::-1][:n_features]
    top_names = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, n_features * 0.3)))
    ax.barh(range(len(top_names)), top_importances[::-1], color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    return fig
