"""Feature importance and SHAP plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _enrich_feature_names(
    feature_names: list[str],
    feature_labels: dict[str, dict] | None = None,
) -> list[str]:
    """Enrich feature names with human-readable labels.

    For OHE features like 'gender_F', looks up the base feature 'gender'
    in feature_labels and produces 'gender: F'.

    For features with value mappings like {0: "No", 1: "Yes"},
    the labels are available for plots that show feature values.
    """
    if not feature_labels:
        return feature_names

    enriched = []
    for name in feature_names:
        found = False
        for base_feat, label_map in feature_labels.items():
            is_binary = len(label_map) <= 2

            if name.startswith(f"{base_feat}_"):
                # OHE-expanded name like "marital_status_2"
                suffix = name[len(base_feat) + 1 :]
                for code, label in label_map.items():
                    if str(code) == suffix or label == suffix:
                        if is_binary:
                            # Binary: just use the feature name
                            enriched.append(base_feat)
                        else:
                            enriched.append(f"{base_feat}: {label}")
                        found = True
                        break
                if not found:
                    enriched.append(f"{base_feat}: {suffix}")
                    found = True
                break
            elif name == base_feat:
                # Exact match (e.g. ordinal-encoded binary feature)
                # Binary: just the feature name. Multi: show mapping.
                if is_binary:
                    enriched.append(name)
                else:
                    desc = ", ".join(f"{k}={v}" for k, v in label_map.items())
                    enriched.append(f"{name} ({desc})")
                found = True
                break
        if not found:
            enriched.append(name)
    return enriched


def plot_feature_importance(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot feature importance (from model or permutation-based)."""
    n_features = kwargs.get("n_features", 20)
    feature_labels = kwargs.get("feature_labels")

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
            importances = np.abs(estimator.coef_).mean(axis=0)

    if importances is None:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(estimator, X, y, n_repeats=5, random_state=0, n_jobs=-1)
        importances = result.importances_mean

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Enrich names with labels
    display_names = _enrich_feature_names(feature_names, feature_labels)

    # Sort and take top N
    indices = np.argsort(importances)[::-1][:n_features]
    top_names = [display_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, n_features * 0.3)))
    ax.barh(range(len(top_names)), top_importances[::-1], color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    return fig


def plot_permutation_importance(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot permutation importance (model-agnostic).

    Randomly shuffles each feature and measures the drop in score.
    More reliable than impurity-based importance for correlated features.
    """
    from sklearn.inspection import permutation_importance

    n_features = kwargs.get("n_features", 20)
    n_repeats = kwargs.get("n_repeats", 10)
    scoring = kwargs.get("scoring", "accuracy")
    feature_labels = kwargs.get("feature_labels")

    result = permutation_importance(
        estimator, X, y, n_repeats=n_repeats, random_state=0, n_jobs=-1, scoring=scoring
    )

    feature_names = (
        list(X.columns) if hasattr(X, "columns") else [f"F{i}" for i in range(X.shape[1])]
    )

    # Enrich names with labels
    display_names = _enrich_feature_names(feature_names, feature_labels)

    # Sort by mean importance
    indices = np.argsort(result.importances_mean)[::-1][:n_features]
    top_names = [display_names[i] for i in indices]
    top_means = result.importances_mean[indices]
    top_stds = result.importances_std[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, n_features * 0.3)))
    y_pos = range(len(top_names))
    ax.barh(y_pos, top_means[::-1], xerr=top_stds[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Mean accuracy decrease")
    ax.set_title("Permutation Importance")
    plt.tight_layout()
    return fig
