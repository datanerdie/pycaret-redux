"""Lift, gain, and KS statistic plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_lift_chart(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot lift chart for binary classification."""
    probas = estimator.predict_proba(X)[:, 1]
    y_arr = np.array(y)

    # Sort by predicted probability descending
    order = np.argsort(-probas)
    y_sorted = y_arr[order]

    n = len(y_sorted)
    n_pos = np.sum(y_sorted)
    cum_pos = np.cumsum(y_sorted)
    percentiles = np.arange(1, n + 1) / n
    lift = (cum_pos / np.arange(1, n + 1)) / (n_pos / n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(percentiles, lift, color="steelblue")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_xlabel("Proportion of Population")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_gain_chart(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot cumulative gain chart."""
    probas = estimator.predict_proba(X)[:, 1]
    y_arr = np.array(y)

    order = np.argsort(-probas)
    y_sorted = y_arr[order]

    n = len(y_sorted)
    n_pos = np.sum(y_sorted)
    cum_pos = np.cumsum(y_sorted) / n_pos
    percentiles = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(percentiles, cum_pos, color="steelblue", label="Model")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Random")
    ax.set_xlabel("Proportion of Population")
    ax.set_ylabel("Cumulative Gain")
    ax.set_title("Cumulative Gain Chart")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_ks_statistic(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot Kolmogorov-Smirnov statistic."""
    probas = estimator.predict_proba(X)[:, 1]
    y_arr = np.array(y)

    thresholds = np.linspace(0, 1, 200)
    tpr_list, fpr_list = [], []

    for t in thresholds:
        preds = (probas >= t).astype(int)
        tp = np.sum((preds == 1) & (y_arr == 1))
        fp = np.sum((preds == 1) & (y_arr == 0))
        fn = np.sum((preds == 0) & (y_arr == 1))
        tn = np.sum((preds == 0) & (y_arr == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    ks_values = tpr_arr - fpr_arr
    ks_max_idx = np.argmax(ks_values)
    ks_stat = ks_values[ks_max_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, tpr_arr, label="TPR", color="steelblue")
    ax.plot(thresholds, fpr_arr, label="FPR", color="coral")
    ax.plot(thresholds, ks_values, label=f"KS = {ks_stat:.3f}", color="green", linestyle="--")
    ax.axvline(x=thresholds[ks_max_idx], color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("KS Statistic Plot")
    ax.legend()
    plt.tight_layout()
    return fig
