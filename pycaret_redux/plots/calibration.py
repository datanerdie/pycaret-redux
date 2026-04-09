"""Calibration curve plot."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay


def plot_calibration_curve(estimator: Any, X: Any, y: Any, **kwargs) -> plt.Figure:
    """Plot calibration curve (reliability diagram)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_estimator(estimator, X, y, ax=ax, n_bins=10)
    ax.set_title("Calibration Curve")
    plt.tight_layout()
    return fig
