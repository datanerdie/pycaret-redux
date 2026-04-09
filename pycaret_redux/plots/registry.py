"""Plot registry and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt


@dataclass
class PlotEntry:
    """Registration record for a single plot type."""

    id: str
    name: str
    func: Callable
    requires_proba: bool = False
    binary_only: bool = False


class PlotRegistry:
    """Central registry for all classification plots."""

    def __init__(self) -> None:
        self._plots: dict[str, PlotEntry] = {}

    def register(self, entry: PlotEntry) -> None:
        self._plots[entry.id] = entry

    def get(self, plot_id: str) -> PlotEntry:
        if plot_id not in self._plots:
            raise KeyError(
                f"Plot '{plot_id}' not found. Available: {', '.join(self._plots.keys())}"
            )
        return self._plots[plot_id]

    def available(self, is_multiclass: bool = False) -> dict[str, str]:
        """Return available plot IDs and names."""
        result = {}
        for pid, entry in self._plots.items():
            if entry.binary_only and is_multiclass:
                continue
            result[pid] = entry.name
        return result

    def render(
        self,
        plot_id: str,
        estimator: Any,
        X: Any,
        y: Any,
        is_multiclass: bool = False,
        save: str | None = None,
        **kwargs,
    ) -> Any:
        """Render a plot by ID."""
        entry = self.get(plot_id)
        if entry.binary_only and is_multiclass:
            raise ValueError(f"Plot '{plot_id}' is only available for binary classification.")

        fig = entry.func(estimator, X, y, **kwargs)

        if save and fig is not None:
            fig.savefig(save, bbox_inches="tight", dpi=150)
            plt.close(fig)

        return fig


def build_default_registry() -> PlotRegistry:
    """Build the default plot registry with all built-in plots."""
    from pycaret_redux.plots.calibration import plot_calibration_curve
    from pycaret_redux.plots.classification import (
        plot_auc,
        plot_class_report,
        plot_confusion_matrix,
        plot_precision_recall,
        plot_prediction_error,
        plot_threshold,
    )
    from pycaret_redux.plots.evaluation import (
        plot_learning_curve,
        plot_validation_curve,
    )
    from pycaret_redux.plots.feature import (
        plot_feature_importance,
        plot_permutation_importance,
    )
    from pycaret_redux.plots.gains import plot_gain_chart, plot_ks_statistic, plot_lift_chart

    registry = PlotRegistry()
    entries = [
        PlotEntry("auc", "AUC", plot_auc, requires_proba=True),
        PlotEntry("confusion_matrix", "Confusion Matrix", plot_confusion_matrix),
        PlotEntry("threshold", "Threshold", plot_threshold, requires_proba=True, binary_only=True),
        PlotEntry("pr", "Precision Recall", plot_precision_recall, requires_proba=True),
        PlotEntry("error", "Prediction Error", plot_prediction_error),
        PlotEntry("class_report", "Class Report", plot_class_report),
        PlotEntry("feature", "Feature Importance", plot_feature_importance),
        PlotEntry("permutation", "Permutation Importance", plot_permutation_importance),
        PlotEntry("learning", "Learning Curve", plot_learning_curve),
        PlotEntry("vc", "Validation Curve", plot_validation_curve),
        PlotEntry("calibration", "Calibration Curve", plot_calibration_curve, requires_proba=True),
        PlotEntry("lift", "Lift Chart", plot_lift_chart, requires_proba=True, binary_only=True),
        PlotEntry("gain", "Gain Chart", plot_gain_chart, requires_proba=True, binary_only=True),
        PlotEntry(
            "ks", "KS Statistic Plot", plot_ks_statistic, requires_proba=True, binary_only=True
        ),
    ]
    for entry in entries:
        registry.register(entry)
    return registry
