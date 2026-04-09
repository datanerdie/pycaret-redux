"""Metric registry: MetricEntry dataclass and MetricRegistry class."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class MetricEntry:
    """Registration record for a single classification metric."""

    id: str
    name: str
    display_name: str
    score_func: Callable
    greater_is_better: bool = True
    needs_proba: bool = False
    needs_threshold: bool = False
    supports_multiclass: bool = True
    is_custom: bool = False
    scorer_kwargs: dict[str, Any] = field(default_factory=dict)


class MetricRegistry:
    """Central registry for classification metrics."""

    def __init__(self, is_multiclass: bool = False):
        self._metrics: dict[str, MetricEntry] = {}
        self._is_multiclass = is_multiclass

    def register(self, entry: MetricEntry) -> None:
        """Register a metric entry."""
        self._metrics[entry.id] = entry

    def register_defaults(self) -> None:
        """Register all default metrics."""
        from pycaret_redux.metrics.definitions import get_default_metrics

        for entry in get_default_metrics():
            self.register(entry)

    def remove(self, id_or_name: str) -> None:
        """Remove a metric by ID or name."""
        if id_or_name in self._metrics:
            del self._metrics[id_or_name]
            return
        for mid, entry in self._metrics.items():
            if entry.name == id_or_name:
                del self._metrics[mid]
                return
        raise KeyError(f"Metric '{id_or_name}' not found.")

    def get(self, id_or_name: str) -> MetricEntry:
        """Get a metric entry by ID or name."""
        if id_or_name in self._metrics:
            return self._metrics[id_or_name]
        for entry in self._metrics.values():
            if entry.name == id_or_name:
                return entry
        raise KeyError(
            f"Metric '{id_or_name}' not found. Available: {', '.join(self._metrics.keys())}"
        )

    def get_active(self) -> dict[str, MetricEntry]:
        """Return all active (non-multiclass-filtered) metrics."""
        if not self._is_multiclass:
            return dict(self._metrics)
        return {k: v for k, v in self._metrics.items() if v.supports_multiclass}

    def to_dataframe(self) -> pd.DataFrame:
        """Return metrics as a DataFrame."""
        rows = []
        for entry in self._metrics.values():
            rows.append(
                {
                    "ID": entry.id,
                    "Name": entry.name,
                    "Display Name": entry.display_name,
                    "Greater is Better": entry.greater_is_better,
                    "Needs Proba": entry.needs_proba,
                    "Multiclass": entry.supports_multiclass,
                    "Custom": entry.is_custom,
                }
            )
        return pd.DataFrame(rows).set_index("ID")

    def __contains__(self, id_or_name: str) -> bool:
        if id_or_name in self._metrics:
            return True
        return any(e.name == id_or_name for e in self._metrics.values())

    def __len__(self) -> int:
        return len(self._metrics)
