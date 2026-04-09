"""Model registry: ModelEntry dataclass and ModelRegistry class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TuningSpace:
    """Hyperparameter search space for a model."""

    grid: dict[str, list[Any]] = field(default_factory=dict)
    distributions: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEntry:
    """Registration record for a single classifier."""

    id: str
    name: str
    class_def: type
    default_args: dict[str, Any] = field(default_factory=dict)
    tuning: TuningSpace = field(default_factory=TuningSpace)
    is_turbo: bool = True
    is_special: bool = False
    supports_predict_proba: bool = True
    shap_type: str | None = None  # None, "type1", "type2"


class ModelRegistry:
    """Central registry for all classification models."""

    def __init__(self, seed: int = 0, n_jobs: int | None = -1):
        self._models: dict[str, ModelEntry] = {}
        self._seed = seed
        self._n_jobs = n_jobs

    def register(self, entry: ModelEntry) -> None:
        """Register a model entry."""
        self._models[entry.id] = entry

    def register_defaults(self) -> None:
        """Register all default classifiers."""
        from pycaret_redux.models.definitions import get_default_classifiers

        for entry in get_default_classifiers(self._seed, self._n_jobs):
            self.register(entry)

    def get(self, model_id: str) -> ModelEntry:
        """Get a model entry by ID."""
        if model_id not in self._models:
            raise KeyError(
                f"Model '{model_id}' not found. Available: {', '.join(self._models.keys())}"
            )
        return self._models[model_id]

    def list_models(self, turbo_only: bool = False, include_special: bool = False) -> pd.DataFrame:
        """Return a DataFrame of available models."""
        rows = []
        for entry in self._models.values():
            if entry.is_special and not include_special:
                continue
            if turbo_only and not entry.is_turbo:
                continue
            rows.append(
                {
                    "ID": entry.id,
                    "Name": entry.name,
                    "Turbo": entry.is_turbo,
                }
            )
        return pd.DataFrame(rows).set_index("ID")

    def create_instance(self, id_or_estimator: str | Any, **kwargs) -> Any:
        """Create a model instance from ID or return estimator as-is."""
        if isinstance(id_or_estimator, str):
            entry = self.get(id_or_estimator)
            args = {**entry.default_args, **kwargs}
            return entry.class_def(**args)
        # Assume it's already an estimator
        return id_or_estimator

    def get_ids(self, turbo_only: bool = False, include_special: bool = False) -> list[str]:
        """Return list of model IDs."""
        return [
            e.id
            for e in self._models.values()
            if (not e.is_special or include_special) and (not turbo_only or e.is_turbo)
        ]

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models

    def __len__(self) -> int:
        return len(self._models)
