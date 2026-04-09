"""Base experiment logger protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ExperimentLogger(Protocol):
    """Protocol for experiment tracking backends."""

    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters."""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        ...

    def log_model(self, model: Any, model_name: str) -> None:
        """Log a trained model."""
        ...

    def log_artifact(self, path: str) -> None:
        """Log a file artifact."""
        ...

    def log_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Log a DataFrame as artifact."""
        ...

    def start_run(self, run_name: str | None = None) -> None:
        """Start a tracking run."""
        ...

    def end_run(self) -> None:
        """End the current tracking run."""
        ...
