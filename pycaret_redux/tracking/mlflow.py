"""MLflow experiment logger implementation."""

from __future__ import annotations

from typing import Any

import pandas as pd


class MLflowLogger:
    """Experiment logger backed by MLflow.

    Usage::

        logger = MLflowLogger(experiment_name="my_experiment")
        logger.start_run(run_name="lr_baseline")
        logger.log_params({"model": "lr", "fold": 10})
        logger.log_metrics({"accuracy": 0.95})
        logger.log_model(model, "best_model")
        logger.end_run()
    """

    def __init__(self, experiment_name: str = "pycaret_redux", tracking_uri: str | None = None):
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "MLflow is required for experiment tracking. "
                "Install with: uv add mlflow"
            )

        self._mlflow = mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, model_name: str) -> None:
        self._mlflow.sklearn.log_model(model, model_name)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def log_dataframe(self, df: pd.DataFrame, name: str) -> None:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, f"{name}.csv")
            df.to_csv(filepath)
            self._mlflow.log_artifact(filepath)

    def start_run(self, run_name: str | None = None) -> None:
        self._mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        self._mlflow.end_run()
