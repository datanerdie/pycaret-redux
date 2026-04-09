"""Save and load models with joblib.

Models are saved as a dict containing both the preprocessing pipeline
and the fitted estimator, so a single file is all you need for deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    """Bundle of preprocessing pipeline + fitted estimator."""

    estimator: Any
    pipeline: Pipeline | None = None
    target_name: str = ""
    feature_names_in: list[str] | None = None
    is_multiclass: bool = False


def save_model(
    estimator: Any,
    model_name: str,
    pipeline: Pipeline | None = None,
    target_name: str = "",
    feature_names_in: list[str] | None = None,
    is_multiclass: bool = False,
    verbose: bool = True,
) -> Path:
    """Save a model + preprocessing pipeline to disk.

    Parameters
    ----------
    estimator : fitted estimator
        Trained model to save.
    model_name : str
        File path (without extension). '.joblib' will be appended.
    pipeline : Pipeline, optional
        Preprocessing pipeline. If provided, bundled with the model.
    target_name : str
        Name of the target column (for reference).
    feature_names_in : list[str], optional
        Expected input feature names.
    is_multiclass : bool
        Whether the task is multiclass.
    verbose : bool
        Print confirmation.

    Returns
    -------
    Path to the saved file.
    """
    path = Path(model_name)
    if path.suffix != ".joblib":
        path = path.with_suffix(".joblib")

    logger.info("Saving model to %s (includes_pipeline=%s)", path, pipeline is not None)

    artifact = ModelArtifact(
        estimator=estimator,
        pipeline=pipeline,
        target_name=target_name,
        feature_names_in=feature_names_in,
        is_multiclass=is_multiclass,
    )
    joblib.dump(artifact, path)

    if verbose:
        print(f"Model saved to: {path}")
        if pipeline is not None:
            print("  Includes: preprocessing pipeline + estimator")
        else:
            print("  Includes: estimator only")
    return path


def load_model(
    model_name: str,
    verbose: bool = True,
) -> ModelArtifact:
    """Load a model artifact from disk.

    Parameters
    ----------
    model_name : str
        File path (with or without .joblib extension).
    verbose : bool
        Print confirmation.

    Returns
    -------
    ModelArtifact with .estimator and .pipeline attributes.
    """
    path = Path(model_name)
    if not path.exists():
        path = path.with_suffix(".joblib")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_name}")

    logger.info("Loading model from %s", path)
    loaded = joblib.load(path)

    # Backward compatibility: if saved before bundling was added
    if not isinstance(loaded, ModelArtifact):
        loaded = ModelArtifact(estimator=loaded)

    if verbose:
        print(f"Model loaded from: {path}")
        if loaded.pipeline is not None:
            print("  Loaded: preprocessing pipeline + estimator")
        else:
            print("  Loaded: estimator only")
    return loaded


def predict_from_artifact(
    artifact: ModelArtifact,
    data: Any,
) -> Any:
    """Make predictions using a saved ModelArtifact.

    Applies the preprocessing pipeline (if present) then predicts.

    Parameters
    ----------
    artifact : ModelArtifact
        Loaded model artifact.
    data : DataFrame-like
        Raw input data (before preprocessing).

    Returns
    -------
    Predictions array.
    """
    import pandas as pd

    X = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

    if artifact.pipeline is not None:
        X = artifact.pipeline.transform(X)

    return artifact.estimator.predict(X)
