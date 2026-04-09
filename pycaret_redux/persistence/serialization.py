"""Save and load models with joblib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_model(
    estimator: Any,
    model_name: str,
    verbose: bool = True,
) -> Path:
    """Save a model to disk using joblib.

    Parameters
    ----------
    estimator : fitted estimator
        Model to save.
    model_name : str
        File path (without extension). '.joblib' will be appended.
    verbose : bool
        Print confirmation.

    Returns
    -------
    Path to the saved file.
    """
    path = Path(model_name)
    if path.suffix != ".joblib":
        path = path.with_suffix(".joblib")

    joblib.dump(estimator, path)

    if verbose:
        print(f"Model saved to: {path}")
    return path


def load_model(
    model_name: str,
    verbose: bool = True,
) -> Any:
    """Load a model from disk.

    Parameters
    ----------
    model_name : str
        File path (with or without .joblib extension).
    verbose : bool
        Print confirmation.

    Returns
    -------
    Loaded estimator.
    """
    path = Path(model_name)
    if not path.exists():
        path = path.with_suffix(".joblib")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_name}")

    model = joblib.load(path)

    if verbose:
        print(f"Model loaded from: {path}")
    return model
