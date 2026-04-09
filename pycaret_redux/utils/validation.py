"""Input validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_dataframe(data: Any) -> pd.DataFrame:
    """Convert data input to a pandas DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        df.columns = [str(c) for c in df.columns]
        return df
    if isinstance(data, list):
        return pd.DataFrame(data)
    raise TypeError(f"Unsupported data type: {type(data)}. Expected DataFrame, ndarray, or list.")


def validate_target(
    data: pd.DataFrame, target: int | str | Any
) -> str:
    """Resolve target to a column name and validate it exists."""
    if isinstance(target, (int, np.integer)):
        target = data.columns[target]
    target = str(target)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    return target


def validate_setup_params(
    train_size: float,
    fold: int,
) -> None:
    """Validate common setup parameters."""
    if not 0.0 < train_size < 1.0:
        raise ValueError(f"train_size must be between 0 and 1, got {train_size}.")
    if fold < 2:
        raise ValueError(f"fold must be >= 2, got {fold}.")
