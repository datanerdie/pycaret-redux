"""Normalization and power transformation helpers."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def build_normalizer(method: str) -> BaseEstimator:
    """Build a normalizer/scaler from method name.

    Parameters
    ----------
    method : str
        One of: "zscore", "minmax", "maxabs", "robust".
    """
    match method.lower():
        case "zscore":
            return StandardScaler()
        case "minmax":
            return MinMaxScaler()
        case "maxabs":
            return MaxAbsScaler()
        case "robust":
            return RobustScaler()
        case _:
            raise ValueError(
                f"Unknown normalize_method: '{method}'. "
                "Choose from: zscore, minmax, maxabs, robust."
            )


def build_power_transformer(method: str, seed: int = 0) -> BaseEstimator:
    """Build a power transformer from method name.

    Parameters
    ----------
    method : str
        One of: "yeo-johnson", "quantile".
    seed : int
        Random state for quantile transformer.
    """
    match method.lower():
        case "yeo-johnson":
            return PowerTransformer(method="yeo-johnson", standardize=False)
        case "quantile":
            return QuantileTransformer(
                random_state=seed, output_distribution="normal"
            )
        case _:
            raise ValueError(
                f"Unknown transformation_method: '{method}'. "
                "Choose from: yeo-johnson, quantile."
            )
