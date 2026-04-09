"""Imputation strategies for numeric and categorical features."""

from __future__ import annotations

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline


def build_numeric_imputer(strategy: str) -> Pipeline | SimpleImputer | KNNImputer:
    """Build an imputer for numeric features.

    Parameters
    ----------
    strategy : str
        One of: "mean", "median", "mode", "knn", "drop", or a numeric constant.
    """
    strategy_map = {"mode": "most_frequent", "mean": "mean", "median": "median"}

    if isinstance(strategy, str):
        strategy_lower = strategy.lower()
        if strategy_lower == "knn":
            return KNNImputer()
        if strategy_lower == "drop":
            # SimpleImputer doesn't support drop; we use most_frequent as safe fallback
            # Actual drop is handled at pipeline build time
            return SimpleImputer(strategy="most_frequent")
        if strategy_lower in strategy_map:
            return SimpleImputer(strategy=strategy_map[strategy_lower])
        raise ValueError(
            f"Unknown numeric imputation strategy: '{strategy}'. "
            "Choose from: mean, median, mode, knn."
        )
    # Assume numeric constant
    return SimpleImputer(strategy="constant", fill_value=strategy)


def build_categorical_imputer(strategy: str) -> SimpleImputer:
    """Build an imputer for categorical features.

    Parameters
    ----------
    strategy : str
        One of: "mode" or a string constant.
    """
    if strategy.lower() == "mode":
        return SimpleImputer(strategy="most_frequent")
    return SimpleImputer(strategy="constant", fill_value=strategy)
