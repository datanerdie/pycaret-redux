"""Categorical encoding strategies."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_categorical_encoder(
    max_encoding_ohe: int = 25,
    ordinal_features: dict[str, list] | None = None,
    encoding_method: Any = None,
) -> BaseEstimator:
    """Build the appropriate categorical encoder.

    Parameters
    ----------
    max_encoding_ohe : int
        Max unique values for one-hot encoding. Features exceeding this use
        target encoding (via category_encoders).
    ordinal_features : dict, optional
        Mapping of feature name -> ordered categories for ordinal encoding.
    encoding_method : Any, optional
        Custom encoder instance. If provided, used as-is.
    """
    if encoding_method is not None:
        return encoding_method

    return OneHotEncoder(
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        max_categories=max_encoding_ohe,
        drop=None,
    )


def build_ordinal_encoder(ordinal_features: dict[str, list]) -> OrdinalEncoder:
    """Build an ordinal encoder with specified category ordering."""
    categories = list(ordinal_features.values())
    return OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group rare categories into a single value.

    Categories appearing less than `threshold` fraction of samples are
    replaced with `rare_value`.
    """

    def __init__(self, threshold: float = 0.05, rare_value: str = "rare"):
        self.threshold = threshold
        self.rare_value = rare_value
        self._rare_categories: dict[str, set] = {}

    def fit(self, X, y=None):
        import pandas as pd

        X = pd.DataFrame(X)
        self._rare_categories = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            rare = set(freq[freq < self.threshold].index)
            if rare:
                self._rare_categories[col] = rare
        return self

    def transform(self, X):
        import pandas as pd

        X = pd.DataFrame(X).copy()
        for col, rare_cats in self._rare_categories.items():
            if col in X.columns:
                X[col] = X[col].where(~X[col].isin(rare_cats), self.rare_value)
        return X
