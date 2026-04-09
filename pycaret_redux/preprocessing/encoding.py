"""Categorical encoding strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class SmartEncoder(BaseEstimator, TransformerMixin):
    """Smart categorical encoder that handles binary and multi-category features differently.

    - Binary features (2 unique values): ordinal encoded (0/1), no OHE needed.
    - Multi-category features: one-hot encoded.
    - Optionally drops the first OHE column to avoid the dummy variable trap
      (important for linear models like logistic regression).

    Parameters
    ----------
    max_categories : int
        Max unique values for one-hot encoding.
    drop_first : bool
        If True, drops the first category in OHE to avoid multicollinearity.
        Recommended for linear models (LR, Ridge, LDA).
    """

    def __init__(self, max_categories: int = 25, drop_first: bool = False):
        self.max_categories = max_categories
        self.drop_first = drop_first
        self._binary_cols: list[str | int] = []
        self._multi_cols: list[str | int] = []
        self._binary_encoder: OrdinalEncoder | None = None
        self._multi_encoder: OneHotEncoder | None = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        self._binary_cols = []
        self._multi_cols = []
        self.n_features_in_ = X.shape[1]  # sklearn fitted marker

        for col in X.columns:
            n_unique = X[col].nunique()
            if n_unique <= 2:
                self._binary_cols.append(col)
            else:
                self._multi_cols.append(col)

        # Fit binary encoder (simple ordinal 0/1)
        if self._binary_cols:
            self._binary_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            self._binary_encoder.fit(X[self._binary_cols])

        # Fit OHE for multi-category columns
        if self._multi_cols:
            self._multi_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="infrequent_if_exist",
                max_categories=self.max_categories,
                drop="first" if self.drop_first else None,
            )
            self._multi_encoder.fit(X[self._multi_cols])

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        parts = []

        if self._binary_cols and self._binary_encoder is not None:
            existing = [c for c in self._binary_cols if c in X.columns]
            if existing:
                binary_encoded = self._binary_encoder.transform(X[existing])
                parts.append(pd.DataFrame(binary_encoded, columns=existing, index=X.index))

        if self._multi_cols and self._multi_encoder is not None:
            existing = [c for c in self._multi_cols if c in X.columns]
            if existing:
                multi_encoded = self._multi_encoder.transform(X[existing])
                ohe_names = self._multi_encoder.get_feature_names_out(existing)
                parts.append(pd.DataFrame(multi_encoded, columns=ohe_names, index=X.index))

        if not parts:
            return X

        return pd.concat(parts, axis=1)

    def get_feature_names_out(self, input_features=None):
        names = list(self._binary_cols)
        if self._multi_encoder is not None:
            names.extend(self._multi_encoder.get_feature_names_out(self._multi_cols))
        return np.array(names)

    def __sklearn_is_fitted__(self):
        """Tell sklearn this estimator is fitted after fit() is called."""
        return hasattr(self, "_binary_cols") and hasattr(self, "_multi_cols")


def build_categorical_encoder(
    max_encoding_ohe: int = 25,
    ordinal_features: dict[str, list] | None = None,
    encoding_method: Any = None,
    drop_first: bool = False,
) -> BaseEstimator:
    """Build the appropriate categorical encoder.

    Parameters
    ----------
    max_encoding_ohe : int
        Max unique values for one-hot encoding.
    ordinal_features : dict, optional
        Mapping of feature name -> ordered categories for ordinal encoding.
    encoding_method : Any, optional
        Custom encoder instance. If provided, used as-is.
    drop_first : bool
        Drop first OHE column to avoid dummy variable trap.
    """
    if encoding_method is not None:
        return encoding_method

    return SmartEncoder(max_categories=max_encoding_ohe, drop_first=drop_first)


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

    Categories appearing less than `threshold`` fraction of samples are
    replaced with ``rare_value``.
    """

    def __init__(self, threshold: float = 0.05, rare_value: str = "rare"):
        self.threshold = threshold
        self.rare_value = rare_value
        self._rare_categories: dict[str, set] = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self._rare_categories = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            rare = set(freq[freq < self.threshold].index)
            if rare:
                self._rare_categories[col] = rare
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, rare_cats in self._rare_categories.items():
            if col in X.columns:
                X[col] = X[col].where(~X[col].isin(rare_cats), self.rare_value)
        return X
