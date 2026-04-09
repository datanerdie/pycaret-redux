"""Feature engineering transformers: polynomial, date, binning, grouping."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


class ExtractDateTimeFeatures(BaseEstimator, TransformerMixin):
    """Extract date/time components from datetime columns.

    Parameters
    ----------
    features : list[str]
        Which components to extract. Options: day, month, year, hour, minute,
        second, dayofweek, dayofyear, quarter, weekofyear.
    """

    def __init__(self, features: list[str] | None = None):
        self.features = features or ["day", "month", "year"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        new_cols = {}
        drop_cols = []

        for col in X.columns:
            if not pd.api.types.is_datetime64_any_dtype(X[col]):
                try:
                    X[col] = pd.to_datetime(X[col])
                except (ValueError, TypeError):
                    continue

            dt = X[col].dt
            for feat in self.features:
                attr = feat.lower()
                if attr == "weekofyear":
                    attr = "isocalendar"
                    new_cols[f"{col}_{feat}"] = dt.isocalendar().week.astype(int)
                elif hasattr(dt, attr):
                    new_cols[f"{col}_{feat}"] = getattr(dt, attr)
            drop_cols.append(col)

        for name, values in new_cols.items():
            X[name] = values
        X = X.drop(columns=drop_cols)
        return X


class GroupFeatures(BaseEstimator, TransformerMixin):
    """Aggregate grouped features (mean, median, std).

    Parameters
    ----------
    groups : dict[str, list[str]]
        Mapping of group name to list of feature columns.
    drop : bool
        Whether to drop the original group columns.
    """

    def __init__(self, groups: dict[str, list[str]], drop: bool = False):
        self.groups = groups
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for group_name, cols in self.groups.items():
            existing = [c for c in cols if c in X.columns]
            if not existing:
                continue
            subset = X[existing]
            X[f"{group_name}_mean"] = subset.mean(axis=1)
            X[f"{group_name}_median"] = subset.median(axis=1)
            X[f"{group_name}_std"] = subset.std(axis=1)
            if self.drop:
                X = X.drop(columns=existing)
        return X


def build_binning_transformer(columns: list[str], n_bins: int = 5) -> KBinsDiscretizer:
    """Build a binning transformer for specified numeric columns."""
    return KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile",
        subsample=None,
    )
