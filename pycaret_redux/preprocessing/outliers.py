"""Outlier detection and removal."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers from training data using IsolationForest or other methods.

    This transformer only removes outliers during fit_transform (training).
    During transform (prediction), it passes data through unchanged.

    Parameters
    ----------
    method : str
        Detection method: "iforest" (IsolationForest).
    threshold : float
        Contamination fraction (proportion of outliers). Default 0.05.
    seed : int
        Random state.
    """

    def __init__(self, method: str = "iforest", threshold: float = 0.05, seed: int = 0):
        self.method = method
        self.threshold = threshold
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # During transform (inference), pass through unchanged
        return X

    def fit_transform(self, X, y=None):
        """Fit and remove outliers. Returns filtered X (and y if provided)."""
        X = pd.DataFrame(X)
        numeric_cols = X.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return X

        match self.method.lower():
            case "iforest":
                detector = IsolationForest(
                    contamination=self.threshold,
                    random_state=self.seed,
                    n_jobs=-1,
                )
            case _:
                raise ValueError(f"Unknown outlier method: '{self.method}'")

        predictions = detector.fit_predict(X[numeric_cols])
        mask = predictions != -1

        if y is not None:
            return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
        return X[mask].reset_index(drop=True)
