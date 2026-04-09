"""Auto-detect and transform skewed numeric features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class SkewTransformer(BaseEstimator, TransformerMixin):
    """Detect skewed features and apply power transform automatically.

    Only transforms features whose absolute skewness exceeds the threshold.
    Uses Yeo-Johnson (handles negative values).

    Parameters
    ----------
    threshold : float
        Minimum absolute skewness to trigger transformation. Default 1.0.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self._skewed_cols: list[int] = []
        self._transformer: PowerTransformer | None = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        n_cols = X_arr.shape[1]

        # Compute skewness per column
        self._skewed_cols = []
        for i in range(n_cols):
            col = X_arr[:, i]
            col_clean = col[~np.isnan(col)]
            if len(col_clean) > 2:
                skew = float(pd.Series(col_clean).skew())
                if abs(skew) > self.threshold:
                    self._skewed_cols.append(i)

        if self._skewed_cols:
            self._transformer = PowerTransformer(method="yeo-johnson", standardize=False)
            self._transformer.fit(X_arr[:, self._skewed_cols])

        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=float).copy()
        if self._skewed_cols and self._transformer is not None:
            X_arr[:, self._skewed_cols] = self._transformer.transform(X_arr[:, self._skewed_cols])
        if hasattr(X, "columns"):
            return pd.DataFrame(X_arr, columns=X.columns, index=X.index)
        return X_arr
