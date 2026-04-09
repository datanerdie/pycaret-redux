"""Standalone imbalance handling, applied per CV fold (not in pipeline)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def resample(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    method: str | Any = "SMOTE",
    seed: int = 0,
) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
    """Resample training data to handle class imbalance.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    method : str or sampler instance
        Resampling method name or pre-configured sampler.
        String options: SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler,
        SMOTEENN, SMOTETomek, BorderlineSMOTE.
    seed : int
        Random state.

    Returns
    -------
    X_resampled, y_resampled
    """
    if not isinstance(method, str):
        # Assume pre-configured sampler
        return method.fit_resample(X, y)

    try:
        import importlib.util

        if importlib.util.find_spec("imblearn") is None:
            raise ImportError
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for fix_imbalance. "
            "Install with: uv add imbalanced-learn"
        )

    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import (
        ADASYN,
        SMOTE,
        BorderlineSMOTE,
        RandomOverSampler,
    )
    from imblearn.under_sampling import RandomUnderSampler

    samplers = {
        "smote": SMOTE(random_state=seed),
        "adasyn": ADASYN(random_state=seed),
        "randomoversampler": RandomOverSampler(random_state=seed),
        "randomundersampler": RandomUnderSampler(random_state=seed),
        "smoteenn": SMOTEENN(random_state=seed),
        "smotetomek": SMOTETomek(random_state=seed),
        "borderlinesmote": BorderlineSMOTE(random_state=seed),
    }

    method_lower = method.lower()
    if method_lower not in samplers:
        raise ValueError(
            f"Unknown resampling method: '{method}'. "
            f"Choose from: {', '.join(samplers.keys())}"
        )

    sampler = samplers[method_lower]
    return sampler.fit_resample(X, y)
