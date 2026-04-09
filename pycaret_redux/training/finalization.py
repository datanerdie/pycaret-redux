"""finalize_model and predict_model logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from pycaret_redux.config import ExperimentConfig
from pycaret_redux.training.cross_validation import _build_full_pipeline, _extract_estimator


def finalize_model(
    estimator: Any,
    config: ExperimentConfig,
) -> Any:
    """Retrain estimator on full dataset (train + test).

    Parameters
    ----------
    estimator : fitted estimator
        Model to finalize.
    config : ExperimentConfig
        Experiment config with train/test data.

    Returns
    -------
    Refitted estimator on combined train+test data.
    """
    X_full = pd.concat([config.X_train, config.X_test], ignore_index=True)
    y_full = pd.concat([config.y_train, config.y_test], ignore_index=True)

    pipeline = _build_full_pipeline(config.pipeline, clone(estimator))
    pipeline.fit(X_full, y_full)
    return _extract_estimator(pipeline)


def predict_model(
    estimator: Any,
    config: ExperimentConfig,
    data: pd.DataFrame | np.ndarray | None = None,
    probability_threshold: float | None = None,
    raw_score: bool = False,
    round_to: int = 4,
) -> pd.DataFrame:
    """Make predictions using a fitted model.

    Parameters
    ----------
    estimator : fitted estimator
        Trained model.
    config : ExperimentConfig
        Experiment config.
    data : DataFrame, optional
        New data. If None, uses test set.
    probability_threshold : float, optional
        Custom threshold for binary classification.
    raw_score : bool
        Include raw probability scores.
    round_to : int
        Decimal places for probabilities.

    Returns
    -------
    DataFrame with predictions appended.
    """
    if data is not None:
        X = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
    else:
        X = config.X_test.copy()

    # Apply preprocessing
    if config.pipeline is not None:
        X_transformed = config.pipeline.transform(X)
    else:
        X_transformed = X

    # Predict
    predictions = estimator.predict(X_transformed)

    result = X.copy()
    result["prediction_label"] = predictions

    # Predict probabilities if available
    if hasattr(estimator, "predict_proba"):
        try:
            probas = estimator.predict_proba(X_transformed)

            if probability_threshold is not None and probas.shape[1] == 2:
                # Binary classification with custom threshold
                result["prediction_label"] = (probas[:, 1] >= probability_threshold).astype(int)

            if raw_score:
                for i in range(probas.shape[1]):
                    result[f"prediction_score_{i}"] = np.round(probas[:, i], round_to)
            else:
                # Just include the score for the predicted class
                result["prediction_score"] = np.round(
                    np.max(probas, axis=1), round_to
                )
        except Exception:
            pass

    return result
