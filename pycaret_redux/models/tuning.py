"""Data-aware hyperparameter search space generation.

Adapts search grids based on dataset characteristics (n_samples, n_features,
n_classes) so the search space is appropriate for the problem.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def adapt_search_space(
    grid: dict[str, list[Any]],
    estimator_id: str,
    n_samples: int,
    n_features: int,
    n_classes: int = 2,
) -> dict[str, list[Any]]:
    """Adapt a hyperparameter search grid based on dataset characteristics.

    Parameters
    ----------
    grid : dict
        Original search grid from model definitions.
    estimator_id : str
        Model ID (e.g. "rf", "dt", "lr").
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features.
    n_classes : int
        Number of target classes.

    Returns
    -------
    dict
        Adapted search grid.
    """
    if not grid:
        return grid

    adapted = dict(grid)

    # --- max_depth: cap based on log2(n_samples) ---
    if "max_depth" in adapted:
        sensible_max = max(3, int(math.log2(n_samples)))
        adapted["max_depth"] = [d for d in adapted["max_depth"] if d is None or d <= sensible_max]
        if not adapted["max_depth"]:
            adapted["max_depth"] = list(range(1, sensible_max + 1))
        logger.debug("Adapted max_depth to max %d (n_samples=%d)", sensible_max, n_samples)

    # --- n_estimators: scale with dataset size ---
    if "n_estimators" in adapted:
        if n_samples < 500:
            adapted["n_estimators"] = [10, 30, 50, 100]
        elif n_samples < 5000:
            adapted["n_estimators"] = [50, 100, 200, 300]
        else:
            adapted["n_estimators"] = [100, 200, 300, 500, 1000]

    # --- min_samples_leaf: percentage-based ---
    if "min_samples_leaf" in adapted:
        min_leaf_1pct = max(2, int(n_samples * 0.01))
        min_leaf_5pct = max(5, int(n_samples * 0.05))
        adapted["min_samples_leaf"] = sorted(
            {2, 5, min_leaf_1pct, min_leaf_5pct, max(10, min_leaf_1pct * 2)}
        )

    # --- min_samples_split: at least 2x min_samples_leaf ---
    if "min_samples_split" in adapted:
        min_split_2pct = max(2, int(n_samples * 0.02))
        adapted["min_samples_split"] = sorted({2, 5, min_split_2pct, min_split_2pct * 2})

    # --- C / alpha for regularized models: scale with feature range ---
    if "C" in adapted:
        # Wider range for more features (more regularization may be needed)
        if n_features > 100:
            adapted["C"] = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        elif n_features > 20:
            adapted["C"] = [0.001, 0.01, 0.1, 1, 5, 10]
        # else keep default

    if "alpha" in adapted and estimator_id in ("svm", "ridge"):
        if n_features > 100:
            adapted["alpha"] = [1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0]

    # --- max_features: cap at actual feature count ---
    if "max_features" in adapted:
        adapted["max_features"] = [
            f
            for f in adapted["max_features"]
            if not isinstance(f, (int, float)) or f <= n_features
        ]
        if not adapted["max_features"]:
            adapted["max_features"] = [1.0, "sqrt", "log2"]

    # --- num_leaves for LightGBM: cap at 2^max_depth ---
    if "num_leaves" in adapted and "max_depth" in adapted:
        max_d = max((d for d in adapted["max_depth"] if isinstance(d, int)), default=10)
        max_leaves = 2**max_d
        adapted["num_leaves"] = [nl for nl in adapted["num_leaves"] if nl <= max_leaves]
        if not adapted["num_leaves"]:
            adapted["num_leaves"] = [15, 31]

    # --- hidden_layer_sizes for MLP ---
    if "hidden_layer_sizes" in adapted:
        if n_features < 10:
            adapted["hidden_layer_sizes"] = [(50,), (25, 25), (50, 25)]
        elif n_features > 100:
            adapted["hidden_layer_sizes"] = [
                (100,),
                (200,),
                (100, 100),
                (200, 100),
            ]

    logger.info(
        "Adapted search space for %s: n_samples=%d, n_features=%d, n_classes=%d",
        estimator_id,
        n_samples,
        n_features,
        n_classes,
    )
    return adapted
