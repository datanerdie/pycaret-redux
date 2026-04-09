"""Statistical comparison of models using CV fold scores."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def compare_model_stats(
    model_a_scores: np.ndarray,
    model_b_scores: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    test: str = "wilcoxon",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Statistical test: is model A significantly different from model B?

    Parameters
    ----------
    model_a_scores : array-like
        Per-fold scores for model A.
    model_b_scores : array-like
        Per-fold scores for model B.
    model_a_name : str
        Display name for model A.
    model_b_name : str
        Display name for model B.
    test : str
        "wilcoxon" (non-parametric, default) or "ttest" (parametric paired t-test).
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    dict with keys: test, statistic, p_value, significant, conclusion,
    model_a_mean, model_b_mean, difference.
    """
    a = np.asarray(model_a_scores)
    b = np.asarray(model_b_scores)

    if len(a) != len(b):
        raise ValueError("Both models must have the same number of fold scores.")

    if len(a) < 3:
        raise ValueError("Need at least 3 folds for a statistical test.")

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    diff = mean_a - mean_b

    if test == "ttest":
        statistic, p_value = stats.ttest_rel(a, b)
        test_name = "Paired t-test"
    elif test == "wilcoxon":
        # Wilcoxon needs non-zero differences
        if np.allclose(a, b):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(a, b)
        test_name = "Wilcoxon signed-rank test"
    else:
        raise ValueError(f"Unknown test: '{test}'. Use 'ttest' or 'wilcoxon'.")

    significant = p_value < alpha
    if significant:
        better = model_a_name if mean_a > mean_b else model_b_name
        conclusion = f"{better} is significantly better (p={p_value:.4f})"
    else:
        conclusion = f"No significant difference (p={p_value:.4f})"

    return {
        "test": test_name,
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "alpha": alpha,
        "significant": significant,
        "model_a": model_a_name,
        "model_b": model_b_name,
        "model_a_mean": round(mean_a, 4),
        "model_b_mean": round(mean_b, 4),
        "difference": round(diff, 4),
        "conclusion": conclusion,
    }
