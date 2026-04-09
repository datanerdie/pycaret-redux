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


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    alpha: float = 0.05,
    correction: bool = True,
) -> dict[str, Any]:
    """McNemar's test for comparing two classifiers.

    Uses the 2x2 contingency table of disagreements between two models.
    More appropriate than fold-based tests when comparing classifiers
    on the same dataset (Raschka 2018, arXiv:1811.12808).

    Parameters
    ----------
    y_true : array-like
        True labels.
    preds_a : array-like
        Predictions from model A.
    preds_b : array-like
        Predictions from model B.
    model_a_name : str
        Display name for model A.
    model_b_name : str
        Display name for model B.
    alpha : float
        Significance level.
    correction : bool
        Apply Edwards continuity correction.

    Returns
    -------
    dict with test results.
    """
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    # Build disagreement table
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true

    # b: A correct, B wrong | c: A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    if b + c == 0:
        return {
            "test": "McNemar's test",
            "statistic": 0.0,
            "p_value": 1.0,
            "alpha": alpha,
            "significant": False,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "b_count": int(b),
            "c_count": int(c),
            "conclusion": "Models make identical errors — no difference.",
        }

    # Chi-squared statistic (with optional continuity correction)
    if correction:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)

    p_value = float(1 - stats.chi2.cdf(chi2, df=1))
    significant = p_value < alpha

    if significant:
        better = model_a_name if b > c else model_b_name
        conclusion = f"{better} is significantly better (p={p_value:.4f})"
    else:
        conclusion = f"No significant difference (p={p_value:.4f})"

    return {
        "test": "McNemar's test",
        "statistic": round(float(chi2), 4),
        "p_value": round(p_value, 4),
        "alpha": alpha,
        "significant": significant,
        "model_a": model_a_name,
        "model_b": model_b_name,
        "b_count": int(b),
        "c_count": int(c),
        "conclusion": conclusion,
    }
