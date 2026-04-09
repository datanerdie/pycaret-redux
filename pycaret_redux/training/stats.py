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


def cochrans_q_test(
    y_true: np.ndarray,
    predictions_list: list[np.ndarray],
    model_names: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cochran's Q test for comparing 3+ classifiers.

    Tests null hypothesis: all classifiers have the same error rate.
    Extends McNemar's test to multiple classifiers.

    Parameters
    ----------
    y_true : array-like
        True labels.
    predictions_list : list of array-like
        Predictions from each classifier.
    model_names : list of str, optional
        Display names for the models.
    alpha : float
        Significance level.

    Returns
    -------
    dict with test results.
    """
    y_true = np.asarray(y_true)
    k = len(predictions_list)

    if k < 3:
        raise ValueError("Cochran's Q requires at least 3 classifiers.")

    if model_names is None:
        model_names = [f"Model_{i}" for i in range(k)]

    # Build correctness matrix: (n_samples, k_classifiers)
    correct = np.column_stack([np.asarray(p) == y_true for p in predictions_list])

    # Row sums and column sums
    row_sums = correct.sum(axis=1)  # L_i
    col_sums = correct.sum(axis=0)  # G_j

    T = float(correct.sum())

    numerator = (k - 1) * (k * float((col_sums**2).sum()) - T**2)
    denominator = k * T - float((row_sums**2).sum())

    if denominator == 0:
        return {
            "test": "Cochran's Q test",
            "statistic": 0.0,
            "p_value": 1.0,
            "alpha": alpha,
            "significant": False,
            "k_classifiers": k,
            "model_names": model_names,
            "conclusion": "All classifiers perform identically.",
        }

    q_stat = numerator / denominator
    p_value = float(1 - stats.chi2.cdf(q_stat, df=k - 1))
    significant = p_value < alpha

    conclusion = (
        f"Significant difference among {k} classifiers (p={p_value:.4f})"
        if significant
        else f"No significant difference among {k} classifiers (p={p_value:.4f})"
    )

    return {
        "test": "Cochran's Q test",
        "statistic": round(float(q_stat), 4),
        "p_value": round(p_value, 4),
        "alpha": alpha,
        "significant": significant,
        "k_classifiers": k,
        "model_names": model_names,
        "conclusion": conclusion,
    }


def five_by_two_cv_f_test(
    estimator_a: Any,
    estimator_b: Any,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Dietterich's 5x2cv paired F-test.

    More powerful than paired t-test with lower false positive rate.
    Repeats 50/50 split 5 times and computes F-statistic.

    Based on: Dietterich (1998), Alpaydin (1999), Raschka (2018).

    Parameters
    ----------
    estimator_a : estimator
        First classifier (unfitted, will be cloned).
    estimator_b : estimator
        Second classifier (unfitted, will be cloned).
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    seed : int
        Random state.
    alpha : float
        Significance level.

    Returns
    -------
    dict with test results.
    """
    import pandas as pd
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedShuffleSplit

    # Keep as DataFrame/Series if provided (for pipeline compatibility)
    use_iloc = isinstance(X, pd.DataFrame)
    y_arr = np.asarray(y)  # for split stratification only

    diffs = np.zeros((5, 2))
    variances = np.zeros(5)

    for i in range(5):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + i)
        for fold_idx, (idx1, idx2) in enumerate(splitter.split(X, y_arr)):
            if use_iloc:
                X1, X2 = X.iloc[idx1], X.iloc[idx2]
                y1, y2 = y.iloc[idx1], y.iloc[idx2]
            else:
                X1, X2 = X[idx1], X[idx2]
                y1, y2 = y[idx1], y[idx2]

            # Train A on set1, test on set2
            a1 = clone(estimator_a).fit(X1, y1)
            b1 = clone(estimator_b).fit(X1, y1)
            score_a1 = float(np.mean(np.asarray(a1.predict(X2)) == np.asarray(y2)))
            score_b1 = float(np.mean(np.asarray(b1.predict(X2)) == np.asarray(y2)))
            diffs[i, 0] = score_a1 - score_b1

            # Train A on set2, test on set1
            a2 = clone(estimator_a).fit(X2, y2)
            b2 = clone(estimator_b).fit(X2, y2)
            score_a2 = float(np.mean(np.asarray(a2.predict(X1)) == np.asarray(y1)))
            score_b2 = float(np.mean(np.asarray(b2.predict(X1)) == np.asarray(y1)))
            diffs[i, 1] = score_a2 - score_b2

        variances[i] = np.var([diffs[i, 0], diffs[i, 1]], ddof=0)

    sum_sq = float(np.sum(diffs**2))
    sum_var = float(np.sum(variances))

    if sum_var == 0:
        return {
            "test": "5x2cv paired F-test",
            "statistic": 0.0,
            "p_value": 1.0,
            "alpha": alpha,
            "significant": False,
            "model_a": type(estimator_a).__name__,
            "model_b": type(estimator_b).__name__,
            "conclusion": "No difference detected.",
        }

    f_stat = sum_sq / (2 * sum_var)
    p_value = float(1 - stats.f.cdf(f_stat, dfn=10, dfd=5))
    significant = p_value < alpha

    conclusion = (
        f"Significant difference (p={p_value:.4f})"
        if significant
        else f"No significant difference (p={p_value:.4f})"
    )

    return {
        "test": "5x2cv paired F-test",
        "statistic": round(float(f_stat), 4),
        "p_value": round(p_value, 4),
        "alpha": alpha,
        "significant": significant,
        "model_a": type(estimator_a).__name__,
        "model_b": type(estimator_b).__name__,
        "conclusion": conclusion,
    }
