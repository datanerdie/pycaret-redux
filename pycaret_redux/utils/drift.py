"""Data drift detection using statistical tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_test: str = "ks",
    categorical_test: str = "chi2",
    alpha: float = 0.05,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Detect feature drift between reference (train) and current (new) data.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference data (typically training data).
    current : pd.DataFrame
        New data to check for drift.
    numeric_test : str
        "ks" (Kolmogorov-Smirnov) or "psi" (Population Stability Index).
    categorical_test : str
        "chi2" (Chi-squared test).
    alpha : float
        Significance level for statistical tests.
    numeric_cols : list[str], optional
        Numeric columns to check. Auto-detected if None.
    categorical_cols : list[str], optional
        Categorical columns to check. Auto-detected if None.

    Returns
    -------
    pd.DataFrame with columns: Feature, Type, Test, Statistic, P-Value, Drifted.
    """
    common_cols = list(set(reference.columns) & set(current.columns))

    if numeric_cols is None:
        numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(reference[c])]
    if categorical_cols is None:
        categorical_cols = [
            c
            for c in common_cols
            if c not in numeric_cols and pd.api.types.is_object_dtype(reference[c])
        ]

    results: list[dict[str, Any]] = []

    # Numeric drift
    for col in numeric_cols:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        if numeric_test == "ks":
            stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            test_name = "KS Test"
        elif numeric_test == "psi":
            stat = _psi(ref_vals, cur_vals)
            p_value = None
            test_name = "PSI"
        else:
            raise ValueError(f"Unknown numeric_test: '{numeric_test}'")

        drifted = (p_value < alpha) if p_value is not None else (stat > 0.2)
        results.append(
            {
                "Feature": col,
                "Type": "Numeric",
                "Test": test_name,
                "Statistic": round(stat, 4),
                "P-Value": round(p_value, 4) if p_value is not None else "N/A",
                "Drifted": drifted,
            }
        )

    # Categorical drift
    for col in categorical_cols:
        ref_vals = reference[col].dropna()
        cur_vals = current[col].dropna()

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        # Build contingency table
        all_cats = sorted(set(ref_vals.unique()) | set(cur_vals.unique()))
        ref_counts = ref_vals.value_counts().reindex(all_cats, fill_value=0)
        cur_counts = cur_vals.value_counts().reindex(all_cats, fill_value=0)

        contingency = np.array([ref_counts.values, cur_counts.values])
        # Remove zero columns to avoid chi2 issues
        nonzero = contingency.sum(axis=0) > 0
        contingency = contingency[:, nonzero]

        if contingency.shape[1] < 2:
            continue

        stat, p_value, _, _ = stats.chi2_contingency(contingency)
        drifted = p_value < alpha
        results.append(
            {
                "Feature": col,
                "Type": "Categorical",
                "Test": "Chi-squared",
                "Statistic": round(stat, 4),
                "P-Value": round(p_value, 4),
                "Drifted": drifted,
            }
        )

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("Drifted", ascending=False).reset_index(drop=True)
    return df


def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Population Stability Index."""
    # Create bins from reference
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Normalize to proportions, add small epsilon to avoid log(0)
    eps = 1e-6
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)
