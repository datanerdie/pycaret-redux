"""Data profiling utilities shown during setup(profile=True)."""

from __future__ import annotations

import pandas as pd


def profile_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_types: dict[str, list[str]],
) -> None:
    """Display a data profile summary.

    Shows missing values, class distribution, numeric statistics,
    and feature correlations.
    """
    from pycaret_redux.utils.display import _in_notebook, _ipython_display

    sections: list[tuple[str, pd.DataFrame]] = []

    # 1. Class distribution
    class_dist = y_train.value_counts().reset_index()
    class_dist.columns = ["Class", "Count"]
    class_dist["Proportion"] = (class_dist["Count"] / len(y_train)).round(4)
    sections.append(("Target Distribution", class_dist))

    # 2. Missing values
    missing = X_train.isnull().sum()
    missing_pct = (missing / len(X_train) * 100).round(2)
    missing_df = pd.DataFrame({"Missing": missing, "% Missing": missing_pct})
    missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)
    if len(missing_df) > 0:
        sections.append(("Missing Values", missing_df))

    # 3. Numeric feature statistics
    numeric_cols = feature_types.get("Numeric", [])
    if numeric_cols:
        existing_cols = [c for c in numeric_cols if c in X_train.columns]
        if existing_cols:
            stats_df = X_train[existing_cols].describe().T
            stats_df = stats_df[["mean", "std", "min", "25%", "50%", "75%", "max"]]
            stats_df = stats_df.round(4)
            sections.append(("Numeric Feature Statistics", stats_df))

    # 4. Categorical feature cardinality
    cat_cols = feature_types.get("Categorical", [])
    if cat_cols:
        existing_cols = [c for c in cat_cols if c in X_train.columns]
        if existing_cols:
            card_data = []
            for col in existing_cols:
                card_data.append(
                    {
                        "Feature": col,
                        "Unique": X_train[col].nunique(),
                        "Top Value": X_train[col].mode().iloc[0]
                        if len(X_train[col].mode()) > 0
                        else "N/A",
                        "Top Freq": X_train[col].value_counts().iloc[0]
                        if len(X_train[col]) > 0
                        else 0,
                    }
                )
            sections.append(("Categorical Feature Summary", pd.DataFrame(card_data)))

    # 5. Top correlations with numeric features
    if numeric_cols and len(numeric_cols) >= 2:
        existing_cols = [c for c in numeric_cols if c in X_train.columns]
        if len(existing_cols) >= 2:
            corr = X_train[existing_cols].corr()
            # Get top correlated pairs (excluding self-correlation)
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    pairs.append(
                        {
                            "Feature A": corr.columns[i],
                            "Feature B": corr.columns[j],
                            "Correlation": round(corr.iloc[i, j], 4),
                        }
                    )
            if pairs:
                corr_df = pd.DataFrame(pairs)
                corr_df["Abs Correlation"] = corr_df["Correlation"].abs()
                corr_df = corr_df.sort_values("Abs Correlation", ascending=False).head(10)
                corr_df = corr_df.drop(columns="Abs Correlation")
                sections.append(("Top Feature Correlations", corr_df))

    # 6. Class imbalance check
    min_class = class_dist["Proportion"].min()
    max_class = class_dist["Proportion"].max()
    imbalance_ratio = round(max_class / min_class, 2) if min_class > 0 else float("inf")
    if imbalance_ratio > 3:
        warning_df = pd.DataFrame(
            [
                {
                    "Warning": f"Class imbalance detected (ratio {imbalance_ratio}:1). "
                    "Consider using fix_imbalance=True in setup()."
                }
            ]
        )
        sections.append(("Warnings", warning_df))

    # Display
    if _in_notebook():
        for title, df in sections:
            styled = df.style.set_caption(title)
            _ipython_display(styled)
    else:
        for title, df in sections:
            print(f"\n--- {title} ---")
            print(df.to_string())
        print()
