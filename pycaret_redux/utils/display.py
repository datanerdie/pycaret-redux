"""Display helpers replicating original PyCaret style.

Uses pandas Styler for Jupyter output (yellow highlights, lightgreen booleans)
and plain text for terminal.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _in_notebook() -> bool:
    """Detect if running inside a Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


def display_data_source(method_name: str, data_used: str) -> None:
    """Print a short data source banner before method output.

    Parameters
    ----------
    method_name : str
        Name of the method (e.g. "compare_models").
    data_used : str
        Description of which data is used.
    """
    if _in_notebook():
        from IPython.display import HTML, display

        display(
            HTML(
                f'<div style="background-color:#e8f4f8; color:#1a5276; '
                f"padding:4px 10px; border-left:3px solid #2980b9; "
                f'font-size:12px; margin-bottom:6px; font-family:monospace;">'
                f"<b>{method_name}</b> &mdash; {data_used}</div>"
            )
        )
    else:
        print(f"[{method_name}] {data_used}")


# Theme-agnostic color styles: each pairs a background with an explicit
# text color so the output is readable on both light and dark backgrounds.
_STYLES = {
    "highlight": "background-color: #fff3cd; color: #664d03",  # warm amber
    "setup_true": "background-color: #d1e7dd; color: #0f5132",  # green
    "timing": "background-color: #e2e3e5; color: #41464b",  # grey
    "bold": "font-weight: bold; text-align: left",
}


def _ipython_display(obj: Any) -> None:
    """Display an object via IPython if available, else print."""
    if _in_notebook():
        from IPython.display import display

        display(obj)
    else:
        print(obj)


# ---------------------------------------------------------------------------
# Setup summary — PyCaret style: Description | Value with lightgreen bools
# ---------------------------------------------------------------------------


def display_setup_summary(config: Any) -> None:
    """Display setup summary as a two-column DataFrame, PyCaret style."""
    cfg = config
    sc = cfg.setup_config

    container: list[list[Any]] = []
    container.append(["Session id", cfg.seed])
    container.append(["Target", cfg.target_name])
    container.append(["Target type", "Multiclass" if cfg.is_multiclass else "Binary"])
    n_total = len(cfg.X_train) + len(cfg.X_test)
    n_feat = len(cfg.feature_names_in)
    container.append(["Original data shape", f"({n_total}, {n_feat + 1})"])
    container.append(["Transformed train set shape", f"({len(cfg.X_train)}, {n_feat})"])
    container.append(["Transformed test set shape", f"({len(cfg.X_test)}, {n_feat})"])

    # Feature counts by type
    for fx_name in ["Numeric", "Categorical", "Ordinal", "Date", "Text"]:
        cols = cfg.feature_types.get(fx_name, [])
        if cols:
            container.append([f"{fx_name} features", len(cols)])

    if sc.preprocess:
        container.append(["Preprocess", sc.preprocess])
        container.append(["Imputation type", sc.imputation_type or "None"])
        if sc.imputation_type == "simple":
            container.append(["Numeric imputation", sc.numeric_imputation])
            container.append(["Categorical imputation", sc.categorical_imputation])
        if sc.normalize:
            container.append(["Normalize", sc.normalize])
            container.append(["Normalize method", sc.normalize_method])
        if sc.transformation:
            container.append(["Transformation", sc.transformation])
            container.append(["Transformation method", sc.transformation_method])
        if sc.pca:
            container.append(["PCA", sc.pca])
            container.append(["PCA method", sc.pca_method])
            container.append(["PCA components", sc.pca_components])
        if sc.feature_selection:
            container.append(["Feature selection", sc.feature_selection])
            container.append(["Feature selection method", sc.feature_selection_method])
        if sc.fix_imbalance:
            container.append(["Fix imbalance", sc.fix_imbalance])
            container.append(["Fix imbalance method", sc.fix_imbalance_method])
        if sc.remove_outliers:
            container.append(["Remove outliers", sc.remove_outliers])
            container.append(["Outliers threshold", sc.outliers_threshold])
        if sc.remove_multicollinearity:
            container.append(["Remove multicollinearity", sc.remove_multicollinearity])
            container.append(
                [
                    "Multicollinearity threshold",
                    sc.multicollinearity_threshold,
                ]
            )
        if sc.polynomial_features:
            container.append(["Polynomial features", sc.polynomial_features])
            container.append(["Polynomial degree", sc.polynomial_degree])
        if sc.low_variance_threshold is not None:
            container.append(["Low variance threshold", sc.low_variance_threshold])

    container.append(
        [
            "Fold Generator",
            cfg.fold_generator.__class__.__name__ if cfg.fold_generator else "None",
        ]
    )
    container.append(["Fold Number", sc.fold])
    container.append(["CPU Jobs", sc.n_jobs])
    container.append(["Use GPU", sc.use_gpu])

    setup_df = pd.DataFrame(container, columns=["Description", "Value"])

    if _in_notebook():
        styled = setup_df.style.apply(_highlight_setup).hide(axis="index")
        _ipython_display(styled)
    else:
        pd.set_option("display.max_rows", 100)
        print(setup_df.to_string(index=False))
        pd.reset_option("display.max_rows")


def _highlight_setup(column: pd.Series) -> list[str]:
    """Highlight True/Yes values with lightgreen background (PyCaret style)."""
    s = _STYLES["setup_true"]
    return [s if v is True or v == "Yes" else "" for v in column]


# ---------------------------------------------------------------------------
# CV fold scores — yellow background on Mean row
# ---------------------------------------------------------------------------


def display_fold_scores(fold_scores: pd.DataFrame, model_name: str = "") -> None:
    """Display CV fold scores with yellow-highlighted Mean row."""
    if _in_notebook():
        styled = _color_df(fold_scores, _STYLES["highlight"], ["Mean"], axis=1)
        styled = styled.format(precision=4)
        _ipython_display(styled)
    else:
        print(fold_scores.to_string())


def _color_df(
    df: pd.DataFrame,
    style: str,
    names: list,
    axis: int = 1,
) -> pd.io.formats.style.Styler:
    """Apply CSS style to specific rows (axis=1) or columns (axis=0).

    Parameters
    ----------
    style : str
        Full CSS style string (e.g. "background-color: #fff3cd; color: #664d03").
    """
    return df.style.apply(
        lambda x: [style if (x.name in names) else "" for _ in x],
        axis=axis,
    )


# ---------------------------------------------------------------------------
# Compare models — yellow on best-per-column, lightgrey on TT (Sec)
# ---------------------------------------------------------------------------


def display_comparison(
    comparison_df: pd.DataFrame,
    sort_col: str | None = None,
    highlight_best: bool = True,
) -> None:
    """Display comparison table with yellow-highlighted best values."""
    if _in_notebook():
        styled = _highlight_comparison(comparison_df)
        styled = styled.format(
            precision=4,
            subset=[c for c in comparison_df.columns if c != "Model"],
        )
        styled = styled.hide(axis="index")
        _ipython_display(styled)
    else:
        print(f"\n{comparison_df.to_string(index=False)}\n")


def _highlight_comparison(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply PyCaret-style highlighting to comparison DataFrame.

    Uses theme-agnostic colors with explicit text colors for readability
    on both light and dark backgrounds.
    """
    metric_cols = [c for c in df.columns if c not in ("Model", "TT (Sec)")]

    highlight_style = _STYLES["highlight"]

    def highlight_max(s: pd.Series) -> list[str]:
        is_best = s == s.max()
        return [highlight_style if v else "" for v in is_best]

    styler = df.style
    if metric_cols:
        styler = styler.apply(highlight_max, subset=metric_cols)

    if "TT (Sec)" in df.columns:
        styler = styler.map(lambda _: _STYLES["timing"], subset=["TT (Sec)"])

    styler = styler.map(lambda _: _STYLES["bold"], subset=["Model"])
    return styler


# ---------------------------------------------------------------------------
# Evaluation summary
# ---------------------------------------------------------------------------


def display_evaluation(
    scores: dict[str, float],
    metric_names: dict[str, str],
    ci_map: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Display evaluation metrics with optional bootstrap confidence intervals."""
    rows = []
    for metric_id, score in scores.items():
        name = metric_names.get(metric_id, metric_id)
        row = {"Metric": name, "Score": score}
        if ci_map and metric_id in ci_map:
            lower, upper = ci_map[metric_id]
            row["95% CI"] = f"[{lower}, {upper}]"
        rows.append(row)

    eval_df = pd.DataFrame(rows)

    if _in_notebook():
        format_dict = {"Score": "{:.4f}"}
        styled = eval_df.style.format(format_dict).hide(axis="index")
        _ipython_display(styled)
    else:
        print("\nModel Evaluation on Test Set:")
        for metric_id, score in scores.items():
            name = metric_names.get(metric_id, metric_id)
            ci_str = ""
            if ci_map and metric_id in ci_map:
                lower, upper = ci_map[metric_id]
                ci_str = f"  [{lower}, {upper}]"
            print(f"  {name:15s}: {score:.4f}{ci_str}")


# ---------------------------------------------------------------------------
# Progress indicator for compare_models
# ---------------------------------------------------------------------------


def create_progress(total: int, description: str = "Comparing models"):
    """Create a tqdm progress bar with fallback."""
    if _in_notebook():
        try:
            from tqdm.notebook import tqdm

            return tqdm(total=total, desc=description)
        except ImportError:
            # ipywidgets not available, fall back to plain tqdm
            from tqdm import tqdm

            return tqdm(total=total, desc=description)
    else:
        from tqdm import tqdm

        return tqdm(total=total, desc=description, ncols=80)
