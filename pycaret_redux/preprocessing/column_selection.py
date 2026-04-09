"""Feature type detection and column routing."""

from __future__ import annotations

import pandas as pd

from pycaret_redux.config import SetupConfig


def detect_feature_types(
    X: pd.DataFrame,
    setup_cfg: SetupConfig,
) -> dict[str, list[str]]:
    """Detect or apply user-specified feature types.

    Returns a dict with keys: Numeric, Categorical, Ordinal, Date, Text, Ignore, Keep.
    """
    types: dict[str, list[str]] = {
        "Numeric": [],
        "Categorical": [],
        "Ordinal": [],
        "Date": [],
        "Text": [],
        "Ignore": [],
        "Keep": [],
    }

    types["Ignore"] = setup_cfg.ignore_features or []
    types["Keep"] = setup_cfg.keep_features or []
    types["Date"] = setup_cfg.date_features or list(X.select_dtypes(include="datetime").columns)
    types["Text"] = setup_cfg.text_features or []
    types["Ordinal"] = list((setup_cfg.ordinal_features or {}).keys())

    excluded = set(types["Ignore"] + types["Date"] + types["Text"] + types["Ordinal"])

    if setup_cfg.numeric_features is not None:
        types["Numeric"] = [c for c in setup_cfg.numeric_features if c not in excluded]
    else:
        cat_override = set(setup_cfg.categorical_features or [])
        types["Numeric"] = [
            c
            for c in X.select_dtypes(include="number").columns
            if c not in excluded and c not in cat_override
        ]

    if setup_cfg.categorical_features is not None:
        types["Categorical"] = [c for c in setup_cfg.categorical_features if c not in excluded]
    else:
        types["Categorical"] = [
            c
            for c in X.select_dtypes(include=["object", "category", "string"]).columns
            if c not in excluded and c not in set(types["Date"] + types["Text"])
        ]

    return types
