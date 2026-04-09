"""Experiment configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline


@dataclass
class SetupConfig:
    """Immutable record of user choices from setup().

    Captures all preprocessing and experiment configuration parameters
    so they can be referenced throughout the experiment lifecycle.
    """

    # Data splitting
    train_size: float = 0.7
    data_split_shuffle: bool = True
    data_split_stratify: bool | list[str] = True

    # Cross-validation
    fold_strategy: str | BaseCrossValidator = "stratifiedkfold"
    fold: int = 10
    fold_shuffle: bool = False
    fold_groups: str | pd.DataFrame | None = None

    # Feature type overrides
    numeric_features: list[str] | None = None
    categorical_features: list[str] | None = None
    ordinal_features: dict[str, list] | None = None
    date_features: list[str] | None = None
    text_features: list[str] | None = None
    ignore_features: list[str] | None = None
    keep_features: list[str] | None = None

    # Preprocessing toggles
    preprocess: bool = True
    imputation_type: str | None = "simple"
    numeric_imputation: str = "mean"
    categorical_imputation: str = "mode"
    iterative_imputation_iters: int = 5
    text_features_method: str = "tf-idf"
    max_encoding_ohe: int = 25
    encoding_method: Any = None
    rare_to_value: float | None = None
    rare_value: str = "rare"
    drop_first_ohe: bool = False

    # Feature engineering
    polynomial_features: bool = False
    polynomial_degree: int = 2
    low_variance_threshold: float | None = None
    group_features: dict | None = None
    drop_groups: bool = False
    remove_multicollinearity: bool = False
    multicollinearity_threshold: float = 0.9
    bin_numeric_features: list[str] | None = None
    create_date_columns: list[str] = field(default_factory=lambda: ["day", "month", "year"])

    # Outliers
    remove_outliers: bool = False
    outliers_method: str = "iforest"
    outliers_threshold: float = 0.05

    # Imbalance
    fix_imbalance: bool = False
    fix_imbalance_method: str | Any = "SMOTE"

    # Transformations
    transformation: bool = False
    transformation_method: str = "yeo-johnson"
    normalize: bool = False
    normalize_method: str = "zscore"

    # Dimensionality reduction
    pca: bool = False
    pca_method: str = "linear"
    pca_components: int | float | str | None = None

    # Feature selection
    feature_selection: bool = False
    feature_selection_method: str = "classic"
    feature_selection_estimator: str | Any = "lightgbm"
    n_features_to_select: int | float = 0.2

    # Custom pipeline
    custom_pipeline: Any = None
    custom_pipeline_position: int = -1

    # Execution
    n_jobs: int | None = -1
    use_gpu: bool = False
    session_id: int | None = None
    verbose: bool = True


@dataclass
class ExperimentConfig:
    """Mutable experiment state container.

    Holds all runtime state for a classification experiment, replacing
    the scattered instance attributes across PyCaret's 5-level hierarchy.
    """

    # Setup config (set once during setup)
    setup_config: SetupConfig | None = None

    # Random state
    seed: int = 0

    # Data partitions
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None

    # Metadata
    target_name: str = ""
    feature_names_in: list[str] = field(default_factory=list)
    feature_types: dict[str, list[str]] = field(default_factory=dict)
    is_multiclass: bool = False

    # Feature labels: maps feature_name -> {code: label} for display in plots
    # e.g. {"smoking": {0: "No", 1: "Yes"}, "marital": {1: "Married", 2: "Single"}}
    feature_labels: dict[str, dict] = field(default_factory=dict)
    # Target labels: maps target code -> label
    # e.g. {0: "Benign", 1: "Malignant"}
    target_labels: dict = field(default_factory=dict)

    # Pipeline (preprocessing only, no estimator)
    pipeline: Pipeline | None = None

    # CV splitter
    fold_generator: BaseCrossValidator | None = None

    # Lifecycle flag
    is_setup_done: bool = False
