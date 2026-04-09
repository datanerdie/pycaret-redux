"""Build the preprocessing pipeline using sklearn ColumnTransformer."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
    f_classif,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from pycaret_redux.config import ExperimentConfig, SetupConfig
from pycaret_redux.preprocessing.encoding import (
    RareCategoryGrouper,
    build_categorical_encoder,
    build_ordinal_encoder,
)
from pycaret_redux.preprocessing.feature_engineering import (
    ExtractDateTimeFeatures,
    GroupFeatures,
)
from pycaret_redux.preprocessing.imputation import (
    build_categorical_imputer,
    build_numeric_imputer,
)
from pycaret_redux.preprocessing.scaling import build_normalizer, build_power_transformer


def build_preprocessing_pipeline(
    config: ExperimentConfig,
    setup_cfg: SetupConfig,
) -> Pipeline:
    """Build a standard sklearn preprocessing Pipeline.

    The pipeline uses ColumnTransformer for parallel processing of
    numeric and categorical features, followed by optional whole-dataset
    transforms (polynomial, PCA, feature selection, etc.).

    Parameters
    ----------
    config : ExperimentConfig
        Experiment state with feature_types populated.
    setup_cfg : SetupConfig
        User configuration from setup().

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps: list[tuple[str, Any]] = []

    numeric_cols = config.feature_types.get("Numeric", [])
    categorical_cols = config.feature_types.get("Categorical", [])
    ordinal_cols = config.feature_types.get("Ordinal", [])
    date_cols = config.feature_types.get("Date", [])

    # --- Stage 1: Column-parallel preprocessing ---
    transformers: list[tuple[str, Any, list[str]]] = []

    # Numeric branch: imputation → optional power transform → optional normalization
    if numeric_cols:
        num_steps: list[tuple[str, Any]] = []
        if setup_cfg.imputation_type == "simple":
            num_steps.append(("imputer", build_numeric_imputer(setup_cfg.numeric_imputation)))
        if setup_cfg.transformation:
            num_steps.append((
                "power_transform",
                build_power_transformer(setup_cfg.transformation_method, config.seed),
            ))
        if setup_cfg.normalize:
            num_steps.append(("normalizer", build_normalizer(setup_cfg.normalize_method)))
        if num_steps:
            transformers.append(("numeric", Pipeline(num_steps), numeric_cols))
        else:
            transformers.append(("numeric", "passthrough", numeric_cols))

    # Categorical branch: optional rare grouping → imputation → encoding
    if categorical_cols:
        cat_steps: list[tuple[str, Any]] = []
        if setup_cfg.rare_to_value is not None:
            cat_steps.append((
                "rare_grouper",
                RareCategoryGrouper(
                    threshold=setup_cfg.rare_to_value,
                    rare_value=setup_cfg.rare_value,
                ),
            ))
        if setup_cfg.imputation_type == "simple":
            cat_imputer = build_categorical_imputer(setup_cfg.categorical_imputation)
            cat_steps.append(("imputer", cat_imputer))
        cat_steps.append((
            "encoder",
            build_categorical_encoder(
                max_encoding_ohe=setup_cfg.max_encoding_ohe,
                encoding_method=setup_cfg.encoding_method,
            ),
        ))
        transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))

    # Ordinal branch
    if ordinal_cols and setup_cfg.ordinal_features:
        ord_steps: list[tuple[str, Any]] = []
        if setup_cfg.imputation_type == "simple":
            ord_imputer = build_categorical_imputer(setup_cfg.categorical_imputation)
            ord_steps.append(("imputer", ord_imputer))
        ord_steps.append(("encoder", build_ordinal_encoder(setup_cfg.ordinal_features)))
        transformers.append(("ordinal", Pipeline(ord_steps), ordinal_cols))

    # Date branch
    if date_cols:
        transformers.append((
            "date",
            ExtractDateTimeFeatures(setup_cfg.create_date_columns),
            date_cols,
        ))

    if transformers:
        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        steps.append(("preprocessor", column_transformer))

    # --- Stage 2: Post-encoding whole-dataset transforms ---

    # Group features
    if setup_cfg.group_features:
        steps.append((
            "group_features",
            GroupFeatures(setup_cfg.group_features, drop=setup_cfg.drop_groups),
        ))

    # Polynomial features
    if setup_cfg.polynomial_features:
        steps.append((
            "polynomial",
            PolynomialFeatures(
                degree=setup_cfg.polynomial_degree,
                interaction_only=False,
                include_bias=False,
            ),
        ))

    # Low variance removal
    if setup_cfg.low_variance_threshold is not None:
        steps.append((
            "low_variance",
            VarianceThreshold(threshold=setup_cfg.low_variance_threshold),
        ))

    # PCA / dimensionality reduction
    if setup_cfg.pca:
        steps.append(("pca", _build_pca(setup_cfg)))

    # Feature selection
    if setup_cfg.feature_selection:
        steps.append(("feature_selection", _build_feature_selector(setup_cfg, config.seed)))

    # If no steps were added, create a minimal passthrough pipeline
    if not steps:
        steps.append(("passthrough", "passthrough"))

    return Pipeline(steps)


def _build_pca(setup_cfg: SetupConfig) -> PCA | KernelPCA | IncrementalPCA:
    """Build a PCA transformer from setup config."""
    n_components = setup_cfg.pca_components
    if isinstance(n_components, str):
        # e.g. "mle"
        pass  # PCA handles string values
    elif isinstance(n_components, float) and 0 < n_components < 1:
        pass  # PCA handles float as variance ratio
    elif n_components is None:
        n_components = 0.99  # Keep 99% variance by default

    match setup_cfg.pca_method.lower():
        case "linear":
            return PCA(n_components=n_components)
        case "kernel":
            n = n_components if isinstance(n_components, int) else None
            return KernelPCA(n_components=n, kernel="rbf")
        case "incremental":
            n = n_components if isinstance(n_components, int) else None
            return IncrementalPCA(n_components=n)
        case _:
            raise ValueError(
                f"Unknown pca_method: '{setup_cfg.pca_method}'. "
                "Choose from: linear, kernel, incremental."
            )


def _build_feature_selector(
    setup_cfg: SetupConfig, seed: int
) -> SelectKBest | SequentialFeatureSelector | SelectFromModel:
    """Build a feature selector from setup config."""
    n_features = setup_cfg.n_features_to_select

    match setup_cfg.feature_selection_method.lower():
        case "classic":
            k = n_features if isinstance(n_features, int) else "all"
            return SelectKBest(score_func=f_classif, k=k)
        case "sequential":
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(max_iter=1000, random_state=seed)
            return SequentialFeatureSelector(
                estimator,
                n_features_to_select=n_features,
                direction="forward",
            )
        case _:
            raise ValueError(
                f"Unknown feature_selection_method: '{setup_cfg.feature_selection_method}'. "
                "Choose from: classic, sequential."
            )
