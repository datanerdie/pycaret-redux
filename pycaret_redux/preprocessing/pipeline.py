"""Build the preprocessing pipeline using sklearn ColumnTransformer."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (
    RFE,
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

logger = logging.getLogger(__name__)


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

    logger.info(
        "Building preprocessing pipeline: %d numeric, %d categorical, %d ordinal, %d date columns",
        len(numeric_cols),
        len(categorical_cols),
        len(ordinal_cols),
        len(date_cols),
    )

    # --- Stage 1: Column-parallel preprocessing ---
    transformers: list[tuple[str, Any, list[str]]] = []

    # Numeric branch: imputation → optional power transform → optional normalization
    if numeric_cols:
        num_steps: list[tuple[str, Any]] = []
        if setup_cfg.imputation_type == "simple":
            num_steps.append(("imputer", build_numeric_imputer(setup_cfg.numeric_imputation)))
        if setup_cfg.transformation:
            if setup_cfg.transformation_method == "auto":
                from pycaret_redux.preprocessing.skew import SkewTransformer

                num_steps.append(("auto_skew_transform", SkewTransformer(threshold=1.0)))
            else:
                num_steps.append(
                    (
                        "power_transform",
                        build_power_transformer(setup_cfg.transformation_method, config.seed),
                    )
                )
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
            cat_steps.append(
                (
                    "rare_grouper",
                    RareCategoryGrouper(
                        threshold=setup_cfg.rare_to_value,
                        rare_value=setup_cfg.rare_value,
                    ),
                )
            )
        if setup_cfg.imputation_type == "simple":
            cat_imputer = build_categorical_imputer(setup_cfg.categorical_imputation)
            cat_steps.append(("imputer", cat_imputer))
        cat_steps.append(
            (
                "encoder",
                build_categorical_encoder(
                    max_encoding_ohe=setup_cfg.max_encoding_ohe,
                    encoding_method=setup_cfg.encoding_method,
                    drop_first=setup_cfg.drop_first_ohe,
                ),
            )
        )
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
        transformers.append(
            (
                "date",
                ExtractDateTimeFeatures(setup_cfg.create_date_columns),
                date_cols,
            )
        )

    if transformers:
        logger.debug("ColumnTransformer branches: %s", [t[0] for t in transformers])
        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        steps.append(("preprocessor", column_transformer))

    # --- Stage 2: Post-encoding whole-dataset transforms ---

    # Group features
    if setup_cfg.group_features:
        steps.append(
            (
                "group_features",
                GroupFeatures(setup_cfg.group_features, drop=setup_cfg.drop_groups),
            )
        )

    # Polynomial features
    if setup_cfg.polynomial_features:
        logger.info("Adding polynomial features (degree=%d)", setup_cfg.polynomial_degree)
        steps.append(
            (
                "polynomial",
                PolynomialFeatures(
                    degree=setup_cfg.polynomial_degree,
                    interaction_only=False,
                    include_bias=False,
                ),
            )
        )

    # Low variance removal
    if setup_cfg.low_variance_threshold is not None:
        steps.append(
            (
                "low_variance",
                VarianceThreshold(threshold=setup_cfg.low_variance_threshold),
            )
        )

    # PCA / dimensionality reduction
    if setup_cfg.pca:
        logger.info(
            "Adding PCA (method=%s, components=%s)",
            setup_cfg.pca_method,
            setup_cfg.pca_components,
        )
        steps.append(("pca", _build_pca(setup_cfg)))

    # Feature selection
    if setup_cfg.feature_selection:
        logger.info(
            "Adding feature selection (method=%s, n_features=%s)",
            setup_cfg.feature_selection_method,
            setup_cfg.n_features_to_select,
        )
        steps.append(("feature_selection", _build_feature_selector(setup_cfg, config.seed)))

    # If no steps were added, create a minimal passthrough pipeline
    if not steps:
        steps.append(("passthrough", "passthrough"))

    # Enable pipeline memory caching for expensive transforms
    memory = None
    if hasattr(setup_cfg, "cache_pipeline") and setup_cfg.cache_pipeline:
        import tempfile

        memory = tempfile.mkdtemp()
        logger.info("Pipeline caching enabled at: %s", memory)

    pipeline = Pipeline(steps, memory=memory)
    # Output DataFrames (not numpy arrays) to preserve feature names
    pipeline.set_output(transform="pandas")
    logger.info(
        "Preprocessing pipeline built with %d steps: %s",
        len(steps),
        [s[0] for s in steps],
    )
    return pipeline


def _build_pca(setup_cfg: SetupConfig) -> Any:
    """Build a dimensionality reduction transformer from setup config.

    Supports PCA variants, random projections, and LDA.
    """
    n_components = setup_cfg.pca_components
    if isinstance(n_components, str):
        pass  # PCA handles string values like "mle"
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
        case "random":
            from sklearn.random_projection import GaussianRandomProjection

            n = n_components if isinstance(n_components, int) else "auto"
            return GaussianRandomProjection(n_components=n)
        case "sparse_random":
            from sklearn.random_projection import SparseRandomProjection

            n = n_components if isinstance(n_components, int) else "auto"
            return SparseRandomProjection(n_components=n)
        case "lda":
            # LDA as supervised dimensionality reduction
            n = n_components if isinstance(n_components, int) else None
            return LinearDiscriminantAnalysis(n_components=n)
        case _:
            raise ValueError(
                f"Unknown pca_method: '{setup_cfg.pca_method}'. "
                "Choose from: linear, kernel, incremental, random, "
                "sparse_random, lda."
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
        case "rfe":
            from sklearn.linear_model import LogisticRegression

            est = setup_cfg.feature_selection_estimator
            if isinstance(est, str):
                est = LogisticRegression(max_iter=1000, random_state=seed)
            n = n_features if isinstance(n_features, int) else None
            return RFE(estimator=est, n_features_to_select=n, step=1)
        case _:
            raise ValueError(
                f"Unknown feature_selection_method: "
                f"'{setup_cfg.feature_selection_method}'. "
                "Choose from: classic, sequential, rfe."
            )
