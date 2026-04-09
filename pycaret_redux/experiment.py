"""ClassificationExperiment — the single public API class."""

from __future__ import annotations

import logging
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from pycaret_redux.config import ExperimentConfig, SetupConfig
from pycaret_redux.metrics.registry import MetricEntry, MetricRegistry
from pycaret_redux.models.registry import ModelRegistry
from pycaret_redux.preprocessing.column_selection import detect_feature_types
from pycaret_redux.preprocessing.outliers import OutlierRemover
from pycaret_redux.preprocessing.pipeline import build_preprocessing_pipeline
from pycaret_redux.utils.types import DATAFRAME_LIKE, TARGET_LIKE
from pycaret_redux.utils.validation import to_dataframe, validate_setup_params, validate_target

logger = logging.getLogger(__name__)


class ClassificationExperiment:
    """Low-code classification experiment.

    Usage::

        exp = ClassificationExperiment()
        exp.setup(data=df, target="target")
        best = exp.compare_models()
        tuned = exp.tune_model(best)
        preds = exp.predict_model(tuned)
        exp.save_model(tuned, "my_model")
    """

    def __init__(self) -> None:
        self._config = ExperimentConfig()
        self._model_registry: ModelRegistry | None = None
        self._metric_registry: MetricRegistry | None = None
        self._last_pull: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_setup_done(self) -> bool:
        return self._config.is_setup_done

    @property
    def is_multiclass(self) -> bool:
        return self._config.is_multiclass

    @property
    def X_train(self) -> pd.DataFrame | None:
        return self._config.X_train

    @property
    def X_test(self) -> pd.DataFrame | None:
        return self._config.X_test

    @property
    def y_train(self) -> pd.Series | None:
        return self._config.y_train

    @property
    def y_test(self) -> pd.Series | None:
        return self._config.y_test

    @property
    def pipeline(self):
        return self._config.pipeline

    @property
    def seed(self) -> int:
        return self._config.seed

    # ------------------------------------------------------------------
    # setup()
    # ------------------------------------------------------------------

    def setup(
        self,
        data: DATAFRAME_LIKE | None = None,
        target: TARGET_LIKE = -1,
        index: bool | int | str = True,
        train_size: float = 0.7,
        test_data: DATAFRAME_LIKE | None = None,
        ordinal_features: dict[str, list] | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        date_features: list[str] | None = None,
        text_features: list[str] | None = None,
        ignore_features: list[str] | None = None,
        keep_features: list[str] | None = None,
        preprocess: bool = True,
        create_date_columns: list[str] | None = None,
        imputation_type: str | None = "simple",
        numeric_imputation: str = "mean",
        categorical_imputation: str = "mode",
        iterative_imputation_iters: int = 5,
        text_features_method: str = "tf-idf",
        max_encoding_ohe: int = 25,
        encoding_method: Any = None,
        rare_to_value: float | None = None,
        rare_value: str = "rare",
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        low_variance_threshold: float | None = None,
        group_features: dict | None = None,
        drop_groups: bool = False,
        remove_multicollinearity: bool = False,
        multicollinearity_threshold: float = 0.9,
        bin_numeric_features: list[str] | None = None,
        remove_outliers: bool = False,
        outliers_method: str = "iforest",
        outliers_threshold: float = 0.05,
        fix_imbalance: bool = False,
        fix_imbalance_method: str | Any = "SMOTE",
        transformation: bool = False,
        transformation_method: str = "yeo-johnson",
        normalize: bool = False,
        normalize_method: str = "zscore",
        pca: bool = False,
        pca_method: str = "linear",
        pca_components: int | float | str | None = None,
        feature_selection: bool = False,
        feature_selection_method: str = "classic",
        feature_selection_estimator: str | Any = "lightgbm",
        n_features_to_select: int | float = 0.2,
        custom_pipeline: Any = None,
        custom_pipeline_position: int = -1,
        data_split_shuffle: bool = True,
        data_split_stratify: bool | list[str] = True,
        fold_strategy: str | Any = "stratifiedkfold",
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: str | pd.DataFrame | None = None,
        n_jobs: int | None = -1,
        use_gpu: bool = False,
        session_id: int | None = None,
        verbose: bool = True,
        profile: bool = False,
    ) -> Self:
        """Initialize the experiment: validate data, preprocess, and split train/test.

        Ingests raw data, detects feature types, builds the preprocessing
        pipeline, removes outliers if requested, and splits into train/test sets.

        Parameters
        ----------
        data : DataFrame-like
            Training data containing features and the target column.
        target : int or str, default -1
            Target column name or positional index. Default ``-1`` (last column).
        index : bool, int, or str, default True
            How to handle the DataFrame index on ingestion.
        train_size : float, default 0.7
            Proportion of data used for training (remainder goes to test).
        test_data : DataFrame-like, optional
            Separate test set. When provided, ``train_size`` is ignored.
        ordinal_features : dict of str to list, optional
            Mapping of ordinal column names to their ordered categories.
        numeric_features : list of str, optional
            Column names to force-treat as numeric.
        categorical_features : list of str, optional
            Column names to force-treat as categorical.
        date_features : list of str, optional
            Column names to parse as dates and expand.
        text_features : list of str, optional
            Column names to vectorize as text.
        ignore_features : list of str, optional
            Column names to drop before modelling.
        keep_features : list of str, optional
            Column names to pass through without transformation.
        preprocess : bool, default True
            Whether to build and fit the preprocessing pipeline.
        create_date_columns : list of str, optional
            Date parts to extract, e.g. ``["day", "month", "year"]``.
        imputation_type : str or None, default "simple"
            ``"simple"`` for univariate or ``"iterative"`` for multivariate imputation.
        numeric_imputation : str, default "mean"
            Strategy for numeric missing values (``"mean"``, ``"median"``, ``"zero"``).
        categorical_imputation : str, default "mode"
            Strategy for categorical missing values (``"mode"``, ``"not_available"``).
        iterative_imputation_iters : int, default 5
            Rounds for iterative imputation when ``imputation_type="iterative"``.
        text_features_method : str, default "tf-idf"
            Vectorization method for text features (``"tf-idf"`` or ``"bow"``).
        max_encoding_ohe : int, default 25
            Maximum unique values for one-hot encoding; above this threshold
            ordinal/target encoding is used instead.
        encoding_method : object, optional
            Custom encoder instance from category_encoders.
        rare_to_value : float or None, optional
            Frequency threshold below which categories are grouped.
        rare_value : str, default "rare"
            Replacement label for rare categories.
        polynomial_features : bool, default False
            Whether to create polynomial interaction features.
        polynomial_degree : int, default 2
            Degree of polynomial features.
        low_variance_threshold : float or None, optional
            Remove features with variance below this threshold.
        group_features : dict, optional
            Groups of features to aggregate (e.g. mean, std).
        drop_groups : bool, default False
            Drop original features after grouping.
        remove_multicollinearity : bool, default False
            Drop features with pairwise correlation above the threshold.
        multicollinearity_threshold : float, default 0.9
            Correlation threshold for multicollinearity removal.
        bin_numeric_features : list of str, optional
            Numeric columns to discretize into bins.
        remove_outliers : bool, default False
            Remove outliers from the training set before modelling.
        outliers_method : str, default "iforest"
            Outlier detection method (``"iforest"`` or ``"ee"``).
        outliers_threshold : float, default 0.05
            Proportion of data to treat as outliers.
        fix_imbalance : bool, default False
            Apply oversampling to balance the target distribution.
        fix_imbalance_method : str or object, default "SMOTE"
            Resampling strategy or sampler instance.
        transformation : bool, default False
            Apply power transformation to numeric features.
        transformation_method : str, default "yeo-johnson"
            Power transform method (``"yeo-johnson"`` or ``"quantile"``).
        normalize : bool, default False
            Normalize (scale) numeric features.
        normalize_method : str, default "zscore"
            Scaling method (``"zscore"``, ``"minmax"``, ``"maxabs"``, ``"robust"``).
        pca : bool, default False
            Apply dimensionality reduction via PCA.
        pca_method : str, default "linear"
            PCA variant (``"linear"``, ``"kernel"``, ``"incremental"``).
        pca_components : int, float, str, or None, optional
            Number or fraction of components to keep.
        feature_selection : bool, default False
            Select a subset of features using an estimator.
        feature_selection_method : str, default "classic"
            Feature selection approach (``"classic"`` or ``"boruta"``).
        feature_selection_estimator : str or object, default "lightgbm"
            Estimator used for feature importance ranking.
        n_features_to_select : int or float, default 0.2
            Number or fraction of features to retain.
        custom_pipeline : object, optional
            A scikit-learn transformer or pipeline to inject.
        custom_pipeline_position : int, default -1
            Position in the pipeline to insert ``custom_pipeline``.
        data_split_shuffle : bool, default True
            Shuffle data before splitting.
        data_split_stratify : bool or list of str, default True
            Stratify by target during train/test split.
        fold_strategy : str or CV splitter, default "stratifiedkfold"
            Cross-validation strategy. Options include ``"stratifiedkfold"``,
            ``"kfold"``, ``"groupkfold"``, ``"timeseries"``.
        fold : int, default 10
            Number of cross-validation folds.
        fold_shuffle : bool, default False
            Shuffle within cross-validation folds.
        fold_groups : str, DataFrame, or None, optional
            Column or array defining groups for group-based CV strategies.
        n_jobs : int or None, default -1
            Number of parallel jobs (``-1`` uses all processors).
        use_gpu : bool, default False
            Use GPU-accelerated estimators where available.
        session_id : int, optional
            Random seed for reproducibility. If ``None``, a random seed is chosen.
        verbose : bool, default True
            Whether to print the setup summary table.
        profile : bool, default False
            Whether to run data profiling after setup.

        Returns
        -------
        self
            The experiment instance, enabling method chaining.

        Examples
        --------
        >>> exp = ClassificationExperiment()
        >>> exp.setup(data=df, target="species", session_id=42)
        >>> exp.is_setup_done
        True
        """
        logger.info(
            "setup() called with data shape=%s, target=%r",
            getattr(data, "shape", "?"),
            target,
        )

        if data is None:
            raise ValueError("data parameter is required.")

        validate_setup_params(train_size, fold)

        # Resolve seed
        seed = session_id if session_id is not None else np.random.randint(0, 10000)
        self._config.seed = seed
        logger.debug("Random seed set to %d", seed)

        # Convert to DataFrame
        df = to_dataframe(data)

        # Resolve target column
        target_name = validate_target(df, target)
        self._config.target_name = target_name

        # Separate X and y
        X = df.drop(columns=[target_name])
        y = df[target_name]

        # Handle ignore_features
        if ignore_features:
            X = X.drop(columns=[c for c in ignore_features if c in X.columns])

        # Detect multiclass
        self._config.is_multiclass = y.nunique() > 2
        logger.info(
            "Target '%s' has %d classes (multiclass=%s)",
            target_name,
            y.nunique(),
            self._config.is_multiclass,
        )

        # Store feature names
        self._config.feature_names_in = list(X.columns)

        # Create setup config
        if create_date_columns is None:
            create_date_columns = ["day", "month", "year"]

        setup_cfg = SetupConfig(
            train_size=train_size,
            data_split_shuffle=data_split_shuffle,
            data_split_stratify=data_split_stratify,
            fold_strategy=fold_strategy,
            fold=fold,
            fold_shuffle=fold_shuffle,
            fold_groups=fold_groups,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            ordinal_features=ordinal_features,
            date_features=date_features,
            text_features=text_features,
            ignore_features=ignore_features,
            keep_features=keep_features,
            preprocess=preprocess,
            imputation_type=imputation_type,
            numeric_imputation=numeric_imputation,
            categorical_imputation=categorical_imputation,
            iterative_imputation_iters=iterative_imputation_iters,
            text_features_method=text_features_method,
            max_encoding_ohe=max_encoding_ohe,
            encoding_method=encoding_method,
            rare_to_value=rare_to_value,
            rare_value=rare_value,
            polynomial_features=polynomial_features,
            polynomial_degree=polynomial_degree,
            low_variance_threshold=low_variance_threshold,
            group_features=group_features,
            drop_groups=drop_groups,
            remove_multicollinearity=remove_multicollinearity,
            multicollinearity_threshold=multicollinearity_threshold,
            bin_numeric_features=bin_numeric_features,
            create_date_columns=create_date_columns,
            remove_outliers=remove_outliers,
            outliers_method=outliers_method,
            outliers_threshold=outliers_threshold,
            fix_imbalance=fix_imbalance,
            fix_imbalance_method=fix_imbalance_method,
            transformation=transformation,
            transformation_method=transformation_method,
            normalize=normalize,
            normalize_method=normalize_method,
            pca=pca,
            pca_method=pca_method,
            pca_components=pca_components,
            feature_selection=feature_selection,
            feature_selection_method=feature_selection_method,
            feature_selection_estimator=feature_selection_estimator,
            n_features_to_select=n_features_to_select,
            custom_pipeline=custom_pipeline,
            custom_pipeline_position=custom_pipeline_position,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            session_id=session_id,
            verbose=verbose,
        )
        self._config.setup_config = setup_cfg

        # --- Train/test split ---
        logger.info(
            "Splitting data with train_size=%.2f, shuffle=%s, stratify=%s",
            train_size,
            data_split_shuffle,
            data_split_stratify,
        )
        stratify = y if data_split_stratify is True else None
        if test_data is not None:
            test_df = to_dataframe(test_data)
            X_test = test_df.drop(columns=[target_name])
            y_test = test_df[target_name]
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=train_size,
                random_state=seed,
                shuffle=data_split_shuffle,
                stratify=stratify,
            )

        self._config.X_train = X_train.reset_index(drop=True)
        self._config.X_test = X_test.reset_index(drop=True)
        self._config.y_train = y_train.reset_index(drop=True)
        self._config.y_test = y_test.reset_index(drop=True)

        # --- Detect feature types ---
        self._config.feature_types = detect_feature_types(X_train, setup_cfg)

        # --- Remove outliers (before pipeline, modifies train data) ---
        if remove_outliers:
            logger.info(
                "Removing outliers with method=%r, threshold=%.3f",
                outliers_method,
                outliers_threshold,
            )
            remover = OutlierRemover(
                method=outliers_method,
                threshold=outliers_threshold,
                seed=seed,
            )
            result = remover.fit_transform(self._config.X_train, self._config.y_train)
            if isinstance(result, tuple):
                self._config.X_train, self._config.y_train = result
            else:
                self._config.X_train = result

        # --- CV splitter ---
        self._config.fold_generator = self._build_fold_generator(
            fold_strategy, fold, fold_shuffle, seed
        )

        # --- Initialize registries ---
        self._model_registry = ModelRegistry(seed=seed, n_jobs=n_jobs)
        self._model_registry.register_defaults()

        self._metric_registry = MetricRegistry(is_multiclass=self._config.is_multiclass)
        self._metric_registry.register_defaults()

        # --- Preprocessing pipeline ---
        if preprocess:
            logger.info("Building and fitting preprocessing pipeline")
            pipeline = build_preprocessing_pipeline(self._config, setup_cfg)
            pipeline.fit(self._config.X_train, self._config.y_train)
            self._config.pipeline = pipeline

        self._config.is_setup_done = True

        if verbose:
            self._print_setup_summary()

        if profile:
            from pycaret_redux.utils.profiling import profile_data

            profile_data(self._config.X_train, self._config.y_train, self._config.feature_types)

        logger.info("Setup complete. Data shape: train=%s, test=%s", X_train.shape, X_test.shape)
        return self

    # ------------------------------------------------------------------
    # Placeholder methods (to be implemented in later phases)
    # ------------------------------------------------------------------

    def compare_models(
        self,
        include: list[str | Any] | None = None,
        exclude: list[str] | None = None,
        fold: int | Any | None = None,
        round: int = 4,
        cross_validation: bool = True,
        sort: str = "Accuracy",
        n_select: int = 1,
        budget_time: float | None = None,
        turbo: bool = True,
        errors: str = "ignore",
        fit_kwargs: dict | None = None,
        verbose: bool = True,
    ) -> Any:
        """Compare all available models using cross-validation.

        Trains every registered model (or a subset) with cross-validation
        and ranks them by the chosen metric.

        Parameters
        ----------
        include : list of str or estimator, optional
            Model IDs or estimator instances to include. If ``None``, all
            registered models are compared.
        exclude : list of str, optional
            Model IDs to exclude from comparison.
        fold : int or CV splitter, optional
            Override the fold configuration set during ``setup()``.
        round : int, default 4
            Number of decimal places for metric scores.
        cross_validation : bool, default True
            Whether to use cross-validation. If ``False``, trains once and
            evaluates on the hold-out test set.
        sort : str, default "Accuracy"
            Metric name to sort the leaderboard by.
        n_select : int, default 1
            Number of top models to return. When greater than 1 a list is returned.
        budget_time : float, optional
            Time budget in minutes. Comparison stops after this limit.
        turbo : bool, default True
            When ``True``, only fast/lightweight models are included.
        errors : str, default "ignore"
            How to handle model-fitting errors (``"ignore"`` or ``"raise"``).
        fit_kwargs : dict, optional
            Extra keyword arguments forwarded to each model's ``fit()`` call.
        verbose : bool, default True
            Print the comparison leaderboard.

        Returns
        -------
        object or list of object
            Best fitted model, or a list of top models when ``n_select > 1``.

        Examples
        --------
        >>> best = exp.compare_models(sort="F1", n_select=3)
        >>> exp.pull()  # leaderboard DataFrame
        """
        self._check_setup()
        logger.info(
            "compare_models() called with sort=%r, n_select=%d, turbo=%s",
            sort,
            n_select,
            turbo,
        )
        from pycaret_redux.training.comparison import compare_models as _compare

        result, self._comparison_df = _compare(
            config=self._config,
            model_registry=self._model_registry,
            metric_registry=self._metric_registry,
            include=include,
            exclude=exclude,
            fold=fold,
            round_to=round,
            cross_validation=cross_validation,
            sort=sort,
            n_select=n_select,
            budget_time=budget_time,
            turbo=turbo,
            errors=errors,
            fit_kwargs=fit_kwargs,
            verbose=verbose,
        )
        self._last_pull = self._comparison_df
        return result

    def create_model(
        self,
        estimator: str | Any,
        fold: int | Any | None = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: dict | None = None,
        return_train_score: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> Any:
        """Train a single model with cross-validation.

        Fits the specified estimator and evaluates it using k-fold
        cross-validation. The per-fold scores are stored and can be
        retrieved via ``pull()``.

        Parameters
        ----------
        estimator : str or estimator
            Model ID (e.g. ``"lr"``, ``"rf"``) or an sklearn-compatible
            estimator instance.
        fold : int or CV splitter, optional
            Override the fold configuration set during ``setup()``.
        round : int, default 4
            Number of decimal places for metric scores.
        cross_validation : bool, default True
            Whether to evaluate with cross-validation.
        fit_kwargs : dict, optional
            Extra keyword arguments forwarded to the model's ``fit()`` call.
        return_train_score : bool, default False
            Whether to include training scores in the output.
        verbose : bool, default True
            Print the cross-validation results table.
        **kwargs
            Additional keyword arguments passed to the estimator constructor.

        Returns
        -------
        object
            The fitted model estimator.

        Examples
        --------
        >>> model = exp.create_model("lr")
        >>> scores = exp.pull()
        """
        self._check_setup()
        logger.info(
            "create_model() called with estimator=%r, cross_validation=%s",
            estimator,
            cross_validation,
        )
        from pycaret_redux.training.creation import create_model as _create

        model, fold_scores, mean_scores, fit_time = _create(
            estimator=estimator,
            config=self._config,
            model_registry=self._model_registry,
            metric_registry=self._metric_registry,
            fold=fold,
            round_to=round,
            cross_validation=cross_validation,
            fit_kwargs=fit_kwargs,
            return_train_score=return_train_score,
            verbose=verbose,
            **kwargs,
        )
        if fold_scores is not None:
            self._last_pull = fold_scores
        return model

    def tune_model(
        self,
        estimator: Any,
        fold: int | Any | None = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: dict[str, list] | None = None,
        optimize: str = "Accuracy",
        choose_better: bool = True,
        search_library: str = "sklearn",
        verbose: bool = True,
        **kwargs,
    ) -> Any:
        """Tune model hyperparameters via random search.

        Searches over a predefined or custom hyperparameter grid and
        returns the best-performing configuration.

        Parameters
        ----------
        estimator : object
            A fitted model returned by ``create_model()`` or ``compare_models()``.
        fold : int or CV splitter, optional
            Override the fold configuration set during ``setup()``.
        round : int, default 4
            Number of decimal places for metric scores.
        n_iter : int, default 10
            Number of random search iterations.
        custom_grid : dict of str to list, optional
            Custom hyperparameter grid. Keys are parameter names, values
            are lists of candidate values.
        optimize : str, default "Accuracy"
            Metric to optimize during the search.
        choose_better : bool, default True
            If ``True``, only return the tuned model when it improves on
            the original; otherwise return the original.
        search_library : str, default "sklearn"
            Search backend to use (``"sklearn"``).
        verbose : bool, default True
            Print tuning results.
        **kwargs
            Additional keyword arguments forwarded to the search algorithm.

        Returns
        -------
        object
            The tuned model estimator.

        Examples
        --------
        >>> tuned_lr = exp.tune_model(model, n_iter=50, optimize="AUC")
        """
        self._check_setup()
        logger.info(
            "tune_model() called with optimize=%r, n_iter=%d, search_library=%r",
            optimize,
            n_iter,
            search_library,
        )
        from pycaret_redux.training.tuning import tune_model as _tune

        tuned, _ = _tune(
            estimator=estimator,
            config=self._config,
            model_registry=self._model_registry,
            metric_registry=self._metric_registry,
            fold=fold,
            round_to=round,
            n_iter=n_iter,
            custom_grid=custom_grid,
            optimize=optimize,
            choose_better=choose_better,
            search_library=search_library,
            verbose=verbose,
            **kwargs,
        )
        return tuned

    def blend_models(
        self,
        estimator_list: list[Any],
        fold: int | Any | None = None,
        round: int = 4,
        method: str = "auto",
        weights: list[float] | None = None,
        optimize: str = "Accuracy",
        choose_better: bool = False,
        verbose: bool = True,
    ) -> Any:
        """Create a voting ensemble from multiple models.

        Combines several fitted models into a ``VotingClassifier`` that
        aggregates their predictions using soft or hard voting.

        Parameters
        ----------
        estimator_list : list of object
            Fitted model estimators to blend.
        fold : int or CV splitter, optional
            Override the fold configuration set during ``setup()``.
        round : int, default 4
            Number of decimal places for metric scores.
        method : str, default "auto"
            Voting method: ``"soft"``, ``"hard"``, or ``"auto"`` (soft when
            all estimators support ``predict_proba``, hard otherwise).
        weights : list of float, optional
            Voting weights for each estimator. Equal weights when ``None``.
        optimize : str, default "Accuracy"
            Metric used to evaluate the blended model.
        choose_better : bool, default False
            If ``True``, return the blend only when it outperforms the best
            individual model; otherwise return the best individual.
        verbose : bool, default True
            Print evaluation results.

        Returns
        -------
        object
            The fitted ``VotingClassifier``.

        Examples
        --------
        >>> blended = exp.blend_models([model1, model2, model3])
        """
        self._check_setup()
        logger.info(
            "blend_models() called with %d estimators, method=%r",
            len(estimator_list),
            method,
        )
        from pycaret_redux.training.ensembles import blend_models as _blend

        return _blend(
            estimator_list=estimator_list,
            config=self._config,
            metric_registry=self._metric_registry,
            fold=fold,
            round_to=round,
            method=method,
            weights=weights,
            optimize=optimize,
            choose_better=choose_better,
            verbose=verbose,
        )

    def stack_models(
        self,
        estimator_list: list[Any],
        meta_model: Any | None = None,
        fold: int | Any | None = None,
        round: int = 4,
        method: str = "auto",
        restack: bool = False,
        optimize: str = "Accuracy",
        choose_better: bool = False,
        verbose: bool = True,
    ) -> Any:
        """Create a stacking ensemble from multiple models.

        Trains a meta-learner on top of the base models' out-of-fold
        predictions using a ``StackingClassifier``.

        Parameters
        ----------
        estimator_list : list of object
            Fitted base-layer model estimators.
        meta_model : object or None, optional
            Meta-learner estimator. Defaults to ``LogisticRegression`` when
            ``None``.
        fold : int or CV splitter, optional
            Override the fold configuration set during ``setup()``.
        round : int, default 4
            Number of decimal places for metric scores.
        method : str, default "auto"
            Method for generating base-layer predictions (``"auto"``,
            ``"predict_proba"``, ``"predict"``).
        restack : bool, default False
            When ``True``, include original features alongside base-layer
            predictions in the meta-learner input.
        optimize : str, default "Accuracy"
            Metric used to evaluate the stacked model.
        choose_better : bool, default False
            If ``True``, return the stack only when it outperforms the best
            individual model.
        verbose : bool, default True
            Print evaluation results.

        Returns
        -------
        object
            The fitted ``StackingClassifier``.

        Examples
        --------
        >>> stacked = exp.stack_models([lr, rf, xgb])
        """
        self._check_setup()
        logger.info("stack_models() called with %d estimators", len(estimator_list))
        from pycaret_redux.training.ensembles import stack_models as _stack

        return _stack(
            estimator_list=estimator_list,
            config=self._config,
            metric_registry=self._metric_registry,
            meta_model=meta_model,
            fold=fold,
            round_to=round,
            method=method,
            restack=restack,
            optimize=optimize,
            choose_better=choose_better,
            verbose=verbose,
        )

    def ensemble_model(
        self,
        estimator: Any,
        method: str = "bagging",
        fold: int | Any | None = None,
        round: int = 4,
        n_estimators: int = 10,
        optimize: str = "Accuracy",
        choose_better: bool = False,
        verbose: bool = True,
    ) -> Any:
        """Create a bagging or boosting ensemble from a single model.

        Wraps the estimator in a ``BaggingClassifier`` (or boosting
        variant) that trains multiple copies on bootstrapped samples.

        Parameters
        ----------
        estimator : object
            Fitted model to ensemble.
        method : str, default "bagging"
            Ensemble method (``"bagging"`` or ``"boosting"``).
        fold : int or CV splitter, optional
            Override the fold configuration set during ``setup()``.
        round : int, default 4
            Number of decimal places for metric scores.
        n_estimators : int, default 10
            Number of base estimators in the ensemble.
        optimize : str, default "Accuracy"
            Metric used to evaluate the ensemble.
        choose_better : bool, default False
            If ``True``, return the ensemble only when it outperforms the
            original model.
        verbose : bool, default True
            Print evaluation results.

        Returns
        -------
        object
            The fitted ensemble estimator.

        Examples
        --------
        >>> bagged = exp.ensemble_model(model, n_estimators=20)
        """
        self._check_setup()
        logger.info(
            "ensemble_model() called with method=%r, n_estimators=%d",
            method,
            n_estimators,
        )
        from pycaret_redux.training.ensembles import ensemble_model as _ensemble

        return _ensemble(
            estimator=estimator,
            config=self._config,
            metric_registry=self._metric_registry,
            method=method,
            fold=fold,
            round_to=round,
            n_estimators=n_estimators,
            optimize=optimize,
            choose_better=choose_better,
            verbose=verbose,
        )

    def plot_model(
        self,
        estimator: Any,
        plot: str = "auc",
        save: str | None = None,
        **kwargs,
    ) -> Any:
        """Generate a model diagnostic plot.

        Renders a visual evaluation of the model using the hold-out test
        set. The plot is displayed inline and optionally saved to disk.

        Parameters
        ----------
        estimator : object
            Fitted model to visualize.
        plot : str, default "auc"
            Plot type. Supported values: ``"auc"``, ``"confusion_matrix"``,
            ``"threshold"``, ``"pr"``, ``"error"``, ``"class_report"``,
            ``"feature"``, ``"learning"``, ``"vc"``, ``"calibration"``,
            ``"lift"``, ``"gain"``, ``"ks"``.
        save : str or None, optional
            File path to save the figure. If ``None``, the plot is only
            displayed.
        **kwargs
            Additional keyword arguments forwarded to the plot renderer.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered figure object.

        Examples
        --------
        >>> exp.plot_model(model, plot="confusion_matrix")
        >>> exp.plot_model(model, plot="auc", save="auc_plot.png")
        """
        self._check_setup()
        from pycaret_redux.plots.registry import build_default_registry

        registry = build_default_registry()

        # Prepare transformed test data for the estimator
        X_test = self._config.X_test
        y_test = self._config.y_test
        if self._config.pipeline is not None:
            X_test = self._config.pipeline.transform(X_test)

        return registry.render(
            plot_id=plot,
            estimator=estimator,
            X=X_test,
            y=y_test,
            is_multiclass=self._config.is_multiclass,
            save=save,
            **kwargs,
        )

    def evaluate_model(self, estimator: Any, **kwargs) -> None:
        """Print an evaluation summary for a model on the test set.

        Computes all registered metrics against the hold-out test data
        and displays a formatted table of results.

        Parameters
        ----------
        estimator : object
            Fitted model to evaluate.
        **kwargs
            Reserved for future use.

        Returns
        -------
        None
            Results are printed to stdout.

        Examples
        --------
        >>> exp.evaluate_model(model)
        """
        self._check_setup()
        from pycaret_redux.metrics.scoring import calculate_metrics
        from pycaret_redux.utils.display import display_evaluation

        X_test = self._config.X_test
        y_test = self._config.y_test
        if self._config.pipeline is not None:
            X_test = self._config.pipeline.transform(X_test)

        y_pred = estimator.predict(X_test)
        y_proba = None
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_test)

        scores = calculate_metrics(y_test, y_pred, y_proba, self._metric_registry)
        metric_names = {
            mid: e.display_name for mid, e in self._metric_registry.get_active().items()
        }
        display_evaluation(scores, metric_names)

    def interpret_model(self, estimator: Any, plot: str = "summary", **kwargs) -> Any:
        """Generate SHAP-based model interpretation plots.

        Uses SHAP (SHapley Additive exPlanations) to explain feature
        contributions to the model's predictions on the test set.

        Parameters
        ----------
        estimator : object
            Fitted model to interpret. Must be compatible with SHAP.
        plot : str, default "summary"
            Plot type: ``"summary"`` for a beeswarm summary or ``"bar"``
            for mean absolute SHAP value bar chart.
        **kwargs
            Additional keyword arguments forwarded to the SHAP explainer.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered SHAP figure.

        Raises
        ------
        ImportError
            If the ``shap`` package is not installed.

        Examples
        --------
        >>> fig = exp.interpret_model(model, plot="summary")
        """
        self._check_setup()
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is required for interpret_model. Install with: uv add shap")

        X_test = self._config.X_test
        if self._config.pipeline is not None:
            X_test = self._config.pipeline.transform(X_test)

        explainer = shap.Explainer(estimator, X_test)
        shap_values = explainer(X_test)

        if plot == "summary":
            shap.summary_plot(shap_values, X_test, show=False)
        elif plot == "bar":
            shap.plots.bar(shap_values, show=False)
        else:
            raise ValueError(f"Unknown interpret plot: '{plot}'. Use 'summary' or 'bar'.")

        import matplotlib.pyplot as plt

        return plt.gcf()

    def calibrate_model(
        self,
        estimator: Any,
        method: str = "sigmoid",
        fold: int | Any | None = None,
        verbose: bool = True,
    ) -> Any:
        """Calibrate predicted probabilities using Platt scaling or isotonic regression.

        Wraps the estimator in a ``CalibratedClassifierCV`` so that
        ``predict_proba`` outputs well-calibrated probabilities.

        Parameters
        ----------
        estimator : object
            Fitted model whose probabilities should be calibrated.
        method : str, default "sigmoid"
            Calibration method: ``"sigmoid"`` (Platt scaling) or
            ``"isotonic"`` (isotonic regression).
        fold : int or CV splitter, optional
            Cross-validation strategy used during calibration. If ``None``,
            uses the fold generator from ``setup()``.
        verbose : bool, default True
            Print a confirmation message after calibration.

        Returns
        -------
        CalibratedClassifierCV
            The calibrated model wrapper.

        Examples
        --------
        >>> calibrated = exp.calibrate_model(model, method="isotonic")
        """
        self._check_setup()
        from sklearn.base import clone
        from sklearn.calibration import CalibratedClassifierCV

        cv = fold if fold is not None else self._config.fold_generator
        calibrated = CalibratedClassifierCV(
            estimator=clone(estimator),
            method=method,
            cv=cv,
        )

        X_train = self._config.X_train
        y_train = self._config.y_train
        if self._config.pipeline is not None:
            X_train = self._config.pipeline.transform(X_train)

        calibrated.fit(X_train, y_train)

        if verbose:
            print(f"Calibrated model created (method={method}).")
        return calibrated

    def optimize_threshold(
        self,
        estimator: Any,
        optimize: str = "F1",
        verbose: bool = True,
    ) -> tuple[Any, float]:
        """Find the optimal decision threshold for binary classification.

        Sweeps thresholds from 0.01 to 0.99 and selects the one that
        maximizes the chosen metric on the hold-out test set.

        Parameters
        ----------
        estimator : object
            Fitted model with a ``predict_proba`` method.
        optimize : str, default "F1"
            Metric to maximize. Supported: ``"F1"``, ``"Accuracy"``,
            ``"Precision"``, ``"Recall"``.
        verbose : bool, default True
            Print the optimal threshold and its score.

        Returns
        -------
        tuple of (object, float)
            The original estimator and the optimal probability threshold.

        Raises
        ------
        ValueError
            If the experiment is multiclass or the estimator lacks
            ``predict_proba``.

        Examples
        --------
        >>> model, threshold = exp.optimize_threshold(model, optimize="F1")
        >>> print(threshold)
        0.42
        """
        self._check_setup()
        if self._config.is_multiclass:
            raise ValueError("optimize_threshold is only for binary classification.")

        if not hasattr(estimator, "predict_proba"):
            raise ValueError("Estimator must support predict_proba.")

        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metric_funcs = {
            "f1": f1_score,
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
        }
        opt_lower = optimize.lower()
        if opt_lower not in metric_funcs:
            raise ValueError(f"optimize must be one of: {list(metric_funcs.keys())}")

        score_func = metric_funcs[opt_lower]

        X_test = self._config.X_test
        y_test = self._config.y_test
        if self._config.pipeline is not None:
            X_test = self._config.pipeline.transform(X_test)

        probas = estimator.predict_proba(X_test)[:, 1]

        best_threshold = 0.5
        best_score = 0.0
        for threshold in np.arange(0.01, 1.0, 0.01):
            preds = (probas >= threshold).astype(int)
            if opt_lower != "accuracy":
                score = score_func(y_test, preds, zero_division=0)
            else:
                score = score_func(y_test, preds)
            if score > best_score:
                best_score = score
                best_threshold = threshold

        if verbose:
            print(f"Optimal threshold: {best_threshold:.2f} ({optimize}={best_score:.4f})")

        return estimator, round(best_threshold, 2)

    def predict_model(
        self,
        estimator: Any,
        data: DATAFRAME_LIKE | None = None,
        probability_threshold: float | None = None,
        raw_score: bool = False,
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Generate predictions on new or hold-out test data.

        When ``data`` is ``None``, predictions are made on the test set
        created during ``setup()``. Preprocessing is applied automatically.

        Parameters
        ----------
        estimator : object
            Fitted model to generate predictions with.
        data : DataFrame-like, optional
            New data to predict on. If ``None``, the hold-out test set is used.
        probability_threshold : float or None, optional
            Custom decision threshold for binary classification. Overrides
            the default 0.5.
        raw_score : bool, default False
            When ``True``, include raw predicted probabilities in the output.
        round : int, default 4
            Number of decimal places for probability scores.
        verbose : bool, default True
            Print prediction summary.

        Returns
        -------
        pd.DataFrame
            DataFrame with original features plus ``prediction_label``
            and (optionally) ``prediction_score`` columns.

        Examples
        --------
        >>> preds = exp.predict_model(model)
        >>> preds = exp.predict_model(model, data=new_df, raw_score=True)
        """
        self._check_setup()
        logger.info(
            "predict_model() called with data=%s",
            "custom" if data is not None else "test_set",
        )
        from pycaret_redux.training.finalization import predict_model as _predict

        return _predict(
            estimator=estimator,
            config=self._config,
            data=data,
            probability_threshold=probability_threshold,
            raw_score=raw_score,
            round_to=round,
        )

    def finalize_model(self, estimator: Any) -> Any:
        """Retrain the model on the full dataset (train + test combined).

        Call this before deployment to use all available data for the
        final fit. The returned model should not be evaluated further
        since no hold-out set remains.

        Parameters
        ----------
        estimator : object
            Fitted model to retrain on the complete dataset.

        Returns
        -------
        object
            The re-fitted model estimator.

        Examples
        --------
        >>> final_model = exp.finalize_model(tuned_model)
        >>> exp.save_model(final_model, "production_model")
        """
        self._check_setup()
        logger.info("finalize_model() called with estimator=%s", type(estimator).__name__)
        from pycaret_redux.training.finalization import finalize_model as _finalize

        return _finalize(estimator=estimator, config=self._config)

    def save_model(self, estimator: Any, model_name: str, verbose: bool = True) -> None:
        """Save the model and preprocessing pipeline to disk.

        The saved artifact bundles the estimator and the preprocessing
        pipeline together, so a single file is all you need for deployment.

        Parameters
        ----------
        estimator : object
            Fitted model to persist.
        model_name : str
            File path (without extension) for the saved artifact.
        verbose : bool, default True
            Print a confirmation message after saving.

        Returns
        -------
        None

        Examples
        --------
        >>> exp.save_model(final_model, "my_classifier")
        """
        from pycaret_redux.persistence.serialization import save_model as _save

        _save(
            estimator,
            model_name,
            pipeline=self._config.pipeline,
            target_name=self._config.target_name,
            feature_names_in=self._config.feature_names_in,
            is_multiclass=self._config.is_multiclass,
            verbose=verbose,
        )

    def load_model(self, model_name: str, verbose: bool = True) -> Any:
        """Load a previously saved model from disk.

        Returns the fitted estimator. The preprocessing pipeline is also
        restored internally via the saved ``ModelArtifact``.

        Parameters
        ----------
        model_name : str
            File path (without extension) of the saved artifact.
        verbose : bool, default True
            Print a confirmation message after loading.

        Returns
        -------
        object
            The fitted model estimator.

        Examples
        --------
        >>> loaded = exp.load_model("my_classifier")
        >>> preds = exp.predict_model(loaded, data=new_df)
        """
        from pycaret_redux.persistence.serialization import load_model as _load

        artifact = _load(model_name, verbose=verbose)
        return artifact.estimator

    def models(self, turbo_only: bool = False, **kwargs) -> pd.DataFrame:
        """List all registered classification models.

        Returns a DataFrame summarizing every model in the registry,
        including its ID, name, and whether it is turbo-eligible.

        Parameters
        ----------
        turbo_only : bool, default False
            When ``True``, only show fast/lightweight models.
        **kwargs
            Reserved for future use.

        Returns
        -------
        pd.DataFrame
            Table of available models and their metadata.

        Examples
        --------
        >>> exp.models(turbo_only=True)
        """
        if self._model_registry is None:
            raise RuntimeError("Call setup() first to initialize the model registry.")
        return self._model_registry.list_models(turbo_only=turbo_only)

    def get_metrics(self, **kwargs) -> pd.DataFrame:
        """List all registered evaluation metrics.

        Returns a DataFrame with metric IDs, display names, and whether
        each metric requires predicted probabilities.

        Parameters
        ----------
        **kwargs
            Reserved for future use.

        Returns
        -------
        pd.DataFrame
            Table of available metrics and their properties.

        Examples
        --------
        >>> exp.get_metrics()
        """
        if self._metric_registry is None:
            raise RuntimeError("Call setup() first to initialize the metric registry.")
        return self._metric_registry.to_dataframe()

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: Any,
        greater_is_better: bool = True,
        needs_proba: bool = False,
        supports_multiclass: bool = True,
        display_name: str | None = None,
        **kwargs,
    ) -> None:
        """Register a custom evaluation metric.

        The metric becomes available in ``compare_models()``,
        ``create_model()``, and other training methods.

        Parameters
        ----------
        id : str
            Unique identifier for the metric (e.g. ``"mcc"``).
        name : str
            Full name of the metric (e.g. ``"Matthews Corr. Coef."``).
        score_func : callable
            Scoring function with signature ``(y_true, y_pred)`` or
            ``(y_true, y_proba)`` when ``needs_proba=True``.
        greater_is_better : bool, default True
            Whether a higher value indicates better performance.
        needs_proba : bool, default False
            Whether the metric requires predicted probabilities.
        supports_multiclass : bool, default True
            Whether the metric supports multiclass targets.
        display_name : str or None, optional
            Short label for display tables. Defaults to ``name``.
        **kwargs
            Reserved for future use.

        Returns
        -------
        None

        Examples
        --------
        >>> from sklearn.metrics import matthews_corrcoef
        >>> exp.add_metric("mcc", "MCC", matthews_corrcoef)
        """
        if self._metric_registry is None:
            raise RuntimeError("Call setup() first.")
        entry = MetricEntry(
            id=id,
            name=name,
            display_name=display_name or name,
            score_func=score_func,
            greater_is_better=greater_is_better,
            needs_proba=needs_proba,
            supports_multiclass=supports_multiclass,
            is_custom=True,
        )
        self._metric_registry.register(entry)

    def remove_metric(self, name_or_id: str) -> None:
        """Remove a metric from the registry.

        The metric will no longer appear in training and evaluation
        results after removal.

        Parameters
        ----------
        name_or_id : str
            The metric ID or display name to remove.

        Returns
        -------
        None

        Examples
        --------
        >>> exp.remove_metric("mcc")
        """
        if self._metric_registry is None:
            raise RuntimeError("Call setup() first.")
        self._metric_registry.remove(name_or_id)

    # ------------------------------------------------------------------
    # OOB evaluation
    # ------------------------------------------------------------------

    def get_oob_score(self, estimator: Any) -> float | None:
        """Get out-of-bag score for bagging/forest estimators.

        OOB evaluation uses the ~37% of samples not seen by each tree,
        providing a free validation estimate without a separate holdout set.

        Parameters
        ----------
        estimator : estimator
            A fitted estimator that supports ``oob_score`` (e.g. RandomForest,
            BaggingClassifier, ExtraTreesClassifier).

        Returns
        -------
        float or None
            OOB accuracy, or None if the estimator doesn't support it.

        Examples
        --------
        >>> rf = exp.create_model("rf")
        >>> exp.get_oob_score(rf)
        0.9523
        """
        self._check_setup()
        if not hasattr(estimator, "oob_score_"):
            # Try refitting with oob_score=True
            from sklearn.base import clone

            est = clone(estimator)
            if hasattr(est, "oob_score"):
                est.set_params(oob_score=True)
                X_train = self._config.X_train
                y_train = self._config.y_train
                if self._config.pipeline is not None:
                    X_train = self._config.pipeline.transform(X_train)
                est.fit(X_train, y_train)
                return round(float(est.oob_score_), 4)
            return None
        return round(float(estimator.oob_score_), 4)

    # ------------------------------------------------------------------
    # Bias-variance diagnostic
    # ------------------------------------------------------------------

    def diagnose_bias_variance(
        self,
        estimator: Any,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Diagnose whether a model suffers from high bias or high variance.

        Analyzes learning curves to determine if the model would benefit
        from more data (high variance) or needs more capacity (high bias).

        Parameters
        ----------
        estimator : estimator
            Fitted model to diagnose.
        verbose : bool
            Print diagnosis.

        Returns
        -------
        dict with keys: train_score, val_score, gap, diagnosis, suggestion.

        Examples
        --------
        >>> diag = exp.diagnose_bias_variance(rf)
        >>> diag["diagnosis"]
        'High variance (overfitting)'
        """
        self._check_setup()
        import numpy as np
        from sklearn.model_selection import learning_curve

        X_train = self._config.X_train
        y_train = self._config.y_train
        if self._config.pipeline is not None:
            X_train = self._config.pipeline.transform(X_train)

        train_sizes, train_scores, val_scores = learning_curve(
            estimator,
            X_train,
            y_train,
            cv=self._config.fold_generator,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring="accuracy",
        )

        final_train = float(np.mean(train_scores[-1]))
        final_val = float(np.mean(val_scores[-1]))
        gap = final_train - final_val

        # Diagnose
        if final_val < 0.7 and gap < 0.05:
            diagnosis = "High bias (underfitting)"
            suggestion = "Try a more complex model, add features, or reduce regularization."
        elif gap > 0.1:
            diagnosis = "High variance (overfitting)"
            suggestion = "Try regularization, feature selection, more data, or a simpler model."
        elif final_val < 0.85 and gap > 0.05:
            diagnosis = "Moderate bias-variance trade-off"
            suggestion = "Consider tuning hyperparameters or ensembling."
        else:
            diagnosis = "Good fit"
            suggestion = "Model appears well-balanced."

        # Check if more data would help (is val score still rising?)
        if len(val_scores) >= 3:
            val_trend = np.mean(val_scores[-1]) - np.mean(val_scores[-3])
            more_data_helps = val_trend > 0.01
        else:
            more_data_helps = False

        result = {
            "train_score": round(final_train, 4),
            "val_score": round(final_val, 4),
            "gap": round(gap, 4),
            "diagnosis": diagnosis,
            "suggestion": suggestion,
            "more_data_helps": more_data_helps,
        }

        if verbose:
            print(f"\nBias-Variance Diagnostic for {type(estimator).__name__}")
            print(f"  Train score: {result['train_score']}")
            print(f"  Val score:   {result['val_score']}")
            print(f"  Gap:         {result['gap']}")
            print(f"  Diagnosis:   {result['diagnosis']}")
            print(f"  Suggestion:  {result['suggestion']}")
            if more_data_helps:
                print("  More data would likely improve performance.")

        return result

    # ------------------------------------------------------------------
    # Nested cross-validation
    # ------------------------------------------------------------------

    def nested_cv(
        self,
        estimator: str | Any,
        param_grid: dict[str, list] | None = None,
        outer_fold: int | Any | None = None,
        inner_fold: int = 5,
        n_iter: int = 10,
        optimize: str = "acc",
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run nested cross-validation for unbiased model evaluation.

        Uses an inner CV loop for hyperparameter tuning and an outer CV
        loop for performance estimation. This prevents optimistic bias
        from using the same data for both tuning and evaluation.

        Based on: Raschka (2018), arXiv:1811.12808.

        Parameters
        ----------
        estimator : str or estimator
            Model ID or sklearn-compatible estimator.
        param_grid : dict, optional
            Hyperparameter search space. If None, uses the model's default
            tuning grid from the registry.
        outer_fold : int or CV splitter, optional
            Outer CV splitter. Defaults to experiment fold_generator.
        inner_fold : int
            Number of inner CV folds for tuning.
        n_iter : int
            Number of random search iterations in inner loop.
        optimize : str
            Metric ID to optimize in inner loop.
        round : int
            Decimal places.
        verbose : bool
            Print results.

        Returns
        -------
        pd.DataFrame
            Outer fold scores with Mean, SD, and 95% CI.

        Examples
        --------
        >>> scores = exp.nested_cv("rf", n_iter=20)
        >>> scores = exp.nested_cv("lr", param_grid={"C": [0.01, 0.1, 1, 10]})
        """
        self._check_setup()
        from pycaret_redux.models.factory import create_estimator
        from pycaret_redux.training.cross_validation import run_nested_cross_validation

        model = create_estimator(estimator, self._model_registry)

        # Resolve param_grid from registry if not provided
        if param_grid is None and isinstance(estimator, str):
            entry = self._model_registry.get(estimator)
            param_grid = entry.tuning.grid if entry.tuning.grid else None

        outer = outer_fold if outer_fold is not None else self._config.fold_generator

        fold_scores, mean_scores = run_nested_cross_validation(
            estimator=model,
            config=self._config,
            metric_registry=self._metric_registry,
            param_grid=param_grid,
            outer_cv=outer,
            inner_cv=inner_fold,
            n_iter=n_iter,
            optimize=optimize,
            round_to=round,
        )

        self._last_pull = fold_scores

        if verbose:
            from pycaret_redux.utils.display import display_fold_scores

            display_fold_scores(fold_scores, f"Nested CV — {type(model).__name__}")

        return fold_scores

    # ------------------------------------------------------------------
    # Data drift
    # ------------------------------------------------------------------

    def check_drift(
        self,
        data: pd.DataFrame,
        numeric_test: str = "ks",
        categorical_test: str = "chi2",
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Check for data drift between training data and new data.

        Compares feature distributions using statistical tests (KS test
        for numeric, Chi-squared for categorical) and flags drifted features.

        Parameters
        ----------
        data : pd.DataFrame
            New data to check against the training set.
        numeric_test : str
            "ks" (Kolmogorov-Smirnov) or "psi" (Population Stability Index).
        categorical_test : str
            "chi2" (Chi-squared test).
        alpha : float
            Significance level.

        Returns
        -------
        pd.DataFrame
            One row per feature with columns for the test statistic,
            p-value, and a boolean ``drifted`` flag.

        Examples
        --------
        >>> drift_report = exp.check_drift(new_data)
        >>> drift_report[drift_report["drifted"]]
        """
        self._check_setup()
        from pycaret_redux.utils.drift import check_drift as _check_drift

        return _check_drift(
            reference=self._config.X_train,
            current=data,
            numeric_test=numeric_test,
            categorical_test=categorical_test,
            alpha=alpha,
            numeric_cols=self._config.feature_types.get("Numeric"),
            categorical_cols=self._config.feature_types.get("Categorical"),
        )

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    def compare_model_stats(
        self,
        model_a: Any,
        model_b: Any,
        metric: str = "Accuracy",
        test: str = "wilcoxon",
        fold: int | Any | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Statistical test: is model A significantly different from model B?

        Runs cross-validation on both models and compares their per-fold
        scores using a paired statistical test.

        Parameters
        ----------
        model_a : estimator
            First model.
        model_b : estimator
            Second model.
        metric : str
            Metric to compare on. Default "Accuracy".
        test : str
            "wilcoxon" (non-parametric), "ttest" (paired t-test), or
            "mcnemar" (McNemar's test on prediction disagreements).
        fold : int or CV splitter, optional
            Override fold configuration. Not used for McNemar's test.
        verbose : bool
            Print the result.

        Returns
        -------
        dict
            Dictionary with keys ``test``, ``model_a``, ``model_b``,
            ``p_value``, ``significant``, and ``conclusion``.

        Examples
        --------
        >>> result = exp.compare_model_stats(lr, rf, test="mcnemar")
        >>> result = exp.compare_model_stats(lr, rf, metric="AUC", test="wilcoxon")
        """
        self._check_setup()

        if test == "mcnemar":
            from pycaret_redux.training.stats import mcnemar_test

            X_test = self._config.X_test
            y_test = self._config.y_test
            if self._config.pipeline is not None:
                X_test = self._config.pipeline.transform(X_test)

            preds_a = model_a.predict(X_test)
            preds_b = model_b.predict(X_test)

            result = mcnemar_test(
                y_true=y_test.values,
                preds_a=preds_a,
                preds_b=preds_b,
                model_a_name=type(model_a).__name__,
                model_b_name=type(model_b).__name__,
            )
        else:
            from pycaret_redux.training.cross_validation import run_cross_validation
            from pycaret_redux.training.stats import compare_model_stats as _compare_stats

            _, scores_a, means_a, _ = run_cross_validation(
                estimator=model_a,
                config=self._config,
                metric_registry=self._metric_registry,
                fold=fold,
            )
            _, scores_b, means_b, _ = run_cross_validation(
                estimator=model_b,
                config=self._config,
                metric_registry=self._metric_registry,
                fold=fold,
            )

            metric_entry = self._metric_registry.get(metric)
            if metric_entry is None:
                for e in self._metric_registry._metrics.values():
                    if (
                        e.name.lower() == metric.lower()
                        or e.display_name.lower() == metric.lower()
                    ):
                        metric_entry = e
                        break
            display_col = metric_entry.display_name

            # Exclude Mean/SD/CI rows, convert to float
            fold_scores_a = scores_a[display_col].iloc[:-3].values.astype(float)
            fold_scores_b = scores_b[display_col].iloc[:-3].values.astype(float)

            result = _compare_stats(
                model_a_scores=fold_scores_a,
                model_b_scores=fold_scores_b,
                model_a_name=type(model_a).__name__,
                model_b_name=type(model_b).__name__,
                test=test,
            )

        if verbose:
            print(f"\nStatistical Model Comparison ({result['test']})")
            if "model_a_mean" in result:
                print(f"  {result['model_a']:30s} mean: {result['model_a_mean']}")
                print(f"  {result['model_b']:30s} mean: {result['model_b_mean']}")
            if "b_count" in result:
                print(f"  A correct & B wrong: {result['b_count']}")
                print(f"  A wrong & B correct: {result['c_count']}")
            print(f"  p-value: {result['p_value']}")
            print(f"  Result:  {result['conclusion']}")

        return result

    def compare_multiple_stats(
        self,
        models: list[Any],
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Cochran's Q test: are 3+ classifiers significantly different?

        Parameters
        ----------
        models : list of estimators
            Three or more fitted models to compare.
        verbose : bool
            Print results.

        Returns
        -------
        dict with Cochran's Q test results.
        """
        self._check_setup()
        from pycaret_redux.training.stats import cochrans_q_test

        X_test = self._config.X_test
        y_test = self._config.y_test
        if self._config.pipeline is not None:
            X_test = self._config.pipeline.transform(X_test)

        preds = [m.predict(X_test) for m in models]
        names = [type(m).__name__ for m in models]

        result = cochrans_q_test(y_test.values, preds, names)

        if verbose:
            print(f"\n{result['test']}")
            print(f"  Classifiers: {', '.join(result['model_names'])}")
            print(f"  Q statistic: {result['statistic']}")
            print(f"  p-value: {result['p_value']}")
            print(f"  Result: {result['conclusion']}")

        return result

    def compare_5x2cv(
        self,
        model_a: Any,
        model_b: Any,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Dietterich's 5x2cv paired F-test for model comparison.

        More powerful than paired t-test with lower false positive rate.
        Uses raw data (not preprocessed) to avoid data leakage.

        Parameters
        ----------
        model_a : estimator
            First classifier.
        model_b : estimator
            Second classifier.
        verbose : bool
            Print results.

        Returns
        -------
        dict with F-test results.
        """
        self._check_setup()
        from sklearn.base import clone

        from pycaret_redux.training.cross_validation import _build_full_pipeline
        from pycaret_redux.training.stats import five_by_two_cv_f_test

        # Build full pipelines so preprocessing is included
        pipe_a = _build_full_pipeline(self._config.pipeline, clone(model_a))
        pipe_b = _build_full_pipeline(self._config.pipeline, clone(model_b))

        X = self._config.X_train
        y = self._config.y_train

        result = five_by_two_cv_f_test(pipe_a, pipe_b, X.values, y.values, seed=self._config.seed)
        # Fix model names to use original estimator names
        result["model_a"] = type(model_a).__name__
        result["model_b"] = type(model_b).__name__

        if verbose:
            print(f"\n{result['test']}")
            print(f"  {result['model_a']} vs {result['model_b']}")
            print(f"  F statistic: {result['statistic']}")
            print(f"  p-value: {result['p_value']}")
            print(f"  Result: {result['conclusion']}")

        return result

    # ------------------------------------------------------------------
    # AutoML
    # ------------------------------------------------------------------

    def automl(
        self,
        optimize: str = "Accuracy",
        turbo: bool = True,
        n_top: int = 3,
        tune_n_iter: int = 10,
        ensemble: str = "blend",
        verbose: bool = True,
    ) -> Any:
        """Automated ML pipeline: compare, tune, ensemble.

        Runs the full pipeline in one call:
        1. Compare all models, pick top N
        2. Tune each top model
        3. Blend or stack the tuned models

        Parameters
        ----------
        optimize : str
            Metric to optimize throughout. Default "Accuracy".
        turbo : bool
            Only use fast models in comparison.
        n_top : int
            Number of top models to tune and ensemble.
        tune_n_iter : int
            Iterations for hyperparameter tuning per model.
        ensemble : str
            "blend" for VotingClassifier, "stack" for StackingClassifier.
        verbose : bool
            Print progress.

        Returns
        -------
        object
            The best ensembled (or single) model after the full pipeline.

        Examples
        --------
        >>> best = exp.automl(optimize="AUC", n_top=5)
        >>> exp.predict_model(best)
        """
        self._check_setup()

        # Step 1: Compare models
        if verbose:
            print("Step 1/3: Comparing models...")
        top_models = self.compare_models(
            sort=optimize, n_select=n_top, turbo=turbo, errors="ignore", verbose=verbose
        )
        if not isinstance(top_models, list):
            top_models = [top_models]

        # Step 2: Tune each model
        if verbose:
            print(f"\nStep 2/3: Tuning top {len(top_models)} models...")
        tuned_models = []
        for i, model in enumerate(top_models):
            if verbose:
                print(f"  Tuning model {i + 1}/{len(top_models)}: {type(model).__name__}")
            tuned = self.tune_model(model, optimize=optimize, n_iter=tune_n_iter, verbose=False)
            tuned_models.append(tuned)

        # Step 3: Ensemble
        if verbose:
            print(f"\nStep 3/3: Creating {ensemble} ensemble...")
        if len(tuned_models) == 1:
            best = tuned_models[0]
        elif ensemble == "stack":
            best = self.stack_models(tuned_models, optimize=optimize, verbose=verbose)
        else:
            best = self.blend_models(tuned_models, optimize=optimize, verbose=verbose)

        if verbose:
            print(f"\nAutoML complete. Final model: {type(best).__name__}")
        return best

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def pull(self) -> pd.DataFrame:
        """Return the last CV or comparison result as a DataFrame.

        After ``create_model()``, returns the per-fold CV scores.
        After ``compare_models()``, returns the model comparison table.

        Returns
        -------
        pd.DataFrame
            Last result table, or empty DataFrame if nothing has run yet.

        Examples
        --------
        >>> exp.create_model("lr")
        >>> exp.pull()  # per-fold CV scores
        """
        if self._last_pull is not None:
            return self._last_pull
        return pd.DataFrame()

    def get_config(self, variable: str | None = None) -> Any:
        """Get experiment configuration or a specific variable.

        Parameters
        ----------
        variable : str, optional
            Specific config attribute to return. If None, returns a dict
            of all configuration. Available keys: seed, target_name,
            X_train, X_test, y_train, y_test, pipeline, fold_generator,
            feature_types, is_multiclass, n_jobs, fold.

        Returns
        -------
        object or dict
            Value of the requested config variable, or a dictionary of
            all configuration when ``variable`` is ``None``.

        Examples
        --------
        >>> exp.get_config("seed")
        42
        >>> exp.get_config("X_train").shape
        (700, 10)
        """
        self._check_setup()
        config_map = {
            "seed": self._config.seed,
            "target_name": self._config.target_name,
            "X_train": self._config.X_train,
            "X_test": self._config.X_test,
            "y_train": self._config.y_train,
            "y_test": self._config.y_test,
            "pipeline": self._config.pipeline,
            "fold_generator": self._config.fold_generator,
            "feature_types": self._config.feature_types,
            "feature_names_in": self._config.feature_names_in,
            "is_multiclass": self._config.is_multiclass,
            "n_jobs": self._config.setup_config.n_jobs,
            "fold": self._config.setup_config.fold,
            "fold_strategy": self._config.setup_config.fold_strategy,
            "preprocess": self._config.setup_config.preprocess,
            "normalize": self._config.setup_config.normalize,
            "transformation": self._config.setup_config.transformation,
            "pca": self._config.setup_config.pca,
            "fix_imbalance": self._config.setup_config.fix_imbalance,
        }
        if variable is not None:
            if variable not in config_map:
                raise ValueError(
                    f"Unknown variable '{variable}'. Available: {', '.join(config_map.keys())}"
                )
            return config_map[variable]
        return config_map

    def get_pipeline(self) -> Any:
        """Return the fitted preprocessing pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline or None
            The preprocessing pipeline, or ``None`` if ``preprocess=False``
            was passed to ``setup()``.

        Examples
        --------
        >>> pipe = exp.get_pipeline()
        >>> pipe.transform(new_data)
        """
        self._check_setup()
        return self._config.pipeline

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_setup(self) -> None:
        if not self._config.is_setup_done:
            raise RuntimeError("Experiment not initialized. Call setup() first.")

    def _build_fold_generator(
        self,
        strategy: str | Any,
        n_folds: int,
        shuffle: bool,
        seed: int,
    ) -> Any:
        rs = seed if shuffle else None
        if isinstance(strategy, str):
            match strategy.lower():
                case "stratifiedkfold":
                    return StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=rs)
                case "kfold":
                    return KFold(n_splits=n_folds, shuffle=shuffle, random_state=rs)
                case "groupkfold":
                    return GroupKFold(n_splits=n_folds)
                case "timeseries":
                    return TimeSeriesSplit(n_splits=n_folds)
                case "repeatedstratifiedkfold":
                    return RepeatedStratifiedKFold(
                        n_splits=n_folds, n_repeats=3, random_state=seed
                    )
                case "repeatedkfold":
                    return RepeatedKFold(n_splits=n_folds, n_repeats=3, random_state=seed)
                case _:
                    raise ValueError(
                        f"Unknown fold_strategy: '{strategy}'. "
                        "Use 'stratifiedkfold', 'kfold', 'groupkfold', "
                        "'timeseries', 'repeatedstratifiedkfold', 'repeatedkfold', "
                        "or pass a CV splitter object."
                    )
        # Assume it's a sklearn CV splitter
        return strategy

    def _print_setup_summary(self) -> None:
        from pycaret_redux.utils.display import display_setup_summary

        display_setup_summary(self._config)
