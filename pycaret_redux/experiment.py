"""ClassificationExperiment — the single public API class."""

from __future__ import annotations

import logging
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

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
    ) -> Self:
        """Initialize the experiment: validate data, preprocess, split train/test.

        Parameters
        ----------
        data : DataFrame-like
            Training data with features and target column.
        target : int or str
            Target column name or index. Default -1 (last column).
        train_size : float
            Proportion of data for training. Default 0.7.
        fold : int
            Number of cross-validation folds. Default 10.
        session_id : int, optional
            Random seed for reproducibility.
        verbose : bool
            Whether to print setup summary.

        Returns
        -------
        self
            The experiment instance for method chaining.
        """
        if data is None:
            raise ValueError("data parameter is required.")

        validate_setup_params(train_size, fold)

        # Resolve seed
        seed = session_id if session_id is not None else np.random.randint(0, 10000)
        self._config.seed = seed

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
        stratify = y if data_split_stratify is True else None
        if test_data is not None:
            test_df = to_dataframe(test_data)
            X_test = test_df.drop(columns=[target_name])
            y_test = test_df[target_name]
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
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
            pipeline = build_preprocessing_pipeline(self._config, setup_cfg)
            pipeline.fit(self._config.X_train, self._config.y_train)
            self._config.pipeline = pipeline

        self._config.is_setup_done = True

        if verbose:
            self._print_setup_summary()

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

        Parameters
        ----------
        sort : str
            Metric to sort by. Default "Accuracy".
        n_select : int
            Number of top models to return.
        turbo : bool
            Only use fast models.

        Returns
        -------
        Best model (or list if n_select > 1).
        """
        self._check_setup()
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

        Parameters
        ----------
        estimator : str or estimator
            Model ID (e.g. "lr") or sklearn-compatible estimator.

        Returns
        -------
        Fitted model.
        """
        self._check_setup()
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

        Parameters
        ----------
        estimator : fitted estimator
            Model to tune.
        n_iter : int
            Number of search iterations.
        optimize : str
            Metric to optimize.
        """
        self._check_setup()
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
        """Create a voting ensemble from multiple models."""
        self._check_setup()
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
        """Create a stacking ensemble."""
        self._check_setup()
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
        """Create a bagging ensemble."""
        self._check_setup()
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
        """Generate model diagnostic plots.

        Parameters
        ----------
        estimator : fitted estimator
            Model to plot.
        plot : str
            Plot type: auc, confusion_matrix, threshold, pr, error,
            class_report, feature, learning, vc, calibration, lift, gain, ks.
        save : str, optional
            File path to save the plot.
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
        """Print evaluation summary for a model."""
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
        """SHAP-based model interpretation."""
        self._check_setup()
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required for interpret_model. "
                "Install with: uv add shap"
            )

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
        """Calibrate predicted probabilities.

        Parameters
        ----------
        estimator : fitted estimator
            Model to calibrate.
        method : str
            "sigmoid" (Platt) or "isotonic".
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
        """Find optimal decision threshold for binary classification.

        Returns
        -------
        (estimator, optimal_threshold)
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
        """Make predictions on new or test data."""
        self._check_setup()
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
        """Retrain model on full dataset (train + test)."""
        self._check_setup()
        from pycaret_redux.training.finalization import finalize_model as _finalize

        return _finalize(estimator=estimator, config=self._config)

    def save_model(self, estimator: Any, model_name: str, verbose: bool = True) -> None:
        """Save model + preprocessing pipeline to disk.

        The saved artifact bundles the estimator and the preprocessing
        pipeline together, so a single file is all you need for deployment.
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
        """Load model from disk.

        Returns the fitted estimator. The preprocessing pipeline is also
        loaded and can be accessed via the returned ModelArtifact when
        using ``load_model`` from ``pycaret_redux.persistence``.
        """
        from pycaret_redux.persistence.serialization import load_model as _load

        artifact = _load(model_name, verbose=verbose)
        return artifact.estimator

    def models(self, turbo_only: bool = False, **kwargs) -> pd.DataFrame:
        """List available models."""
        if self._model_registry is None:
            raise RuntimeError("Call setup() first to initialize the model registry.")
        return self._model_registry.list_models(turbo_only=turbo_only)

    def get_metrics(self, **kwargs) -> pd.DataFrame:
        """List available metrics."""
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
        """Register a custom metric."""
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
        """Remove a metric."""
        if self._metric_registry is None:
            raise RuntimeError("Call setup() first.")
        self._metric_registry.remove(name_or_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_setup(self) -> None:
        if not self._config.is_setup_done:
            raise RuntimeError(
                "Experiment not initialized. Call setup() first."
            )

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
                    return StratifiedKFold(
                        n_splits=n_folds, shuffle=shuffle, random_state=rs
                    )
                case "kfold":
                    return KFold(
                        n_splits=n_folds, shuffle=shuffle, random_state=rs
                    )
                case _:
                    raise ValueError(
                        f"Unknown fold_strategy: '{strategy}'. "
                        "Use 'stratifiedkfold', 'kfold', or pass a CV splitter."
                    )
        # Assume it's a sklearn CV splitter
        return strategy

    def _print_setup_summary(self) -> None:
        from pycaret_redux.utils.display import display_setup_summary

        display_setup_summary(self._config)
