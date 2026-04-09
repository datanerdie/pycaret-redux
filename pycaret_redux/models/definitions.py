"""All default classifier definitions as ModelEntry list."""

from __future__ import annotations

import numpy as np

from pycaret_redux.models.registry import ModelEntry, TuningSpace


def _arange(start: float, stop: float, step: float) -> list[float]:
    """Helper to generate parameter ranges."""
    return list(np.arange(start, stop + step / 2, step).round(6))


def get_default_classifiers(seed: int, n_jobs: int | None = -1) -> list[ModelEntry]:
    """Return all default classifier model entries."""
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        AdaBoostClassifier,
        BaggingClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.linear_model import (
        LogisticRegression,
        RidgeClassifier,
        SGDClassifier,
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    models = [
        # Logistic Regression
        ModelEntry(
            id="lr",
            name="Logistic Regression",
            class_def=LogisticRegression,
            default_args={"max_iter": 1000, "random_state": seed},
            tuning=TuningSpace(
                grid={
                    "C": _arange(0.001, 10, 0.5),
                    "class_weight": ["balanced", None],
                },
            ),
            shap_type=None,
        ),
        # K Neighbors
        ModelEntry(
            id="knn",
            name="K Neighbors Classifier",
            class_def=KNeighborsClassifier,
            default_args={"n_jobs": n_jobs},
            tuning=TuningSpace(
                grid={
                    "n_neighbors": list(range(1, 51)),
                    "weights": ["uniform", "distance"],
                    "metric": ["minkowski", "euclidean", "manhattan"],
                },
            ),
            shap_type=None,
        ),
        # Naive Bayes
        ModelEntry(
            id="nb",
            name="Naive Bayes",
            class_def=GaussianNB,
            tuning=TuningSpace(
                grid={
                    "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 1.0],
                },
            ),
            shap_type=None,
        ),
        # Decision Tree
        ModelEntry(
            id="dt",
            name="Decision Tree Classifier",
            class_def=DecisionTreeClassifier,
            default_args={"random_state": seed},
            tuning=TuningSpace(
                grid={
                    "max_depth": list(range(1, 17)),
                    "max_features": [1.0, "sqrt", "log2"],
                    "min_samples_leaf": [2, 3, 4, 5, 6],
                    "min_samples_split": [2, 5, 7, 9, 10],
                    "criterion": ["gini", "entropy"],
                },
            ),
            shap_type="type1",
        ),
        # SVM - Linear Kernel (SGD)
        ModelEntry(
            id="svm",
            name="SVM - Linear Kernel",
            class_def=SGDClassifier,
            default_args={
                "random_state": seed,
                "n_jobs": n_jobs,
                "tol": 0.001,
                "loss": "hinge",
                "penalty": "l2",
                "eta0": 0.001,
            },
            tuning=TuningSpace(
                grid={
                    "penalty": ["elasticnet", "l2", "l1"],
                    "alpha": [1e-7, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5],
                    "l1_ratio": _arange(0.01, 0.99, 0.1),
                    "fit_intercept": [True, False],
                    "eta0": [0.001, 0.01, 0.05, 0.1, 0.5],
                },
            ),
            supports_predict_proba=False,
            shap_type=None,
        ),
        # SVM - RBF Kernel
        ModelEntry(
            id="rbfsvm",
            name="SVM - Radial Kernel",
            class_def=SVC,
            default_args={
                "random_state": seed,
                "gamma": "auto",
                "kernel": "rbf",
                "probability": True,
            },
            tuning=TuningSpace(
                grid={
                    "C": _arange(0.001, 50, 2.5),
                    "class_weight": ["balanced", None],
                },
            ),
            is_turbo=False,
            shap_type=None,
        ),
        # Gaussian Process
        ModelEntry(
            id="gpc",
            name="Gaussian Process Classifier",
            class_def=GaussianProcessClassifier,
            default_args={"random_state": seed, "max_iter_predict": 100},
            tuning=TuningSpace(
                grid={"max_iter_predict": [100, 200, 300, 500]},
            ),
            is_turbo=False,
            shap_type=None,
        ),
        # MLP
        ModelEntry(
            id="mlp",
            name="MLP Classifier",
            class_def=MLPClassifier,
            default_args={"random_state": seed, "max_iter": 500},
            tuning=TuningSpace(
                grid={
                    "alpha": [1e-5, 1e-4, 1e-3, 0.01, 0.1],
                    "hidden_layer_sizes": [
                        (100,), (50, 50), (100, 50), (50, 100),
                        (100, 100), (200,), (50,),
                    ],
                    "activation": ["relu", "tanh", "logistic"],
                    "learning_rate": ["constant", "invscaling", "adaptive"],
                },
            ),
            is_turbo=False,
            shap_type=None,
        ),
        # Ridge
        ModelEntry(
            id="ridge",
            name="Ridge Classifier",
            class_def=RidgeClassifier,
            default_args={"random_state": seed},
            tuning=TuningSpace(
                grid={
                    "alpha": _arange(0.01, 10, 0.5),
                    "fit_intercept": [True, False],
                    "class_weight": ["balanced", None],
                },
            ),
            supports_predict_proba=False,
            shap_type=None,
        ),
        # Random Forest
        ModelEntry(
            id="rf",
            name="Random Forest Classifier",
            class_def=RandomForestClassifier,
            default_args={"random_state": seed, "n_jobs": n_jobs},
            tuning=TuningSpace(
                grid={
                    "n_estimators": [10, 50, 100, 200, 300],
                    "max_depth": list(range(1, 12)),
                    "min_samples_leaf": [2, 3, 4, 5, 6],
                    "min_samples_split": [2, 5, 7, 9, 10],
                    "max_features": [1.0, "sqrt", "log2"],
                    "criterion": ["gini", "entropy"],
                    "class_weight": ["balanced", "balanced_subsample", None],
                },
            ),
            shap_type="type1",
        ),
        # QDA
        ModelEntry(
            id="qda",
            name="Quadratic Discriminant Analysis",
            class_def=QuadraticDiscriminantAnalysis,
            tuning=TuningSpace(
                grid={"reg_param": _arange(0.0, 1.0, 0.1)},
            ),
            shap_type=None,
        ),
        # AdaBoost
        ModelEntry(
            id="ada",
            name="Ada Boost Classifier",
            class_def=AdaBoostClassifier,
            default_args={"random_state": seed},
            tuning=TuningSpace(
                grid={
                    "n_estimators": [10, 50, 100, 200, 300],
                    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                },
            ),
            shap_type="type2",
        ),
        # Gradient Boosting
        ModelEntry(
            id="gbc",
            name="Gradient Boosting Classifier",
            class_def=GradientBoostingClassifier,
            default_args={"random_state": seed},
            tuning=TuningSpace(
                grid={
                    "n_estimators": [10, 50, 100, 200, 300],
                    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5],
                    "max_depth": list(range(1, 12)),
                    "min_samples_split": [2, 5, 7, 9, 10],
                    "min_samples_leaf": [1, 2, 3, 4, 5],
                    "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
                    "max_features": [1.0, "sqrt", "log2"],
                },
            ),
            shap_type="type2",
        ),
        # LDA
        ModelEntry(
            id="lda",
            name="Linear Discriminant Analysis",
            class_def=LinearDiscriminantAnalysis,
            tuning=TuningSpace(
                grid={
                    "solver": ["lsqr", "eigen"],
                    "shrinkage": [None, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                },
            ),
            shap_type=None,
        ),
        # Extra Trees
        ModelEntry(
            id="et",
            name="Extra Trees Classifier",
            class_def=ExtraTreesClassifier,
            default_args={"random_state": seed, "n_jobs": n_jobs},
            tuning=TuningSpace(
                grid={
                    "n_estimators": [10, 50, 100, 200, 300],
                    "max_depth": list(range(1, 12)),
                    "min_samples_leaf": [2, 3, 4, 5, 6],
                    "min_samples_split": [2, 5, 7, 9, 10],
                    "max_features": [1.0, "sqrt", "log2"],
                    "criterion": ["gini", "entropy"],
                    "class_weight": ["balanced", "balanced_subsample", None],
                },
            ),
            shap_type="type1",
        ),
        # Dummy
        ModelEntry(
            id="dummy",
            name="Dummy Classifier",
            class_def=DummyClassifier,
            default_args={"random_state": seed, "strategy": "prior"},
            is_turbo=False,
            shap_type=None,
        ),
        # Bagging (special)
        ModelEntry(
            id="bagging",
            name="Bagging Classifier",
            class_def=BaggingClassifier,
            default_args={"random_state": seed, "n_jobs": n_jobs},
            is_special=True,
            shap_type=None,
        ),
    ]

    # Optional boosting models
    models.extend(_get_optional_boosters(seed, n_jobs))

    return models


def _get_optional_boosters(seed: int, n_jobs: int | None) -> list[ModelEntry]:
    """Try to import optional boosting libraries."""
    entries: list[ModelEntry] = []

    try:
        from xgboost import XGBClassifier

        entries.append(ModelEntry(
            id="xgboost",
            name="Extreme Gradient Boosting",
            class_def=XGBClassifier,
            default_args={
                "random_state": seed,
                "n_jobs": n_jobs,
                "verbosity": 0,
                "use_label_encoder": False,
            },
            tuning=TuningSpace(
                grid={
                    "n_estimators": [10, 50, 100, 200, 300],
                    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5],
                    "max_depth": list(range(1, 12)),
                    "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.5, 0.7, 0.8, 0.9, 1.0],
                    "min_child_weight": [1, 2, 5, 10],
                    "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
                    "reg_lambda": [0, 0.001, 0.01, 0.1, 1.0],
                },
            ),
            shap_type="type2",
        ))
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        entries.append(ModelEntry(
            id="lightgbm",
            name="Light Gradient Boosting Machine",
            class_def=LGBMClassifier,
            default_args={
                "random_state": seed,
                "n_jobs": n_jobs,
                "verbosity": -1,
            },
            tuning=TuningSpace(
                grid={
                    "n_estimators": [10, 50, 100, 200, 300],
                    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5],
                    "max_depth": list(range(-1, 12)),
                    "num_leaves": [10, 20, 31, 50, 100, 150],
                    "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.5, 0.7, 0.8, 0.9, 1.0],
                    "min_child_samples": [5, 10, 20, 50],
                    "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
                    "reg_lambda": [0, 0.001, 0.01, 0.1, 1.0],
                },
            ),
            shap_type="type2",
        ))
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        entries.append(ModelEntry(
            id="catboost",
            name="CatBoost Classifier",
            class_def=CatBoostClassifier,
            default_args={
                "random_state": seed,
                "verbose": False,
                "thread_count": n_jobs if n_jobs and n_jobs > 0 else -1,
            },
            tuning=TuningSpace(
                grid={
                    "depth": list(range(1, 12)),
                    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5],
                    "iterations": [100, 200, 300, 500],
                    "l2_leaf_reg": [1, 3, 5, 7, 9],
                },
            ),
            is_turbo=False,
            shap_type="type2",
        ))
    except ImportError:
        pass

    return entries
