"""Tests for Phase 4: Training core — create_model, compare_models."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from pycaret_redux import ClassificationExperiment


@pytest.fixture
def setup_exp():
    """Set up a basic classification experiment."""
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame
    df["target"] = (df["target"] == 0).astype(int)

    exp = ClassificationExperiment()
    exp.setup(data=df, target="target", session_id=42, fold=3, verbose=False)
    return exp


class TestCreateModel:
    def test_create_by_id(self, setup_exp):
        model = setup_exp.create_model("lr", verbose=False)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_create_by_estimator(self, setup_exp):
        lr = LogisticRegression(max_iter=500, random_state=0)
        model = setup_exp.create_model(lr, verbose=False)
        assert hasattr(model, "predict")

    def test_create_without_cv(self, setup_exp):
        model = setup_exp.create_model("dt", cross_validation=False, verbose=False)
        assert hasattr(model, "predict")

    def test_create_multiple_models(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        dt = setup_exp.create_model("dt", verbose=False)
        rf = setup_exp.create_model("rf", verbose=False)
        # All should be different types
        assert type(lr) != type(dt)

    def test_create_with_kwargs(self, setup_exp):
        model = setup_exp.create_model("rf", n_estimators=10, verbose=False)
        assert hasattr(model, "n_estimators")
        assert model.n_estimators == 10

    def test_model_is_fitted(self, setup_exp):
        model = setup_exp.create_model("lr", verbose=False)
        # Should be able to predict on test data
        preds = model.predict(setup_exp.X_test)
        assert len(preds) == len(setup_exp.X_test)


class TestCompareModels:
    def test_compare_basic(self, setup_exp):
        best = setup_exp.compare_models(
            include=["lr", "dt", "nb"],
            verbose=False,
        )
        assert hasattr(best, "predict")

    def test_compare_returns_list(self, setup_exp):
        models = setup_exp.compare_models(
            include=["lr", "dt", "nb"],
            n_select=2,
            verbose=False,
        )
        assert isinstance(models, list)
        assert len(models) == 2

    def test_compare_with_exclude(self, setup_exp):
        best = setup_exp.compare_models(
            include=["lr", "dt", "nb", "rf"],
            exclude=["rf"],
            verbose=False,
        )
        assert hasattr(best, "predict")

    def test_compare_sort_by_metric(self, setup_exp):
        best = setup_exp.compare_models(
            include=["lr", "dt"],
            sort="F1",
            verbose=False,
        )
        assert hasattr(best, "predict")

    def test_compare_with_budget(self, setup_exp):
        best = setup_exp.compare_models(
            include=["lr", "dt", "nb"],
            budget_time=0.5,  # 30 seconds
            verbose=False,
        )
        assert hasattr(best, "predict")

    def test_comparison_df_stored(self, setup_exp):
        setup_exp.compare_models(
            include=["lr", "dt", "nb"],
            verbose=False,
        )
        assert hasattr(setup_exp, "_comparison_df")
        assert isinstance(setup_exp._comparison_df, pd.DataFrame)
        assert len(setup_exp._comparison_df) == 3

    def test_compare_errors_ignore(self, setup_exp):
        # Should not raise even if a model fails
        best = setup_exp.compare_models(
            include=["lr", "nb"],
            errors="ignore",
            verbose=False,
        )
        assert best is not None
