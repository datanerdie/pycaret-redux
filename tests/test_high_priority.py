"""Tests for high-priority improvements: nested CV, halving search, McNemar's,
class weights, HistGradientBoosting, confidence intervals."""

import numpy as np
import pandas as pd
import pytest

from pycaret_redux import ClassificationExperiment


@pytest.fixture
def setup_exp():
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame
    df["target"] = (df["target"] == 0).astype(int)

    exp = ClassificationExperiment()
    exp.setup(data=df, target="target", session_id=42, fold=3, verbose=False)
    return exp


class TestNestedCV:
    def test_nested_cv_by_id(self, setup_exp):
        scores = setup_exp.nested_cv("lr", n_iter=3, verbose=False)
        assert isinstance(scores, pd.DataFrame)
        assert "Mean" in scores.index
        assert "SD" in scores.index
        assert "95% CI" in scores.index

    def test_nested_cv_with_grid(self, setup_exp):
        scores = setup_exp.nested_cv(
            "dt",
            param_grid={"max_depth": [2, 3, 5]},
            n_iter=3,
            verbose=False,
        )
        assert isinstance(scores, pd.DataFrame)

    def test_nested_cv_stored_in_pull(self, setup_exp):
        setup_exp.nested_cv("lr", n_iter=3, verbose=False)
        pulled = setup_exp.pull()
        assert len(pulled) > 0


class TestHalvingSearch:
    def test_halving_tune(self, setup_exp):
        dt = setup_exp.create_model("dt", verbose=False)
        tuned = setup_exp.tune_model(
            dt,
            search_library="halving",
            verbose=False,
        )
        assert hasattr(tuned, "predict")


class TestMcNemar:
    def test_mcnemar_test(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        dt = setup_exp.create_model("dt", verbose=False)
        result = setup_exp.compare_model_stats(lr, dt, test="mcnemar", verbose=False)
        assert result["test"] == "McNemar's test"
        assert "p_value" in result
        assert "conclusion" in result
        assert "b_count" in result
        assert "c_count" in result

    def test_wilcoxon_still_works(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        dt = setup_exp.create_model("dt", verbose=False)
        result = setup_exp.compare_model_stats(lr, dt, test="wilcoxon", verbose=False)
        assert result["test"] == "Wilcoxon signed-rank test"


class TestClassWeights:
    def test_class_weight_imbalance(self):
        from sklearn.datasets import load_iris

        data = load_iris(as_frame=True)
        df = data.frame
        df["target"] = (df["target"] == 0).astype(int)

        exp = ClassificationExperiment()
        exp.setup(
            data=df,
            target="target",
            fix_imbalance=True,
            fix_imbalance_method="class_weight",
            session_id=42,
            fold=3,
            verbose=False,
        )
        lr = exp.create_model("lr", verbose=False)
        assert hasattr(lr, "predict")


class TestHistGradientBoosting:
    def test_hgbc_available(self, setup_exp):
        models = setup_exp.models()
        assert "hgbc" in models.index

    def test_hgbc_create(self, setup_exp):
        model = setup_exp.create_model("hgbc", verbose=False)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


class TestConfidenceIntervals:
    def test_ci_in_fold_scores(self, setup_exp):
        setup_exp.create_model("lr", verbose=False)
        scores = setup_exp.pull()
        assert "95% CI" in scores.index
        # CI values should be string like "[0.xx, 0.yy]"
        ci_row = scores.loc["95% CI"]
        assert any("[" in str(v) for v in ci_row.values)
