"""Tests for Phase 5: Tuning, ensembles, finalization, prediction."""

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


class TestTuneModel:
    def test_tune_basic(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        tuned = setup_exp.tune_model(lr, n_iter=3, verbose=False)
        assert hasattr(tuned, "predict")

    def test_tune_with_custom_grid(self, setup_exp):
        dt = setup_exp.create_model("dt", verbose=False)
        tuned = setup_exp.tune_model(
            dt,
            custom_grid={"max_depth": [2, 3, 5]},
            n_iter=3,
            verbose=False,
        )
        assert hasattr(tuned, "predict")

    def test_tune_optimize_f1(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        tuned = setup_exp.tune_model(lr, optimize="F1", n_iter=3, verbose=False)
        assert hasattr(tuned, "predict")


class TestBlendModels:
    def test_blend_basic(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        dt = setup_exp.create_model("dt", verbose=False)
        blended = setup_exp.blend_models([lr, dt], verbose=False)
        assert hasattr(blended, "predict")

    def test_blend_hard_voting(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        dt = setup_exp.create_model("dt", verbose=False)
        blended = setup_exp.blend_models([lr, dt], method="hard", verbose=False)
        assert hasattr(blended, "predict")


class TestStackModels:
    def test_stack_basic(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        dt = setup_exp.create_model("dt", verbose=False)
        stacked = setup_exp.stack_models([lr, dt], verbose=False)
        assert hasattr(stacked, "predict")


class TestEnsembleModel:
    def test_ensemble_bagging(self, setup_exp):
        dt = setup_exp.create_model("dt", verbose=False)
        bagged = setup_exp.ensemble_model(dt, n_estimators=5, verbose=False)
        assert hasattr(bagged, "predict")


class TestPredictModel:
    def test_predict_test_set(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        preds = setup_exp.predict_model(lr)
        assert isinstance(preds, pd.DataFrame)
        assert "prediction_label" in preds.columns
        assert len(preds) == len(setup_exp.X_test)

    def test_predict_with_score(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        preds = setup_exp.predict_model(lr)
        assert "prediction_score" in preds.columns

    def test_predict_raw_score(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        preds = setup_exp.predict_model(lr, raw_score=True)
        assert "prediction_score_0" in preds.columns
        assert "prediction_score_1" in preds.columns

    def test_predict_new_data(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        preds = setup_exp.predict_model(lr, data=setup_exp.X_train.head(5))
        assert len(preds) == 5

    def test_predict_custom_threshold(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        preds = setup_exp.predict_model(lr, probability_threshold=0.9)
        assert "prediction_label" in preds.columns


class TestFinalizeModel:
    def test_finalize(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        final = setup_exp.finalize_model(lr)
        assert hasattr(final, "predict")
        # Should be able to predict
        preds = final.predict(
            setup_exp.pipeline.transform(setup_exp.X_test)
        )
        assert len(preds) == len(setup_exp.X_test)
