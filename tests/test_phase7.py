"""Tests for Phase 7: Tracking, persistence, calibration, threshold."""

import os

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


class TestSaveLoadModel:
    def test_save_load(self, setup_exp, tmp_path):
        lr = setup_exp.create_model("lr", verbose=False)
        path = str(tmp_path / "my_model")
        setup_exp.save_model(lr, path, verbose=False)
        assert os.path.exists(f"{path}.joblib")

        loaded = setup_exp.load_model(path, verbose=False)
        assert hasattr(loaded, "predict")

    def test_load_nonexistent(self, setup_exp):
        with pytest.raises(FileNotFoundError):
            setup_exp.load_model("/nonexistent/model")


class TestCalibrateModel:
    def test_calibrate_sigmoid(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        calibrated = setup_exp.calibrate_model(lr, method="sigmoid", verbose=False)
        assert hasattr(calibrated, "predict_proba")

    def test_calibrate_isotonic(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        calibrated = setup_exp.calibrate_model(lr, method="isotonic", verbose=False)
        assert hasattr(calibrated, "predict_proba")


class TestOptimizeThreshold:
    def test_optimize_f1(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        model, threshold = setup_exp.optimize_threshold(lr, optimize="F1", verbose=False)
        assert 0.0 < threshold < 1.0

    def test_optimize_accuracy(self, setup_exp):
        lr = setup_exp.create_model("lr", verbose=False)
        model, threshold = setup_exp.optimize_threshold(lr, optimize="Accuracy", verbose=False)
        assert 0.0 < threshold < 1.0
