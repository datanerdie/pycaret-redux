"""Tests for Phase 6: Plotting."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

import matplotlib.pyplot as plt
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


@pytest.fixture
def fitted_model(setup_exp):
    return setup_exp.create_model("lr", verbose=False)


class TestPlotModel:
    def test_auc(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="auc")
        assert fig is not None
        plt.close("all")

    def test_confusion_matrix(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="confusion_matrix")
        assert fig is not None
        plt.close("all")

    def test_precision_recall(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="pr")
        assert fig is not None
        plt.close("all")

    def test_threshold(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="threshold")
        assert fig is not None
        plt.close("all")

    def test_prediction_error(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="error")
        assert fig is not None
        plt.close("all")

    def test_class_report(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="class_report")
        assert fig is not None
        plt.close("all")

    def test_feature_importance(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="feature")
        assert fig is not None
        plt.close("all")

    def test_calibration(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="calibration")
        assert fig is not None
        plt.close("all")

    def test_lift_chart(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="lift")
        assert fig is not None
        plt.close("all")

    def test_gain_chart(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="gain")
        assert fig is not None
        plt.close("all")

    def test_ks_statistic(self, setup_exp, fitted_model):
        fig = setup_exp.plot_model(fitted_model, plot="ks")
        assert fig is not None
        plt.close("all")

    def test_unknown_plot_raises(self, setup_exp, fitted_model):
        with pytest.raises(KeyError, match="not found"):
            setup_exp.plot_model(fitted_model, plot="nonexistent")

    def test_save_plot(self, setup_exp, fitted_model, tmp_path):
        save_path = str(tmp_path / "test_plot.png")
        setup_exp.plot_model(fitted_model, plot="confusion_matrix", save=save_path)
        import os
        assert os.path.exists(save_path)
        plt.close("all")


class TestEvaluateModel:
    def test_evaluate(self, setup_exp, fitted_model, capsys):
        setup_exp.evaluate_model(fitted_model)
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out
