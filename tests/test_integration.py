"""End-to-end integration tests with real datasets."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pycaret_redux import ClassificationExperiment


@pytest.fixture
def iris_binary():
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame
    df["target"] = (df["target"] == 0).astype(int)
    return df


@pytest.fixture
def iris_multiclass():
    from sklearn.datasets import load_iris

    return load_iris(as_frame=True).frame


class TestEndToEndBinary:
    """Full PyCaret-style workflow on binary classification."""

    def test_full_workflow(self, iris_binary, tmp_path):
        exp = ClassificationExperiment()

        # 1. Setup
        exp.setup(data=iris_binary, target="target", session_id=42, fold=3, verbose=False)
        assert exp.is_setup_done
        assert not exp.is_multiclass

        # 2. Compare models
        best = exp.compare_models(
            include=["lr", "dt", "nb"],
            verbose=False,
        )
        assert hasattr(best, "predict")

        # 3. Create model
        lr = exp.create_model("lr", verbose=False)
        assert hasattr(lr, "predict_proba")

        # 4. Tune model
        tuned = exp.tune_model(lr, n_iter=3, verbose=False)
        assert hasattr(tuned, "predict")

        # 5. Ensemble
        dt = exp.create_model("dt", verbose=False)
        blended = exp.blend_models([lr, dt], verbose=False)
        assert hasattr(blended, "predict")

        # 6. Stack
        stacked = exp.stack_models([lr, dt], verbose=False)
        assert hasattr(stacked, "predict")

        # 7. Plot
        fig = exp.plot_model(lr, plot="confusion_matrix")
        assert fig is not None
        plt.close("all")

        fig = exp.plot_model(lr, plot="auc")
        assert fig is not None
        plt.close("all")

        # 8. Evaluate
        exp.evaluate_model(lr)

        # 9. Predict
        preds = exp.predict_model(lr)
        assert isinstance(preds, pd.DataFrame)
        assert "prediction_label" in preds.columns
        assert "prediction_score" in preds.columns

        # 10. Optimize threshold
        _, threshold = exp.optimize_threshold(lr, verbose=False)
        assert 0.0 < threshold < 1.0

        # 11. Calibrate
        calibrated = exp.calibrate_model(lr, verbose=False)
        assert hasattr(calibrated, "predict_proba")

        # 12. Finalize
        final = exp.finalize_model(lr)
        assert hasattr(final, "predict")

        # 13. Save & Load
        model_path = str(tmp_path / "final_model")
        exp.save_model(final, model_path, verbose=False)
        loaded = exp.load_model(model_path, verbose=False)
        assert hasattr(loaded, "predict")

        # 14. Models & Metrics
        models_df = exp.models()
        assert isinstance(models_df, pd.DataFrame)
        assert "lr" in models_df.index

        metrics_df = exp.get_metrics()
        assert isinstance(metrics_df, pd.DataFrame)

        # 15. Custom metric
        exp.add_metric("custom", "Custom", lambda y, p: 1.0)
        assert "custom" in exp.get_metrics().index
        exp.remove_metric("custom")


class TestEndToEndMulticlass:
    """Full workflow on multiclass classification."""

    def test_multiclass_workflow(self, iris_multiclass):
        exp = ClassificationExperiment()

        exp.setup(data=iris_multiclass, target="target", session_id=42, fold=3, verbose=False)
        assert exp.is_multiclass

        best = exp.compare_models(include=["lr", "dt"], verbose=False)
        assert hasattr(best, "predict")

        lr = exp.create_model("lr", verbose=False)
        preds = exp.predict_model(lr)
        assert len(preds["prediction_label"].unique()) > 2

        fig = exp.plot_model(lr, plot="confusion_matrix")
        assert fig is not None
        plt.close("all")


class TestMixedDataWorkflow:
    """Test with mixed numeric/categorical features."""

    def test_mixed_features(self):
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            "age": np.random.randint(18, 80, n).astype(float),
            "income": np.random.normal(50000, 15000, n),
            "gender": np.random.choice(["M", "F"], n),
            "education": np.random.choice(["HS", "BS", "MS", "PhD"], n),
            "target": np.random.randint(0, 2, n),
        })
        df.loc[0:5, "age"] = np.nan
        df.loc[3:8, "gender"] = np.nan

        exp = ClassificationExperiment()
        exp.setup(
            data=df, target="target",
            numeric_imputation="median",
            categorical_imputation="mode",
            normalize=True,
            session_id=42, fold=3, verbose=False,
        )

        lr = exp.create_model("lr", verbose=False)
        preds = exp.predict_model(lr)
        assert "prediction_label" in preds.columns
        assert len(preds) == len(exp.X_test)
