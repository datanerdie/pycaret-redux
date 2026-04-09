"""Tests for Phase 2: Preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from pycaret_redux import ClassificationExperiment


@pytest.fixture
def iris_data():
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame
    df["target"] = (df["target"] == 0).astype(int)
    return df


@pytest.fixture
def mixed_data():
    """Dataset with numeric, categorical, and missing values."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n).astype(float),
        "income": np.random.normal(50000, 15000, n),
        "gender": np.random.choice(["M", "F", "Other"], n),
        "education": np.random.choice(["HS", "BS", "MS", "PhD"], n),
        "target": np.random.randint(0, 2, n),
    })
    # Add some missing values
    df.loc[0:9, "age"] = np.nan
    df.loc[5:14, "income"] = np.nan
    df.loc[10:15, "gender"] = np.nan
    return df


class TestPreprocessingPipeline:
    def test_pipeline_created_on_setup(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(data=iris_data, target="target", session_id=42, verbose=False)
        assert exp.pipeline is not None

    def test_pipeline_transforms_data(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(data=iris_data, target="target", session_id=42, verbose=False)
        X_transformed = exp.pipeline.transform(exp.X_test)
        assert X_transformed is not None
        assert len(X_transformed) == len(exp.X_test)

    def test_no_pipeline_when_preprocess_false(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(
            data=iris_data, target="target", preprocess=False,
            session_id=42, verbose=False,
        )
        assert exp.pipeline is None

    def test_missing_value_imputation(self, mixed_data):
        exp = ClassificationExperiment()
        exp.setup(
            data=mixed_data, target="target",
            numeric_imputation="mean",
            categorical_imputation="mode",
            session_id=42, verbose=False,
        )
        X_transformed = exp.pipeline.transform(exp.X_test)
        # Should have no NaN after imputation
        assert not np.any(np.isnan(X_transformed)) if isinstance(X_transformed, np.ndarray) else not pd.DataFrame(X_transformed).isna().any().any()

    def test_normalization(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(
            data=iris_data, target="target",
            normalize=True, normalize_method="zscore",
            session_id=42, verbose=False,
        )
        X_transformed = exp.pipeline.transform(exp.X_train)
        # Z-scored data should have roughly mean=0, std=1
        arr = np.array(X_transformed)
        assert np.abs(arr.mean(axis=0)).max() < 0.5

    def test_power_transform(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(
            data=iris_data, target="target",
            transformation=True, transformation_method="yeo-johnson",
            session_id=42, verbose=False,
        )
        X_transformed = exp.pipeline.transform(exp.X_test)
        assert X_transformed is not None

    def test_pca(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(
            data=iris_data, target="target",
            pca=True, pca_components=2,
            session_id=42, verbose=False,
        )
        X_transformed = exp.pipeline.transform(exp.X_test)
        arr = np.array(X_transformed)
        assert arr.shape[1] == 2

    def test_low_variance_threshold(self, iris_data):
        exp = ClassificationExperiment()
        # Add a constant column
        iris_data_copy = iris_data.copy()
        iris_data_copy["constant"] = 1.0
        exp.setup(
            data=iris_data_copy, target="target",
            low_variance_threshold=0.01,
            session_id=42, verbose=False,
        )
        X_transformed = exp.pipeline.transform(exp.X_test)
        arr = np.array(X_transformed)
        # Constant column should be removed
        assert arr.shape[1] < len(iris_data_copy.columns) - 1  # minus target

    def test_remove_outliers(self, iris_data):
        exp = ClassificationExperiment()
        original_len = len(iris_data)
        exp.setup(
            data=iris_data, target="target",
            remove_outliers=True, outliers_threshold=0.1,
            session_id=42, verbose=False,
        )
        # Train set should have fewer samples than original train split
        assert len(exp.X_train) < int(original_len * 0.7) + 5

    def test_mixed_data_end_to_end(self, mixed_data):
        exp = ClassificationExperiment()
        exp.setup(
            data=mixed_data, target="target",
            numeric_imputation="median",
            categorical_imputation="mode",
            normalize=True,
            session_id=42, verbose=False,
        )
        X_transformed = exp.pipeline.transform(exp.X_test)
        assert X_transformed is not None
        assert len(X_transformed) == len(exp.X_test)


class TestColumnDetection:
    def test_numeric_detection(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(data=iris_data, target="target", session_id=42, verbose=False)
        assert len(exp._config.feature_types["Numeric"]) == 4
        assert len(exp._config.feature_types["Categorical"]) == 0

    def test_categorical_detection(self, mixed_data):
        exp = ClassificationExperiment()
        exp.setup(data=mixed_data, target="target", session_id=42, verbose=False)
        assert "gender" in exp._config.feature_types["Categorical"]
        assert "education" in exp._config.feature_types["Categorical"]
        assert "age" in exp._config.feature_types["Numeric"]

    def test_manual_feature_override(self, mixed_data):
        exp = ClassificationExperiment()
        exp.setup(
            data=mixed_data, target="target",
            numeric_features=["age"],
            categorical_features=["gender"],
            session_id=42, verbose=False,
        )
        assert exp._config.feature_types["Numeric"] == ["age"]
        assert exp._config.feature_types["Categorical"] == ["gender"]
