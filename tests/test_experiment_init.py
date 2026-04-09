"""Tests for Phase 1: ClassificationExperiment skeleton and setup()."""

import numpy as np
import pandas as pd
import pytest

from pycaret_redux import ClassificationExperiment


@pytest.fixture
def iris_data():
    """Simple binary classification dataset."""
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame
    # Make binary: setosa vs not-setosa
    df["target"] = (df["target"] == 0).astype(int)
    return df


@pytest.fixture
def multiclass_data():
    """Multiclass classification dataset."""
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    return data.frame


class TestExperimentInit:
    def test_create_experiment(self):
        exp = ClassificationExperiment()
        assert not exp.is_setup_done
        assert exp.X_train is None
        assert exp.X_test is None

    def test_methods_fail_before_setup(self):
        exp = ClassificationExperiment()
        with pytest.raises(RuntimeError, match="Call setup"):
            exp.compare_models()
        with pytest.raises(RuntimeError, match="Call setup"):
            exp.create_model("lr")


class TestSetup:
    def test_basic_setup(self, iris_data):
        exp = ClassificationExperiment()
        result = exp.setup(data=iris_data, target="target", session_id=42, verbose=False)

        assert result is exp  # returns self
        assert exp.is_setup_done
        assert exp.seed == 42
        assert not exp.is_multiclass

    def test_train_test_split(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(data=iris_data, target="target", train_size=0.8, session_id=0, verbose=False)

        total = len(iris_data)
        assert len(exp.X_train) == pytest.approx(total * 0.8, abs=2)
        assert len(exp.X_test) == pytest.approx(total * 0.2, abs=2)
        assert len(exp.y_train) == len(exp.X_train)
        assert len(exp.y_test) == len(exp.X_test)

    def test_multiclass_detection(self, multiclass_data):
        exp = ClassificationExperiment()
        exp.setup(data=multiclass_data, target="target", session_id=0, verbose=False)
        assert exp.is_multiclass

    def test_target_by_index(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(data=iris_data, target=-1, session_id=0, verbose=False)
        assert exp._config.target_name == "target"

    def test_ignore_features(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(
            data=iris_data,
            target="target",
            ignore_features=["sepal length (cm)"],
            session_id=0,
            verbose=False,
        )
        assert "sepal length (cm)" not in exp.X_train.columns

    def test_custom_test_data(self, iris_data):
        train = iris_data.iloc[:120]
        test = iris_data.iloc[120:]
        exp = ClassificationExperiment()
        exp.setup(data=train, test_data=test, target="target", session_id=0, verbose=False)
        assert len(exp.X_train) == 120
        assert len(exp.X_test) == 30

    def test_numpy_input(self):
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        data = np.column_stack([X, y])

        exp = ClassificationExperiment()
        exp.setup(data=data, target=-1, session_id=0, verbose=False)
        assert exp.is_setup_done
        assert len(exp.X_train) + len(exp.X_test) == 100

    def test_fold_generator_created(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(data=iris_data, target="target", fold=5, session_id=0, verbose=False)
        assert exp._config.fold_generator is not None
        assert exp._config.fold_generator.n_splits == 5


class TestSetupValidation:
    def test_no_data_raises(self):
        exp = ClassificationExperiment()
        with pytest.raises(ValueError, match="data parameter is required"):
            exp.setup()

    def test_invalid_train_size(self, iris_data):
        exp = ClassificationExperiment()
        with pytest.raises(ValueError, match="train_size"):
            exp.setup(data=iris_data, target="target", train_size=1.5)

    def test_invalid_fold(self, iris_data):
        exp = ClassificationExperiment()
        with pytest.raises(ValueError, match="fold"):
            exp.setup(data=iris_data, target="target", fold=1)

    def test_invalid_target(self, iris_data):
        exp = ClassificationExperiment()
        with pytest.raises(ValueError, match="not found"):
            exp.setup(data=iris_data, target="nonexistent")

    def test_invalid_fold_strategy(self, iris_data):
        exp = ClassificationExperiment()
        with pytest.raises(ValueError, match="Unknown fold_strategy"):
            exp.setup(data=iris_data, target="target", fold_strategy="invalid")
