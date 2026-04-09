"""Tests for Phase 3: Model + Metric registries."""

import pandas as pd
import pytest

from pycaret_redux import ClassificationExperiment
from pycaret_redux.metrics.registry import MetricEntry, MetricRegistry
from pycaret_redux.models.registry import ModelEntry, ModelRegistry


@pytest.fixture
def iris_data():
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame
    df["target"] = (df["target"] == 0).astype(int)
    return df


class TestModelRegistry:
    def test_register_defaults(self):
        registry = ModelRegistry(seed=42)
        registry.register_defaults()
        assert len(registry) >= 16  # at least sklearn built-ins

    def test_get_model(self):
        registry = ModelRegistry(seed=42)
        registry.register_defaults()
        lr = registry.get("lr")
        assert lr.name == "Logistic Regression"
        assert lr.id == "lr"

    def test_unknown_model(self):
        registry = ModelRegistry(seed=42)
        registry.register_defaults()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_models(self):
        registry = ModelRegistry(seed=42)
        registry.register_defaults()
        df = registry.list_models()
        assert isinstance(df, pd.DataFrame)
        assert "lr" in df.index
        assert "Name" in df.columns

    def test_list_turbo_only(self):
        registry = ModelRegistry(seed=42)
        registry.register_defaults()
        all_df = registry.list_models()
        turbo_df = registry.list_models(turbo_only=True)
        assert len(turbo_df) <= len(all_df)
        # Non-turbo models like rbfsvm, gpc should be excluded
        if "rbfsvm" in all_df.index:
            assert "rbfsvm" not in turbo_df.index

    def test_create_instance(self):
        registry = ModelRegistry(seed=42)
        registry.register_defaults()
        model = registry.create_instance("lr")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_ids(self):
        registry = ModelRegistry(seed=42)
        registry.register_defaults()
        ids = registry.get_ids()
        assert "lr" in ids
        assert "rf" in ids
        # Special models should not be in default list
        assert "bagging" not in ids

    def test_contains(self):
        registry = ModelRegistry(seed=42)
        registry.register_defaults()
        assert "lr" in registry
        assert "nonexistent" not in registry


class TestMetricRegistry:
    def test_register_defaults(self):
        registry = MetricRegistry()
        registry.register_defaults()
        assert len(registry) == 7

    def test_get_metric(self):
        registry = MetricRegistry()
        registry.register_defaults()
        acc = registry.get("acc")
        assert acc.name == "Accuracy"

    def test_get_by_name(self):
        registry = MetricRegistry()
        registry.register_defaults()
        acc = registry.get("Accuracy")
        assert acc.id == "acc"

    def test_remove_metric(self):
        registry = MetricRegistry()
        registry.register_defaults()
        registry.remove("mcc")
        assert len(registry) == 6
        assert "mcc" not in registry

    def test_to_dataframe(self):
        registry = MetricRegistry()
        registry.register_defaults()
        df = registry.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "acc" in df.index

    def test_custom_metric(self):
        registry = MetricRegistry()
        registry.register_defaults()
        registry.register(MetricEntry(
            id="custom",
            name="Custom Metric",
            display_name="Custom",
            score_func=lambda y_true, y_pred: 1.0,
            is_custom=True,
        ))
        assert len(registry) == 8
        assert "custom" in registry


class TestExperimentRegistries:
    def test_models_method(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(data=iris_data, target="target", session_id=42, verbose=False)
        df = exp.models()
        assert isinstance(df, pd.DataFrame)
        assert "lr" in df.index

    def test_get_metrics_method(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(data=iris_data, target="target", session_id=42, verbose=False)
        df = exp.get_metrics()
        assert isinstance(df, pd.DataFrame)
        assert "acc" in df.index

    def test_add_remove_metric(self, iris_data):
        exp = ClassificationExperiment()
        exp.setup(data=iris_data, target="target", session_id=42, verbose=False)

        exp.add_metric(
            id="custom", name="Custom",
            score_func=lambda y, p: 1.0,
        )
        assert "custom" in exp.get_metrics().index

        exp.remove_metric("custom")
        assert "custom" not in exp.get_metrics().index

    def test_models_before_setup_raises(self):
        exp = ClassificationExperiment()
        with pytest.raises(RuntimeError):
            exp.models()
