# PyCaret Redux

A modernized, low-code classification library inspired by [PyCaret](https://github.com/pycaret/pycaret).

PyCaret Redux takes the beloved PyCaret workflow and rebuilds it from scratch with clean internals, modern dependencies, and no deep inheritance chains.

## Installation

```bash
# Clone and install
git clone https://github.com/datanerdie/pycaret-redux.git
cd pycaret-redux
uv sync

# Optional extras
uv add xgboost lightgbm catboost       # boosting models
uv add imbalanced-learn                 # class imbalance handling (SMOTE etc.)
uv add shap                             # model interpretation
uv add mlflow                           # experiment tracking
uv add optuna                           # Optuna hyperparameter tuning
```

## Quick Start

```python
from pycaret_redux import ClassificationExperiment
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True).frame

exp = ClassificationExperiment()
exp.setup(data=data, target="target", session_id=42)
best = exp.compare_models()
tuned = exp.tune_model(best)
preds = exp.predict_model(tuned)
exp.save_model(tuned, "my_model")
```

Or use AutoML for a fully automated pipeline:

```python
best_model = exp.automl(optimize="Accuracy", n_top=3, ensemble="blend")
```

## Features at a Glance

| Category | Methods |
|----------|---------|
| **Setup** | `setup()` with 35+ preprocessing options, `profile=True` for data profiling |
| **Modeling** | `compare_models()`, `create_model()`, `tune_model()`, `automl()` |
| **Ensembles** | `blend_models()`, `stack_models()`, `ensemble_model()` |
| **Evaluation** | `evaluate_model()`, `predict_model()`, `pull()` |
| **Plots** | `plot_model()` — 14 plot types |
| **Calibration** | `calibrate_model()`, `optimize_threshold()` |
| **Statistics** | `compare_model_stats()`, `compare_multiple_stats()`, `compare_5x2cv()` |
| **Diagnostics** | `diagnose_bias_variance()`, `get_oob_score()`, `check_drift()` |
| **Advanced CV** | `nested_cv()` + 6 fold strategies |
| **Deployment** | `finalize_model()`, `save_model()`, `load_model()`, `predict_from_artifact()` |
| **Inspection** | `models()`, `get_metrics()`, `get_config()`, `get_pipeline()` |
| **Extensibility** | `add_metric()`, `remove_metric()`, custom estimators |

## Guide

### Setup & Data Profiling

```python
exp = ClassificationExperiment()
exp.setup(
    data=df,
    target="target",
    train_size=0.8,
    session_id=42,
    fold=5,

    # Preprocessing
    normalize=True,
    normalize_method="zscore",         # zscore, minmax, maxabs, robust
    transformation=True,
    transformation_method="yeo-johnson",  # yeo-johnson, quantile, or "auto" (per-feature skew detection)

    # Feature engineering
    polynomial_features=True,
    polynomial_degree=2,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.9,

    # Dimensionality reduction
    pca=True,
    pca_method="linear",               # linear, kernel, incremental, random, sparse_random, lda

    # Feature selection
    feature_selection=True,
    feature_selection_method="classic", # classic (SelectKBest), sequential, rfe

    # Missing values
    numeric_imputation="mean",         # mean, median, mode, knn
    categorical_imputation="mode",

    # Outliers
    remove_outliers=True,
    outliers_threshold=0.05,

    # Imbalance handling
    fix_imbalance=True,
    fix_imbalance_method="SMOTE",      # SMOTE, ADASYN, RandomOverSampler, or "class_weight"

    # Data profiling
    profile=True,                      # shows class distribution, missing values, correlations, warnings
)
```

### Comparing Models

```python
best = exp.compare_models(sort="Accuracy", turbo=True)

# Compare specific models (including HistGradientBoosting)
best = exp.compare_models(include=["lr", "rf", "hgbc", "gbc", "et"])

# Return top 3
top3 = exp.compare_models(n_select=3)

# Time budget (minutes)
best = exp.compare_models(budget_time=2.0)

# Pull the comparison table
exp.pull()
```

### Creating & Tuning Models

```python
lr = exp.create_model("lr")
rf = exp.create_model("rf")
hgbc = exp.create_model("hgbc")     # HistGradientBoosting (fast, handles missing values)
dt = exp.create_model("dt", max_depth=5)

# Standard random search
tuned = exp.tune_model(rf, n_iter=50, optimize="F1")

# HalvingRandomSearchCV (successive halving — much faster)
tuned = exp.tune_model(rf, search_library="halving", optimize="Accuracy")

# Custom grid
tuned = exp.tune_model(dt, custom_grid={"max_depth": [3, 5, 7, 10]})
```

Available models: `lr`, `knn`, `nb`, `dt`, `svm`, `rbfsvm`, `gpc`, `mlp`, `ridge`, `rf`, `qda`, `ada`, `gbc`, `hgbc`, `lda`, `et`, `dummy`, and optionally `xgboost`, `lightgbm`, `catboost`.

### Ensembles

```python
blended = exp.blend_models([lr, rf, dt])                  # soft voting
blended = exp.blend_models([lr, rf, dt], method="hard")   # hard voting
stacked = exp.stack_models([lr, rf, dt])                  # stacking
bagged = exp.ensemble_model(dt, n_estimators=20)          # bagging
```

### AutoML

```python
best_model = exp.automl(
    optimize="Accuracy",
    n_top=3,               # tune top 3 models
    tune_n_iter=20,        # 20 iterations per model
    ensemble="blend",      # or "stack"
)
```

### Plots (14 types)

```python
exp.plot_model(model, plot="auc")              # ROC/AUC curve
exp.plot_model(model, plot="confusion_matrix") # Confusion matrix
exp.plot_model(model, plot="pr")               # Precision-recall curve
exp.plot_model(model, plot="threshold")        # Metrics vs threshold (binary)
exp.plot_model(model, plot="feature")          # Feature importance
exp.plot_model(model, plot="permutation")      # Permutation importance (model-agnostic)
exp.plot_model(model, plot="class_report")     # Classification report heatmap
exp.plot_model(model, plot="calibration")      # Calibration curve
exp.plot_model(model, plot="learning")         # Learning curve
exp.plot_model(model, plot="vc")               # Validation curve
exp.plot_model(model, plot="lift")             # Lift chart (binary)
exp.plot_model(model, plot="gain")             # Gain chart (binary)
exp.plot_model(model, plot="ks")               # KS statistic (binary)
exp.plot_model(model, plot="error")            # Prediction error

exp.plot_model(model, plot="auc", save="roc.png")  # save to file
```

### Evaluation & Predictions

```python
exp.evaluate_model(model)
preds = exp.predict_model(model)
preds = exp.predict_model(model, data=new_df)
preds = exp.predict_model(model, raw_score=True)
preds = exp.predict_model(model, probability_threshold=0.7)
```

### Calibration & Threshold Optimization

```python
calibrated = exp.calibrate_model(model, method="sigmoid")  # or "isotonic"
model, threshold = exp.optimize_threshold(model, optimize="F1")
```

### Statistical Model Comparison

Five statistical tests for rigorous model comparison:

```python
# McNemar's test (pairwise, on prediction disagreements)
exp.compare_model_stats(lr, rf, test="mcnemar")

# Wilcoxon signed-rank test (pairwise, on CV fold scores)
exp.compare_model_stats(lr, rf, metric="Accuracy", test="wilcoxon")

# Paired t-test (pairwise, on CV fold scores)
exp.compare_model_stats(lr, rf, test="ttest")

# Cochran's Q test (3+ classifiers, omnibus test)
exp.compare_multiple_stats([lr, rf, dt, hgbc])

# 5x2cv paired F-test (Dietterich's method — most powerful pairwise test)
exp.compare_5x2cv(lr, rf)
```

### Nested Cross-Validation

Unbiased evaluation: inner loop tunes, outer loop evaluates.

```python
scores = exp.nested_cv("rf", n_iter=20)
scores = exp.nested_cv("lr", param_grid={"C": [0.01, 0.1, 1, 10]})
```

### Bias-Variance Diagnostic

```python
diag = exp.diagnose_bias_variance(model)
# Prints: train/val scores, gap, diagnosis (high bias/high variance/good fit), suggestion
```

### OOB Evaluation

Free validation for forest/bagging models:

```python
oob_score = exp.get_oob_score(rf)  # uses out-of-bag samples (~37% per tree)
```

### Data Drift Detection

```python
drift_report = exp.check_drift(new_data)
# Returns DataFrame: Feature, Type, Test (KS/Chi2), Statistic, P-Value, Drifted
```

### Finalize, Save & Load

```python
final_model = exp.finalize_model(model)

# Save (bundles preprocessing pipeline + model in one .joblib file)
exp.save_model(final_model, "production_model")

# Load (returns the estimator)
loaded = exp.load_model("production_model")

# Standalone deployment (no experiment needed)
from pycaret_redux.persistence.serialization import load_model, predict_from_artifact
artifact = load_model("production_model")
predictions = predict_from_artifact(artifact, raw_data)
```

### Inspecting State

```python
exp.pull()                          # last CV or comparison result
exp.get_config("seed")              # any config variable
exp.get_config("X_train")           # training data
exp.get_pipeline()                  # preprocessing pipeline
exp.models()                        # available models
exp.get_metrics()                   # available metrics
```

### Custom Metrics

```python
from sklearn.metrics import balanced_accuracy_score

exp.add_metric(id="bal_acc", name="Balanced Accuracy", score_func=balanced_accuracy_score)
exp.create_model("rf")  # now includes Balanced Accuracy in CV results
exp.remove_metric("bal_acc")
```

### Cross-Validation Strategies

```python
exp.setup(data=df, target="target", fold_strategy="stratifiedkfold")        # default
exp.setup(data=df, target="target", fold_strategy="kfold")
exp.setup(data=df, target="target", fold_strategy="groupkfold")
exp.setup(data=df, target="target", fold_strategy="timeseries")
exp.setup(data=df, target="target", fold_strategy="repeatedstratifiedkfold")
exp.setup(data=df, target="target", fold_strategy="repeatedkfold")

# Or any sklearn CV splitter
from sklearn.model_selection import LeaveOneOut
exp.setup(data=df, target="target", fold_strategy=LeaveOneOut())
```

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
exp.setup(data=df, target="target")  # all steps now logged
```

## What Changed from Original PyCaret

| | Original PyCaret | PyCaret Redux |
|---|---|---|
| Architecture | 5-level inheritance, 220K-line monoliths | Single class, composition |
| Pipeline | imblearn Pipeline | sklearn Pipeline + ColumnTransformer |
| Model registry | 23 container classes | Data-driven `ModelEntry` dataclasses |
| API | Functional + OOP | OOP only (no global state) |
| Deployment | Save model only | Bundles pipeline + model in one file |
| Statistical tests | None | 5 tests (McNemar, Wilcoxon, t-test, Cochran's Q, 5x2cv F) |
| Diagnostics | None | Bias-variance, OOB, drift detection, nested CV |
| Python | 3.8+ | 3.12+ |
| Dependencies | numpy<1.27, pandas<2.2, sklearn<1.5 | numpy 2.x, pandas 2.x, sklearn 1.6+ |

## Development

```bash
uv sync
uv add --dev pytest ruff mypy

# Run tests (107 passing)
uv run pytest

# Lint & format
uv run ruff check pycaret_redux/
uv run ruff format pycaret_redux/
```

## License

MIT
