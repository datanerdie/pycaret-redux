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
| **Evaluation** | `evaluate_model()` with bootstrap CI, `predict_model()`, `pull()` |
| **Plots** | `plot_model()` — 14 plot types including permutation importance |
| **Calibration** | `calibrate_model()` with Brier/log loss, `optimize_threshold()` |
| **Statistics** | 6 tests: McNemar, Wilcoxon, t-test, Cochran's Q, 5x2cv F, DeLong |
| **Diagnostics** | `diagnose_bias_variance()`, `get_oob_score()`, `check_drift()` |
| **Advanced CV** | `nested_cv()` + 6 fold strategies + 95% confidence intervals |
| **Deployment** | `finalize_model()`, `save_model()`, `load_model()`, `predict_from_artifact()` |
| **Inspection** | `models()`, `get_metrics()`, `get_config()`, `get_pipeline()` |
| **Extensibility** | `add_metric()`, `remove_metric()`, custom estimators |
| **Display** | PyCaret-style styled tables, auto dark/light theme detection |

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
    normalize_method="zscore",            # zscore, minmax, maxabs, robust
    transformation=True,
    transformation_method="yeo-johnson",  # yeo-johnson, quantile, or "auto" (per-feature skew detection)

    # Categorical encoding
    max_encoding_ohe=25,                  # OHE for features with <=25 categories
    drop_first_ohe=True,                  # drop first column to avoid dummy trap (for LR, Ridge)
    # Binary features (2 values) are automatically ordinal-encoded, not OHE'd

    # Feature engineering
    polynomial_features=True,
    polynomial_degree=2,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.9,

    # Dimensionality reduction
    pca=True,
    pca_method="linear",                  # linear, kernel, incremental, random, sparse_random, lda

    # Feature selection
    feature_selection=True,
    feature_selection_method="classic",   # classic (SelectKBest), sequential, rfe

    # Missing values
    numeric_imputation="mean",            # mean, median, mode, knn
    categorical_imputation="mode",

    # Outliers
    remove_outliers=True,
    outliers_threshold=0.05,

    # Imbalance handling
    fix_imbalance=True,
    fix_imbalance_method="SMOTE",         # SMOTE, ADASYN, RandomOverSampler, or "class_weight"

    # Data profiling
    profile=True,                         # class distribution, missing values, correlations, warnings
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
exp.plot_model(model, plot="learning")         # Learning curve (bias/variance)
exp.plot_model(model, plot="vc")               # Validation curve
exp.plot_model(model, plot="lift")             # Lift chart (binary)
exp.plot_model(model, plot="gain")             # Gain chart (binary)
exp.plot_model(model, plot="ks")               # KS statistic (binary)
exp.plot_model(model, plot="error")            # Prediction error

exp.plot_model(model, plot="auc", save="roc.png")  # save to file
```

### Evaluation & Predictions

`evaluate_model` computes bootstrap 95% confidence intervals (1000 resamples):

```python
exp.evaluate_model(model)
# Output: Metric | Score | 95% CI
#         Accuracy  0.9561  [0.9211, 0.9825]

preds = exp.predict_model(model)
preds = exp.predict_model(model, data=new_df)
preds = exp.predict_model(model, raw_score=True)
preds = exp.predict_model(model, probability_threshold=0.7)
```

### Calibration & Threshold Optimization

`calibrate_model` shows before/after Brier Score and Log Loss:

```python
calibrated = exp.calibrate_model(model, method="sigmoid")  # or "isotonic"
# Output:          Brier Score   Log Loss
#         Before   0.0412        0.1389
#         After    0.0398        0.1301

model, threshold = exp.optimize_threshold(model, optimize="F1")
```

### Statistical Model Comparison

Six statistical tests for rigorous model comparison:

```python
# McNemar's test (pairwise, on prediction disagreements)
exp.compare_model_stats(lr, rf, test="mcnemar")

# DeLong test (pairwise, compares AUC curves directly)
exp.compare_model_stats(lr, rf, test="delong")

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

Unbiased evaluation: inner loop tunes, outer loop evaluates. Results include 95% CI.

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
| Encoding | Fixed OHE for all categoricals | Smart: binary→ordinal, multi→OHE, optional drop_first |
| API | Functional + OOP | OOP only (no global state) |
| Deployment | Save model only | Bundles pipeline + model in one file |
| Evaluation | Point estimates only | Bootstrap 95% CI on all metrics |
| Calibration | Basic | Shows before/after Brier Score + Log Loss |
| Statistical tests | None | 6 tests (McNemar, DeLong, Wilcoxon, t-test, Cochran's Q, 5x2cv F) |
| Diagnostics | None | Bias-variance, OOB, drift detection, nested CV |
| Tuning | Grid/Random only | + HalvingRandomSearchCV (successive halving) |
| Display | Light theme only | Auto dark/light theme detection |
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
