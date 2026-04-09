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
uv add imbalanced-learn                 # class imbalance handling
uv add shap                             # model interpretation
uv add mlflow                           # experiment tracking
uv add optuna                           # advanced hyperparameter tuning
```

## Quick Start

```python
from pycaret_redux import ClassificationExperiment
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer(as_frame=True).frame

# 1. Setup
exp = ClassificationExperiment()
exp.setup(data=data, target="target", session_id=42)

# 2. Compare models
best = exp.compare_models()

# 3. Tune the best model
tuned = exp.tune_model(best)

# 4. Predict
preds = exp.predict_model(tuned)

# 5. Save for deployment
exp.save_model(tuned, "my_model")
```

## Guide

### Setup & Data Profiling

`setup()` initializes the experiment: validates data, splits train/test, builds the preprocessing pipeline, and registers available models and metrics.

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
    normalize_method="zscore",       # zscore, minmax, maxabs, robust
    transformation=True,
    transformation_method="yeo-johnson",

    # Feature engineering
    polynomial_features=True,
    polynomial_degree=2,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.9,

    # Missing values
    numeric_imputation="mean",       # mean, median, mode, knn
    categorical_imputation="mode",

    # Outliers
    remove_outliers=True,
    outliers_threshold=0.05,

    # Imbalance
    fix_imbalance=True,              # requires: uv add imbalanced-learn
    fix_imbalance_method="SMOTE",

    # Profile the data
    profile=True,                    # shows class distribution, missing values, correlations
)
```

Use `profile=True` to see a summary of your data: target distribution, missing values, numeric statistics, categorical cardinality, top correlations, and class imbalance warnings.

### Comparing Models

Train and rank all available models in one call:

```python
best = exp.compare_models(sort="Accuracy", turbo=True)
```

The output is a styled table with yellow-highlighted best scores per metric, training time, and a progress bar.

```python
# Compare specific models only
best = exp.compare_models(include=["lr", "rf", "dt", "knn", "gbc"])

# Return top 3 models
top3 = exp.compare_models(n_select=3)

# Set a time budget (minutes)
best = exp.compare_models(budget_time=2.0)
```

### Creating & Tuning Models

```python
# Train a specific model by ID
lr = exp.create_model("lr")
rf = exp.create_model("rf")
dt = exp.create_model("dt", max_depth=5)  # pass extra params

# Tune hyperparameters
tuned_rf = exp.tune_model(rf, n_iter=50, optimize="F1")

# Use a custom search grid
tuned_dt = exp.tune_model(dt, custom_grid={"max_depth": [3, 5, 7, 10, 15]})
```

Available model IDs: `lr`, `knn`, `nb`, `dt`, `svm`, `rbfsvm`, `gpc`, `mlp`, `ridge`, `rf`, `qda`, `ada`, `gbc`, `lda`, `et`, `dummy`, and optionally `xgboost`, `lightgbm`, `catboost`.

### Ensembles

```python
# Soft voting (average probabilities)
blended = exp.blend_models([lr, rf, dt])

# Hard voting (majority rule)
blended_hard = exp.blend_models([lr, rf, dt], method="hard")

# Stacking with a meta-learner
stacked = exp.stack_models([lr, rf, dt])

# Bagging
bagged = exp.ensemble_model(dt, n_estimators=20)
```

### AutoML

Run the full pipeline in one call — compare, tune, ensemble:

```python
best_model = exp.automl(
    optimize="Accuracy",
    n_top=3,               # tune top 3 models
    tune_n_iter=20,        # 20 iterations per model
    ensemble="blend",      # or "stack"
)
```

### Plotting

```python
exp.plot_model(model, plot="auc")              # ROC/AUC curve
exp.plot_model(model, plot="confusion_matrix") # Confusion matrix
exp.plot_model(model, plot="pr")               # Precision-recall curve
exp.plot_model(model, plot="threshold")        # Metrics vs threshold (binary)
exp.plot_model(model, plot="feature")          # Feature importance
exp.plot_model(model, plot="class_report")     # Classification report heatmap
exp.plot_model(model, plot="calibration")      # Calibration curve
exp.plot_model(model, plot="learning")         # Learning curve
exp.plot_model(model, plot="vc")               # Validation curve
exp.plot_model(model, plot="lift")             # Lift chart (binary)
exp.plot_model(model, plot="gain")             # Gain chart (binary)
exp.plot_model(model, plot="ks")               # KS statistic (binary)
exp.plot_model(model, plot="error")            # Prediction error

# Save a plot to file
exp.plot_model(model, plot="auc", save="roc_curve.png")
```

### Evaluation & Predictions

```python
# Evaluate on test set
exp.evaluate_model(model)

# Predict on test set
preds = exp.predict_model(model)

# Predict on new data
preds = exp.predict_model(model, data=new_df)

# Predict with raw probability scores
preds = exp.predict_model(model, raw_score=True)

# Custom probability threshold (binary only)
preds = exp.predict_model(model, probability_threshold=0.7)
```

### Probability Calibration & Threshold Optimization

```python
# Calibrate predicted probabilities
calibrated = exp.calibrate_model(model, method="sigmoid")  # or "isotonic"

# Find optimal decision threshold
model, threshold = exp.optimize_threshold(model, optimize="F1")
preds = exp.predict_model(model, probability_threshold=threshold)
```

### Statistical Model Comparison

Test whether one model is *significantly* better than another:

```python
result = exp.compare_model_stats(lr, rf, metric="Accuracy", test="wilcoxon")
# Prints: p-value, conclusion ("Model A is significantly better" or "No significant difference")
```

### Data Drift Detection

Check if new data has drifted from the training distribution:

```python
drift_report = exp.check_drift(new_data)
# Returns DataFrame: Feature, Type, Test, Statistic, P-Value, Drifted
```

Uses KS test for numeric features and Chi-squared for categorical features.

### Finalize, Save & Load

```python
# Retrain on full data (train + test) before deployment
final_model = exp.finalize_model(model)

# Save (bundles preprocessing pipeline + model in one file)
exp.save_model(final_model, "production_model")

# Load
loaded = exp.load_model("production_model")

# For standalone deployment (no experiment needed)
from pycaret_redux.persistence.serialization import load_model, predict_from_artifact
artifact = load_model("production_model")
predictions = predict_from_artifact(artifact, raw_data)
```

### Inspecting State

```python
# Get last CV or comparison results
exp.pull()

# Inspect any config variable
exp.get_config("seed")
exp.get_config("X_train")
exp.get_config("feature_types")

# Get the preprocessing pipeline
pipeline = exp.get_pipeline()

# List available models and metrics
exp.models()
exp.get_metrics()
```

### Custom Metrics

```python
from sklearn.metrics import balanced_accuracy_score

exp.add_metric(id="bal_acc", name="Balanced Accuracy", score_func=balanced_accuracy_score)

# Now used in compare_models, create_model, etc.
exp.create_model("rf")

exp.remove_metric("bal_acc")
```

### Cross-Validation Strategies

```python
exp.setup(data=df, target="target", fold_strategy="stratifiedkfold", fold=10)  # default
exp.setup(data=df, target="target", fold_strategy="kfold", fold=5)
exp.setup(data=df, target="target", fold_strategy="groupkfold", fold=5)
exp.setup(data=df, target="target", fold_strategy="timeseries", fold=5)
exp.setup(data=df, target="target", fold_strategy="repeatedstratifiedkfold", fold=5)
exp.setup(data=df, target="target", fold_strategy="repeatedkfold", fold=5)

# Or pass any sklearn CV splitter
from sklearn.model_selection import LeaveOneOut
exp.setup(data=df, target="target", fold_strategy=LeaveOneOut())
```

### Logging

PyCaret Redux uses Python's standard `logging` module. Enable it to see what's happening under the hood:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now all setup, training, tuning, and persistence steps are logged
exp.setup(data=df, target="target")
```

## What Changed from Original PyCaret

| | Original PyCaret | PyCaret Redux |
|---|---|---|
| Architecture | 5-level inheritance, 220K-line monoliths | Single class, composition, ~3,500 lines |
| Pipeline | imblearn Pipeline | sklearn Pipeline + ColumnTransformer |
| Model registry | 23 container classes | Data-driven `ModelEntry` dataclasses |
| API | Functional + OOP | OOP only (no global state) |
| Deployment | Save model only | Bundles pipeline + model in one file |
| Python | 3.8+ | 3.12+ |
| Dependencies | numpy<1.27, pandas<2.2, sklearn<1.5 | numpy 2.x, pandas 2.x, sklearn 1.6+ |

## Development

```bash
uv sync
uv add --dev pytest ruff mypy

# Run tests (97 passing)
uv run pytest

# Lint & format
uv run ruff check pycaret_redux/
uv run ruff format pycaret_redux/
```

## License

MIT
