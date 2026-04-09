# PyCaret Redux

A modernized, low-code classification library inspired by [PyCaret](https://github.com/pycaret/pycaret).

PyCaret Redux takes the beloved PyCaret workflow and rebuilds it from scratch with clean internals, modern dependencies, and no deep inheritance chains.

## Quick Start

```python
from pycaret_redux import ClassificationExperiment

exp = ClassificationExperiment()
exp.setup(data=df, target="target", session_id=42)
best = exp.compare_models()
tuned = exp.tune_model(best)
exp.plot_model(tuned, plot="auc")
preds = exp.predict_model(tuned)
exp.save_model(tuned, "my_model")
```

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

## Features

| Feature | API |
|---|---|
| Setup & preprocessing | `exp.setup(data, target, normalize=True, ...)` |
| Compare models | `exp.compare_models(sort="Accuracy")` |
| Train a model | `exp.create_model("rf")` |
| Hyperparameter tuning | `exp.tune_model(model, optimize="F1")` |
| Voting ensemble | `exp.blend_models([model1, model2])` |
| Stacking ensemble | `exp.stack_models([model1, model2])` |
| Bagging ensemble | `exp.ensemble_model(model, n_estimators=10)` |
| Predictions | `exp.predict_model(model, data=new_data)` |
| 13 plot types | `exp.plot_model(model, plot="auc")` |
| Probability calibration | `exp.calibrate_model(model)` |
| Threshold optimization | `exp.optimize_threshold(model, optimize="F1")` |
| Finalize & deploy | `exp.finalize_model(model)` |
| Save / load | `exp.save_model(model, "name")` |
| Custom metrics | `exp.add_metric(id, name, score_func)` |
| Experiment tracking | MLflow integration |

### Available Models

18+ classifiers out of the box: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM, Random Forest, Gradient Boosting, AdaBoost, Extra Trees, LDA, QDA, Ridge, MLP, Gaussian Process, and optionally XGBoost, LightGBM, CatBoost.

### Available Plots

`auc` | `confusion_matrix` | `pr` | `threshold` | `error` | `class_report` | `feature` | `learning` | `vc` | `calibration` | `lift` | `gain` | `ks`

## What Changed from Original PyCaret

| | Original PyCaret | PyCaret Redux |
|---|---|---|
| Architecture | 5-level inheritance, 220K-line monoliths | Single class, composition, ~2,500 lines |
| Pipeline | imblearn Pipeline | sklearn Pipeline + ColumnTransformer |
| Model registry | 23 container classes | Data-driven `ModelEntry` dataclasses |
| API | Functional + OOP | OOP only (no global state) |
| Python | 3.8+ | 3.12+ |
| Dependencies | numpy<1.27, pandas<2.2, sklearn<1.5 | numpy 2.x, pandas 2.x, sklearn 1.6+ |

## Development

```bash
uv sync
uv add --dev pytest ruff mypy

# Run tests (95 passing)
uv run pytest

# Lint
uv run ruff check pycaret_redux/
```

## License

MIT
