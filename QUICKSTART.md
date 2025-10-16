# Quick Start Guide

Get started with ml_store in 5 minutes!

## Installation

```bash
pip install ml-store
```

## Basic Usage

### 1. Train a Classifier

```python
import polars as pl
from ml_store import create_modelling_data, train_model, evaluate_model

# Load your data
df = pl.read_csv("data.csv")

# Create configuration
config = {
    "model_type": "classifier",
    "target_column": "target",
    "feature_columns": ["feature1", "feature2", "feature3"],
    "categorical_features": ["feature1"],
    "train_ratio": 0.7,
    "test_ratio": 0.3,
    "random_seed": 42,
    "model_params": {
        "learning_params": {
            "iterations": 1000,
            "learning_rate": 0.1,
            "depth": 6
        }
    }
}

# Prepare data
X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(df, config)

# Train model
model = train_model(X_train, y_train, config=config, categorical_features=cat_indices)

# Evaluate
metrics = evaluate_model(model, X_test, y_test, config=config)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

### 2. Config-Driven Workflow

**config.json:**
```json
{
  "model_type": "classifier",
  "target_column": "target",
  "data_path": "data/train.csv",
  "feature_columns": ["age", "income", "score"],
  "categorical_features": [],
  "train_ratio": 0.7,
  "test_ratio": 0.3,
  "random_seed": 42,
  "plotting_backend": "plotly",
  "model_params": {
    "learning_params": {
      "iterations": 1000,
      "learning_rate": 0.1
    }
  }
}
```

**train.py:**
```python
from ml_store import load_config, load_data, create_modelling_data, train_model

# Load config
config = load_config("config.json")

# Load data
df = load_data(config)

# Prepare and train
X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(df, config)
model = train_model(X_train, y_train, config=config, categorical_features=cat_indices)

# Save model
from ml_store import save_model
save_model(model, "model.cbm", config=config)
```

### 3. Model Evaluation

```python
from ml_store import ml_evaluation

# Get feature importance
importance = ml_evaluation.get_feature_importance(model, config=config)
print(importance.to_polars().head(10))

# Plot feature importance
fig = ml_evaluation.plot_feature_importance(importance, top_n=15, config=config)
fig.show()  # If using plotly backend

# Calculate SHAP values
shap_result = ml_evaluation.calculate_shap_values(model, X_test, config=config)

# Plot SHAP summary
fig = ml_evaluation.plot_shap_summary(shap_result, config=config)
fig.show()

# Create full evaluation report
report = ml_evaluation.create_evaluation_report(
    model,
    X_test,
    y_test,
    config=config,
    output_dir='./evaluation_output',
    top_n_features=15
)
```

### 4. Load Data from SQL

**Snowflake:**
```python
config = {
    "data_path": "SELECT * FROM customers WHERE year = 2024",
    "data_type": "sql",
    "db_type": "snowflake",
    "db_connection": "snowflake://user:pass@account/database/schema?warehouse=wh",
    "model_type": "classifier",
    "target_column": "churn",
    "feature_columns": ["age", "tenure", "monthly_charges"]
}

df = load_data(config)
```

**MS SQL Server:**
```python
config = {
    "data_path": "SELECT * FROM sales WHERE date >= '2024-01-01'",
    "data_type": "sql",
    "db_type": "mssql",
    "db_connection": "mssql://user:pass@server/database",
    "model_type": "regressor",
    "target_column": "sales_amount",
    "feature_columns": ["price", "quantity", "discount"]
}

df = load_data(config)
```

### 5. Make Predictions

```python
from ml_store import load_model, predict

# Load saved model
model = load_model("model.cbm", model_type="classifier")

# Make predictions on new data
new_data = pl.read_csv("new_data.csv")
predictions = predict(model, new_data, config=config)

# Add predictions to dataframe
new_data = new_data.with_columns(pl.Series("predictions", predictions))
```

### 6. MLFlow Integration

```python
from ml_store import log_to_mlflow

# Train model
model = train_model(X_train, y_train, config=config, categorical_features=cat_indices)

# Log to MLFlow
config["mlflow"] = {
    "experiment_name": "my_experiment",
    "run_name": "run_1"
}

log_to_mlflow(
    model=model,
    config=config,
    metrics={"accuracy": 0.95, "auc": 0.98},
    params={"learning_rate": 0.1, "depth": 6}
)
```

## Common Patterns

### Pattern 1: Complete Pipeline

```python
from ml_store import *

# 1. Load config
config = load_config("config.json")

# 2. Load data
df = load_data(config)

# 3. Prepare data
X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(df, config)

# 4. Train
model = train_model(X_train, y_train, config=config, categorical_features=cat_indices)

# 5. Evaluate
metrics = evaluate_model(model, X_test, y_test, config=config)

# 6. Save
save_model(model, "model.cbm", config=config)

# 7. Log to MLFlow
log_to_mlflow(model, config, metrics)
```

### Pattern 2: Cross-Validation

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_seed=42)
scores = []

for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = train_model(X_train, y_train, config=config, categorical_features=cat_indices)
    metrics = evaluate_model(model, X_val, y_val, config=config)
    scores.append(metrics['accuracy'])

print(f"CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### Pattern 3: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'iterations': [500, 1000, 1500]
}

model = CatBoostClassifier(cat_features=cat_indices, verbose=0)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Update config with best params
config['model_params']['learning_params'].update(grid_search.best_params_)
```

## Next Steps

- Read the full [README](README.md)
- Explore [examples](examples/)
- Check out [SQL data loading guide](docs/SQL_DATA_LOADING.md)
- Review [API documentation](ml_store/)

## Tips

1. **Use config files** for reproducibility
2. **Set random seeds** for consistent results
3. **Log to MLFlow** for experiment tracking
4. **Use plotly backend** for interactive plots
5. **Leverage SQL loading** for large datasets
6. **Run tests** before deploying models

## Common Issues

**Import Error:**
```bash
pip install --upgrade ml-store
```

**MLFlow not tracking:**
```bash
mlflow ui  # Start MLFlow UI
# Then check http://localhost:5000
```

**SQL connection fails:**
```bash
# Install SQL dependencies
pip install ml-store[sql]
```

## Getting Help

- 📖 [Full Documentation](README.md)
- 🐛 [Report Issues](https://github.com/Wicks-Analytics/ml_store/issues)
- 💬 [Discussions](https://github.com/Wicks-Analytics/ml_store/discussions)
