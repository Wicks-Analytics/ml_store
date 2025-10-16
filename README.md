# ML Store

Machine learning module with standalone functions using **MLFlow**, **CatBoost**, **Polars**, and **analytics_store** for efficient model training, evaluation, and tracking.

## Features

- 🚀 **Standalone Functions**: Easy-to-use functions for the complete ML workflow
- 📊 **Polars Integration**: Fast data loading and processing with Polars DataFrames
- 🎯 **CatBoost Models**: Support for both classification and regression tasks
- 📈 **MLFlow Tracking**: Automatic experiment tracking and model logging
- ⚙️ **JSON Configuration**: Flexible model configuration via JSON files
- 💾 **Model Persistence**: Save and load trained models easily
- 📉 **analytics_store Integration**: Advanced model validation and visualization
- 🔄 **Streamlined API**: One-step data preparation with `create_modelling_data`

## Installation

This project uses `uv` for virtual environment management. Install dependencies:

```bash
uv sync
```

## Project Structure

```
ml_store/
├── ml_store/              # Main package
│   ├── __init__.py
│   └── ml_functions.py    # Standalone ML functions
├── config/                # Configuration files
│   ├── classifier_config.json
│   └── regressor_config.json
├── examples/              # Example scripts
│   └── train_example.py
├── models/                # Saved models (created at runtime)
├── data/                  # Data directory (create as needed)
└── pyproject.toml         # Project dependencies
```

## Quick Start

### 1. Create a Configuration File

Create a JSON config file (see `config/regressor_config.json` for example):

```json
{
  "model_type": "regressor",
  "target_column": "Insurance_Premium",
  "feature_columns": ["Driver_Age", "Driver_Experience", "Annual_Mileage"],
  "train_ratio": 0.8,
  "test_ratio": 0.2,
  "random_seed": 42,
  "model_params": {
    "learning_params": {
      "iterations": 1500,
      "learning_rate": 0.05,
      "depth": 6,
      "loss_function": "RMSE",
      "eval_metric": "MAE",
      "verbose": 100
    },
    "early_stopping_rounds": 10
  },
  "mlflow": {
    "experiment_name": "my_experiment",
    "run_name": "run_1"
  }
}
```

**Note:** 
- The top-level `random_seed` is automatically used for both data splitting and model training
- `learning_params` contains model initialization parameters
- `early_stopping_rounds` is a training parameter (passed to `.fit()`)

### 2. Simple Workflow (Recommended)

```python
import polars as pl
from ml_store import (
    load_config,
    create_modelling_data,
    train_model,
    evaluate_model,
    log_to_mlflow,
    save_model
)

# Load configuration
config = load_config("config/regressor_config.json")

# Load data
df = pl.read_csv("data/dataset.csv")

# One-step data preparation: split + feature extraction
X_train, y_train, X_test, y_test, features, cat_idx = create_modelling_data(df, config)

# Train model with validation
model = train_model(X_train, y_train, config, cat_idx, X_test, y_test)

# Evaluate (returns Polars DataFrame)
metrics_df = evaluate_model(model, X_test, y_test, config["model_type"])
print(metrics_df)

# Log to MLFlow
run_id = log_to_mlflow(model, config, metrics_df, features)

# Save model
save_model(model, "models/my_model.cbm")
```

### 3. Advanced Workflow (Manual Control)

```python
from ml_store import (
    load_config,
    assign_split,
    prepare_features,
    train_model,
    evaluate_model
)

# Load and split data manually
config = load_config("config/regressor_config.json")
df = pl.read_csv("data/dataset.csv")

# Split into train/test sets
train_df, test_df = assign_split(df, config)

# Prepare features separately
X_train, y_train, features, cat_idx = prepare_features(train_df, config, is_training=True)
X_test, y_test, _, _ = prepare_features(test_df, config, is_training=False)

# Train and evaluate
model = train_model(X_train, y_train, config, cat_idx, X_test, y_test)
metrics_df = evaluate_model(model, X_test, y_test, config["model_type"])
```

## Key Functions

### `create_modelling_data` (Recommended)

The **one-step data preparation** function that combines splitting and feature extraction:

```python
X_train, y_train, X_test, y_test, features, cat_idx = create_modelling_data(df, config)
```

**Returns:**
- `X_train`, `y_train`: Training features and target (numpy arrays)
- `X_test`, `y_test`: Test features and target (numpy arrays)
- `features`: List of feature names
- `cat_idx`: List of categorical feature indices

**Benefits:**
- ✅ Single function call for complete data preparation
- ✅ Handles both 2-way (train/test) and 3-way (train/test/holdout) splits
- ✅ Automatically extracts features and categorical indices
- ✅ Config-driven with sensible defaults

See `CREATE_MODELLING_DATA_GUIDE.md` for detailed documentation.

## Available Functions

### Configuration & Data

- **`load_config(config_path)`**: Load model configuration from JSON file
- **`create_modelling_data(df, config, ...)`**: One-step data preparation (split + feature extraction)
- **`assign_split(df, config, ...)`**: Split DataFrame into train/test/holdout sets
- **`prepare_features(df, config, is_training)`**: Prepare features and target from DataFrame

### Model Training & Evaluation

- **`train_model(X_train, y_train, config, categorical_features, X_val, y_val)`**: Train a CatBoost model
- **`evaluate_model(model, X, y, model_type)`**: Evaluate model performance (returns Polars DataFrame)
- **`predict(model, X, return_proba)`**: Make predictions

### MLFlow & Persistence

- **`log_to_mlflow(model, config, metrics_df, feature_names, additional_params)`**: Log model and metrics to MLFlow
- **`save_model(model, save_path, model_format)`**: Save model to disk
- **`load_model(model_path, model_type)`**: Load model from disk

## Configuration Options

### Required Fields

- `model_type`: `"classifier"` or `"regressor"`
- `target_column`: Name of the target column

### Optional Fields

- `feature_columns`: List of feature columns (if not specified, uses all columns except target)
- `categorical_features`: List of categorical feature names
- `random_seed`: Random seed for reproducibility (used for both splitting and training, default: 42)
- `split_column`: Name of column containing predefined split assignments
- `split_values`: Dictionary mapping "train", "test", "holdout" to column values
- `train_ratio`, `test_ratio`, `holdout_ratio`: Custom split ratios (defaults: 0.8, 0.2 for 2-way split)
- `model_params`: Model configuration (see below)
- `mlflow`: MLFlow configuration (experiment_name, run_name)

### Model Parameters Structure

The `model_params` section has two parts:

```json
{
  "model_params": {
    "learning_params": {
      "iterations": 1500,
      "learning_rate": 0.05,
      "depth": 6,
      "loss_function": "RMSE",
      "eval_metric": "MAE",
      "verbose": 100
    },
    "early_stopping_rounds": 10
  }
}
```

**`learning_params`** - Model initialization parameters:
- `iterations`: Number of boosting iterations
- `learning_rate`: Learning rate
- `depth`: Depth of trees
- `loss_function`: Loss function (e.g., "Logloss", "RMSE", "RMSEWithUncertainty")
- `eval_metric`: Evaluation metric (e.g., "AUC", "MAE")
- `verbose`: Verbosity level (0 = silent, 100 = print every 100 iterations)

**`early_stopping_rounds`** - Training parameter (default: 25):
- Number of iterations without improvement before stopping

See [CatBoost documentation](https://catboost.ai/docs/concepts/python-reference_parameters-list.html) for all available parameters.

**Note:** The top-level `random_seed` is automatically added to `learning_params` if not explicitly set.

## Metrics

The `evaluate_model` function returns a Polars DataFrame with metrics calculated using **analytics_store**.

### Classification Metrics (from analytics_store)

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- ROC AUC (binary classification only)

### Regression Metrics (from analytics_store)

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MSE (Mean Squared Error)

**Example:**
```python
metrics_df = evaluate_model(model, X_test, y_test, "regressor")
print(metrics_df)
# Output:
# ┌──────────┬──────────┬──────────┬──────────┐
# │ mae      │ rmse     │ r2       │ mse      │
# ├──────────┼──────────┼──────────┼──────────┤
# │ 45.23    │ 58.49    │ 0.8234   │ 3421.56  │
# └──────────┴──────────┴──────────┴──────────┘

# Access specific metric
mae = metrics_df["mae"][0]
```

## Data Splitting

The `assign_split` function provides flexible data splitting with two modes:

### 1. Random Splits

Split data randomly based on ratios with **priority order**:

**Ratios:**
1. **Config file ratios** (highest priority)
2. **Function parameters**
3. **Defaults** (0.7, 0.15, 0.15)

**Random Seed:**
1. **Config file** (highest priority)
2. **Function parameter**
3. **Default** (42)

**Option A: Use config file ratios and seed**

```json
{
  "train_ratio": 0.6,
  "test_ratio": 0.2,
  "holdout_ratio": 0.2,
  "random_seed": 123
}
```

```python
# Config ratios and seed will be used
train_df, test_df, holdout_df = assign_split(df, config)
```

**Option B: Use function parameters**

```python
# Function parameters will be used (no ratios in config)
train_df, test_df, holdout_df = assign_split(
    df, 
    config,
    train_ratio=0.7,
    test_ratio=0.15,
    holdout_ratio=0.15,
    random_seed=42
)
```

**Option C: Use defaults**

```python
# Defaults (0.7, 0.15, 0.15) will be used
train_df, test_df, holdout_df = assign_split(df, config, random_seed=42)
```

### 2. Column-Based Splits

Use a predefined split column in your data. Add to your config:

```json
{
  "split_column": "dataset_split",
  "split_values": {
    "train": "TRAIN",
    "test": "TEST",
    "holdout": "HOLDOUT"
  }
}
```

Then simply call:

```python
train_df, test_df, holdout_df = assign_split(df, config)
```

The function will automatically detect the split column and use it.

## MLFlow Tracking

The `log_to_mlflow` function automatically logs:

- **Learning parameters** (prefixed with `learning_`):
  - `learning_iterations`
  - `learning_learning_rate`
  - `learning_depth`
  - `learning_loss_function`
  - etc.
- **Training parameters**:
  - `early_stopping_rounds`
  - `random_seed`
- **Evaluation metrics** (from analytics_store)
- **Model artifact** (trained CatBoost model)
- **Feature importance** (top 10 features)
- **Additional metadata** (number of features, dataset sizes, etc.)

View your experiments:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

## Example Workflows

### Simple Example

See `examples/simple_example.py` for a minimal example:

```python
import polars as pl
from ml_store import load_config, create_modelling_data, train_model, evaluate_model

config = load_config("config/regressor_config.json")
df = pl.read_csv("data/dataset.csv")

# One-step data preparation
X_train, y_train, X_test, y_test, features, cat_idx = create_modelling_data(df, config)

# Train
model = train_model(X_train, y_train, config, cat_idx, X_test, y_test)

# Evaluate
metrics_df = evaluate_model(model, X_test, y_test, config["model_type"])
print(f"Test MAE: {metrics_df['mae'][0]:.2f}")
```

### Complete Example

See `examples/train_example.py` for a complete workflow:

1. Loading configuration
2. One-step data preparation with `create_modelling_data`
3. Training with validation set and early stopping
4. Evaluating performance (returns Polars DataFrame)
5. Logging to MLFlow with all parameters
6. Saving and loading models
7. Making predictions

### Advanced Validation Example

See `examples/advanced_validation_example.py` for analytics_store integration:

1. Basic metrics with `evaluate_model`
2. Using analytics_store for advanced analysis
3. Generating diagnostic plots
4. Residual analysis
5. Error distribution analysis

## Data Format

The module works with **Polars DataFrames**. Load your data using Polars:

```python
import polars as pl

# CSV
df = pl.read_csv("data.csv")

# Parquet
df = pl.read_parquet("data.parquet")

# JSON
df = pl.read_json("data.json")
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
