# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-10-16

### Added
- Initial release of ml_store package
- Core ML functions module (`ml_functions.py`)
  - `load_config()` - Load configuration from JSON files
  - `load_data()` - Load data from multiple formats (CSV, Parquet, JSON, SQL)
  - `prepare_features()` - Prepare features and target from DataFrame
  - `assign_split()` - Assign train/val/test splits
  - `create_modelling_data()` - Complete data preparation pipeline
  - `train_model()` - Train CatBoost models (classifier/regressor)
  - `evaluate_model()` - Evaluate model performance
  - `predict()` - Make predictions
  - `save_model()` - Save trained models
  - `load_model()` - Load saved models
  - `log_to_mlflow()` - Log experiments to MLFlow

- ML evaluation module (`ml_evaluation.py`)
  - `get_feature_importance()` - Calculate feature importance
  - `plot_feature_importance()` - Visualize feature importance
  - `calculate_shap_values()` - Calculate SHAP values
  - `plot_shap_summary()` - SHAP summary plots
  - `plot_shap_dependence()` - SHAP dependence plots
  - `plot_shap_force()` - SHAP force plots
  - `calculate_partial_dependence()` - Calculate partial dependence
  - `plot_partial_dependence()` - Partial dependence plots
  - `plot_multiple_partial_dependence()` - Multiple PD plots
  - `create_evaluation_report()` - Comprehensive evaluation report
  - `create_pool_from_config()` - Create CatBoost Pool from config

- SQL database support
  - Snowflake connector integration
  - Microsoft SQL Server connector integration
  - Connection string parsing
  - Query execution via `load_data()`

- Plotting backend configuration
  - Support for matplotlib (default)
  - Support for plotly (interactive)
  - Backend selection via config file
  - Backend override via function parameters

- Configuration-driven workflows
  - JSON-based configuration files
  - Feature column specification
  - Categorical feature handling
  - Train/val/test split configuration
  - Model parameter configuration
  - MLFlow experiment configuration
  - Plotting backend configuration

- Comprehensive test suite
  - Unit tests for all core functions
  - Integration tests for end-to-end workflows
  - Test fixtures and utilities
  - pytest configuration

- Documentation
  - README with usage examples
  - SQL data loading guide
  - Example configurations
  - Example scripts
  - API documentation in docstrings

- Example configurations
  - Classifier configuration
  - Regressor configuration
  - Configuration with split column
  - Snowflake SQL configuration
  - MS SQL Server configuration

- Example scripts
  - Basic training example
  - Training with split column
  - Data loading examples
  - SQL data loading examples

### Dependencies
- mlflow >= 2.18.0
- catboost >= 1.2.7
- polars >= 1.13.0
- scikit-learn >= 1.5.2
- numpy >= 2.1.3
- analytics-store (from GitHub)
- shap >= 0.41.0
- matplotlib >= 3.4.0
- plotly >= 5.0.0

### Optional Dependencies
- snowflake-connector-python >= 3.0.0 (for Snowflake support)
- pyodbc >= 4.0.0 (for MS SQL Server support)

[Unreleased]: https://github.com/Wicks-Analytics/ml_store/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Wicks-Analytics/ml_store/releases/tag/v0.1.0
