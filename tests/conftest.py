"""Pytest configuration and fixtures for ml_store tests."""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import tempfile
import json


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    df = pl.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'cat_feature1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat_feature2': np.random.choice(['X', 'Y'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return df


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    X3 = np.random.randn(n_samples)
    
    # Create target with some relationship to features
    y = 2 * X1 + 3 * X2 - 1.5 * X3 + np.random.randn(n_samples) * 0.5
    
    df = pl.DataFrame({
        'feature1': X1,
        'feature2': X2,
        'feature3': X3,
        'cat_feature': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': y
    })
    
    return df


@pytest.fixture
def classifier_config():
    """Create sample classifier configuration."""
    return {
        "model_type": "classifier",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2", "feature3", "cat_feature1", "cat_feature2"],
        "categorical_features": ["cat_feature1", "cat_feature2"],
        "train_ratio": 0.7,
        "test_ratio": 0.3,
        "random_seed": 42,
        "plotting_backend": "matplotlib",
        "model_params": {
            "learning_params": {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 4,
                "verbose": 0
            }
        }
    }


@pytest.fixture
def regressor_config():
    """Create sample regressor configuration."""
    return {
        "model_type": "regressor",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2", "feature3", "cat_feature"],
        "categorical_features": ["cat_feature"],
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "random_seed": 42,
        "plotting_backend": "matplotlib",
        "model_params": {
            "learning_params": {
                "iterations": 100,
                "learning_rate": 0.05,
                "depth": 3,
                "verbose": 0
            }
        }
    }


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config = {
        "model_type": "classifier",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2"],
        "categorical_features": [],
        "train_ratio": 0.7,
        "test_ratio": 0.3,
        "random_seed": 42
    }
    
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    return config_path


@pytest.fixture
def temp_csv_file(tmp_path, sample_classification_data):
    """Create a temporary CSV file."""
    csv_path = tmp_path / "test_data.csv"
    sample_classification_data.write_csv(csv_path)
    return csv_path


@pytest.fixture
def temp_parquet_file(tmp_path, sample_classification_data):
    """Create a temporary Parquet file."""
    parquet_path = tmp_path / "test_data.parquet"
    sample_classification_data.write_parquet(parquet_path)
    return parquet_path


@pytest.fixture
def trained_classifier(sample_classification_data, classifier_config):
    """Create a trained classifier model."""
    from ml_store import create_modelling_data, train_model
    
    X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
        sample_classification_data, classifier_config
    )
    
    model = train_model(
        X_train, y_train,
        config=classifier_config,
        categorical_features=cat_indices,
        verbose=0
    )
    
    return {
        'model': model,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'cat_indices': cat_indices
    }


@pytest.fixture
def trained_regressor(sample_regression_data, regressor_config):
    """Create a trained regressor model."""
    from ml_store import create_modelling_data, train_model
    
    X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
        sample_regression_data, regressor_config
    )
    
    model = train_model(
        X_train, y_train,
        config=regressor_config,
        categorical_features=cat_indices,
        verbose=0
    )
    
    return {
        'model': model,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'cat_indices': cat_indices
    }
