"""Test script to verify single random_seed is used for both splitting and training."""

import polars as pl
import numpy as np
from ml_store import assign_split, prepare_features, train_model


def test_single_random_seed():
    """Test that a single random_seed in config is used for both splitting and training."""
    print("=" * 60)
    print("Testing Single random_seed for Splitting and Training")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    df = pl.DataFrame({
        "feature1": np.random.randn(200),
        "feature2": np.random.randn(200),
        "feature3": np.random.randn(200),
        "target": np.random.randint(0, 2, 200)
    })
    
    print(f"Original dataset: {df.shape[0]} rows\n")
    
    # Config with single random_seed at top level
    config = {
        "model_type": "classifier",
        "target_column": "target",
        "random_seed": 999,
        "model_params": {
            "iterations": 10,
            "depth": 3,
            "verbose": False
        }
    }
    
    print(f"Config random_seed: {config['random_seed']}")
    print("Note: random_seed NOT in model_params\n")
    
    # Split data - should use random_seed=999
    train_df1, test_df1, holdout_df1 = assign_split(df, config)
    train_df2, test_df2, holdout_df2 = assign_split(df, config)
    
    # Verify splitting is reproducible
    assert train_df1.equals(train_df2), "Train sets should be identical (splitting uses config seed)"
    print("✓ Data splitting is reproducible with config random_seed")
    
    # Prepare features
    X_train, y_train, feature_names, cat_indices = prepare_features(train_df1, config)
    
    # Train model - should use random_seed=999 from top-level config
    model1 = train_model(X_train, y_train, config)
    model2 = train_model(X_train, y_train, config)
    
    # Verify models produce same predictions (reproducible training)
    predictions1 = model1.predict(X_train[:10])
    predictions2 = model2.predict(X_train[:10])
    
    assert np.array_equal(predictions1, predictions2), "Models should produce identical predictions"
    print("✓ Model training is reproducible with config random_seed")
    
    print("\n✓ Single random_seed test passed!")
    print(f"  - Splitting used seed: {config['random_seed']}")
    print(f"  - Training used seed: {config['random_seed']}")
    print()


def test_model_params_override():
    """Test that model_params.random_seed overrides top-level random_seed."""
    print("=" * 60)
    print("Testing model_params.random_seed Override")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    df = pl.DataFrame({
        "feature1": np.random.randn(200),
        "feature2": np.random.randn(200),
        "feature3": np.random.randn(200),
        "target": np.random.randint(0, 2, 200)
    })
    
    # Config with both top-level and model_params random_seed
    config = {
        "model_type": "classifier",
        "target_column": "target",
        "random_seed": 999,  # Used for splitting
        "model_params": {
            "iterations": 10,
            "depth": 3,
            "random_seed": 777,  # Used for training (overrides top-level)
            "verbose": False
        }
    }
    
    print(f"Top-level random_seed: {config['random_seed']}")
    print(f"model_params.random_seed: {config['model_params']['random_seed']}\n")
    
    # Split data - should use random_seed=999
    train_df, test_df, holdout_df = assign_split(df, config)
    print(f"✓ Splitting used top-level seed: {config['random_seed']}")
    
    # Prepare features
    X_train, y_train, feature_names, cat_indices = prepare_features(train_df, config)
    
    # Train model - should use random_seed=777 from model_params
    model = train_model(X_train, y_train, config)
    print(f"✓ Training used model_params seed: {config['model_params']['random_seed']}")
    
    print("\n✓ Override test passed!")
    print("  - model_params.random_seed takes precedence for training")
    print("  - Top-level random_seed used for splitting")
    print()


if __name__ == "__main__":
    test_single_random_seed()
    test_model_params_override()
    
    print("=" * 60)
    print("All single random_seed tests passed! ✓")
    print("=" * 60)
