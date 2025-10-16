"""Example script demonstrating how to use assign_split function."""

import polars as pl
from pathlib import Path

from ml_store import (
    load_config,
    load_data,
    assign_split,
    prepare_features,
    train_model,
    evaluate_model,
    log_to_mlflow,
    save_model,
)


def main():
    # 1. Load configuration
    config_path = Path("config/classifier_config.json")
    config = load_config(config_path)
    print(f"Loaded config: {config['model_type']} model")
    
    # 2. Load data (example with CSV)
    # Replace with your actual data path
    data_path = Path("data/full_dataset.csv")
    df = load_data(data_path, file_type="csv")
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 3. Split data using assign_split
    # This will perform random splits by default
    print("\nSplitting data...")
    train_df, test_df, holdout_df = assign_split(
        df,
        config,
        train_ratio=0.7,
        test_ratio=0.15,
        holdout_ratio=0.15,
        random_seed=42
    )
    
    print(f"Train set: {train_df.shape[0]} rows")
    print(f"Test set: {test_df.shape[0]} rows")
    print(f"Holdout set: {holdout_df.shape[0]} rows")
    
    # 4. Prepare features for each split
    X_train, y_train, feature_names, cat_indices = prepare_features(
        train_df, config, is_training=True
    )
    X_test, y_test, _, _ = prepare_features(
        test_df, config, is_training=True
    )
    X_holdout, y_holdout, _, _ = prepare_features(
        holdout_df, config, is_training=True
    )
    
    print(f"\nFeatures: {len(feature_names)}, Categorical: {len(cat_indices)}")
    
    # 5. Train model with test set as validation
    print("\nTraining model...")
    model = train_model(
        X_train, y_train,
        config=config,
        categorical_features=cat_indices,
        X_val=X_test,
        y_val=y_test,
        verbose=True
    )
    
    # 6. Evaluate on all sets
    print("\nEvaluating model...")
    train_metrics = evaluate_model(model, X_train, y_train, config["model_type"])
    test_metrics = evaluate_model(model, X_test, y_test, config["model_type"])
    holdout_metrics = evaluate_model(model, X_holdout, y_holdout, config["model_type"])
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nHoldout Metrics:")
    for metric, value in holdout_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 7. Log to MLFlow with all metrics
    print("\nLogging to MLFlow...")
    all_metrics = {
        **{"train_" + k: v for k, v in train_metrics.items()},
        **{"test_" + k: v for k, v in test_metrics.items()},
        **{"holdout_" + k: v for k, v in holdout_metrics.items()},
    }
    
    run_id = log_to_mlflow(
        model=model,
        config=config,
        metrics=all_metrics,
        feature_names=feature_names,
        additional_params={
            "train_size": len(X_train),
            "test_size": len(X_test),
            "holdout_size": len(X_holdout)
        }
    )
    print(f"MLFlow run ID: {run_id}")
    
    # 8. Save model
    model_path = Path("models/catboost_model_with_splits.cbm")
    save_model(model, model_path)
    print(f"\nModel saved to: {model_path}")


def example_with_split_column():
    """Example using a predefined split column in the data."""
    
    # Load config that specifies split column
    config_path = Path("config/classifier_with_split_column.json")
    config = load_config(config_path)
    print(f"Loaded config with split column: {config.get('split_column')}")
    
    # Load data that has a split column
    data_path = Path("data/dataset_with_splits.csv")
    df = load_data(data_path, file_type="csv")
    
    # Split based on the column values
    train_df, test_df, holdout_df = assign_split(df, config)
    
    print(f"Train set: {train_df.shape[0]} rows")
    print(f"Test set: {test_df.shape[0]} rows")
    print(f"Holdout set: {holdout_df.shape[0]} rows")
    
    # Continue with training as in main()...


if __name__ == "__main__":
    # Run the main example with random splits
    main()
    
    # Uncomment to run example with split column
    # example_with_split_column()
