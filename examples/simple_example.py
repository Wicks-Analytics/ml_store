"""Simple example using create_modelling_data for quick model training."""

import polars as pl
from pathlib import Path
from ml_store import load_config, create_modelling_data, train_model, evaluate_model


def main():
    # Load config and data
    config = load_config(Path("../config/regressor_config.json"))
    df = pl.read_csv(Path("data/car_insurance_premium_dataset.csv"))
    
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target: {config['target_column']}\n")
    
    # One-step data preparation: split + feature extraction
    X_train, y_train, X_test, y_test, features, cat_idx, exp_train, exp_test = create_modelling_data(df, config)
    
    print(f"✓ Data prepared: {len(X_train)} train, {len(X_test)} test")
    print(f"✓ Features: {len(features)}\n")
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train, config, cat_idx, X_test, y_test, exp_train, exp_test)
    
    # Evaluate
    train_metrics_df = evaluate_model(model, X_train, y_train, config["model_type"])
    test_metrics_df = evaluate_model(model, X_test, y_test, config["model_type"])
    
    print("\nResults:")
    print(f"  Train MAE: {train_metrics_df['mae'][0]:.2f}")
    print(f"  Test MAE: {test_metrics_df['mae'][0]:.2f}")
    print(f"  Train RMSE: {train_metrics_df['rmse'][0]:.2f}")
    print(f"  Test RMSE: {test_metrics_df['rmse'][0]:.2f}")


if __name__ == "__main__":
    main()
