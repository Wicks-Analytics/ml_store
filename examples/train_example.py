"""Example script demonstrating how to use ml_store functions."""

import polars as pl
from pathlib import Path

from ml_store import (
    load_config,
    load_data,
    create_modelling_data,
    train_model,
    evaluate_model,
    log_to_mlflow,
    save_model,
    load_model,
    predict,
)


def main():
    # 1. Load configuration
    config_path = Path("../config/regressor_config.json")
    config = load_config(config_path)
    print(f"Loaded config: {config['model_type']} model")
    print(f"Target: {config['target_column']}\n")
    
    # 2. Load data
    # Option A: Load data manually
    data_path = Path("data/car_insurance_premium_dataset.csv")
    df = pl.read_csv(data_path)
    
    # Option B: Load data from config (if config has "data_path" and optionally "data_type")
    # df = load_data(config)
    
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns}\n")
    
    # 3. Prepare modeling data (split + feature preparation in one step)
    print("Preparing modeling data...")
    X_train, y_train, X_test, y_test, feature_names, cat_indices, exposure_train, exposure_test = create_modelling_data(df, config)
    
    print(f"Train set: {len(X_train)} rows ({len(X_train)/df.shape[0]*100:.1f}%)")
    print(f"Test set: {len(X_test)} rows ({len(X_test)/df.shape[0]*100:.1f}%)")
    print(f"Features: {len(feature_names)}, Categorical: {len(cat_indices)}")
    if exposure_train is not None:
        print(f"Using exposure variable: {config.get('exposure_column')}\n")
    else:
        print()
    
    # 5. Train model
    print("\nTraining model...")
    model = train_model(
        X_train, y_train,
        config=config,
        categorical_features=cat_indices,
        X_val=X_test,
        y_val=y_test,
        exposure_train=exposure_train,
        exposure_val=exposure_test,
        verbose=True
    )
    
    # 6. Evaluate model
    print("\nEvaluating model...")
    train_metrics_df = evaluate_model(model, X_train, y_train, config["model_type"])
    test_metrics_df = evaluate_model(model, X_test, y_test, config["model_type"])
    
    print("\nTraining Metrics:")
    print(train_metrics_df)
    
    print("\nTest Metrics:")
    print(test_metrics_df)
    
    # Optional: Get DataFrame for advanced analysis with analytics_store
    # results_df = evaluate_model(model, X_test, y_test, config["model_type"], return_dataframe=True)
    # from analytics_store import validation_plots
    # validation_plots.plot_regression_diagnostics(results_df, "actual", "predicted")
    
    # 7. Log to MLFlow
    print("\nLogging to MLFlow...")
    run_id = log_to_mlflow(
        model=model,
        config=config,
        metrics=test_metrics_df,
        feature_names=feature_names,
        additional_params={"train_size": len(X_train), "test_size": len(X_test)}
    )
    print(f"MLFlow run ID: {run_id}")
    
    # 8. Save model
    model_path = Path("../models/car_insurance_premium_model.cbm")
    model_path.parent.mkdir(exist_ok=True)
    save_model(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # 9. Load model and make predictions
    print("\nLoading model and making predictions...")
    loaded_model = load_model(model_path, model_type=config["model_type"])
    predictions = predict(loaded_model, X_test[:5])
    print(f"\nSample predictions (first 5):")
    print(f"  Predicted: {predictions}")
    print(f"  Actual: {y_test[:5]}")


if __name__ == "__main__":
    main()
