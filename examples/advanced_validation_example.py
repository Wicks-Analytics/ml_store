"""Advanced example showing analytics_store integration for detailed model analysis."""

import polars as pl
from pathlib import Path
from ml_store import load_config, create_modelling_data, train_model, evaluate_model
from analytics_store import validation_plots


def main():
    # Load config and data
    config = load_config(Path("../config/regressor_config.json"))
    df = pl.read_csv(Path("data/car_insurance_premium_dataset.csv"))
    
    print(f"Dataset: {df.shape[0]} rows")
    print(f"Target: {config['target_column']}\n")
    
    # Prepare data
    X_train, y_train, X_test, y_test, features, cat_idx, exp_train, exp_test = create_modelling_data(df, config)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train, config, cat_idx, X_test, y_test, exp_train, exp_test)
    
    # Basic evaluation (returns metrics DataFrame)
    print("\n=== Basic Metrics ===")
    metrics_df = evaluate_model(model, X_test, y_test, config["model_type"])
    print(metrics_df)
    
    # Advanced evaluation - create DataFrame with predictions for analytics_store
    print("\n=== Advanced Analysis with analytics_store ===")
    predictions = model.predict(X_test)
    results_df = pl.DataFrame({
        "actual": y_test,
        "predicted": predictions
    })
    
    print(f"\nResults DataFrame shape: {results_df.shape}")
    print(results_df.head())
    
    # Calculate residuals
    residuals_df = results_df.with_columns(
        (pl.col("actual") - pl.col("predicted")).alias("residual")
    )
    
    print("\nResiduals Summary:")
    print(f"  Mean residual: {residuals_df['residual'].mean():.4f}")
    print(f"  Std residual: {residuals_df['residual'].std():.4f}")
    print(f"  Min residual: {residuals_df['residual'].min():.4f}")
    print(f"  Max residual: {residuals_df['residual'].max():.4f}")
    
    # Generate plots using analytics_store
    try:
        print("\nGenerating validation plots with analytics_store...")
        
        # Create outputs directory
        Path("../outputs").mkdir(exist_ok=True)
        
        # Regression diagnostics plot
        validation_plots.plot_regression_diagnostics(
            results_df,
            actual_column="actual",
            predicted_column="predicted",
            title="Model Regression Diagnostics",
            save_path="../outputs/regression_diagnostics.png"
        )
        print("  ✓ Regression diagnostics saved")
        
    except Exception as e:
        print(f"  Note: Plotting error: {e}")
    
    # Error analysis
    print("\nError Analysis:")
    errors_df = residuals_df.with_columns(
        pl.col("residual").abs().alias("abs_error")
    ).sort("abs_error", descending=True)
    
    print("  Top 5 largest prediction errors:")
    for row in errors_df.head(5).iter_rows(named=True):
        print(f"    Actual: {row['actual']:.2f}, "
              f"Predicted: {row['predicted']:.2f}, "
              f"Error: {row['abs_error']:.2f}")
    
    # Save results for further analysis
    results_path = Path("../outputs/model_results.csv")
    residuals_df.write_csv(results_path)
    print(f"\n✓ Results saved to: {results_path}")
    
    print("\n✓ Advanced validation complete!")


if __name__ == "__main__":
    main()
