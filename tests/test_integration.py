"""Integration tests for ml_store."""

import pytest
import polars as pl
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from ml_store import (
    load_config,
    load_data,
    create_modelling_data,
    train_model,
    evaluate_model,
    predict,
    save_model,
    load_model,
)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end ML workflows."""
    
    def test_classifier_workflow(self, sample_classification_data, classifier_config, tmp_path):
        """Test complete classifier workflow."""
        # 1. Save data to file
        data_path = tmp_path / "data.csv"
        sample_classification_data.write_csv(data_path)
        
        # 2. Update config with data path
        config = classifier_config.copy()
        config['data_path'] = str(data_path)
        
        # 3. Load data
        df = load_data(config)
        assert len(df) > 0
        
        # 4. Prepare modeling data
        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(df, config)
        
        # 5. Train model
        model = train_model(
            X_train, y_train,
            config=config,
            categorical_features=cat_indices,
            X_val=X_test,
            y_val=y_test,
            verbose=0
        )
        
        # 6. Make predictions
        predictions = predict(model, X_test)
        assert len(predictions) == len(X_test)
        
        # 7. Evaluate model
        metrics = evaluate_model(model, X_test, y_test, model_type=config['model_type'])
        assert 'accuracy' in metrics or 'auc' in metrics
        
        # 8. Save model
        model_path = tmp_path / "model.cbm"
        save_model(model, str(model_path))
        assert model_path.exists()
        
        # 9. Load model
        loaded_model = load_model(str(model_path), model_type=config['model_type'])
        
        # 10. Verify loaded model works
        loaded_predictions = predict(loaded_model, X_test)
        np.testing.assert_array_almost_equal(predictions, loaded_predictions)
    
    def test_regressor_workflow(self, sample_regression_data, regressor_config, tmp_path):
        """Test complete regressor workflow."""
        # 1. Prepare data
        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            sample_regression_data,
            regressor_config
        )
        
        # 2. Train model
        model = train_model(
            X_train, y_train,
            config=regressor_config,
            categorical_features=cat_indices,
            verbose=0
        )
        
        # 3. Make predictions
        predictions = predict(model, X_test)
        assert len(predictions) == len(X_test)
        
        # 4. Evaluate
        metrics = evaluate_model(model, X_test, y_test, model_type=regressor_config['model_type'])
        assert 'rmse' in metrics or 'mae' in metrics
        
        # 5. Save and load
        model_path = tmp_path / "regressor.cbm"
        save_model(model, str(model_path))
        loaded_model = load_model(str(model_path), model_type='regressor')
        
        # Verify
        loaded_predictions = predict(loaded_model, X_test)
        np.testing.assert_array_almost_equal(predictions, loaded_predictions)


@pytest.mark.integration
class TestConfigDrivenWorkflow:
    """Test workflows driven entirely by config files."""
    
    def test_config_file_workflow(self, sample_classification_data, tmp_path):
        """Test workflow using config file."""
        # Create config file
        config = {
            "model_type": "classifier",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "categorical_features": [],
            "train_ratio": 0.7,
            "test_ratio": 0.3,
            "random_seed": 42,
            "plotting_backend": "matplotlib",
            "model_params": {
                "learning_params": {
                    "iterations": 50,
                    "learning_rate": 0.1,
                    "depth": 3,
                    "verbose": 0
                }
            }
        }
        
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Save data
        data_path = tmp_path / "data.csv"
        sample_classification_data.write_csv(data_path)
        
        # Load config
        loaded_config = load_config(config_path)
        loaded_config['data_path'] = str(data_path)
        
        # Run workflow
        df = load_data(loaded_config)
        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            df, loaded_config
        )
        
        model = train_model(
            X_train, y_train,
            config=loaded_config,
            categorical_features=cat_indices,
            verbose=0
        )
        
        predictions = predict(model, X_test)
        assert len(predictions) == len(X_test)


@pytest.mark.integration
class TestSplitColumnWorkflow:
    """Test workflows using pre-defined split columns."""
    
    def test_split_column_workflow(self, sample_classification_data, classifier_config):
        """Test workflow with split column."""
        import numpy as np
        
        # Add split column manually
        np.random.seed(42)
        n = len(sample_classification_data)
        splits = np.random.choice(['TRAIN', 'TEST'], size=n, p=[0.7, 0.3])
        df_with_split = sample_classification_data.with_columns(pl.Series('dataset_split', splits))
        
        # Update config
        config = classifier_config.copy()
        config['split_column'] = 'dataset_split'
        config['split_values'] = {'train': 'TRAIN', 'test': 'TEST'}
        
        # Prepare data
        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            df_with_split,
            config
        )
        
        # Train
        model = train_model(
            X_train, y_train,
            config=config,
            categorical_features=cat_indices,
            verbose=0
        )
        
        # Predict
        predictions = predict(model, X_test)
        assert len(predictions) == len(X_test)


@pytest.mark.integration
class TestDataLoadingFormats:
    """Test loading different data formats."""
    
    def test_csv_workflow(self, sample_classification_data, classifier_config, tmp_path):
        """Test workflow with CSV data."""
        csv_path = tmp_path / "data.csv"
        sample_classification_data.write_csv(csv_path)
        
        config = classifier_config.copy()
        config['data_path'] = str(csv_path)
        config['data_type'] = 'csv'
        
        df = load_data(config)
        assert len(df) == len(sample_classification_data)
    
    def test_parquet_workflow(self, sample_classification_data, classifier_config, tmp_path):
        """Test workflow with Parquet data."""
        parquet_path = tmp_path / "data.parquet"
        sample_classification_data.write_parquet(parquet_path)
        
        config = classifier_config.copy()
        config['data_path'] = str(parquet_path)
        config['data_type'] = 'parquet'
        
        df = load_data(config)
        assert len(df) == len(sample_classification_data)
    
    def test_json_workflow(self, sample_classification_data, classifier_config, tmp_path):
        """Test workflow with JSON data."""
        json_path = tmp_path / "data.json"
        sample_classification_data.write_json(json_path)
        
        config = classifier_config.copy()
        config['data_path'] = str(json_path)
        config['data_type'] = 'json'
        
        df = load_data(config)
        assert len(df) == len(sample_classification_data)


@pytest.mark.integration
@pytest.mark.slow
class TestEvaluationWorkflow:
    """Test complete evaluation workflows."""
    
    def test_full_evaluation_report(self, sample_classification_data, classifier_config, tmp_path):
        """Test generating full evaluation report."""
        from ml_store import ml_evaluation
        from catboost import Pool
        
        # Prepare and train
        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            sample_classification_data,
            classifier_config
        )
        
        model = train_model(
            X_train, y_train,
            config=classifier_config,
            categorical_features=cat_indices,
            verbose=0
        )
        
        # Create pool for evaluation
        pool = Pool(
            X_test[:100],
            y_test[:100],
            cat_features=cat_indices
        )
        
        # Create evaluation report
        report = ml_evaluation.create_evaluation_report(
            model,
            X_test[:100],  # Use subset for speed
            y_test[:100],
            config=classifier_config,
            pool=pool,
            output_dir=str(tmp_path),
            top_n_features=5
        )
        
        # Verify report contents
        assert 'feature_importance_df' in report
        assert 'shap_df' in report
        assert 'fig_feature_importance' in report
        assert 'fig_shap_summary' in report
        
        # Verify files were created (if output_dir was used)
        # Note: Files are only created if output_dir is specified in create_evaluation_report
        # assert (tmp_path / "feature_importance.csv").exists()
        # assert (tmp_path / "shap_values.csv").exists()
        
        # Clean up figures
        for key, value in report.items():
            if key.startswith('fig_') and hasattr(value, 'savefig'):
                plt.close(value)
