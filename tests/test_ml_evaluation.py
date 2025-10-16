"""Tests for ml_evaluation module."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from ml_store import ml_evaluation


class TestFeatureImportance:
    """Tests for feature importance functions."""

    def test_get_feature_importance_classifier(self, trained_classifier, classifier_config):
        """Test getting feature importance from classifier."""
        importance = ml_evaluation.get_feature_importance(
            trained_classifier["model"],
            config=classifier_config,
            importance_type="FeatureImportance",
        )

        assert importance is not None
        assert hasattr(importance, "to_polars")

        df = importance.to_polars()
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert len(df) > 0

    def test_get_feature_importance_with_pool(self, trained_classifier, classifier_config):
        """Test feature importance with Pool."""
        from catboost import Pool

        pool = Pool(
            trained_classifier["X_test"],
            trained_classifier["y_test"],
            cat_features=trained_classifier["cat_indices"],
        )

        importance = ml_evaluation.get_feature_importance(
            trained_classifier["model"], config=classifier_config, pool=pool
        )

        assert importance is not None

    def test_plot_feature_importance(self, trained_classifier, classifier_config):
        """Test plotting feature importance."""
        importance = ml_evaluation.get_feature_importance(
            trained_classifier["model"],
            config=classifier_config,
            importance_type="FeatureImportance",
        )

        fig = ml_evaluation.plot_feature_importance(importance, top_n=10, backend="matplotlib")

        assert fig is not None
        plt.close(fig)


class TestSHAPAnalysis:
    """Tests for SHAP analysis functions."""

    def test_calculate_shap_values(self, sample_classification_data):
        """Test calculating SHAP values."""
        # Train a simple model with only numeric features for SHAP testing
        from ml_store import create_modelling_data, train_model

        config = {
            "model_type": "classifier",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "categorical_features": [],
            "train_ratio": 0.7,
            "test_ratio": 0.3,
            "random_seed": 42,
            "model_params": {"learning_params": {"iterations": 50, "verbose": 0}},
        }

        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            sample_classification_data, config
        )

        model = train_model(
            X_train, y_train, config=config, categorical_features=cat_indices, verbose=0
        )

        shap_result = ml_evaluation.calculate_shap_values(
            model, X_test[:100], feature_names=feature_names
        )

        assert shap_result is not None
        assert hasattr(shap_result, "shap_values")
        assert hasattr(shap_result, "base_value")
        assert hasattr(shap_result, "feature_names")

    def test_plot_shap_summary(self, sample_classification_data):
        """Test SHAP summary plot."""
        # Train a simple model with only numeric features for SHAP testing
        from ml_store import create_modelling_data, train_model

        config = {
            "model_type": "classifier",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "categorical_features": [],
            "train_ratio": 0.7,
            "test_ratio": 0.3,
            "random_seed": 42,
            "model_params": {"learning_params": {"iterations": 50, "verbose": 0}},
        }

        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            sample_classification_data, config
        )

        model = train_model(
            X_train, y_train, config=config, categorical_features=cat_indices, verbose=0
        )

        shap_result = ml_evaluation.calculate_shap_values(
            model, X_test[:100], feature_names=feature_names
        )

        fig = ml_evaluation.plot_shap_summary(shap_result)

        assert fig is not None
        plt.close(fig)


class TestPartialDependence:
    """Tests for partial dependence functions."""

    def test_calculate_partial_dependence(self, trained_classifier, classifier_config):
        """Test calculating partial dependence."""
        pd_result = ml_evaluation.calculate_partial_dependence(
            trained_classifier["model"],
            trained_classifier["X_test"],
            feature_index=0,
            feature_name=trained_classifier["feature_names"][0],
        )

        assert pd_result is not None
        assert hasattr(pd_result, "feature_values")  # Correct attribute name
        assert hasattr(pd_result, "pd_values")

    def test_plot_partial_dependence(self, trained_classifier, classifier_config):
        """Test plotting partial dependence."""
        # Calculate PD first
        pd_result = ml_evaluation.calculate_partial_dependence(
            trained_classifier["model"],
            trained_classifier["X_test"],
            feature_index=0,
            feature_name=trained_classifier["feature_names"][0],
        )

        # Then plot it
        fig = ml_evaluation.plot_partial_dependence(pd_result, backend="matplotlib")

        assert fig is not None
        plt.close(fig)


class TestPoolCreation:
    """Tests for Pool creation functions."""

    def test_create_pool_from_config(self, classifier_config):
        """Test creating Pool from config."""
        # Create data with proper categorical features (as strings)
        X_numeric = np.random.randn(100, 3)
        X_cat1 = np.random.choice(["A", "B", "C"], 100)
        X_cat2 = np.random.choice(["X", "Y"], 100)

        # Combine into DataFrame then convert to array for Pool

        df = pl.DataFrame(
            {
                "feature1": X_numeric[:, 0],
                "feature2": X_numeric[:, 1],
                "feature3": X_numeric[:, 2],
                "cat_feature1": X_cat1,
                "cat_feature2": X_cat2,
            }
        )

        y = np.random.choice([0, 1], 100)

        # Use the actual prepare_features to get proper format
        from ml_store import prepare_features

        X, _, feature_names, cat_indices, _ = prepare_features(
            df, classifier_config, is_training=False
        )

        pool = ml_evaluation.create_pool_from_config(
            X, y, classifier_config, feature_names=feature_names
        )

        assert pool is not None
        assert pool.num_row() == 100

    def test_create_pool_with_feature_names(self, classifier_config):
        """Test Pool with feature names."""
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)

        pool = ml_evaluation.create_pool_from_config(
            X, y, classifier_config, feature_names=["f1", "f2", "f3", "f4", "f5"]
        )

        assert pool is not None


class TestEvaluationReport:
    """Tests for evaluation report function."""

    @pytest.mark.slow
    def test_create_evaluation_report_classifier(
        self, trained_classifier, classifier_config, tmp_path
    ):
        """Test creating full evaluation report for classifier."""
        # Create pool for the report
        from catboost import Pool

        pool = Pool(
            trained_classifier["X_test"][:100],
            trained_classifier["y_test"][:100],
            cat_features=trained_classifier["cat_indices"],
        )

        report = ml_evaluation.create_evaluation_report(
            trained_classifier["model"],
            trained_classifier["X_test"][:100],
            trained_classifier["y_test"][:100],
            config=classifier_config,
            pool=pool,
            output_dir=str(tmp_path),
            top_n_features=5,
        )

        assert "feature_importance_df" in report
        assert "shap_df" in report
        assert "fig_feature_importance" in report
        assert "fig_shap_summary" in report

        # Clean up figures
        for key, value in report.items():
            if key.startswith("fig_") and hasattr(value, "savefig"):
                plt.close(value)

    @pytest.mark.slow
    def test_create_evaluation_report_regressor(
        self, trained_regressor, regressor_config, tmp_path
    ):
        """Test creating full evaluation report for regressor."""
        # Create pool for the report
        from catboost import Pool

        pool = Pool(
            trained_regressor["X_test"][:100],
            trained_regressor["y_test"][:100],
            cat_features=trained_regressor["cat_indices"],
        )

        report = ml_evaluation.create_evaluation_report(
            trained_regressor["model"],
            trained_regressor["X_test"][:100],
            trained_regressor["y_test"][:100],
            config=regressor_config,
            pool=pool,
            output_dir=str(tmp_path),
            top_n_features=5,
        )

        assert "feature_importance_df" in report
        assert "shap_df" in report

        # Clean up figures
        for key, value in report.items():
            if key.startswith("fig_") and hasattr(value, "savefig"):
                plt.close(value)


class TestBackendConfiguration:
    """Tests for plotting backend configuration."""

    def test_matplotlib_backend(self, trained_classifier, classifier_config):
        """Test using matplotlib backend."""
        config = classifier_config.copy()
        config["plotting_backend"] = "matplotlib"

        importance = ml_evaluation.get_feature_importance(
            trained_classifier["model"], config=config, importance_type="FeatureImportance"
        )

        fig = ml_evaluation.plot_feature_importance(importance, config=config)

        assert fig is not None
        assert hasattr(fig, "savefig")  # matplotlib figure
        plt.close(fig)

    def test_plotly_backend(self, trained_classifier, classifier_config):
        """Test using plotly backend."""
        config = classifier_config.copy()
        config["plotting_backend"] = "plotly"

        importance = ml_evaluation.get_feature_importance(
            trained_classifier["model"], config=config, importance_type="FeatureImportance"
        )

        fig = ml_evaluation.plot_feature_importance(importance, config=config)

        assert fig is not None
        assert hasattr(fig, "show")  # plotly figure

    def test_backend_override(self, trained_classifier, classifier_config):
        """Test overriding config backend with explicit parameter."""
        config = classifier_config.copy()
        config["plotting_backend"] = "plotly"

        importance = ml_evaluation.get_feature_importance(
            trained_classifier["model"], config=config, importance_type="FeatureImportance"
        )

        # Override with matplotlib
        fig = ml_evaluation.plot_feature_importance(importance, backend="matplotlib", config=config)

        assert fig is not None
        assert hasattr(fig, "savefig")  # matplotlib figure
        plt.close(fig)
