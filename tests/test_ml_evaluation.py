"""Tests for ml_evaluation module."""

import pytest
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from ml_store import ml_evaluation


class TestFeatureImportance:
    """Tests for feature importance functions."""
    
    def test_get_feature_importance_classifier(self, trained_classifier, classifier_config):
        """Test getting feature importance from classifier."""
        importance = ml_evaluation.get_feature_importance(
            trained_classifier['model'],
            config=classifier_config
        )
        
        assert importance is not None
        assert hasattr(importance, 'to_polars')
        
        df = importance.to_polars()
        assert 'feature' in df.columns
        assert 'importance' in df.columns
        assert len(df) > 0
    
    def test_get_feature_importance_with_pool(self, trained_classifier, classifier_config):
        """Test feature importance with Pool."""
        from catboost import Pool
        
        pool = Pool(
            trained_classifier['X_test'],
            trained_classifier['y_test'],
            cat_features=trained_classifier['cat_indices']
        )
        
        importance = ml_evaluation.get_feature_importance(
            trained_classifier['model'],
            config=classifier_config,
            pool=pool
        )
        
        assert importance is not None
    
    def test_plot_feature_importance(self, trained_classifier, classifier_config):
        """Test plotting feature importance."""
        importance = ml_evaluation.get_feature_importance(
            trained_classifier['model'],
            config=classifier_config
        )
        
        fig = ml_evaluation.plot_feature_importance(
            importance,
            top_n=10,
            backend='matplotlib'
        )
        
        assert fig is not None
        plt.close(fig)


class TestSHAPAnalysis:
    """Tests for SHAP analysis functions."""
    
    def test_calculate_shap_values(self, trained_classifier, classifier_config):
        """Test calculating SHAP values."""
        shap_result = ml_evaluation.calculate_shap_values(
            trained_classifier['model'],
            trained_classifier['X_test'][:100],  # Use subset for speed
            config=classifier_config
        )
        
        assert shap_result is not None
        assert hasattr(shap_result, 'shap_values')
        assert hasattr(shap_result, 'base_value')
        assert hasattr(shap_result, 'feature_names')
    
    def test_plot_shap_summary(self, trained_classifier, classifier_config):
        """Test SHAP summary plot."""
        shap_result = ml_evaluation.calculate_shap_values(
            trained_classifier['model'],
            trained_classifier['X_test'][:100],
            config=classifier_config
        )
        
        fig = ml_evaluation.plot_shap_summary(
            shap_result,
            backend='matplotlib'
        )
        
        assert fig is not None
        plt.close(fig)


class TestPartialDependence:
    """Tests for partial dependence functions."""
    
    def test_calculate_partial_dependence(self, trained_classifier, classifier_config):
        """Test calculating partial dependence."""
        pd_result = ml_evaluation.calculate_partial_dependence(
            trained_classifier['model'],
            trained_classifier['X_test'],
            feature_indices=[0],
            config=classifier_config
        )
        
        assert pd_result is not None
        assert hasattr(pd_result, 'grid_values')
        assert hasattr(pd_result, 'pd_values')
    
    def test_plot_partial_dependence(self, trained_classifier, classifier_config):
        """Test plotting partial dependence."""
        fig = ml_evaluation.plot_partial_dependence(
            trained_classifier['model'],
            trained_classifier['X_test'],
            feature_indices=[0, 1],
            config=classifier_config,
            backend='matplotlib'
        )
        
        assert fig is not None
        plt.close(fig)


class TestPoolCreation:
    """Tests for Pool creation functions."""
    
    def test_create_pool_from_config(self, classifier_config):
        """Test creating Pool from config."""
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        pool = ml_evaluation.create_pool_from_config(X, y, classifier_config)
        
        assert pool is not None
        assert pool.num_row() == 100
    
    def test_create_pool_with_cat_features(self, classifier_config):
        """Test Pool with categorical features."""
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        pool = ml_evaluation.create_pool_from_config(
            X, y, classifier_config,
            cat_features=[3, 4]
        )
        
        assert pool is not None


class TestEvaluationReport:
    """Tests for evaluation report function."""
    
    @pytest.mark.slow
    def test_create_evaluation_report_classifier(self, trained_classifier, classifier_config, tmp_path):
        """Test creating full evaluation report for classifier."""
        report = ml_evaluation.create_evaluation_report(
            trained_classifier['model'],
            trained_classifier['X_test'][:100],  # Use subset for speed
            trained_classifier['y_test'][:100],
            config=classifier_config,
            output_dir=str(tmp_path),
            top_n_features=5
        )
        
        assert 'feature_importance_df' in report
        assert 'shap_df' in report
        assert 'fig_feature_importance' in report
        assert 'fig_shap_summary' in report
        
        # Clean up figures
        for key, value in report.items():
            if key.startswith('fig_') and hasattr(value, 'savefig'):
                plt.close(value)
    
    @pytest.mark.slow
    def test_create_evaluation_report_regressor(self, trained_regressor, regressor_config, tmp_path):
        """Test creating full evaluation report for regressor."""
        report = ml_evaluation.create_evaluation_report(
            trained_regressor['model'],
            trained_regressor['X_test'][:100],
            trained_regressor['y_test'][:100],
            config=regressor_config,
            output_dir=str(tmp_path),
            top_n_features=5
        )
        
        assert 'feature_importance_df' in report
        assert 'shap_df' in report
        
        # Clean up figures
        for key, value in report.items():
            if key.startswith('fig_') and hasattr(value, 'savefig'):
                plt.close(value)


class TestBackendConfiguration:
    """Tests for plotting backend configuration."""
    
    def test_matplotlib_backend(self, trained_classifier, classifier_config):
        """Test using matplotlib backend."""
        config = classifier_config.copy()
        config['plotting_backend'] = 'matplotlib'
        
        importance = ml_evaluation.get_feature_importance(
            trained_classifier['model'],
            config=config
        )
        
        fig = ml_evaluation.plot_feature_importance(
            importance,
            config=config
        )
        
        assert fig is not None
        assert hasattr(fig, 'savefig')  # matplotlib figure
        plt.close(fig)
    
    def test_plotly_backend(self, trained_classifier, classifier_config):
        """Test using plotly backend."""
        config = classifier_config.copy()
        config['plotting_backend'] = 'plotly'
        
        importance = ml_evaluation.get_feature_importance(
            trained_classifier['model'],
            config=config
        )
        
        fig = ml_evaluation.plot_feature_importance(
            importance,
            config=config
        )
        
        assert fig is not None
        assert hasattr(fig, 'show')  # plotly figure
    
    def test_backend_override(self, trained_classifier, classifier_config):
        """Test overriding config backend with explicit parameter."""
        config = classifier_config.copy()
        config['plotting_backend'] = 'plotly'
        
        importance = ml_evaluation.get_feature_importance(
            trained_classifier['model'],
            config=config
        )
        
        # Override with matplotlib
        fig = ml_evaluation.plot_feature_importance(
            importance,
            backend='matplotlib',
            config=config
        )
        
        assert fig is not None
        assert hasattr(fig, 'savefig')  # matplotlib figure
        plt.close(fig)
