"""Tests for ml_functions module."""

import pytest
import polars as pl
import numpy as np
import json
from pathlib import Path
from ml_store import (
    load_config,
    load_data,
    prepare_features,
    assign_split,
    create_modelling_data,
    train_model,
    predict,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_success(self, temp_config_file):
        """Test loading a valid config file."""
        config = load_config(temp_config_file)

        assert config["model_type"] == "classifier"
        assert config["target_column"] == "target"
        assert "feature_columns" in config

    def test_load_config_missing_file(self):
        """Test loading a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")

    def test_load_config_missing_required_field(self, tmp_path):
        """Test config with missing required field."""
        bad_config = {"feature_columns": ["f1", "f2"]}
        config_path = tmp_path / "bad_config.json"

        with open(config_path, "w") as f:
            json.dump(bad_config, f)

        with pytest.raises(ValueError, match="Missing required field"):
            load_config(config_path)

    def test_load_config_invalid_model_type(self, tmp_path):
        """Test config with invalid model_type."""
        bad_config = {"model_type": "invalid_type", "target_column": "target"}
        config_path = tmp_path / "bad_config.json"

        with open(config_path, "w") as f:
            json.dump(bad_config, f)

        with pytest.raises(ValueError, match="Invalid model_type"):
            load_config(config_path)


class TestLoadData:
    """Tests for load_data function."""

    def test_load_csv(self, temp_csv_file):
        """Test loading CSV file."""
        config = {"data_path": str(temp_csv_file), "data_type": "csv"}

        df = load_data(config)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_parquet(self, temp_parquet_file):
        """Test loading Parquet file."""
        config = {"data_path": str(temp_parquet_file), "data_type": "parquet"}

        df = load_data(config)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_auto_detect_csv(self, temp_csv_file):
        """Test auto-detecting CSV file type."""
        config = {"data_path": str(temp_csv_file)}

        df = load_data(config)
        assert isinstance(df, pl.DataFrame)

    def test_load_auto_detect_parquet(self, temp_parquet_file):
        """Test auto-detecting Parquet file type."""
        config = {"data_path": str(temp_parquet_file)}

        df = load_data(config)
        assert isinstance(df, pl.DataFrame)

    def test_load_missing_file(self):
        """Test loading non-existent file."""
        config = {"data_path": "nonexistent_file.csv", "data_type": "csv"}

        with pytest.raises(FileNotFoundError):
            load_data(config)

    def test_load_missing_data_path(self):
        """Test config without data_path."""
        config = {"data_type": "csv"}

        with pytest.raises(ValueError, match="Missing required field"):
            load_data(config)

    def test_load_unsupported_type(self, temp_csv_file):
        """Test unsupported data type."""
        config = {"data_path": str(temp_csv_file), "data_type": "unsupported_format"}

        with pytest.raises((ValueError, RuntimeError)):
            load_data(config)


class TestPrepareFeatures:
    """Tests for prepare_features function."""

    def test_prepare_features_training(self, sample_classification_data, classifier_config):
        """Test preparing features for training."""
        X, y, feature_names, cat_indices, exposure = prepare_features(
            sample_classification_data, classifier_config, is_training=True
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert len(feature_names) == X.shape[1]
        assert len(cat_indices) == len(classifier_config["categorical_features"])

    def test_prepare_features_prediction(self, sample_classification_data, classifier_config):
        """Test preparing features for prediction."""
        X, y, feature_names, cat_indices, exposure = prepare_features(
            sample_classification_data, classifier_config, is_training=False
        )

        assert isinstance(X, np.ndarray)
        assert y is None
        assert len(feature_names) == X.shape[1]

    def test_prepare_features_missing_target(self, sample_classification_data, classifier_config):
        """Test with missing target column during training."""
        df_no_target = sample_classification_data.drop("target")

        with pytest.raises(ValueError):
            prepare_features(df_no_target, classifier_config, is_training=True)


class TestAssignSplit:
    """Tests for assign_split function."""

    def test_assign_split_basic(self, sample_classification_data):
        """Test basic split assignment."""
        config = {"train_ratio": 0.7, "test_ratio": 0.3, "random_seed": 42}
        train_df, test_df = assign_split(sample_classification_data, config)

        total = len(train_df) + len(test_df)
        assert abs(len(train_df) / total - 0.7) < 0.05
        assert abs(len(test_df) / total - 0.3) < 0.05

    def test_assign_split_with_holdout(self, sample_classification_data):
        """Test split with holdout set."""
        config = {"train_ratio": 0.6, "test_ratio": 0.2, "holdout_ratio": 0.2, "random_seed": 42}
        train_df, test_df, holdout_df = assign_split(sample_classification_data, config)

        total = len(train_df) + len(test_df) + len(holdout_df)
        assert abs(len(train_df) / total - 0.6) < 0.05
        assert abs(len(test_df) / total - 0.2) < 0.05
        assert abs(len(holdout_df) / total - 0.2) < 0.05

    def test_assign_split_reproducibility(self, sample_classification_data):
        """Test that same seed produces same split."""
        config = {"train_ratio": 0.7, "test_ratio": 0.3, "random_seed": 42}
        train_df1, test_df1 = assign_split(sample_classification_data, config)
        train_df2, test_df2 = assign_split(sample_classification_data, config)

        assert len(train_df1) == len(train_df2)
        assert len(test_df1) == len(test_df2)

    def test_assign_split_invalid_ratios(self, sample_classification_data):
        """Test with invalid ratio sum."""
        config = {"train_ratio": 0.7, "test_ratio": 0.5}
        with pytest.raises(ValueError):
            assign_split(sample_classification_data, config)


class TestCreateModellingData:
    """Tests for create_modelling_data function."""

    def test_create_modelling_data_basic(self, sample_classification_data, classifier_config):
        """Test basic data preparation."""
        X_train, y_train, X_test, y_test, feature_names, cat_indices, exp_train, exp_test = (
            create_modelling_data(sample_classification_data, classifier_config)
        )

        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_test, np.ndarray)

        total_samples = len(X_train) + len(X_test)
        assert abs(len(X_train) / total_samples - 0.7) < 0.05

    def test_create_modelling_data_with_split_column(
        self, sample_classification_data, classifier_config
    ):
        """Test with pre-existing split column."""
        # Create split column manually for this test
        import numpy as np

        np.random.seed(42)
        n = len(sample_classification_data)
        splits = np.random.choice(["TRAIN", "TEST"], size=n, p=[0.7, 0.3])
        df_with_split = sample_classification_data.with_columns(pl.Series("dataset_split", splits))

        config_with_split = classifier_config.copy()
        config_with_split["split_column"] = "dataset_split"
        config_with_split["split_values"] = {"train": "TRAIN", "test": "TEST"}

        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            df_with_split, config_with_split
        )

        assert len(X_train) > 0
        assert len(X_test) > 0


class TestTrainModel:
    """Tests for train_model function."""

    def test_train_classifier(self, sample_classification_data, classifier_config):
        """Test training a classifier."""
        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            sample_classification_data, classifier_config
        )

        model = train_model(
            X_train, y_train, config=classifier_config, categorical_features=cat_indices, verbose=0
        )

        assert model is not None
        assert hasattr(model, "predict")

    def test_train_regressor(self, sample_regression_data, regressor_config):
        """Test training a regressor."""
        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            sample_regression_data, regressor_config
        )

        model = train_model(
            X_train, y_train, config=regressor_config, categorical_features=cat_indices, verbose=0
        )

        assert model is not None
        assert hasattr(model, "predict")

    def test_train_with_validation(self, sample_classification_data, classifier_config):
        """Test training with validation set."""
        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(
            sample_classification_data, classifier_config
        )

        model = train_model(
            X_train,
            y_train,
            config=classifier_config,
            categorical_features=cat_indices,
            X_val=X_test,
            y_val=y_test,
            verbose=0,
        )

        assert model is not None


class TestPredict:
    """Tests for predict function."""

    def test_predict_classifier(self, trained_classifier):
        """Test predictions from classifier."""
        predictions = predict(trained_classifier["model"], trained_classifier["X_test"])

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(trained_classifier["X_test"])
        assert all((predictions >= 0) & (predictions <= 1))

    def test_predict_regressor(self, trained_regressor):
        """Test predictions from regressor."""
        predictions = predict(trained_regressor["model"], trained_regressor["X_test"])

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(trained_regressor["X_test"])

    def test_predict_with_array(self, trained_classifier):
        """Test predictions with array input."""
        predictions = predict(trained_classifier["model"], trained_classifier["X_test"])

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(trained_classifier["X_test"])
