"""ML Store - Machine Learning utilities with MLFlow, CatBoost, and Polars."""

__version__ = "0.1.0"

from .ml_functions import (
    assign_split,
    create_modelling_data,
    evaluate_model,
    load_config,
    load_data,
    load_model,
    log_to_mlflow,
    predict,
    prepare_features,
    save_model,
    train_model,
)

__all__ = [
    "load_config",
    "load_data",
    "prepare_features",
    "assign_split",
    "create_modelling_data",
    "train_model",
    "evaluate_model",
    "log_to_mlflow",
    "save_model",
    "load_model",
    "predict",
]
