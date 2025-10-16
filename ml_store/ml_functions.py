"""Standalone machine learning functions using MLFlow, CatBoost, and Polars."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.catboost
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from analytics_store import model_validation


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load model configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Dictionary containing the configuration

    Example config structure:
        {
            "model_type": "classifier",  # or "regressor"
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"],
            "categorical_features": ["cat_feature1"],
            "data_path": "data/train.csv",  # Optional: for load_data function
            "data_type": "csv",  # Optional: auto-detected if not provided
            "data_options": {},  # Optional: additional read options
            "plotting_backend": "plotly",  # Optional: "matplotlib" or "plotly"
            "model_params": {
                "iterations": 1000,
                "learning_rate": 0.1,
                "depth": 6
            },
            "mlflow": {
                "experiment_name": "my_experiment",
                "run_name": "run_1"
            }
        }
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ["model_type", "target_column"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    if config["model_type"] not in ["classifier", "regressor"]:
        raise ValueError(
            f"Invalid model_type: {config['model_type']}. Must be 'classifier' or 'regressor'"
        )

    return config


def load_data(config: Dict[str, Any]) -> pl.DataFrame:
    """
    Load data from file or database based on configuration.

    Args:
        config: Configuration dictionary containing:
            - data_path: Path to the data file OR SQL query (if data_type is "sql")
            - data_type: Type of data source - "csv", "parquet", "json", "excel", "feather", "ipc", "sql"
                        (optional, auto-detected from extension if not provided)
            - data_options: Optional dictionary of additional options to pass to the read function
            - db_connection: Database connection string (required if data_type is "sql")
            - db_type: Database type - "snowflake" or "mssql" (required if data_type is "sql")

    Returns:
        Polars DataFrame containing the loaded data

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data_path is missing or data_type is unsupported

    Example config for file:
        {
            "data_path": "data/train.csv",
            "data_type": "csv",
            "data_options": {
                "separator": ",",
                "null_values": ["NA", "NULL"]
            }
        }

    Example config for SQL:
        {
            "data_path": "SELECT * FROM sales WHERE year = 2024",
            "data_type": "sql",
            "db_type": "snowflake",
            "db_connection": "snowflake://user:pass@account/database/schema"
        }
    """
    if "data_path" not in config:
        raise ValueError("Missing required field in config: data_path")

    # Get data type from config or infer from file extension
    data_type = config.get("data_type")

    # For SQL, we don't check file existence
    if data_type and data_type.lower() == "sql":
        data_path_or_query = config["data_path"]
    else:
        data_path = Path(config["data_path"])

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data_path_or_query = data_path

        if data_type is None:
            # Infer from file extension
            extension = data_path.suffix.lower().lstrip(".")
            data_type = extension

    # Get optional data reading options
    data_options = config.get("data_options", {})

    # Load data based on type
    data_type = data_type.lower()

    try:
        if data_type == "csv":
            df = pl.read_csv(data_path_or_query, **data_options)
        elif data_type == "parquet":
            df = pl.read_parquet(data_path_or_query, **data_options)
        elif data_type == "json":
            df = pl.read_json(data_path_or_query, **data_options)
        elif data_type in ["excel", "xlsx", "xls"]:
            df = pl.read_excel(data_path_or_query, **data_options)
        elif data_type in ["feather", "ipc", "arrow"]:
            df = pl.read_ipc(data_path_or_query, **data_options)
        elif data_type == "ndjson":
            df = pl.read_ndjson(data_path_or_query, **data_options)
        elif data_type == "avro":
            df = pl.read_avro(data_path_or_query, **data_options)
        elif data_type == "sql":
            # Handle SQL database connections
            if "db_type" not in config:
                raise ValueError(
                    "Missing required field for SQL: db_type (must be 'snowflake' or 'mssql')"
                )
            if "db_connection" not in config:
                raise ValueError("Missing required field for SQL: db_connection")

            db_type = config["db_type"].lower()
            connection_string = config["db_connection"]
            query = data_path_or_query  # For SQL, data_path contains the query

            if db_type == "snowflake":
                # Use Polars read_database with Snowflake connector
                try:
                    import snowflake.connector
                except ImportError:
                    raise ImportError(
                        "snowflake-connector-python is required for Snowflake connections. "
                        "Install it with: pip install snowflake-connector-python"
                    )

                # Parse connection string or use connection dict
                if isinstance(connection_string, str):
                    # Parse snowflake://user:pass@account/database/schema?warehouse=wh
                    from urllib.parse import urlparse, parse_qs

                    parsed = urlparse(connection_string)

                    conn_params = {
                        "user": parsed.username,
                        "password": parsed.password,
                        "account": parsed.hostname,
                    }

                    # Parse path for database/schema
                    path_parts = parsed.path.strip("/").split("/")
                    if len(path_parts) >= 1 and path_parts[0]:
                        conn_params["database"] = path_parts[0]
                    if len(path_parts) >= 2 and path_parts[1]:
                        conn_params["schema"] = path_parts[1]

                    # Parse query params for warehouse, role, etc.
                    query_params = parse_qs(parsed.query)
                    if "warehouse" in query_params:
                        conn_params["warehouse"] = query_params["warehouse"][0]
                    if "role" in query_params:
                        conn_params["role"] = query_params["role"][0]

                    # Allow overriding with data_options
                    conn_params.update(data_options)
                else:
                    conn_params = connection_string

                # Create connection and execute query
                conn = snowflake.connector.connect(**conn_params)
                df = pl.read_database(query, connection=conn)
                conn.close()

            elif db_type == "mssql":
                # Use Polars read_database with pyodbc/pymssql
                try:
                    import pyodbc
                except ImportError:
                    raise ImportError(
                        "pyodbc is required for MS SQL Server connections. "
                        "Install it with: pip install pyodbc"
                    )

                # Connection string can be passed directly or constructed
                if "://" in connection_string:
                    # Parse mssql://user:pass@server/database
                    from urllib.parse import urlparse

                    parsed = urlparse(connection_string)

                    driver = data_options.get("driver", "{ODBC Driver 17 for SQL Server}")
                    conn_str = (
                        f"DRIVER={driver};"
                        f"SERVER={parsed.hostname};"
                        f"DATABASE={parsed.path.strip('/')};"
                        f"UID={parsed.username};"
                        f"PWD={parsed.password}"
                    )

                    # Add additional options
                    if "port" in data_options:
                        conn_str += f";PORT={data_options['port']}"
                    if "trusted_connection" in data_options:
                        conn_str += f";Trusted_Connection={data_options['trusted_connection']}"
                else:
                    conn_str = connection_string

                # Create connection and execute query
                conn = pyodbc.connect(conn_str)
                df = pl.read_database(query, connection=conn)
                conn.close()

            else:
                raise ValueError(
                    f"Unsupported db_type: {db_type}. Supported types: snowflake, mssql"
                )
        else:
            raise ValueError(
                f"Unsupported data_type: {data_type}. "
                f"Supported types: csv, parquet, json, ndjson, excel, feather, ipc, arrow, avro, sql"
            )
    except Exception as e:
        if data_type == "sql":
            raise RuntimeError(f"Error executing SQL query: {str(e)}")
        else:
            raise RuntimeError(
                f"Error reading {data_type} file from {data_path_or_query}: {str(e)}"
            )

    return df


def prepare_features(
    df: pl.DataFrame, config: Dict[str, Any], is_training: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], List[int], Optional[np.ndarray]]:
    """
    Prepare features and target from a Polars DataFrame.

    Args:
        df: Polars DataFrame
        config: Configuration dictionary
        is_training: Whether this is for training (includes target) or prediction

    Returns:
        Tuple of (X, y, feature_names, categorical_feature_indices, exposure)
        - X: Feature matrix as numpy array
        - y: Target array (None if is_training=False)
        - feature_names: List of feature column names
        - categorical_feature_indices: Indices of categorical features
        - exposure: Exposure array (None if not specified in config)
    """
    # Get feature columns
    if "feature_columns" in config and config["feature_columns"]:
        feature_columns = config["feature_columns"]
    else:
        # Use all columns except target and exposure
        exclude_cols = [config["target_column"]]
        if "exposure_column" in config:
            exclude_cols.append(config["exposure_column"])
        feature_columns = [col for col in df.columns if col not in exclude_cols]

    # Extract features
    X = df.select(feature_columns).to_numpy()

    # Extract target if training
    y = None
    if is_training:
        if config["target_column"] not in df.columns:
            raise ValueError(f"Target column '{config['target_column']}' not found in DataFrame")
        y = df[config["target_column"]].to_numpy()

    # Extract exposure if specified (for regression models, e.g., insurance)
    exposure = None
    if "exposure_column" in config and config["exposure_column"]:
        if config["exposure_column"] not in df.columns:
            raise ValueError(
                f"Exposure column '{config['exposure_column']}' not found in DataFrame"
            )
        exposure = df[config["exposure_column"]].to_numpy()

    # Get categorical feature indices
    categorical_features = config.get("categorical_features", [])
    categorical_indices = [
        i for i, col in enumerate(feature_columns) if col in categorical_features
    ]

    return X, y, feature_columns, categorical_indices, exposure


def assign_split(
    df: pl.DataFrame,
    config: Dict[str, Any],
    train_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    holdout_ratio: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Union[Tuple[pl.DataFrame, pl.DataFrame], Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
    """
    Split a DataFrame into train, test, and optionally holdout sets.

    Can perform either:
    1. Random splits based on ratios (from config or function parameters)
    2. Splits based on a split column with predefined values

    Holdout split is optional:
    - If only train_ratio and test_ratio are specified (sum to 1.0), returns (train_df, test_df)
    - If train_ratio, test_ratio, and holdout_ratio are specified, returns (train_df, test_df, holdout_df)

    Priority order for ratios:
    1. Config file (train_ratio, test_ratio, holdout_ratio)
    2. Function parameters
    3. Defaults (0.7, 0.15, 0.15) - includes holdout by default

    Priority order for random_seed:
    1. Config file (random_seed)
    2. Function parameter
    3. Default (42)

    Args:
        df: Polars DataFrame to split
        config: Configuration dictionary that may contain:
            - "split_column": Name of column containing split assignments
            - "split_values": Dict with keys "train", "test", "holdout" mapping to values
            - "train_ratio": Training set ratio (optional)
            - "test_ratio": Test set ratio (optional)
            - "holdout_ratio": Holdout set ratio (optional, omit for 2-way split)
            - "random_seed": Random seed for reproducibility (optional)
        train_ratio: Ratio for training set (default 0.7 if not in config)
        test_ratio: Ratio for test set (default 0.15 if not in config)
        holdout_ratio: Ratio for holdout set (default 0.15 if not in config, None for 2-way split)
        random_seed: Random seed for reproducibility (default 42 if not in config)

    Returns:
        Tuple of (train_df, test_df) if holdout not specified, or
        Tuple of (train_df, test_df, holdout_df) if holdout is specified

    Example config for 2-way split:
        {
            "train_ratio": 0.8,
            "test_ratio": 0.2,
            "random_seed": 42
        }

    Example config for 3-way split:
        {
            "train_ratio": 0.6,
            "test_ratio": 0.2,
            "holdout_ratio": 0.2,
            "random_seed": 123
        }

    Example config for column-based split:
        {
            "split_column": "dataset_split",
            "split_values": {
                "train": "TRAIN",
                "test": "TEST",
                "holdout": "HOLDOUT"
            }
        }
    """
    # Check if config specifies a split column
    if "split_column" in config and config["split_column"]:
        split_column = config["split_column"]

        if split_column not in df.columns:
            raise ValueError(f"Split column '{split_column}' not found in DataFrame")

        # Get split values from config
        split_values = config.get(
            "split_values", {"train": "train", "test": "test", "holdout": "holdout"}
        )

        # Split based on column values
        train_df = df.filter(pl.col(split_column) == split_values.get("train"))
        test_df = df.filter(pl.col(split_column) == split_values.get("test"))

        # Validate that we got data in train and test
        if train_df.height == 0:
            raise ValueError(
                f"No rows found for train split with value '{split_values.get('train')}'"
            )
        if test_df.height == 0:
            raise ValueError(
                f"No rows found for test split with value '{split_values.get('test')}'"
            )

        # Check if holdout is specified
        if "holdout" in split_values:
            holdout_df = df.filter(pl.col(split_column) == split_values.get("holdout"))
            if holdout_df.height == 0:
                raise ValueError(
                    f"No rows found for holdout split with value '{split_values.get('holdout')}'"
                )
            return train_df, test_df, holdout_df
        else:
            return train_df, test_df

    else:
        # Perform random split
        # Priority: config > function parameters > defaults
        # Check if holdout is specified in config or parameters
        has_holdout_in_config = "holdout_ratio" in config
        has_holdout_in_params = holdout_ratio is not None

        # Determine if we should use holdout
        use_holdout = (
            has_holdout_in_config
            or has_holdout_in_params
            or (
                "train_ratio" not in config
                and "test_ratio" not in config
                and train_ratio is None
                and test_ratio is None
            )
        )

        if use_holdout:
            # 3-way split
            final_train_ratio = config.get(
                "train_ratio", train_ratio if train_ratio is not None else 0.7
            )
            final_test_ratio = config.get(
                "test_ratio", test_ratio if test_ratio is not None else 0.15
            )
            final_holdout_ratio = config.get(
                "holdout_ratio", holdout_ratio if holdout_ratio is not None else 0.15
            )
            final_random_seed = config.get(
                "random_seed", random_seed if random_seed is not None else 42
            )

            # Validate ratios
            total_ratio = final_train_ratio + final_test_ratio + final_holdout_ratio
            if not np.isclose(total_ratio, 1.0):
                raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

            # Shuffle the dataframe with seed
            df = df.sample(fraction=1.0, seed=final_random_seed, shuffle=True)

            # Calculate split indices
            n = df.height
            train_end = int(n * final_train_ratio)
            test_end = train_end + int(n * final_test_ratio)

            # Split the dataframe
            train_df = df[:train_end]
            test_df = df[train_end:test_end]
            holdout_df = df[test_end:]

            return train_df, test_df, holdout_df
        else:
            # 2-way split
            final_train_ratio = config.get(
                "train_ratio", train_ratio if train_ratio is not None else 0.8
            )
            final_test_ratio = config.get(
                "test_ratio", test_ratio if test_ratio is not None else 0.2
            )
            final_random_seed = config.get(
                "random_seed", random_seed if random_seed is not None else 42
            )

            # Validate ratios
            total_ratio = final_train_ratio + final_test_ratio
            if not np.isclose(total_ratio, 1.0):
                raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

            # Shuffle the dataframe with seed
            df = df.sample(fraction=1.0, seed=final_random_seed, shuffle=True)

            # Calculate split indices
            n = df.height
            train_end = int(n * final_train_ratio)

            # Split the dataframe
            train_df = df[:train_end]
            test_df = df[train_end:]

            return train_df, test_df


def create_modelling_data(
    df: pl.DataFrame,
    config: Dict[str, Any],
    train_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    holdout_ratio: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Union[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[str],
        List[int],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ],
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[str],
        List[int],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ],
]:
    """
    Complete data preparation pipeline: split data and prepare features for modeling.

    This function combines assign_split() and prepare_features() to provide a one-step
    data preparation solution.

    Args:
        df: Polars DataFrame containing all data
        config: Configuration dictionary (may include "exposure_column" for regression)
        train_ratio: Training set ratio (optional)
        test_ratio: Test set ratio (optional)
        holdout_ratio: Holdout set ratio (optional, omit for 2-way split)
        random_seed: Random seed for reproducibility (optional)

    Returns:
        For 2-way split: (X_train, y_train, X_test, y_test, feature_names, cat_indices, exposure_train, exposure_test)
        For 3-way split: (X_train, y_train, X_test, y_test, X_holdout, y_holdout, feature_names, cat_indices, exposure_train, exposure_test, exposure_holdout)

        Exposure arrays are None if no exposure_column is specified in config.

    Example:
        # 2-way split without exposure
        config = {"train_ratio": 0.8, "test_ratio": 0.2, "target_column": "price"}
        X_train, y_train, X_test, y_test, features, cat_idx, exp_train, exp_test = create_modelling_data(df, config)

        # 2-way split with exposure (insurance use case)
        config = {"train_ratio": 0.8, "test_ratio": 0.2, "target_column": "claims", "exposure_column": "exposure"}
        X_train, y_train, X_test, y_test, features, cat_idx, exp_train, exp_test = create_modelling_data(df, config)
    """
    # Split the data
    split_result = assign_split(df, config, train_ratio, test_ratio, holdout_ratio, random_seed)

    # Check if 2-way or 3-way split
    if len(split_result) == 2:
        # 2-way split
        train_df, test_df = split_result

        # Prepare features for each split
        X_train, y_train, feature_names, cat_indices, exposure_train = prepare_features(
            train_df, config, is_training=True
        )
        X_test, y_test, _, _, exposure_test = prepare_features(test_df, config, is_training=True)

        return (
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            cat_indices,
            exposure_train,
            exposure_test,
        )
    else:
        # 3-way split
        train_df, test_df, holdout_df = split_result

        # Prepare features for each split
        X_train, y_train, feature_names, cat_indices, exposure_train = prepare_features(
            train_df, config, is_training=True
        )
        X_test, y_test, _, _, exposure_test = prepare_features(test_df, config, is_training=True)
        X_holdout, y_holdout, _, _, exposure_holdout = prepare_features(
            holdout_df, config, is_training=True
        )

        return (
            X_train,
            y_train,
            X_test,
            y_test,
            X_holdout,
            y_holdout,
            feature_names,
            cat_indices,
            exposure_train,
            exposure_test,
            exposure_holdout,
        )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    categorical_features: Optional[List[int]] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    exposure_train: Optional[np.ndarray] = None,
    exposure_val: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Union[CatBoostClassifier, CatBoostRegressor]:
    """
    Train a CatBoost model.

    Args:
        X_train: Training features
        y_train: Training target
        config: Configuration dictionary
        categorical_features: List of categorical feature indices
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        exposure_train: Training exposure (optional, for regression models like insurance)
        exposure_val: Validation exposure (optional)
        verbose: Whether to print training progress

    Returns:
        Trained CatBoost model

    Config structure:
        {
            "model_params": {
                "learning_params": {
                    "iterations": 1000,
                    "learning_rate": 0.1,
                    "depth": 6,
                    ...
                },
                "early_stopping_rounds": 10
            },
            "random_seed": 42,
            "exposure_column": "exposure"  # Optional, for insurance models
        }

    Note: For regression models with exposure, the log(exposure) is used as the baseline
          in the CatBoost Pool, which is standard practice in insurance modeling.
    """
    model_params_config = config.get("model_params", {})

    # Extract learning_params (model initialization parameters)
    learning_params = model_params_config.get("learning_params", {}).copy()

    # Add top-level random_seed if not in learning_params
    if "random_seed" in config and "random_seed" not in learning_params:
        learning_params["random_seed"] = config["random_seed"]

    # Override verbose if specified in function call
    if verbose is not None:
        learning_params["verbose"] = verbose

    # Create model based on type with learning_params
    if config["model_type"] == "classifier":
        model = CatBoostClassifier(**learning_params)
    else:
        model = CatBoostRegressor(**learning_params)

    # Prepare baseline (log of exposure) for regression models
    baseline_train = None
    if config["model_type"] == "regressor" and exposure_train is not None:
        baseline_train = np.log(exposure_train)

    # Create training pool
    train_pool = Pool(X_train, y_train, cat_features=categorical_features, baseline=baseline_train)

    # Create validation pool if provided
    eval_set = None
    if X_val is not None and y_val is not None:
        baseline_val = None
        if config["model_type"] == "regressor" and exposure_val is not None:
            baseline_val = np.log(exposure_val)

        eval_set = Pool(X_val, y_val, cat_features=categorical_features, baseline=baseline_val)

    # Get early_stopping_rounds from config (default to 25 if not specified)
    early_stopping_rounds = model_params_config.get("early_stopping_rounds", 25)

    # Train model with early stopping
    if eval_set is not None:
        model.fit(train_pool, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)
    else:
        model.fit(train_pool)

    return model


def evaluate_model(
    model: Union[CatBoostClassifier, CatBoostRegressor],
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
) -> pl.DataFrame:
    """
    Evaluate a trained model using analytics_store.model_validation.

    Args:
        model: Trained CatBoost model
        X: Features
        y: True target values
        model_type: 'classifier' or 'regressor'

    Returns:
        Polars DataFrame with metrics

    Example:
        # Get metrics DataFrame
        metrics_df = evaluate_model(model, X_test, y_test, "regressor")
        print(metrics_df)

        # Access specific metric
        mae = metrics_df["mae"][0]

    """
    # Get predictions
    predictions = model.predict(X)

    # Create DataFrame for analytics_store functions
    df = pl.DataFrame({"actual": y, "predicted": predictions})

    # Calculate metrics based on model type using analytics_store
    if model_type == "classifier":
        # Classification metrics using analytics_store
        classification_metrics = model_validation.calculate_classification_metrics(
            df, true_labels="actual", predicted_labels="predicted"
        )

        metrics = classification_metrics.to_polars()

        # Add ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            try:
                proba = model.predict_proba(X)[:, 1]
                df_with_proba = df.with_columns(pl.Series("score", proba))
                roc_result = model_validation.calculate_roc_curve(
                    df_with_proba, target_column="actual", score_column="score"
                )
                # Add ROC AUC to metrics DataFrame
                metrics = metrics.with_columns(pl.lit(roc_result.auc_score).alias("roc_auc"))
            except Exception:
                pass
    else:
        # Regression metrics using analytics_store
        regression_metrics = model_validation.calculate_regression_metrics(
            df, actual_column="actual", predicted_column="predicted"
        )

        metrics = regression_metrics.to_polars()

    return metrics


def log_to_mlflow(
    model: Union[CatBoostClassifier, CatBoostRegressor],
    config: Dict[str, Any],
    metrics: pl.DataFrame,
    feature_names: Optional[List[str]] = None,
    additional_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Log model, parameters, and metrics to MLFlow.

    Args:
        model: Trained CatBoost model
        config: Configuration dictionary
        metrics: Polars DataFrame of evaluation metrics (from evaluate_model)
        feature_names: List of feature names
        additional_params: Additional parameters to log

    Returns:
        MLFlow run ID
    """
    mlflow_config = config.get("mlflow", {})

    # Set experiment
    experiment_name = mlflow_config.get("experiment_name", "default")
    mlflow.set_experiment(experiment_name)

    # Start run
    run_name = mlflow_config.get("run_name", None)

    with mlflow.start_run(run_name=run_name) as run:
        # Log learning parameters from model_params.learning_params
        model_params_config = config.get("model_params", {})
        learning_params = model_params_config.get("learning_params", {})

        # Flatten learning_params for MLflow logging
        for param_name, param_value in learning_params.items():
            mlflow.log_param(f"learning_{param_name}", param_value)

        # Log early_stopping_rounds if present
        if "early_stopping_rounds" in model_params_config:
            mlflow.log_param("early_stopping_rounds", model_params_config["early_stopping_rounds"])

        # Log additional config parameters
        mlflow.log_param("model_type", config["model_type"])
        mlflow.log_param("target_column", config["target_column"])

        if "random_seed" in config:
            mlflow.log_param("random_seed", config["random_seed"])

        if feature_names:
            mlflow.log_param("num_features", len(feature_names))

        if additional_params:
            mlflow.log_params(additional_params)

        # Log metrics - convert DataFrame to dict
        metrics_dict = metrics.to_dicts()[0] if len(metrics) > 0 else {}
        mlflow.log_metrics(metrics_dict)

        # Log model
        mlflow.catboost.log_model(model, "model")

        # Log feature importance if available
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
            if feature_names and len(feature_names) == len(feature_importance):
                importance_dict = dict(zip(feature_names, feature_importance))
                # Log top 10 features
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
                for feat, imp in sorted_features:
                    mlflow.log_metric(f"importance_{feat}", imp)

        return run.info.run_id


def save_model(
    model: Union[CatBoostClassifier, CatBoostRegressor],
    save_path: Union[str, Path],
    model_format: str = "cbm",
) -> None:
    """
    Save a CatBoost model to disk.

    Args:
        model: Trained CatBoost model
        save_path: Path to save the model
        model_format: Format to save ('cbm', 'json', 'onnx', 'cpp', 'python')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(save_path), format=model_format)


def load_model(
    model_path: Union[str, Path], model_type: str = "classifier"
) -> Union[CatBoostClassifier, CatBoostRegressor]:
    """
    Load a CatBoost model from disk.

    Args:
        model_path: Path to the saved model
        model_type: 'classifier' or 'regressor'

    Returns:
        Loaded CatBoost model
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_type == "classifier":
        model = CatBoostClassifier()
    else:
        model = CatBoostRegressor()

    model.load_model(str(model_path))

    return model


def predict(
    model: Union[CatBoostClassifier, CatBoostRegressor], X: np.ndarray, return_proba: bool = False
) -> np.ndarray:
    """
    Make predictions using a trained model.

    Args:
        model: Trained CatBoost model
        X: Features
        return_proba: Whether to return probabilities (classifier only)

    Returns:
        Predictions array
    """
    if return_proba and isinstance(model, CatBoostClassifier):
        return model.predict_proba(X)
    else:
        return model.predict(X)
