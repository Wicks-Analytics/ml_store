"""Example demonstrating the load_data function."""

from pathlib import Path
from ml_store import load_config, load_data

# Example 1: Load data with explicit data_path and data_type in config
print("=" * 60)
print("Example 1: Load CSV data from config")
print("=" * 60)

config = {
    "model_type": "classifier",
    "target_column": "target",
    "data_path": "data/train.csv",
    "data_type": "csv",
    "data_options": {
        "separator": ",",
        "null_values": ["NA", "NULL"]
    }
}

# Load data using config
df = load_data(config)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
print(f"Columns: {df.columns}")
print()

# Example 2: Auto-detect file type from extension
print("=" * 60)
print("Example 2: Auto-detect file type from extension")
print("=" * 60)

config_auto = {
    "model_type": "regressor",
    "target_column": "price",
    "data_path": "data/housing.parquet"  # No data_type specified
}

# File type will be auto-detected as 'parquet' from the extension
df_parquet = load_data(config_auto)
print(f"Auto-detected parquet file")
print(f"Loaded {df_parquet.shape[0]} rows and {df_parquet.shape[1]} columns")
print()

# Example 3: Load from config file
print("=" * 60)
print("Example 3: Load from config file")
print("=" * 60)

config_path = Path("../config/example_with_data_loading.json")
config_from_file = load_config(config_path)

# Load data specified in the config file
df_from_config = load_data(config_from_file)
print(f"Loaded data from config file")
print(f"Data path: {config_from_file['data_path']}")
print(f"Data type: {config_from_file.get('data_type', 'auto-detected')}")
print(f"Shape: {df_from_config.shape}")
print()

# Example 4: Different file types
print("=" * 60)
print("Example 4: Supported file types")
print("=" * 60)

supported_types = {
    "CSV": {"data_path": "data.csv", "data_type": "csv"},
    "Parquet": {"data_path": "data.parquet", "data_type": "parquet"},
    "JSON": {"data_path": "data.json", "data_type": "json"},
    "NDJSON": {"data_path": "data.ndjson", "data_type": "ndjson"},
    "Excel": {"data_path": "data.xlsx", "data_type": "excel"},
    "Feather/IPC": {"data_path": "data.feather", "data_type": "feather"},
    "Avro": {"data_path": "data.avro", "data_type": "avro"},
    "SQL (Snowflake)": {"data_path": "SELECT * FROM table", "data_type": "sql", "db_type": "snowflake"},
    "SQL (MS SQL Server)": {"data_path": "SELECT * FROM table", "data_type": "sql", "db_type": "mssql"}
}

print("Supported file types:")
for name, example in supported_types.items():
    print(f"  - {name}: data_type='{example['data_type']}'")
print()

# Example 5: With additional read options
print("=" * 60)
print("Example 5: CSV with custom options")
print("=" * 60)

config_custom = {
    "model_type": "classifier",
    "target_column": "target",
    "data_path": "data/custom.csv",
    "data_type": "csv",
    "data_options": {
        "separator": ";",  # Semicolon separator
        "has_header": True,
        "null_values": ["NA", "NULL", "N/A", ""],
        "skip_rows": 1,  # Skip first row
        "n_rows": 10000,  # Only read first 10k rows
        "infer_schema_length": 5000  # Use first 5k rows for schema inference
    }
}

print("Config with custom CSV options:")
print(f"  - Separator: {config_custom['data_options']['separator']}")
print(f"  - Null values: {config_custom['data_options']['null_values']}")
print(f"  - Skip rows: {config_custom['data_options']['skip_rows']}")
print(f"  - Max rows: {config_custom['data_options']['n_rows']}")
print()

print("=" * 60)
print("Complete workflow example")
print("=" * 60)

# Complete workflow: config -> load data -> prepare features -> train
from ml_store import create_modelling_data, train_model

# 1. Load config
config = load_config(Path("../config/example_with_data_loading.json"))
print(f"✓ Loaded config: {config['model_type']} model")

# 2. Load data from config
df = load_data(config)
print(f"✓ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# 3. Prepare modeling data
X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(df, config)
print(f"✓ Prepared data: {len(X_train)} train, {len(X_test)} test samples")

# 4. Train model
model = train_model(X_train, y_train, config=config, categorical_features=cat_indices)
print(f"✓ Model trained successfully")

print("\nWorkflow complete!")
