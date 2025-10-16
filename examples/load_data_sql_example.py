"""Example demonstrating SQL data loading with Snowflake and MS SQL Server."""

from pathlib import Path
from ml_store import load_config, load_data, create_modelling_data, train_model

print("=" * 70)
print("SQL Data Loading Examples")
print("=" * 70)
print()

# =============================================================================
# Example 1: Snowflake Connection
# =============================================================================
print("Example 1: Loading data from Snowflake")
print("-" * 70)

snowflake_config = {
    "model_type": "classifier",
    "target_column": "churn",
    "data_path": "SELECT * FROM customers WHERE year = 2024",
    "data_type": "sql",
    "db_type": "snowflake",
    "db_connection": "snowflake://user:pass@account/database/schema?warehouse=wh",
    "data_options": {
        "warehouse": "COMPUTE_WH",
        "role": "ANALYST_ROLE"
    },
    "feature_columns": ["age", "tenure", "monthly_charges"],
    "categorical_features": []
}

print("Config:")
print(f"  Database: Snowflake")
print(f"  Query: {snowflake_config['data_path'][:50]}...")
print(f"  Warehouse: {snowflake_config['data_options']['warehouse']}")
print()

# Load data from Snowflake
try:
    df_snowflake = load_data(snowflake_config)
    print(f"✓ Successfully loaded {df_snowflake.shape[0]} rows from Snowflake")
    print(f"  Columns: {df_snowflake.columns}")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    print("  Note: Requires snowflake-connector-python package")

print()

# =============================================================================
# Example 2: MS SQL Server Connection
# =============================================================================
print("Example 2: Loading data from MS SQL Server")
print("-" * 70)

mssql_config = {
    "model_type": "regressor",
    "target_column": "sales_amount",
    "data_path": """
        SELECT 
            product_id,
            price,
            quantity,
            discount,
            sales_amount
        FROM sales
        WHERE date >= '2024-01-01'
    """,
    "data_type": "sql",
    "db_type": "mssql",
    "db_connection": "mssql://username:password@server/database",
    "data_options": {
        "driver": "{ODBC Driver 17 for SQL Server}",
        "trusted_connection": "no"
    },
    "feature_columns": ["price", "quantity", "discount"],
    "categorical_features": []
}

print("Config:")
print(f"  Database: MS SQL Server")
print(f"  Query: {mssql_config['data_path'].strip()[:50]}...")
print(f"  Driver: {mssql_config['data_options']['driver']}")
print()

# Load data from MS SQL Server
try:
    df_mssql = load_data(mssql_config)
    print(f"✓ Successfully loaded {df_mssql.shape[0]} rows from MS SQL Server")
    print(f"  Columns: {df_mssql.columns}")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    print("  Note: Requires pyodbc package")

print()

# =============================================================================
# Example 3: Alternative Connection String Formats
# =============================================================================
print("Example 3: Alternative connection string formats")
print("-" * 70)

# Snowflake with dict instead of connection string
snowflake_dict_config = {
    "data_path": "SELECT * FROM table",
    "data_type": "sql",
    "db_type": "snowflake",
    "db_connection": {
        "user": "username",
        "password": "password",
        "account": "account_identifier",
        "database": "my_database",
        "schema": "my_schema",
        "warehouse": "COMPUTE_WH",
        "role": "ANALYST_ROLE"
    }
}

print("Snowflake with connection dict:")
print(f"  Account: {snowflake_dict_config['db_connection']['account']}")
print(f"  Database: {snowflake_dict_config['db_connection']['database']}")
print()

# MS SQL Server with raw ODBC connection string
mssql_odbc_config = {
    "data_path": "SELECT * FROM table",
    "data_type": "sql",
    "db_type": "mssql",
    "db_connection": (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=server_name;"
        "DATABASE=database_name;"
        "UID=username;"
        "PWD=password"
    )
}

print("MS SQL Server with raw ODBC string:")
print(f"  Connection: {mssql_odbc_config['db_connection'][:60]}...")
print()

# =============================================================================
# Example 4: Complete Workflow with SQL Data
# =============================================================================
print("Example 4: Complete ML workflow with SQL data")
print("-" * 70)

# Load config from file
config_path = Path("../config/example_snowflake_config.json")
if config_path.exists():
    config = load_config(config_path)
    print(f"✓ Loaded config from {config_path.name}")
    
    # Load data from database
    try:
        df = load_data(config)
        print(f"✓ Loaded {df.shape[0]} rows from {config['db_type']}")
        
        # Prepare modeling data
        X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(df, config)
        print(f"✓ Prepared data: {len(X_train)} train, {len(X_test)} test samples")
        
        # Train model
        model = train_model(X_train, y_train, config=config, categorical_features=cat_indices)
        print(f"✓ Model trained successfully")
        
        print("\nComplete workflow executed successfully!")
    except Exception as e:
        print(f"✗ Error in workflow: {str(e)}")
else:
    print(f"Config file not found: {config_path}")

print()

# =============================================================================
# Example 5: Security Best Practices
# =============================================================================
print("Example 5: Security best practices")
print("-" * 70)
print("""
Best practices for SQL connections:

1. Use environment variables for credentials:
   import os
   config = {
       "db_connection": f"snowflake://{os.getenv('SF_USER')}:{os.getenv('SF_PASS')}@..."
   }

2. Use connection config files (not in git):
   # Store in .env or separate config file
   # Add to .gitignore

3. Use key-pair authentication (Snowflake):
   config = {
       "db_connection": {
           "user": "username",
           "account": "account",
           "private_key_path": "/path/to/key.p8"
       }
   }

4. Use Windows Authentication (MS SQL Server):
   config = {
       "db_connection": "mssql://server/database",
       "data_options": {
           "trusted_connection": "yes"
       }
   }

5. Limit query scope:
   - Use WHERE clauses to filter data
   - Select only needed columns
   - Use LIMIT/TOP for testing
""")

print("=" * 70)
print("Examples complete!")
print("=" * 70)
