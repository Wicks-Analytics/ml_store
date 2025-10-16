# SQL Data Loading Guide

The `load_data()` function supports loading data directly from SQL databases, including Snowflake and Microsoft SQL Server.

## Table of Contents
- [Overview](#overview)
- [Supported Databases](#supported-databases)
- [Installation Requirements](#installation-requirements)
- [Configuration](#configuration)
- [Examples](#examples)
- [Security Best Practices](#security-best-practices)

## Overview

When `data_type` is set to `"sql"`, the `data_path` field is treated as a SQL query instead of a file path. The function executes the query against the specified database and returns the results as a Polars DataFrame.

## Supported Databases

### 1. Snowflake
- **db_type**: `"snowflake"`
- **Connector**: `snowflake-connector-python`
- **Features**: Full Snowflake SQL support, warehouse/role specification

### 2. Microsoft SQL Server
- **db_type**: `"mssql"`
- **Connector**: `pyodbc`
- **Features**: Full T-SQL support, Windows Authentication support

## Installation Requirements

### For Snowflake
```bash
pip install snowflake-connector-python
```

### For MS SQL Server
```bash
pip install pyodbc
```

**Note**: For MS SQL Server, you also need an ODBC driver installed:
- Windows: [Microsoft ODBC Driver for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
- Linux: Follow [Microsoft's installation guide](https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server)

## Configuration

### Snowflake Configuration

#### Option 1: Connection String
```json
{
  "data_path": "SELECT * FROM customers WHERE year = 2024",
  "data_type": "sql",
  "db_type": "snowflake",
  "db_connection": "snowflake://username:password@account_identifier/database/schema?warehouse=compute_wh&role=analyst_role"
}
```

#### Option 2: Connection Dictionary
```json
{
  "data_path": "SELECT * FROM customers WHERE year = 2024",
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
```

#### Connection String Format
```
snowflake://user:pass@account/database/schema?warehouse=wh&role=role_name
```

**Components:**
- `user`: Snowflake username
- `pass`: Password
- `account`: Account identifier (e.g., `xy12345.us-east-1`)
- `database`: Database name
- `schema`: Schema name
- `warehouse`: Compute warehouse (optional, can be in query params)
- `role`: Role to use (optional, can be in query params)

### MS SQL Server Configuration

#### Option 1: Connection String
```json
{
  "data_path": "SELECT * FROM sales WHERE date >= '2024-01-01'",
  "data_type": "sql",
  "db_type": "mssql",
  "db_connection": "mssql://username:password@server_name/database_name",
  "data_options": {
    "driver": "{ODBC Driver 17 for SQL Server}"
  }
}
```

#### Option 2: Raw ODBC Connection String
```json
{
  "data_path": "SELECT * FROM sales WHERE date >= '2024-01-01'",
  "data_type": "sql",
  "db_type": "mssql",
  "db_connection": "DRIVER={ODBC Driver 17 for SQL Server};SERVER=server_name;DATABASE=database_name;UID=username;PWD=password"
}
```

#### Windows Authentication
```json
{
  "data_path": "SELECT * FROM sales",
  "data_type": "sql",
  "db_type": "mssql",
  "db_connection": "mssql://server_name/database_name",
  "data_options": {
    "trusted_connection": "yes"
  }
}
```

## Examples

### Example 1: Basic Snowflake Query
```python
from ml_store import load_data

config = {
    "data_path": "SELECT * FROM customers WHERE active = TRUE",
    "data_type": "sql",
    "db_type": "snowflake",
    "db_connection": "snowflake://user:pass@account/db/schema?warehouse=wh"
}

df = load_data(config)
print(f"Loaded {len(df)} rows")
```

### Example 2: Complex Query with Joins
```python
config = {
    "data_path": """
        SELECT 
            c.customer_id,
            c.age,
            c.tenure,
            o.total_orders,
            o.total_spent
        FROM customers c
        LEFT JOIN order_summary o ON c.customer_id = o.customer_id
        WHERE c.signup_date >= '2024-01-01'
    """,
    "data_type": "sql",
    "db_type": "snowflake",
    "db_connection": {...}
}

df = load_data(config)
```

### Example 3: MS SQL Server with Parameters
```python
config = {
    "data_path": "SELECT * FROM sales WHERE year = 2024 AND region = 'North'",
    "data_type": "sql",
    "db_type": "mssql",
    "db_connection": "mssql://user:pass@server/database",
    "data_options": {
        "driver": "{ODBC Driver 17 for SQL Server}",
        "port": 1433
    }
}

df = load_data(config)
```

### Example 4: Complete ML Workflow
```python
from ml_store import load_config, load_data, create_modelling_data, train_model

# Load config with SQL settings
config = load_config("config_snowflake.json")

# Load data from database
df = load_data(config)
print(f"Loaded {len(df)} rows from Snowflake")

# Prepare modeling data
X_train, y_train, X_test, y_test, feature_names, cat_indices, _, _ = create_modelling_data(df, config)

# Train model
model = train_model(X_train, y_train, config=config, categorical_features=cat_indices)
```

## Security Best Practices

### 1. Use Environment Variables
Never hardcode credentials in config files:

```python
import os

config = {
    "db_connection": f"snowflake://{os.getenv('SF_USER')}:{os.getenv('SF_PASS')}@{os.getenv('SF_ACCOUNT')}/db/schema"
}
```

### 2. Use .env Files
Store credentials in a `.env` file (add to `.gitignore`):

```bash
# .env
SF_USER=myusername
SF_PASS=mypassword
SF_ACCOUNT=xy12345.us-east-1
```

Load with python-dotenv:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Use Key-Pair Authentication (Snowflake)
For production, use key-pair authentication:

```python
config = {
    "db_connection": {
        "user": "username",
        "account": "account",
        "private_key_path": "/secure/path/to/key.p8",
        "database": "db",
        "schema": "schema"
    }
}
```

### 4. Use Windows Authentication (MS SQL Server)
When possible, use Windows Authentication:

```python
config = {
    "db_connection": "mssql://server/database",
    "data_options": {
        "trusted_connection": "yes"
    }
}
```

### 5. Limit Query Scope
Always limit the data you query:

```sql
-- Good: Specific columns and filters
SELECT customer_id, age, tenure 
FROM customers 
WHERE year = 2024 
LIMIT 100000

-- Avoid: SELECT * without filters
SELECT * FROM large_table
```

### 6. Use Read-Only Accounts
Create database users with read-only permissions for ML workflows.

### 7. Connection Pooling
For repeated queries, consider connection pooling (advanced usage).

## Troubleshooting

### Snowflake Connection Issues
```
Error: 250001: Could not connect to Snowflake backend
```
**Solution**: Check account identifier format, ensure network access, verify credentials.

### MS SQL Server Driver Issues
```
Error: Data source name not found and no default driver specified
```
**Solution**: Install ODBC Driver 17 for SQL Server, specify correct driver in config.

### Query Timeout
For large queries, increase timeout in `data_options`:

```python
config = {
    "data_options": {
        "timeout": 300  # 5 minutes
    }
}
```

## Performance Tips

1. **Use WHERE clauses** to filter data at the database level
2. **Select only needed columns** instead of `SELECT *`
3. **Use LIMIT/TOP** for testing queries
4. **Create database views** for complex queries
5. **Use appropriate indexes** on filtered columns
6. **Consider materialized views** for frequently used queries

## Additional Resources

- [Snowflake Python Connector Documentation](https://docs.snowflake.com/en/user-guide/python-connector.html)
- [pyodbc Documentation](https://github.com/mkleehammer/pyodbc/wiki)
- [Polars read_database Documentation](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.read_database.html)
