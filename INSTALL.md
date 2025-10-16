# Installation Guide

## Requirements

- Python >= 3.12
- pip or uv package manager

## Installation Methods

### 1. Install from PyPI (Recommended)

```bash
pip install ml-store
```

### 2. Install with Optional Dependencies

#### SQL Support (Snowflake and MS SQL Server)
```bash
pip install ml-store[sql]
```

#### Development Dependencies
```bash
pip install ml-store[dev]
```

#### All Optional Dependencies
```bash
pip install ml-store[all]
```

### 3. Install from GitHub

#### Latest Release
```bash
pip install git+https://github.com/Wicks-Analytics/ml_store.git
```

#### Specific Branch
```bash
pip install git+https://github.com/Wicks-Analytics/ml_store.git@develop
```

#### Specific Tag/Version
```bash
pip install git+https://github.com/Wicks-Analytics/ml_store.git@v0.1.0
```

### 4. Install from Source (Development)

```bash
# Clone the repository
git clone https://github.com/Wicks-Analytics/ml_store.git
cd ml_store

# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --dev

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 5. Install with uv (Fast Package Manager)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ml-store
uv pip install ml-store

# Or create a project with ml-store
uv init my-ml-project
cd my-ml-project
uv add ml-store
```

## Verifying Installation

```python
import ml_store

# Check version
print(ml_store.__version__)

# Import main functions
from ml_store import (
    load_config,
    load_data,
    train_model,
    evaluate_model,
)

print("ml_store installed successfully!")
```

## Dependencies

### Core Dependencies
- mlflow >= 2.18.0
- catboost >= 1.2.7
- polars >= 1.13.0
- scikit-learn >= 1.5.2
- numpy >= 2.1.3
- analytics-store
- shap >= 0.41.0
- matplotlib >= 3.4.0
- plotly >= 5.0.0

### Optional Dependencies

#### SQL Support
- snowflake-connector-python >= 3.0.0
- pyodbc >= 4.0.0

#### Development
- pytest >= 8.0.0
- pytest-cov >= 4.0.0
- black >= 23.0.0
- ruff >= 0.1.0
- ipykernel >= 7.0.0

## Platform-Specific Notes

### Windows

For MS SQL Server support, you need to install the ODBC Driver:
1. Download [Microsoft ODBC Driver for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
2. Install the driver
3. Then install ml-store with SQL support:
   ```bash
   pip install ml-store[sql]
   ```

### macOS

For MS SQL Server support on macOS:
```bash
# Install ODBC driver using Homebrew
brew install unixodbc
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew install msodbcsql18 mssql-tools18

# Then install ml-store
pip install ml-store[sql]
```

### Linux

For MS SQL Server support on Linux:
```bash
# Ubuntu/Debian
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Then install ml-store
pip install ml-store[sql]
```

## Troubleshooting

### Import Errors

If you encounter import errors:
```bash
# Reinstall with --force-reinstall
pip install --force-reinstall ml-store

# Or clear cache and reinstall
pip cache purge
pip install ml-store
```

### Dependency Conflicts

If you have dependency conflicts:
```bash
# Create a fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate
pip install ml-store
```

### MLFlow Issues

If MLFlow tracking doesn't work:
```bash
# Check MLFlow installation
mlflow --version

# Reinstall MLFlow
pip install --upgrade mlflow
```

### CatBoost Issues

If CatBoost fails to install:
```bash
# Install build dependencies
pip install --upgrade pip setuptools wheel

# Then install CatBoost
pip install catboost

# Finally install ml-store
pip install ml-store
```

## Upgrading

### Upgrade to Latest Version
```bash
pip install --upgrade ml-store
```

### Upgrade with Optional Dependencies
```bash
pip install --upgrade ml-store[all]
```

## Uninstalling

```bash
pip uninstall ml-store
```

## Docker Installation

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install ml-store
RUN pip install ml-store[all]

# Set working directory
WORKDIR /app

# Copy your code
COPY . /app

CMD ["python", "your_script.py"]
```

Build and run:
```bash
docker build -t ml-store-app .
docker run ml-store-app
```

## Next Steps

After installation:
1. Check out the [README](README.md) for usage examples
2. Explore [example scripts](examples/)
3. Read the [SQL data loading guide](docs/SQL_DATA_LOADING.md)
4. Try the [example configurations](config/)

## Getting Help

If you encounter issues:
1. Check the [troubleshooting section](#troubleshooting)
2. Search [existing issues](https://github.com/Wicks-Analytics/ml_store/issues)
3. Open a [new issue](https://github.com/Wicks-Analytics/ml_store/issues/new)
