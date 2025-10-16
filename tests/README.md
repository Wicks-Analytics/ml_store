# ML Store Tests

This directory contains comprehensive tests for the ml_store package.

## Test Structure

```
tests/
├── __init__.py                 # Package marker
├── conftest.py                 # Pytest fixtures and configuration
├── test_ml_functions.py        # Tests for ml_functions module
├── test_ml_evaluation.py       # Tests for ml_evaluation module
├── test_integration.py         # Integration tests
└── README.md                   # This file
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/test_ml_functions.py
```

### Run specific test class
```bash
pytest tests/test_ml_functions.py::TestLoadConfig
```

### Run specific test
```bash
pytest tests/test_ml_functions.py::TestLoadConfig::test_load_config_success
```

### Run tests by marker
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Run with coverage (if pytest-cov is installed)
```bash
pytest --cov=ml_store --cov-report=html
```

## Test Categories

### Unit Tests
- **test_ml_functions.py**: Tests for individual functions in ml_functions module
  - Config loading and validation
  - Data loading (CSV, Parquet, JSON, SQL)
  - Feature preparation
  - Split assignment
  - Model training
  - Predictions

- **test_ml_evaluation.py**: Tests for evaluation and analysis functions
  - Feature importance
  - SHAP analysis
  - Partial dependence
  - Evaluation reports
  - Plotting backends

### Integration Tests
- **test_integration.py**: End-to-end workflow tests
  - Complete classifier workflow
  - Complete regressor workflow
  - Config-driven workflows
  - Split column workflows
  - Different data format workflows
  - Full evaluation workflows

## Fixtures

The `conftest.py` file provides reusable fixtures:

- `sample_classification_data`: Sample binary classification dataset
- `sample_regression_data`: Sample regression dataset
- `classifier_config`: Sample classifier configuration
- `regressor_config`: Sample regressor configuration
- `temp_config_file`: Temporary config file
- `temp_csv_file`: Temporary CSV file
- `temp_parquet_file`: Temporary Parquet file
- `trained_classifier`: Pre-trained classifier model
- `trained_regressor`: Pre-trained regressor model

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (slower, test multiple components)
- `@pytest.mark.slow`: Slow running tests (can be skipped for quick testing)

## Writing New Tests

### Example test structure:
```python
import pytest
from ml_store import function_to_test

class TestFeatureName:
    """Tests for specific feature."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic use case."""
        result = function_to_test(fixture_name)
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Best practices:
1. Use descriptive test names
2. Test one thing per test
3. Use fixtures for common setup
4. Test both success and failure cases
5. Clean up resources (close figures, delete temp files)
6. Add markers for categorization

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Ensure:
- All dependencies are installed
- Tests are deterministic (use random seeds)
- Temporary files are cleaned up
- Tests can run in parallel (when possible)

## Coverage Goals

Aim for:
- **Unit tests**: >80% code coverage
- **Integration tests**: Cover all major workflows
- **Edge cases**: Test error handling and validation

## Troubleshooting

### Tests fail with import errors
```bash
# Install package in development mode
pip install -e .
```

### Tests fail with missing dependencies
```bash
# Install test dependencies
pip install pytest pytest-cov
```

### Slow tests timeout
```bash
# Skip slow tests
pytest -m "not slow"
```

### Random test failures
- Check if tests use random seeds
- Ensure tests don't depend on execution order
- Clean up shared resources between tests
