# Contributing to ML Store

Thank you for your interest in contributing to ML Store! This document provides guidelines and instructions for contributing.

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Wicks-Analytics/ml_store.git
cd ml_store
```

### 2. Create Virtual Environment
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Using uv
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks (Optional)
```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ml_store --cov-report=html

# Run specific test file
pytest tests/test_ml_functions.py

# Run tests by marker
pytest -m unit
pytest -m integration
```

### Code Formatting
```bash
# Format code with black
black ml_store tests

# Lint with ruff
ruff check ml_store tests
```

### Type Checking (Optional)
```bash
mypy ml_store
```

## Contribution Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md if adding new features
- Add examples for new functionality

### Testing
- Write tests for all new features
- Maintain or improve code coverage
- Ensure all tests pass before submitting PR
- Add integration tests for end-to-end workflows

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Reference issue numbers when applicable

Example:
```
Add SQL data loading support for Snowflake

- Implement load_data function with SQL support
- Add Snowflake connection string parsing
- Update documentation with SQL examples

Fixes #123
```

### Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

3. **Run Tests**
   ```bash
   pytest
   black ml_store tests
   ruff check ml_store tests
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add your feature"
   ```

5. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Go to GitHub and create a PR
   - Fill out the PR template
   - Link related issues
   - Wait for review

### PR Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

## Types of Contributions

### Bug Reports
- Use the issue tracker
- Include minimal reproducible example
- Specify Python version and dependencies
- Include error messages and stack traces

### Feature Requests
- Use the issue tracker
- Describe the use case
- Explain why it would be useful
- Consider implementation approach

### Documentation
- Fix typos and clarify explanations
- Add examples and tutorials
- Improve API documentation
- Update README and guides

### Code Contributions
- Bug fixes
- New features
- Performance improvements
- Refactoring

## Project Structure

```
ml_store/
├── ml_store/           # Main package
│   ├── __init__.py
│   ├── ml_functions.py # Core ML functions
│   └── ml_evaluation.py # Evaluation utilities
├── tests/              # Test suite
│   ├── conftest.py
│   ├── test_ml_functions.py
│   ├── test_ml_evaluation.py
│   └── test_integration.py
├── examples/           # Usage examples
├── config/             # Sample configurations
├── docs/               # Documentation
├── pyproject.toml      # Package configuration
└── README.md           # Main documentation
```

## Code Review Process

1. Maintainers will review PRs within 1-2 weeks
2. Address review comments
3. Once approved, maintainer will merge
4. Your contribution will be in the next release!

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

## Getting Help

- Open an issue for questions
- Check existing issues and documentation
- Reach out to maintainers

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- Release notes
- README.md contributors section
- GitHub contributors page

Thank you for contributing to ML Store! 🎉
