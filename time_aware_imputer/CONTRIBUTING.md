# Contributing to Time-Aware Missing Data Imputer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
```bash
   git clone https://github.com/ontedduabhishakereddy/time-aware-imputer.git
   cd time-aware-imputer
```
3. **Add upstream remote**:
```bash
   git remote add upstream https://github.com/original/time-aware-imputer.git
```

## Development Setup

1. **Create a virtual environment**:
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install development dependencies**:
```bash
   pip install -r requirements-dev.txt
   pip install -e .
```

3. **Verify installation**:
```bash
   python -c "import time_aware_imputer; print(time_aware_imputer.__version__)"
```

## Making Changes

1. **Create a new branch**:
```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
```

2. **Make your changes** following our code style guidelines

3. **Add tests** for new functionality

4. **Update documentation** if needed

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=time_aware_imputer --cov-report=html

# Run specific test file
pytest tests/test_spline.py

# Run specific test
pytest tests/test_spline.py::TestSplineImputer::test_fit_transform_linear
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Aim for >90% code coverage

Example test:
```python
def test_my_feature():
    """Test description."""
    imputer = SplineImputer(method='cubic')
    result = imputer.fit_transform(data)
    assert not result['value'].isna().any()
```

## Code Style

We follow strict code quality standards:

### Formatting
```bash
# Format code with Black
black time_aware_imputer tests

# Sort imports with isort
isort time_aware_imputer tests
```

### Linting
```bash
# Check with flake8
flake8 time_aware_imputer tests

# Type check with mypy
mypy time_aware_imputer
```

### Style Guidelines

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted with isort
- **Type hints**: Required for all functions
- **Docstrings**: Google style for all public methods
- **Comments**: Explain "why", not "what"

Example function:
```python
def my_function(param: str, value: int = 0) -> bool:
    """Short description.

    Longer description if needed.

    Parameters
    ----------
    param : str
        Description of param.
    value : int, optional (default=0)
        Description of value.

    Returns
    -------
    result : bool
        Description of return value.
    """
    # Implementation
    return True
```

## Submitting Changes

1. **Run all quality checks**:
```bash
   # Format
   black time_aware_imputer tests
   isort time_aware_imputer tests
   
   # Check
   flake8 time_aware_imputer tests
   mypy time_aware_imputer
   
   # Test
   pytest
```

2. **Commit your changes**:
```bash
   git add .
   git commit -m "Add feature: brief description"
```
   
   Commit message format:
   - `Add feature: description` - New features
   - `Fix: description` - Bug fixes
   - `Docs: description` - Documentation changes
   - `Refactor: description` - Code refactoring
   - `Test: description` - Test additions/changes

3. **Push to your fork**:
```bash
   git push origin feature/your-feature-name
```

4. **Create a Pull Request**:
   - Go to GitHub and create a PR from your fork
   - Fill out the PR template
   - Link any related issues
   - Wait for review

## Pull Request Guidelines

- **One feature per PR**: Keep changes focused
- **Update tests**: Add/modify tests for your changes
- **Update docs**: Update README or docstrings if needed
- **Pass CI**: All checks must pass
- **Respond to feedback**: Address review comments promptly

## Types of Contributions

### Bug Reports

- Use GitHub Issues
- Include minimal reproducible example
- Specify version and environment
- Describe expected vs actual behavior

### Feature Requests

- Use GitHub Issues
- Explain the use case
- Describe proposed solution
- Discuss alternatives if any

### Code Contributions

Priority areas:
- Bug fixes
- Performance improvements
- Documentation improvements
- Test coverage
- New interpolation methods
- Additional imputation strategies

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Security issues**: Email maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉