# Contributing to Gyraph

Thank you for your interest in contributing to the Directed Graph Signal Processing Framework! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear and descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, package versions)
- Any relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear and descriptive title
- A detailed description of the proposed enhancement
- Any relevant examples or use cases
- Why this enhancement would be useful to most users

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install dependencies**: `pip install -e ".[dev]"`
3. **Make your changes**:
   - Write clear, commented code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
4. **Run tests**: Ensure all tests pass with `python -m unittest discover -s tests/`
5. **Run linting**: Use pre-commit hooks or run `pre-commit run --all-files`
6. **Commit your changes**: Follow conventional commit format (see below)
7. **Push to your fork** and submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/digraph_GSP_framework.git
cd digraph_GSP_framework

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- Follow PEP 8 style guide for Python code
- Use descriptive variable names
- Add docstrings to all public functions and classes
- Keep functions focused and concise
- Comment complex algorithms or non-obvious code

### Docstring Format

Use NumPy-style docstrings:

```python
def function_name(param1, param2):
    """
    Brief description of function.

    More detailed description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    return_type
        Description of return value

    Examples
    --------
    >>> function_name(1, 2)
    3
    """
    return param1 + param2
```

## Commit Message Format

This project uses the automatic commit message refiner. Your commits should follow the conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(filters): add Chebyshev polynomial filter

Implement Chebyshev polynomial approximation for graph filters
to enable efficient computation on large graphs.

Closes #123
```

## Testing

- Write unit tests for all new functionality
- Ensure tests are independent and can run in any order
- Use descriptive test names that explain what is being tested
- Place tests in the `tests/` directory with matching structure to source
- Aim for high code coverage (target: >70%)

```bash
# Run all tests
python -m unittest discover -s tests/

# Run specific test file
python -m unittest tests/test_filters.py

# Run with coverage
coverage run -m unittest discover -s tests/
coverage report -m
```

## Documentation

- Update documentation for any user-facing changes
- Add examples for new features
- Update the CHANGELOG.md for significant changes
- Documentation files are in `docs/`
- Example notebooks go in `examples/`

## Project Structure

```
digraph_GSP_framework/
├── gyraph/              # Main package source code
│   ├── graphs/           # Graph construction and management
│   ├── operators/        # Graph operators
│   ├── filters/          # Filtering implementations
│   ├── sampling/         # Sampling strategies
│   ├── learning/         # Graph learning algorithms
│   ├── surrogates/       # Random surrogate generation
│   ├── source_separation/# Signal separation
│   ├── neural_net/       # Neural network models
│   └── utils/            # Utility functions
├── tests/                # Unit tests
├── examples/             # Example scripts and notebooks
│   ├── basic/            # Basic usage examples
│   ├── advanced/         # Advanced techniques
│   └── applications/     # Real-world applications
├── docs/                 # Documentation
├── data/                 # Sample datasets
└── scripts/              # Utility scripts

```

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose:

1. A maintainer will review your PR
2. They may request changes or ask questions
3. Address any feedback
4. Once approved, your PR will be merged

## Questions?

Feel free to open an issue for any questions about contributing!

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
