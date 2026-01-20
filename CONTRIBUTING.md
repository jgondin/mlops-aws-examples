# Contributing to MLOps AWS Examples

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear, descriptive title
- Detailed steps to reproduce the issue
- Expected vs actual behavior
- Your environment (Python version, AWS region, etc.)
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear, descriptive title
- Detailed description of the proposed functionality
- Use cases and benefits
- Any relevant examples or mockups

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding standards** described below
3. **Add tests** for any new functionality
4. **Update documentation** to reflect your changes
5. **Ensure all tests pass** before submitting
6. **Write a clear commit message** describing your changes

#### Coding Standards

- Follow PEP 8 style guide for Python code
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular
- Add comments for complex logic

#### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run linting
flake8 src/ examples/
black --check src/ examples/
mypy src/
```

## Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mlops-aws-examples.git
cd mlops-aws-examples
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **Configure AWS credentials**
```bash
aws configure
```

## Project Structure

```
mlops-aws-examples/
├── examples/           # Complete example implementations
├── infrastructure/     # IaC templates (CloudFormation, Terraform)
├── src/               # Shared utilities and libraries
├── tests/             # Unit and integration tests
└── docs/              # Documentation
```

## Adding New Examples

When adding a new example:

1. Create a new directory under `examples/`
2. Include a comprehensive README.md
3. Add all necessary code, configs, and requirements
4. Ensure the example is self-contained and runnable
5. Include cleanup scripts to remove AWS resources
6. Document estimated costs
7. Add tests if applicable

## Documentation

- Use clear, concise language
- Include code examples
- Document AWS permissions required
- Provide cost estimates
- Include troubleshooting sections

## Commit Messages

Write clear, concise commit messages:

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat: Add batch inference example with Lambda

- Implement serverless batch inference pipeline
- Add CloudFormation template for Lambda deployment
- Include monitoring and error handling
- Add cost optimization with spot instances

Closes #123
```

## Review Process

1. Maintainers will review your PR within 7 days
2. Address any requested changes
3. Once approved, your PR will be merged

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue with your question or contact the maintainers directly.

Thank you for contributing! 🚀
