# Contributing to DeepSteer

Welcome! We appreciate your interest in contributing to DeepSteer.

## How to Contribute

1. **Fork** the repository on GitHub and clone it with a blobless filter (the history
   carries paper artifacts you do not need up front):
   `git clone --filter=blob:none https://github.com/<you>/deepsteer.git`
2. **Create a branch** for your change (`git checkout -b my-feature`)
3. **Make your changes** and commit them with clear, descriptive messages
4. **Run the tests** to ensure nothing is broken (see Testing below)
5. **Push** your branch and open a **Pull Request** against `main`

## Code Style

- Python 3.10+, type hints on all public functions
- Use `from __future__ import annotations` in every file
- Line length: 100 characters (enforced by ruff)
- Docstrings: Google style on all public functions and classes
- Use `logging.getLogger(__name__)` instead of `print()` in library code

Run the linter before submitting:

```bash
ruff check .    # same command and scope as CI (papers/ is excluded in pyproject.toml)
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -m "not slow and not regression"   # fast tests, what CI runs
pytest tests/ -m slow                             # downloads real models
pytest tests/ -m regression                       # reproduces paper outputs (needs weights)
```

## Contributor License Agreement

Before we can accept your contribution, you must agree to our Contributor
License Agreement (CLA). This ensures that Distiller Labs LLC has the
necessary rights to distribute your contribution under the Apache 2.0
license.

By submitting a pull request, you agree to the terms in [CLA.md](CLA.md).
