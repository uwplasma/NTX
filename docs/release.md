# Release

NTX is now set up to be shipped as a normal Python package rather than only as a
research checkout.

## Release Checklist

1. Run the local verification suite:

   ```bash
   python -m ruff check .
   python -m mypy src/ntx
   python -m pytest -q
   python -m sphinx -b html docs docs/_build/html
   python -m build
   python -m twine check dist/*
   ```

2. Smoke-test the built wheel in a clean environment:

   ```bash
   python -m venv .venv-release
   . .venv-release/bin/activate
   python -m pip install --upgrade pip
   python -m pip install dist/*.whl
   ntx --help
   python -m ntx --help
   ```

3. Update
   [CHANGELOG.md](https://github.com/uwplasma/NTX/blob/main/CHANGELOG.md).

4. Push `main` and confirm the `tests` and `package` workflows are green.

5. Create an annotated tag:

   ```bash
   git tag -a v0.1.0 -m "NTX 0.1.0"
   git push origin v0.1.0
   ```

6. Let the GitHub release workflow build the distributions and attach them to
   the tag release.

## CI/CD Release Path

- `tests.yml` covers lint, type checking, unit/integration tests, and docs.
- `package.yml` builds the wheel and sdist, runs `twine check`, and smoke-tests
  installation across the supported Python versions.
- `release.yml` runs on `v*` tags and publishes the built artifacts to the
  GitHub release.

## Scope

The current shipping target is the monoenergetic NTX package itself. Shipping a
full external geometry stack remains optional and is handled through extras such
as `.[geometry]` and `.[io]`.
