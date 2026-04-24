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

6. Let the GitHub release workflow build the distributions, attach them to the
   tag release, and publish them to PyPI through Trusted Publishing.

## CI/CD Release Path

- `tests.yml` covers lint, type checking, unit/integration tests, and docs.
- `package.yml` builds the wheel and sdist, runs `twine check`, and smoke-tests
  installation across the supported Python versions.
- `release.yml` runs on `v*` tags, publishes the built artifacts to the GitHub
  release, and uploads the same artifacts to PyPI through the protected `pypi`
  environment.

## PyPI Readiness

Before publishing to PyPI:

1. keep the base dependency set restricted to packages installable from normal
   indexes;
2. keep Git direct references out of package metadata; geometry-coupled
   workflows remain documentation-only optional installs until the upstream
   packages are available from standard indexes under stable version
   constraints;
3. keep build and publish jobs separate so the exact tested artifacts are the
   artifacts that are uploaded;
4. configure PyPI Trusted Publishing for the GitHub repository, release
   workflow, and a protected `pypi` environment.

The repository-side Trusted Publishing job is now present in
`.github/workflows/release.yml`. It is tag-gated, downloads the exact
distribution artifact built by the release job, and publishes through
`pypa/gh-action-pypi-publish` without a long-lived API token. On
2026-04-24, `python -m pip index versions ntx` returned no matching
distribution, so the intended package name was not visible on the default PyPI
index from this workstation. The PyPI project and trusted publisher still need
to be created/configured in PyPI before the first tag release.

The intended public install surface is:

```bash
python -m pip install ntx
python -m pip install "ntx[io]"
```

Geometry-coupled examples should remain documented optional workflows until the
upstream geometry packages are available from standard package indexes under
stable version constraints.

## Scope

The current shipping target is the monoenergetic NTX package itself. Shipping a
full external geometry stack remains optional and is documented as an external
install, while repository-owned file I/O support remains available through
`.[io]`.
