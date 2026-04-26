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

5. Confirm the version is not already tagged.

6. Create an annotated tag:

   ```bash
   git tag -a v0.2.3 -m "NTX 0.2.3"
   git push origin v0.2.3
   ```

7. Let the GitHub release workflow build the distributions, attach them to the
   tag release, and publish them to PyPI through Trusted Publishing.

## Current Release

The current release is `0.2.3`.

Verified locally on 2026-04-26:

- `python -m ruff check .`
- `python -m mypy src/ntx`
- `python -m pytest -q`: `336 passed, 5 skipped`
- `python -m sphinx -b html docs docs/_build/html`
- `python -m build`
- `python -m twine check dist/*`
- clean-venv wheel smoke test for `ntx --help`, `python -m ntx --help`, and
  `python -c "import ntx; from ntx import GridSpec"`

The previous `v0.2.0` tag workflows were green for `tests`, `package`, and
`release`, and published successfully through Trusted Publishing. The `v0.2.3`
release uses the same tag-gated workflow and supersedes `v0.2.2` with
file-backed NetCDF output, CLI plotting, and output-path selected
NetCDF/NPZ/HDF5 serialization:

- [GitHub release v0.2.3](https://github.com/uwplasma/NTX/releases/tag/v0.2.3)
- [PyPI ntx 0.2.3](https://pypi.org/project/ntx/0.2.3/)

## CI/CD Release Path

- `tests.yml` covers lint, type checking, unit/integration tests, and docs.
- `package.yml` builds the wheel and sdist, runs `twine check`, and smoke-tests
  installation across the supported Python versions.
- `release.yml` runs on `v*` tags, publishes the built artifacts to the GitHub
  release, and uploads the same artifacts to PyPI through the protected `pypi`
  environment.

## PyPI Publishing

Before publishing a release to PyPI:

1. keep the base dependency set restricted to packages installable from normal
   indexes;
2. keep Git direct references out of package metadata; geometry-coupled
   workflows remain documentation-only optional installs until the upstream
   packages are available from standard indexes under stable version
   constraints;
3. keep build and publish jobs separate so the exact tested artifacts are the
   artifacts that are uploaded;
4. configure PyPI Trusted Publishing for the GitHub repository and release
   workflow. The GitHub release workflow publishes through the repository
   `pypi` environment; PyPI may accept any environment name when the trusted
   publisher entry is configured that way.

The repository-side Trusted Publishing job is present in
`.github/workflows/release.yml`. It is tag-gated, downloads the exact
distribution artifact built by the release job, and publishes through
`pypa/gh-action-pypi-publish` without a long-lived API token.

On 2026-04-24, PyPI Trusted Publishing was configured for repository
`uwplasma/NTX`, workflow `release.yml`, and project name `NTX` with no
environment-name restriction. The `v0.2.0` tag release published successfully
to PyPI the same day.

The intended public install surface is:

```bash
pip install ntx
pip install "ntx[io]"
```

Geometry-coupled examples should remain documented optional workflows until the
upstream geometry packages are available from standard package indexes under
stable version constraints.

## Scope

The current shipping target is the monoenergetic NTX package itself. Shipping a
full external geometry stack remains optional and is documented as an external
install, while repository-owned file I/O support remains available through
`.[io]`.
