# Ship Checklist

This page contains release decisions only. Research priorities and implementation
order belong in the repository-root
[`plan.md`](https://github.com/uwplasma/NTX/blob/main/plan.md); promoted and
planned scientific claims belong in the [benchmark matrix](benchmark-matrix.md).

## Release Scope

Before tagging a release, confirm that:

- public APIs and documented workflows match the package being built;
- every promoted physics claim has a script, test, artifact, threshold, and
  documentation owner in the benchmark matrix;
- research diagnostics that do not pass their promotion gates are described as
  non-shipping and are absent from release claims;
- the canonical checkout is clean and contains no generated caches, local
  profiling dumps, or large unowned fixtures;
- CI coverage remains at least 95%, with physics-driven tests preferred over
  low-value line coverage.

## Verification

Run the same bounded checks used by CI:

```bash
python -m ruff check .
python -m mypy src/ntx
python scripts/test_lane_manifest.py --check
python scripts/check_physics_gates.py
python scripts/build_benchmark_matrix.py
python scripts/build_manuscript_artifacts.py
python -m sphinx -W -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

Run the registered test shards rather than one monolithic test process. The
shard manifest keeps memory bounded and assigns every test to exactly one CI
lane. Confirm that all GitHub `tests` and `package` jobs finish successfully and
that aggregate repository-owned coverage remains at least 95%.

## Artifact Review

1. Regenerate only artifacts owned by changed scripts or gates.
2. Compare JSON metrics and provenance with the previous committed versions.
3. Visually inspect every changed PNG and PDF.
4. Confirm that publication and benchmark manifests reference the same files.
5. Reject generated artifacts without an owning script, test, and documentation
   entry.

## Package Smoke Test

Install the built wheel in a clean environment and verify both entry points:

```bash
python -m venv .venv-release
. .venv-release/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/*.whl
ntx --help
python -m ntx --help
python -c "import ntx; from ntx import GridSpec"
```

Also inspect wheel and source-distribution contents and run the repository-size
guard. Generated documentation, validation images, and external geometry inputs
must not enter the installed wheel.

## Tag And Publish

1. Update the version in `pyproject.toml` and the fallback version in
   `src/ntx/__init__.py`.
2. Update `CHANGELOG.md` and the versioned release notes.
3. Merge only after required checks pass on the release commit.
4. Create and push an annotated `vX.Y.Z` tag.
5. Confirm that `release.yml` publishes the exact tested distributions to the
   GitHub release and PyPI through Trusted Publishing.
6. Install `ntx==X.Y.Z` from PyPI in a fresh environment and rerun the import and
   CLI smoke tests.

The current package and automation details are maintained in
[Release](release.md).
