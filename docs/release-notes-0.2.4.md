# NTX 0.2.4 Release Notes

NTX `0.2.4` is a repository-size and package-distribution hardening release on
top of `0.2.3`.

## Shipping Scope

This release keeps the public Python, CLI, and solver APIs stable. The main
change is operational: fresh clones and PyPI downloads are smaller, while the
current README/docs figures and validation artifacts remain available in the
repository.

## User-Facing Changes

- Fresh repository clones no longer carry old large NetCDF fixture blobs or
  repeated historical `docs/_static` generated-artifact blobs.
- Built distributions exclude `docs/_static`, so PyPI wheel and sdist downloads
  stay focused on installable source code.
- CI now rejects tracked files larger than `2 MiB`, caps the tracked tree size,
  and caps committed docs artifacts.
- Repository hygiene docs now explain the rewritten history and collaborator
  reset/re-clone guidance.

## History-Rewrite Note

The public Git history was rewritten before this release. Existing clones should
be replaced with a fresh clone or deliberately reset against the new
`origin/main`.

## Release Checks

Verified locally on 2026-05-26 before tagging:

- `python -m ruff check .`
- `python scripts/test_lane_manifest.py --check`
- `JAX_ENABLE_X64=True python -m pytest tests/test_repository_size.py tests/test_ci_lane_manifest.py -q`
- `JAX_ENABLE_X64=True python -m pytest tests/test_benchmark_matrix.py tests/test_physics_gates.py tests/test_repository_size.py -q`
- `python -m sphinx -b html docs docs/_build/html`
- `python -m build --outdir /tmp/ntx-dist-0.2.4`
- `python -m twine check /tmp/ntx-dist-0.2.4/*`
- clean-venv wheel smoke test for `ntx --help`, `python -m ntx --help`, and
  `python -c "import ntx; from ntx import GridSpec"`
- built artifact sizes: `151K` wheel and `557K` sdist
- clean installed package footprint: about `1.6M` under `site-packages/ntx`
- `docs/_static` absent from both wheel and sdist

GitHub `tests` and `package` workflows were green on the rewritten `main`
before tagging.

## Tagging

The published tag is `v0.2.4`.
