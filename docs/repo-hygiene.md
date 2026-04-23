# Repository Hygiene

This page records the current cleanup policy for keeping NTX reviewable while
preserving benchmark reproducibility.

## Current State

The tracked worktree is clean after the latest ship-readiness commits. No
tracked benchmark artifact was removed during the cleanup pass.

The following reproducible local artifacts were removed from the NTX worktree:

- Python bytecode caches,
- `.mypy_cache`, `.pytest_cache`, and `.ruff_cache`,
- local `.DS_Store` files,
- `docs/_build`,
- ignored `examples/outputs/*` rerun directories and sample output files.

The 2.9 GB literature/reference checkout
`20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current`
was moved out of the NTX repository root to:

```text
/Users/rogeriojorge/local/20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current
```

That keeps the reference material available locally without making the NTX
worktree look dirty or inflating repository-local tooling scans.

## Artifact Policy

Keep generated artifacts only when they are tied to all of the following:

1. a script that regenerates them,
2. a test or benchmark-matrix row that checks them,
3. documentation or manuscript text that states the claim they support.

Ignored rerun output under `examples/outputs/` is disposable. The committed
publication and validation artifacts under `docs/_static/` are the reviewed
artifact surface.

Do not commit local caches, profiling scratch directories, generated docs HTML,
wheel/sdist build directories, or external reference-code checkouts.

## CI Lane Manifest

Every `tests/test_*.py` file must belong to exactly one maintained lane:

```bash
python scripts/test_lane_manifest.py --check
python scripts/test_lane_manifest.py core_foundation
python scripts/test_lane_manifest.py core_cli_workflows
python scripts/test_lane_manifest.py core_io_workflows
python scripts/test_lane_manifest.py core_parallel_workflows
python scripts/test_lane_manifest.py core_neopax_workflows
python scripts/test_lane_manifest.py core_profile_workflows
python scripts/test_lane_manifest.py core_autodiff_profile_workflows
python scripts/test_lane_manifest.py core_validation
python scripts/test_lane_manifest.py integration_examples
python scripts/test_lane_manifest.py heavy_examples_profiles
python scripts/test_lane_manifest.py heavy_examples_derivatives
python scripts/test_lane_manifest.py heavy_examples_boundary
python scripts/test_lane_manifest.py heavy_examples_publication
```

The split core lanes replace the previous monolithic `core` lane:

- `core_foundation`: algebra, geometry, operator, solver, and helper unit tests,
- `core_cli_workflows`: small public API, CLI, namespace, packaging, and
  example-discovery tests,
- `core_io_workflows`: input-file, profile-script, and VMEC scan workflow tests,
- `core_parallel_workflows`: CPU/GPU script-dispatch and multiprocessing
  workflow tests,
- `core_neopax_workflows`: imported-database mapping and HDF5 round-trip tests,
- `core_profile_workflows`: profile and primitive-force workflow tests,
- `core_autodiff_profile_workflows`: autodiff profile and robust-bootstrap
  workflow tests,
- `core_validation`: benchmark registry, validation, artifact, and physics-gate
  tests.

Local measured shard checks during the split:

- `core_foundation`: `97 passed` in about `1:23` with coverage,
- `core_cli_workflows`: `20 passed` in about `0:22` with coverage,
- `core_io_workflows`: `12 passed` in about `0:15` with coverage,
- `core_parallel_workflows`: `3 passed`, `2 deselected` in about `0:22` with coverage,
- `core_neopax_workflows`: `20 passed` in about `0:23` with coverage,
- `core_profile_workflows`: `16 passed` in about `4:08` with coverage,
- `core_autodiff_profile_workflows`: `2 passed` in about `0:41` with coverage,
- `core_validation`: `56 passed` in about `0:31` with coverage.

If any workflow shard grows beyond the CI wall-time target, split it again
before adding more example-style coverage.

## Verification Order

Use this sequence before merging or tagging:

```bash
python scripts/test_lane_manifest.py --check
python -m ruff check .
python -m mypy src/ntx
python scripts/build_benchmark_matrix.py
python scripts/build_manuscript_artifacts.py
python -m sphinx -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

Then run the relevant lane for the commit being prepared, for example:

```bash
python scripts/test_lane_manifest.py core_validation \
  | xargs python -m pytest -q -m "not gpu"
```

The boundary/equilibrium rerun lane remains opt-in for artifact refreshes:

```bash
NTX_RUN_HEAVY_BOUNDARY_EXAMPLES=1 \
  python scripts/test_lane_manifest.py heavy_examples_boundary \
  | xargs python -m pytest -q -m "not gpu"
```
