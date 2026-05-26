# Repository Hygiene

This page records the current cleanup policy for keeping NTX reviewable while
preserving benchmark reproducibility.

## Current State

The tracked worktree is clean after the latest ship-readiness commits. No
tracked benchmark artifact was removed during the cleanup pass.

The canonical local checkout for commits, pushes, and collaborator-facing
relative paths is:

```text
/Users/rogeriojorge/local/NTX
```

Do not maintain a second writable NTX checkout under nearby code directories.
If a sibling project needs `../NTX/...`, use a symlink to the canonical checkout
rather than a stale clone. The local NEOPAX sibling path currently follows that
policy:

```text
/Users/rogeriojorge/local/several_codes/NTX -> /Users/rogeriojorge/local/NTX
```

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

## Size Policy

The current tracked tree is intentionally small:

- no tracked file in `HEAD` is larger than `2 MiB`,
- the tracked logical tree size is about `19.5 MiB`,
- `docs/_static` is about `17 MiB`,
- `tests/fixtures` is about `68 KiB`.

`tests/test_repository_size.py` enforces the current policy:

- every tracked file must stay `<= 2 MiB`,
- the tracked tree must stay `<= 25 MiB`,
- `docs/_static` must stay `<= 20 MiB`.

If a new validation input, profile, trace, figure, or database would exceed
those limits, do not commit it directly. Prefer one of:

- regenerate it from a script and store only the compact JSON summary and a
  small PNG,
- keep it as an ignored local artifact under `examples/outputs/`,
- attach it to a GitHub release or external archive and link it from docs,
- or add a tiny fixture that exercises the same code path.

The local working directory can still grow much larger because ignored
profiling traces and rerun outputs are useful during development. They are safe
to remove with:

```bash
git clean -fdX
```

The current clone size is larger than the tracked tree because Git history still
contains older large blobs. The largest historical blobs are removed NetCDF
fixtures under `tests/fixtures/`, and the largest accumulated history category
is repeated generated artifacts under `docs/_static/`. A normal commit cannot
remove those historical blobs from clone transfer.

To make fresh clones substantially smaller, the project would need a coordinated
history rewrite, for example with `git filter-repo`, followed by a force push
and collaborator re-clones. That is a repository-maintenance decision, not a
normal feature commit. Until then, users who only need the latest source can use
a shallow or partial clone:

```bash
git clone --depth 1 https://github.com/uwplasma/NTX.git
git clone --filter=blob:none https://github.com/uwplasma/NTX.git
```

## CI Lane Manifest

Every `tests/test_*.py` file must be covered by the maintained lane manifest.
Files can be assigned as whole files or, for slow modules, split by explicit
pytest node ids. The manifest validator rejects duplicate node selections,
missing node selections inside split files, and mixed whole-file/node-id
assignments for the same file:

```bash
python scripts/test_lane_manifest.py --check
python scripts/test_lane_manifest.py core_foundation
python scripts/test_lane_manifest.py core_cli_workflows
python scripts/test_lane_manifest.py core_io_workflows
python scripts/test_lane_manifest.py core_parallel_workflows
python scripts/test_lane_manifest.py core_neopax_workflows
python scripts/test_lane_manifest.py core_profile_audit_workflow
python scripts/test_lane_manifest.py core_profile_basic_workflows
python scripts/test_lane_manifest.py core_profile_optimization_workflows
python scripts/test_lane_manifest.py core_profile_transport_workflows
python scripts/test_lane_manifest.py core_autodiff_uncertainty_workflow
python scripts/test_lane_manifest.py core_robust_bootstrap_workflow
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
- `core_profile_audit_workflow`: primitive-force audit artifact workflow test,
- `core_profile_basic_workflows`: ambipolar-profile and profile-family tests,
- `core_profile_optimization_workflows`: scalar and basis profile-control tests,
- `core_profile_transport_workflows`: conservative profile-transport loop tests,
- `core_autodiff_uncertainty_workflow`: autodiff uncertainty artifact workflow,
- `core_robust_bootstrap_workflow`: robust-bootstrap artifact workflow,
- `core_validation`: benchmark registry, validation, artifact, and physics-gate
  tests.

Local measured shard checks during the split:

- `core_foundation`: `97 passed` in about `1:23` with coverage,
- `core_cli_workflows`: `20 passed` in about `0:22` with coverage,
- `core_io_workflows`: `12 passed` in about `0:15` with coverage,
- `core_parallel_workflows`: `3 passed`, `2 deselected` in about `0:22` with coverage,
- `core_neopax_workflows`: `20 passed` in about `0:23` with coverage,
- `core_profile_audit_workflow`: `1 passed` in about `0:02` with coverage,
- `core_profile_basic_workflows`: `5 passed` in about `0:29` with coverage,
- `core_profile_optimization_workflows`: `4 passed` in about `2:10` with coverage,
- `core_profile_transport_workflows`: `6 passed` in about `1:27` with coverage,
- `core_autodiff_uncertainty_workflow`: `1 passed` in about `0:08` with coverage,
- `core_robust_bootstrap_workflow`: `1 passed` in about `0:19` with coverage,
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
