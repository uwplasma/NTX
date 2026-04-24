# Testing And Quality Assurance

NTX is validated at four levels:

1. unit tests of geometry, operators, and solver algebra
2. regression tests of file-driven workflows and publication examples
3. imported-workflow tests for autodiff, NEOPAX, and JAX geometry backends
4. CPU/GPU runtime and smoke checks

The repository now treats these as separate execution lanes, not one monolithic
developer loop:

- fast PR lane:
  - lint
  - type checking
  - sharded core unit/workflow/validation tests
  - selected integration/example tests
  - docs build
- benchmark lane:
  - literature reproductions
  - long-running fixed-field and W7-X audits
  - scaling/profiling studies
- hardware lane:
  - GPU smoke
  - office workstation profiling

This split is intentional. It keeps pull-request feedback fast while still
tracking literature-grade validation and throughput studies in reproducible
scripts.

## Current Testing Priorities

The fast lane has already reached the coverage target, so new tests should be
chosen for scientific value first. Do not add slow examples only to increase the
headline coverage number.

Near-term high-value gates are:

- monoenergetic convergence ladders for `D11`, `D31`, `D33`, and
  `onsager_residual` on small repository-owned geometries,
- symmetric-limit checks where a constant-field Boozer surface has no radial
  transport channels but retains the finite parallel-conductivity branch,
- artifact-backed reproductions of larger W7-X, precise-QS, and QI-family
  literature cases,
- derivative audits that compare direct AD, prepared implicit-adjoint
  derivatives, forward-mode geometry controls, and centered finite differences
  on the same scalar outputs,
- profile and bootstrap-current workflow tests that are run only after the
  underlying coefficient and derivative gates are already green,
- explicit tests that fixed-field closure comparisons remain stress gates unless
  they also pass the integrated W7-X transfer gate.

Every new benchmark-like test must declare its lane before it is added to CI:

- `core_foundation`: small algebra, geometry, operator, solver, and helper unit tests,
- `core_cli_workflows`: small public API, CLI, namespace, packaging, and example-discovery tests,
- `core_io_workflows`: input-file, profile-script, and VMEC scan workflow tests,
- `core_parallel_workflows`: CPU/GPU script-dispatch and multiprocessing workflow tests,
- `core_neopax_workflows`: imported-database mapping and HDF5 round-trip tests,
- `core_profile_audit_workflow`: primitive-force audit artifact workflow test,
- `core_profile_basic_workflows`: ambipolar-profile and profile-family tests,
- `core_profile_optimization_workflows`: scalar and basis profile-control tests,
- `core_profile_transport_workflows`: conservative profile-transport loop tests,
- `core_autodiff_uncertainty_workflow`: autodiff uncertainty artifact workflow,
- `core_robust_bootstrap_workflow`: robust-bootstrap artifact workflow,
- `core_validation`: small validation, artifact-registry, and physics-gate tests,
- `integration_examples`: representative imported workflow tests,
- `heavy_examples_profiles`: slower profile examples,
- `heavy_examples_derivatives`: local derivative benchmark examples,
- `heavy_examples_boundary`: env-gated imported boundary/equilibrium derivative
  examples,
- `heavy_examples_publication`: publication/manuscript figure examples,
- manual benchmark: long literature reproduction or hardware profiling.

The GitHub Actions sharding is driven by the maintained manifest:

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

Any new `tests/test_*.py` file must be added to the lane manifest. Slow modules
may be split by explicit pytest node ids, but every discovered test function in
that file must be listed exactly once. This prevents new benchmark scripts from
silently landing in the fast core shard.

The imported boundary/equilibrium example reruns are intentionally opt-in:

```bash
NTX_RUN_HEAVY_BOUNDARY_EXAMPLES=1 \
  python scripts/test_lane_manifest.py heavy_examples_boundary \
  | xargs python -m pytest -q -m "not gpu"
```

Normal CI still checks the committed artifacts through the benchmark matrix and
manuscript-artifact tests. The expensive boundary reruns are used when updating
those artifacts.

The first fast owned-geometry coefficient gate is in
`tests/test_physics_gates.py`. It checks `D11`, `D31`, `D33`, the Onsager
residual, and a coarse-to-fine angular-grid transfer on the analytic Boozer
surface. Larger literature-family convergence ladders remain artifact-backed
work rather than default pull-request tests.

The same file also carries the symmetric-limit normalization gates: a
constant-field Boozer surface must have zero radial transport channels, finite
parallel conductivity, exact agreement with the `D33_spitzer` branch, and the
expected inverse-collisionality scaling of that Spitzer branch.

The operator unit tests also include a derivative gate for the implicit-adjoint
lane: the hand-coded `dD_k/dnu_hat` and `dD_k/depsi_hat` blocks must match
JAX differentiation of the assembled Legendre-space operator. This is a fast
way to protect sensitivity-analysis, inverse-design, and uncertainty workflows
from normalization drift in the collisionality and radial-electric-field terms.

The physics-facing gate structure is documented separately in
[`physics-gates.md`](physics-gates.md). The test suite and benchmark scripts are
meant to enforce that gate hierarchy, not to replace it.

The maintained claim-to-artifact map is documented in
[`benchmark-matrix.md`](benchmark-matrix.md) and can be regenerated with:

```bash
python scripts/build_benchmark_matrix.py
```

## Running The Suite

Full local suite:

```bash
python -m pytest -q
```

Coverage:

```bash
python -m pytest --cov=src/ntx --cov-report=term-missing -q
```

Module-wise coverage summary from a `coverage json` file:

```bash
coverage json -o coverage.json
python scripts/build_coverage_report.py \
  --json-input coverage.json \
  --json-output coverage-report.json \
  --text-output coverage-report.txt
```

Lint and type-check:

```bash
python -m ruff check .
python -m mypy src/ntx
```

Docs build:

```bash
python -m sphinx -b html docs docs/_build/html
```

## What The Tests Cover

Representative test groups:

- geometry and Fourier series:
  - `tests/test_geometry.py`
  - `tests/test_vmec.py`
  - `tests/test_vmec_jax_vmec.py`
- operator assembly:
  - `tests/test_operators.py`
- dense solver and scans:
  - `tests/test_solver.py`
  - `tests/test_parallel.py`
  - `tests/test_multiprocess_parallel.py`
- CLI and `.npz` outputs:
  - `tests/test_cli.py`
  - `tests/test_inputfiles.py`
- NEOPAX mapping:
  - `tests/test_neopax_adapter.py`
  - `tests/test_neopax_arrays.py`
- autodiff and optimization helpers:
  - `tests/test_autodiff.py`
  - `tests/test_autodiff_profile_uncertainty_example.py`
  - `tests/test_bootstrap_current_robust_optimization_example.py`
  - `tests/test_file_backed_geometry_control_derivative_benchmark_example.py`
  - `tests/test_explicit_relaxed_boundary_current_derivative_benchmark_example.py`
- profile workflows:
  - `tests/test_profiles_unit.py`
  - `tests/test_profiles_workflows.py`
  - `tests/test_profile_force_reconstruction_audit_example.py`
- coverage and validation summaries:
  - `tests/test_build_coverage_report_script.py`
  - `tests/test_physics_gates.py`
  - `tests/test_benchmark_matrix.py`
- example and figure scripts:
  - `tests/test_make_publication_figures.py`
  - `tests/test_validation_summary_example.py`
  - `tests/test_bootstrap_current_optimization_example.py`

## CI Coverage

The GitHub Actions `tests` workflow now combines coverage across the 3.11 test
shards and publishes:

- raw `.coverage` data
- `coverage.json`
- `coverage-report.json`
- `coverage-report.txt`

The coverage report is intended to answer two different questions:

1. what the current repository-owned line coverage actually is by module,
2. which large files need restructuring or additional tests first.

It should not be interpreted as a physics-validation substitute. Physics gates
and literature benchmarks remain separate acceptance surfaces.

Recent hardening work used this report to target the refactored workflow
modules directly. The current full CI coverage report for the split lanes is
above the release threshold:

- overall repository-owned coverage is `99.0%`,
- `src/ntx/neopax.py`, `src/ntx/_neopax_io.py`, `src/ntx/_neopax_types.py`,
  and `src/ntx/_neopax_bridge.py` are now at or above `97%`,
- `src/ntx/_neopax_field.py`, the imported `vmec_jax`/`booz_xform_jax` field
  bridge, is now at `98.1%`,
- `src/ntx/_autodiff_workflows.py`, `src/ntx/_profiles_eval.py`,
  `src/ntx/_profiles_transport.py`, and `src/ntx/_profiles_controls.py` are all
  above `98%`,
- `src/ntx/_solver_scan.py`, `src/ntx/parallel.py`, `src/ntx/cli.py`,
  `src/ntx/io.py`, and `src/ntx/database.py` are at or above `98%`.

Those gains come from narrow branch tests in the unit/workflow lanes, not from
adding slower benchmark execution to the default developer loop.

The coverage-report helper accepts both absolute and relative `src/ntx/...`
paths from `coverage json`, which keeps local reports and GitHub Actions
summaries comparable. Coverage is a release gate, not a replacement for the
literature-anchored physics gates and benchmark artifacts.

## GPU Validation

GPU-only smoke tests are marked and can be run with:

```bash
python -m pytest -m gpu -q
```

The runtime probes are:

```bash
python scripts/run_gpu_regression.py --output-json gpu-smoke-results.json
python scripts/profile_runtime.py --backend gpu --output-json runtime-profile-gpu.json
python scripts/profile_parallel_runtime.py --output-json parallel-runtime.json
python scripts/profile_multiprocess_runtime.py --backend gpu --workers 2
```

## Cross-Checks Against Independent Workflows

NTX is designed to stand on its own, but it is still valuable to compare its
output against independent neoclassical workflows.

The repository therefore keeps:

- direct W7-X bootstrap-current convergence audits
- optional NEOPAX-coupled checks
- optional SFINCS-JAX-based consistency studies when that package is available
- profile-workflow regression tests split into:
  - fast unit/workflow tests
  - artifact-backed benchmark-family audits

These comparisons are used as trust-building validation, not as the definition
of NTX itself.

## Physics Gate Report

The tracked benchmark gates can be summarized from committed artifacts with:

```bash
python scripts/check_physics_gates.py
```

This is the fastest way to distinguish:

- analytical identities and exact-recovery gates,
- independent-code comparison gates,
- integrated-workflow transfer gates,
- and closure stress metrics that are monitored but not promoted to parity
  claims.

## What To Check Before Claiming A New Physics Result

Before publishing a new equilibrium scan or optimization result:

1. converge `N_xi` at the lowest collisionality used in the study
2. converge `N_theta` and `N_zeta` on `D31`
3. check `onsager_residual`
4. compare serial and parallel scan results on a subset
5. inspect the output `.npz` graphically with `plot_output_npz.py`
6. add or update the benchmark-matrix entry before promoting the result
7. if the workflow feeds NEOPAX, regenerate the monoenergetic database and
   inspect the resulting radial profiles
8. if the workflow uses automatic differentiation, compare the reported
   derivative with centered finite differences on the same scalar output
9. if the workflow uses imported equilibrium or Boozer transformations, state
   whether the gate is projected-boundary, explicitly relaxed equilibrium, or
   implicit-equilibrium sensitivity
