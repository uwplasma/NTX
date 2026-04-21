# Testing And Quality Assurance

NTX is validated at four levels:

1. unit tests of geometry, operators, and solver algebra
2. regression tests of file-driven workflows and publication examples
3. imported-workflow tests for autodiff, NEOPAX, and JAX geometry backends
4. CPU/GPU runtime and smoke checks

The physics-facing gate structure is documented separately in
[`physics-gates.md`](physics-gates.md). The test suite and benchmark scripts are
meant to enforce that gate hierarchy, not to replace it.

## Running The Suite

Full local suite:

```bash
python -m pytest -q
```

Coverage:

```bash
python -m pytest --cov=src/ntx --cov-report=term-missing -q
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
- example and figure scripts:
  - `tests/test_make_publication_figures.py`
  - `tests/test_validation_summary_example.py`
  - `tests/test_bootstrap_current_optimization_example.py`

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
6. if the workflow feeds NEOPAX, regenerate the monoenergetic database and
   inspect the resulting radial profiles
