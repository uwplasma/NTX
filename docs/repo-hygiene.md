# Repository Hygiene

This page records the current local worktree audit and the cleanup plan for
turning the active research changes into reviewable commits without losing
benchmark artifacts.

## Current Cleanup Decision

The local worktree had two zero-byte top-level files:

- `NTX`
- `booz_xform_jax`

They were not directories, source files, or generated artifacts, so they were
removed during this hygiene pass.

No tracked file was reverted. No benchmark artifact was deleted.

Current status of this pass:

- no temporary coverage, package-build, or cache artifacts remain in the
  repository root;
- the remaining untracked files are intentional source, tests, docs, examples,
  or generated benchmark artifacts listed below;
- local verification now includes manifest validation, benchmark-matrix
  generation, manuscript-artifact generation, Sphinx, full Ruff, full Mypy,
  package build, `twine check`, and targeted tests for the touched CI,
  coverage, benchmark, physics-gate, and imported-workflow files.
- `python -m build`, `python -m twine check dist/*`, and a clean-venv wheel
  smoke test were run after removing Git direct references from package
  metadata; temporary `dist/`, `build/`, and smoke-venv artifacts were removed.

## Commit Batches

The current changes should be split and reviewed in this order:

1. **CI lane manifest**
   - `.github/workflows/tests.yml`
   - `scripts/test_lane_manifest.py`
   - `tests/test_ci_lane_manifest.py`
   - `docs/testing.md`
2. **Benchmark matrix**
   - `src/ntx/validation/`
   - `scripts/build_benchmark_matrix.py`
   - `tests/test_benchmark_matrix.py`
   - `docs/benchmark-matrix.md`
   - `docs/_static/benchmark_matrix.json`
3. **Differentiable imported geometry and field workflows**
   - `src/ntx/vmec_jax_backend.py`
   - `src/ntx/_neopax_field.py`
   - `src/ntx/_neopax_types.py`
   - `src/ntx/_neopax_bridge.py`
   - `src/ntx/neopax.py`
   - `src/ntx/__init__.py`
   - `src/ntx/core/`
   - `src/ntx/workflows/`
   - boundary, geometry-control, and differentiable-field tests
4. **Examples and artifacts**
   - new derivative benchmark examples
   - corresponding `docs/_static/*derivative_benchmark*` artifacts
   - `examples/make_publication_figures.py`
5. **Manuscript and documentation refresh**
   - `docs/autodiff.md`
   - `docs/manuscript.md`
   - `docs/validation.md`
   - `docs/literature.md`
   - `docs/performance.md`
   - `docs/release.md`
   - `docs/research-roadmap.md`
   - `docs/source-map.md`
   - `plan.md`
   - `scripts/build_manuscript_artifacts.py`
   - `docs/_static/manuscript_*`
6. **Generated static figure refresh**
   - review all changed PDFs/PNGs for visual or data changes;
   - revert or regenerate deterministic metadata-only PDF changes before the
     final commit if they do not correspond to a real artifact update.

## Modified Tracked Paths

These tracked files are modified and should stay under review:

```text
.github/workflows/tests.yml
docs/_static/ambipolar_profile.pdf
docs/_static/ambipolar_profile_family.pdf
docs/_static/autodiff_inverse_problem.pdf
docs/_static/autodiff_neopax_profiles.pdf
docs/_static/autodiff_neopax_profiles.png
docs/_static/autodiff_profile_uncertainty.pdf
docs/_static/bootstrap_current_from_vmec_or_boozmn.pdf
docs/_static/bootstrap_current_optimization.json
docs/_static/bootstrap_current_optimization.pdf
docs/_static/bootstrap_current_optimization.png
docs/_static/bootstrap_current_reference_audit_w7x.json
docs/_static/bootstrap_current_reference_audit_w7x.pdf
docs/_static/bootstrap_current_reference_audit_w7x.png
docs/_static/bootstrap_current_robust_optimization.pdf
docs/_static/closure_validation_report.pdf
docs/_static/derivative_path_benchmark.json
docs/_static/derivative_path_benchmark.pdf
docs/_static/derivative_path_benchmark.png
docs/_static/manuscript_artifacts.json
docs/_static/manuscript_claims.md
docs/_static/manuscript_tables.md
docs/_static/performance_scaling_heavy.pdf
docs/_static/performance_scaling_smoke.pdf
docs/_static/primitive_profile_transport.pdf
docs/_static/profile_basis_optimization.pdf
docs/_static/profile_control_optimization.pdf
docs/_static/profile_force_reconstruction_audit.pdf
docs/_static/profile_transport_loop.pdf
docs/_static/publication_figure_manifest.json
docs/_static/validation_summary.pdf
docs/autodiff.md
docs/conf.py
docs/index.md
docs/literature.md
docs/manuscript.md
docs/performance.md
docs/release.md
docs/research-roadmap.md
docs/source-map.md
docs/testing.md
docs/validation.md
examples/make_publication_figures.py
plan.md
pyproject.toml
scripts/build_manuscript_artifacts.py
src/ntx/__init__.py
src/ntx/_geometry_types.py
src/ntx/_neopax_bridge.py
src/ntx/_neopax_types.py
src/ntx/neopax.py
src/ntx/vmec_jax_backend.py
tests/test_make_publication_figures.py
tests/test_manuscript_artifacts_script.py
tests/test_profile_force_reconstruction_audit_example.py
```

Interpretation:

- `docs/_static/*.json`, `docs/_static/*.png`, and `docs/_static/*.pdf` are
  generated benchmark/manuscript artifacts. They should be kept only when the
  corresponding script, test, and docs entry are present.
- text docs and `plan.md` are intentional planning/manuscript/validation
  updates.
- source and test changes are tied to differentiable imported workflows,
  namespace exports, benchmark matrix, and CI sharding.

## Untracked Paths To Keep Under Review

These untracked paths are not temporary files. They should be validated and then
committed in the batches above:

```text
docs/_static/benchmark_matrix.json
docs/_static/boundary_forward_mode_current_derivative_benchmark.json
docs/_static/boundary_forward_mode_current_derivative_benchmark.pdf
docs/_static/boundary_forward_mode_current_derivative_benchmark.png
docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.json
docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.pdf
docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.png
docs/_static/file_backed_geometry_control_derivative_benchmark.json
docs/_static/file_backed_geometry_control_derivative_benchmark.pdf
docs/_static/file_backed_geometry_control_derivative_benchmark.png
docs/_static/geometry_control_derivative_benchmark.json
docs/_static/geometry_control_derivative_benchmark.pdf
docs/_static/geometry_control_derivative_benchmark.png
docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.json
docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.pdf
docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.png
docs/benchmark-matrix.md
examples/boundary_forward_mode_current_derivative_benchmark.py
examples/explicit_relaxed_boundary_current_derivative_benchmark.py
examples/file_backed_geometry_control_derivative_benchmark.py
examples/geometry_control_derivative_benchmark.py
examples/implicit_equilibrium_forward_mode_derivative_benchmark.py
scripts/build_benchmark_matrix.py
scripts/test_lane_manifest.py
src/ntx/_neopax_field.py
src/ntx/core/
src/ntx/validation/
src/ntx/workflows/
tests/test_benchmark_matrix.py
tests/test_boundary_forward_mode_current_derivative_benchmark_example.py
tests/test_ci_lane_manifest.py
tests/test_differentiable_neopax_field.py
tests/test_explicit_relaxed_boundary_current_derivative_benchmark_example.py
tests/test_file_backed_geometry_control_derivative_benchmark_example.py
tests/test_geometry_control_derivative_benchmark_example.py
tests/test_implicit_equilibrium_forward_mode_derivative_benchmark_example.py
tests/test_namespace_imports.py
```

Interpretation:

- `docs/_static/*derivative_benchmark*` artifacts are research artifacts tied to
  the new derivative examples. Keep them if the corresponding tests pass and the
  benchmark matrix references them.
- `docs/_static/benchmark_matrix.json` is generated by
  `scripts/build_benchmark_matrix.py`. Keep it with the benchmark matrix.
- `src/ntx/core`, `src/ntx/workflows`, and `src/ntx/validation` are namespace
  organization changes, not generated files.
- the new tests are lane-owned by `scripts/test_lane_manifest.py` so they no
  longer drift into the core shard accidentally.

## Verification Order

Use this verification sequence before committing:

```bash
python scripts/test_lane_manifest.py --check
python -m pytest -q tests/test_ci_lane_manifest.py tests/test_benchmark_matrix.py tests/test_namespace_imports.py
python -m ruff check scripts/test_lane_manifest.py tests/test_ci_lane_manifest.py
python scripts/build_benchmark_matrix.py
python scripts/build_manuscript_artifacts.py
python -m sphinx -b html docs docs/_build/html
```

After that, run the relevant test lane for the commit batch being prepared:

```bash
python scripts/test_lane_manifest.py core | xargs python -m pytest -q -m "not gpu"
python scripts/test_lane_manifest.py integration_examples | xargs python -m pytest -q -m "not gpu"
python scripts/test_lane_manifest.py heavy_examples_profiles | xargs python -m pytest -q -m "not gpu"
python scripts/test_lane_manifest.py heavy_examples_derivatives | xargs python -m pytest -q -m "not gpu"
python scripts/test_lane_manifest.py heavy_examples_boundary | xargs python -m pytest -q -m "not gpu"
python scripts/test_lane_manifest.py heavy_examples_publication | xargs python -m pytest -q -m "not gpu"
```

Only run the heavy lanes when preparing or reviewing artifact changes.
The boundary/equilibrium rerun lane is env-gated in normal CI; use:

```bash
NTX_RUN_HEAVY_BOUNDARY_EXAMPLES=1 \
  python scripts/test_lane_manifest.py heavy_examples_boundary \
  | xargs python -m pytest -q -m "not gpu"
```

Current measured local shard times after the manifest split:

- `core`: `217 passed, 2 deselected` in about `3:37`
- `integration_examples`: `15 passed` in about `2:56`
- `heavy_examples_profiles`: `6 passed` in about `2:15`
- `heavy_examples_derivatives`: `4 passed` in about `1:00`
- `heavy_examples_publication`: `20 passed` in about `1:03`
- `heavy_examples_boundary`: default CI path skips the expensive reruns unless
  `NTX_RUN_HEAVY_BOUNDARY_EXAMPLES=1`

This keeps the normal CI wall time controlled while preserving an explicit
artifact-refresh lane for expensive imported boundary/equilibrium examples.
