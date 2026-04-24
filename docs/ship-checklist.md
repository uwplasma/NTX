# Pre-Merge Ship Checklist

NTX should not be merged, tagged, or shipped until the open research and
software lanes below are either closed or explicitly moved to documented future
work with a clear reason.

## Blocking Lanes

| Lane | Current Status | Required Before Merge |
| --- | --- | --- |
| Code refactoring | In progress | Keep public facades stable while moving implementation ownership toward `ntx.core`, `ntx.workflows`, `ntx.validation`, `ntx.geometry`, and `ntx.io`; each move needs tests and docs. |
| Repository hygiene | In progress | Split the dirty worktree into reviewable commit batches; remove only confirmed temporary files; keep benchmark artifacts only when tied to scripts/tests/docs. |
| CI runtime and coverage | Closed; monitor | Maintain `>=95%` repository-owned coverage, module floors, and a normal CI wall time near `5-10` minutes. |
| Literature-anchored physics gates | In progress | Add or preserve fast gates for convergence, Onsager residuals, exact low-order recovery, coefficient sign/normalization, and artifact-backed literature comparisons. |
| Fixed-field `NTX+NEOPAX` closure | Scoped stress gate | Keep as a monitored stress metric for the first release; do not claim fixed-field parity unless a physics-derived closure improves QA/QH without regressing integrated W7-X. No fitted bridge constants. |
| Multi-CPU and multi-GPU algorithms | Open performance lane | Add measured CPU/GPU/multiprocess crossover maps on production grids; promote only algorithms that beat serial batched JAX on the target workload. |
| `vmec_jax` and `booz_xform_jax` integration | Partly closed | Keep projected-boundary and explicit-relaxed lanes; close or document implicit-equilibrium derivative gaps before making broad optimization claims. |
| SFINCS comparisons | Partly closed | Add more artifact-backed comparisons with aligned physics settings and normalizations; distinguish parity gates from monitored stress gates. |
| Documentation | In progress | Keep docs synchronized with source layout, benchmark matrix, test lanes, performance guidance, examples, and release path. |
| Implicit-equilibrium derivative lane | Open | Boozer and NTX transport observables must match centered finite differences, not just equilibrium volume. |
| Broader W7-X/QI/omnigenous families | Planned | Add owned inputs, convergence ladders, artifacts, and benchmark-matrix rows before promotion. |
| PyPI/release automation | Repo-side configured | Finish external PyPI project setup, Trusted Publishing environment approval, and first tagged release rehearsal. |

## Acceptance Criteria

1. `python scripts/test_lane_manifest.py --check` passes and every test belongs
   to exactly one CI lane.
2. Normal CI lanes remain bounded:
   - `core_foundation`
   - `core_cli_workflows`
   - `core_io_workflows`
   - `core_parallel_workflows`
   - `core_neopax_workflows`
   - `core_profile_audit_workflow`
   - `core_profile_basic_workflows`
   - `core_profile_optimization_workflows`
   - `core_profile_transport_workflows`
   - `core_autodiff_uncertainty_workflow`
   - `core_robust_bootstrap_workflow`
   - `core_validation`
   - `integration_examples`
   - `heavy_examples_profiles`
   - `heavy_examples_derivatives`
   - `heavy_examples_boundary`
   - `heavy_examples_publication`
3. `python -m ruff check .` and `python -m mypy src/ntx` pass.
4. Coverage is `>=95%` for `src/ntx`, with weak modules called out explicitly.
5. `python scripts/build_benchmark_matrix.py` reports no incomplete active gates.
6. `python scripts/build_manuscript_artifacts.py` completes from committed
   artifacts.
7. `python -m sphinx -b html docs docs/_build/html` passes.
8. `python -m build` and `python -m twine check dist/*` pass.
9. Generated figures and JSON artifacts are reviewed against their owning
   scripts/tests/docs.
10. Any lane not closed before release is documented as future work and excluded
    from release claims.

## Current Audit Notes

- Full split-lane CI coverage is currently above the release threshold at
  `99.0%`.
- The maintained coverage-report script now accepts both absolute and relative
  `src/ntx/...` paths from `coverage json`, so local and CI module tables are
  comparable.
- The validation registry now has additional direct unit coverage.
- Fast synthetic imported-workflow tests exercise the imported field/database
  bridge without rerunning expensive boundary/equilibrium artifacts in every
  shard; the current CI report has `_neopax_field.py` at `98.1%` and
  `neopax.py` at `100%`.
- A fast owned-surface physics gate now checks `D11`, `D31`, `D33`, Onsager
  residual, and coarse-to-fine angular-grid transfer on the analytic Boozer
  surface.
- A symmetric-limit physics gate now checks zero radial transport and the
  inverse-collisionality Spitzer parallel-conductivity normalization on a
  constant-field Boozer surface.
- `python scripts/test_lane_manifest.py --check` passes with split core lanes,
  8 integration examples, and 18 opt-in heavy example tests.
- `python scripts/build_benchmark_matrix.py` reports every active benchmark
  gate complete and keeps the broader geometry/autodiff breadth lanes planned.
- `python -m ruff check .`, `python -m mypy src/ntx`, documentation build,
  manuscript-artifact build, package build, and `twine check` pass locally.
- A clean-venv wheel smoke test passes for `ntx --help`, `python -m ntx --help`,
  and importing `GridSpec`.
- Public package metadata no longer exposes Git direct references; optional
  geometry-coupled workflows document direct upstream installs until those
  packages are available from standard package indexes.
- The repository-side PyPI Trusted Publishing job is present and tag-gated; the
  remaining setup is the external PyPI trusted publisher/project configuration.
- The fixed-field `NTX+NEOPAX` lane is explicitly scoped out of first-release
  parity claims. The release claim is the positive W7-X integrated transfer and
  the fixed-field Redl/SFINCS gate; the reduced-closure current mismatch remains
  a monitored stress metric.
- The next valuable coverage work should be opportunistic and physics-driven;
  coverage is no longer a blocking lane.
- The expensive boundary/equilibrium artifact reruns remain opt-in through
  `NTX_RUN_HEAVY_BOUNDARY_EXAMPLES=1`.

## Immediate Next Order

1. Keep the CI lane manifest and benchmark matrix locked as new tests are added.
2. Add the next high-value physics gate rather than low-value coverage tests.
3. Expand owned geometry-family benchmark artifacts only from committed
   scripts/tests/docs.
4. Finish release automation and tag only after all blocking lanes above are
   either closed or explicitly scoped out of the release.
