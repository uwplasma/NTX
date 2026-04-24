# NTX Code Plan

## Goal

Operate NTX as a research-grade JAX-native implementation of the monoenergetic
transport formulation described in Javier Escoto's PhD thesis:
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

This plan is code-facing only. It tracks solver, workflow, validation, and
performance work needed to keep NTX technically strong and useful for open
research problems.

## Current Base State

- [x] JAX-native monoenergetic transport solver
- [x] CLI workflow via `ntx input.toml`
- [x] imported Python API for direct solves and scans
- [x] DKES-style, magnetic-configuration, VMEC, and Boozer geometry lanes
- [x] differentiable imported solve lane
- [x] prepared dense solve and implicit-adjoint derivative path
- [x] direct NEOPAX scan and HDF5 mapping helpers
- [x] CPU, GPU, and multiprocess scan utilities
- [x] publication-quality example figures and figure-bundle generation
- [x] W7-X bootstrap-current convergence audit
- [x] profile-grade ambipolar, control, and transport proxy workflows
- [x] NTX remains scoped to monoenergetic coefficients and flux channels, with
  bootstrap-current closure delegated to NEOPAX

## Current Validation Summary

- local test suite passes
- documentation builds locally
- CPU and GPU smoke/regression workflows are available
- office GPU validation has been closed on the NTX-owned smoke cases
- current local validation status:
  - `ruff`
  - `mypy`
  - `pytest`
  - `sphinx`
- current bootstrap-current interpretation:
  - use `NTX+NEOPAX` for bootstrap-current workflows
  - keep fixed-field Redl/SFINCS audits separate from finite-beta integrated
    workflow audits
  - the README now carries the fixed-field precise-QS comparison figure as the
    current validation status view, but the QA momentum-correction closure is
    still an active audit lane rather than a closed parity claim
  - the validation surface is now codified as explicit physics gates:
    - analytical identities and exact `P=2` recovery
    - independent-code comparison gates
    - integrated-workflow transfer gates
    - monitored closure stress tests
  - current decision:
    - hold the code on the present Escoto-based closure model
    - treat the fixed-field QA/QH gap as a documented reduced-closure stress
      test
    - do not reopen closure-model derivation work unless there is a concrete,
      physically motivated implementation change to test

## Current Delivery Decision

- [x] adopt the current validated model family as the manuscript baseline
- [x] keep W7-X integrated transfer and Redl precise-QS agreement as the main
  positive validation claims
- [x] present fixed-field `NTX+NEOPAX` as a monitored closure stress test
- [x] shift near-term work toward CI speed, packaging, manuscript clarity, and
  reproducibility rather than new closure-model physics

## Pre-Merge Gate Decision

Do not merge, tag, or ship this branch until the open-lane checklist in
[`docs/ship-checklist.md`](docs/ship-checklist.md) is satisfied. The release
order is:

1. close repository hygiene and commit batching,
2. lock CI runtime and `>=95%` coverage,
3. strengthen literature-anchored physics gates and validation artifacts,
4. keep the fixed-field `NTX+NEOPAX` closure lane explicitly scoped as a
   monitored stress metric unless a physics-derived closure closes QA/QH
   without regressing W7-X,
5. finish the `vmec_jax`/`booz_xform_jax` derivative gates that are claimed,
6. add broader W7-X/QI/omnigenous lanes only as artifact-backed planned work
   unless the owned inputs and convergence ladders are already present,
7. finish docs, examples, package checks, and PyPI release automation,
8. then merge, tag, and ship.

## 2026-04 Full Ship-Readiness Audit

This pass consolidates the current plan, git history, source tree, comparison
codes, and literature into a single execution order. The guiding rule is that
NTX should be shipped as a small, accurate, differentiable monoenergetic
transport code, not as an opaque wrapper around heavier neoclassical or
equilibrium stacks.

### Literature And Code Audit Conclusions

- The promoted NTX equation family remains the finite Legendre/Sonine
  monoenergetic transport formulation in Escoto's thesis and the associated
  Fortran reference implementation. The code should keep testing convergence in
  `Pmax`, `N_xi`, `N_theta`, and `N_zeta` rather than changing the physics model
  without a literature-backed reason.
- The International Collaboration on Neoclassical Transport in Stellarators
  benchmark paper remains the broadest independent benchmark target for
  `D11`, `D31`, and `D33`, because it compares the three monoenergetic
  coefficients across several numerical methods and magnetic-optimization
  strategies.
- SFINCS remains the high-fidelity comparison lane, but it solves a broader
  radially local drift-kinetic problem with trajectory and collision-operator
  options. Therefore SFINCS comparisons are parity/stress gates only when the
  physics settings and normalizations are explicitly aligned.
- The Landreman trajectory/collision-operator comparisons set an important
  acceptance rule: do not expect order-`1e-2` agreement in regimes where the
  monoenergetic approximation, electric-field resonance physics, or momentum
  conservation model are intentionally different. In those regimes the gate is
  convergence, sign, ordering, and bounded relative error, not exact parity.
- The closure audit has a hard rule: no fitted bridge constants are allowed in
  production. Any momentum-correction change must be derived from the same
  moment definitions used by the runtime closure and must improve the fixed-field
  QA/QH stress cases without regressing the integrated W7-X transfer gate.
- The differentiable geometry stack is now realistic because `vmec_jax` and
  `booz_xform_jax` are JAX-native and packageable. NTX should use them as
  optional workflows, with derivative gates that distinguish projected-boundary,
  explicitly relaxed equilibrium, and implicit-equilibrium sensitivities.
- The 2026 piecewise-omnigenous and low-bootstrap-current literature raises the
  bar for future novelty claims: NTX should include owned hidden-symmetry,
  omnigenous, and piecewise-omnigenous geometry families before claiming broad
  design relevance beyond the current W7-X and precise-QS validation surfaces.

### Repository State From Git History

Recent commits show that the project already completed the first source split
and fast-coverage pass:

- internal solver, profile, autodiff, geometry, input-file, and NEOPAX helper
  modules were split out of the older facade files;
- coverage is measured by module in CI and the fast core shard reached the
  target `>=95%` threshold;
- benchmark-matrix artifacts now exist and are regenerated from
  `scripts/build_benchmark_matrix.py`;
- derivative, profile, robust-optimization, validation-summary, and W7-X audit
  artifacts are already committed as reproducibility anchors.

The remaining work is therefore not broad low-value coverage. The remaining
work is to keep the fast CI clean while adding high-value literature gates,
geometry-family benchmarks, derivative audits, packaging release automation,
and docs that explain which claims are closed versus monitored.

### Immediate Execution Order

1. **Repository hygiene and CI classification**
   - Review untracked directories and generated artifacts before editing them.
   - Remove only obvious throwaway files such as caches after confirming they
     are not benchmark artifacts.
   - Classify every new example test into `core_foundation`,
     `core_cli_workflows`, `core_io_workflows`, `core_parallel_workflows`,
     `core_neopax_workflows`, `core_profile_audit_workflow`,
     `core_profile_basic_workflows`, `core_profile_optimization_workflows`,
     `core_profile_transport_workflows`, `core_autodiff_uncertainty_workflow`,
     `core_robust_bootstrap_workflow`, `core_validation`,
     `integration_examples`, `heavy_examples`, or manual benchmark lanes so the
     PR workflow stays within `5-10` minutes.
   - Replace the CI core-shard exclusion one-liner with a maintained test-lane
     manifest or pytest markers before adding more benchmark tests.
   - Current status: closed for the current branch. The maintained manifest is
     in place, core tests are split into explicit workflow/validation lanes,
     generated caches have been removed, and new artifacts are tied to
     scripts/tests/docs. Each test-shard job is now capped at ten minutes, and
     subprocess-based parallel smoke tests have their own explicit subprocess
     timeouts so CI stalls fail as bounded test errors.

2. **Benchmark matrix hardening**
   - Require every promoted claim to map to:
     `literature reference -> script -> test -> artifact -> docs/manuscript figure`.
   - Add explicit rows for:
     W7-X EIM, W7-X KJM, CIEMAT-QI or successor QI cases, precise-QS QA/QH,
     low-bootstrap-current omnigenous/piecewise-omnigenous families, and the
     three differentiable geometry-control lanes.
   - Keep fixed-field `NTX+NEOPAX` as a stress gate until the closure branch
     passes both fixed-field QA/QH and integrated W7-X simultaneously.
   - First-release decision: fixed-field `NTX+NEOPAX` remains scoped as a
     monitored closure stress metric, not a promoted parity claim. The release
     claim stays on W7-X integrated transfer and fixed-field Redl/SFINCS.
   - Current status: the matrix builder reports every active gate complete and
     keeps the broader geometry/autodiff breadth work as planned lanes.

3. **High-value physics tests**
   - Add convergence-ladder tests for `D11`, `D31`, `D33`, and Onsager residuals
     on small owned geometry cases.
   - Add artifact-backed literature reproductions for larger cases rather than
     running them in every PR.
   - Keep analytical and algebraic tests fast:
     Fourier derivative identities, nullspace condition, operator block
     structure, source-mode parity, transport coefficient sign/normalization,
     and exact low-order recovery.
   - Treat Redl/SFINCS/bootstrap-current comparisons as workflow validation
     only after the monoenergetic coefficient gates are independently green.
   - Current status: repository artifact gates now assert that the positive
     W7-X transfer and Redl/SFINCS checks pass while the fixed-field closure and
     higher-order `Pmax` checks remain monitored stress metrics.
   - Current artifact-gate addition: the committed monoenergetic validation
     summary is now checked by `evaluate_artifact_gates`; the DKES-style and
     VMEC finest plotted `N_\xi` convergence errors must stay below `2.5e-1`
     before that figure supports the promoted monoenergetic claim.
   - Current fast-lane addition: `tests/test_physics_gates.py` now includes an
     owned analytic Boozer-surface gate for `D11`, `D31`, `D33`, Onsager
     residual, and angular-grid transfer.
   - Current symmetric-limit addition: the same physics-gate file now includes
     constant-field Boozer-surface gates requiring zero radial transport
     channels, positive Spitzer-consistent parallel conductivity, and
     inverse-collisionality scaling of the Spitzer branch. It now also sweeps
     the normalized radial electric field in the same constant-field limit to
     require that electric-field advection does not create radial transport or
     change the parallel-conductivity branch when the magnetic-drift drive is
     absent.
   - Current source-projection addition: `tests/test_operators.py` now checks
     the finite Legendre source projection directly, requiring the
     magnetic-drift drive to occupy only the `k=0` and `k=2` rows with the
     runtime `2/3` and `1/3` weights and the parallel drive to occupy only the
     `k=1` row as physical `B`.
   - Current derivative-consistency addition: `tests/test_operators.py` now
     requires the hand-coded `dD_k/dnu_hat` and `dD_k/depsi_hat` blocks used by
     the implicit-adjoint path to match JAX differentiation of the assembled
     Legendre-space operator.
   - Current profile-force addition: `tests/test_profiles_unit.py` now verifies
     the primitive-profile thermodynamic-force reconstruction for constant
     density/temperature with finite radial electric field and for exponential
     density/temperature profiles at the central finite-difference point.
   - Current prepared-derivative artifact addition:
     `docs/_static/derivative_path_benchmark.json` is now checked by
     `evaluate_artifact_gates`; the prepared custom-VJP derivative must remain
     within `1e-4` relative mismatch of direct reverse-mode, while speedup is
     kept as supporting performance evidence.
   - Current geometry/autodiff artifact addition: the analytic geometry-control,
     file-backed geometry-control, boundary-projected current-derivative, and
     explicit-relaxed boundary-to-current JSON artifacts are now checked by
     `evaluate_artifact_gates` with finite-difference agreement thresholds. The
     implicit-equilibrium derivative artifact is kept as a monitored open lane,
     because only the equilibrium-volume derivative closes on the committed
     diagnostic.

4. **Differentiability gates**
   - Keep direct AD, forward-mode boundary controls, prepared implicit-adjoint
     solves, and finite differences side by side on the same scalar outputs.
   - Promote only the derivative lanes that pass centered finite differences on
     repository-owned cases.
   - Keep implicit-equilibrium Boozer and NTX transport sensitivities open until
     they pass, because the current diagnostic only closes the equilibrium
     volume scalar.
   - Add optimization/UQ examples only when their gradients have a prior
     derivative-audit artifact.

5. **Performance and memory**
   - Profile compile time and steady-state time separately.
   - Prefer stable shapes, prepared geometry reuse, reusable compiled callables,
     and batched scans before adding new dependencies.
   - Use `jax.vmap` for independent scan axes, `jax.lax.scan` for fixed-length
     iterative loops that would otherwise unroll inside `jit`, and buffer
     donation where ownership is clear.
   - Use persistent compilation cache as a reproducibility aid, not as the only
     speed fix, since the W7-X closure profile showed the current bottleneck is
     mostly retracing/static-shape discipline.
   - Use device-memory profiling and explicit GPU memory policies for heavy GPU
     campaigns; do not run broad XLA dump passes by default on this workstation.
   - Evaluate Lineax for repeated structured dense solves or Jacobian-linear
     operators only after the existing prepared solve profile shows a clear
     benefit. Evaluate Equinox only for PyTree/module ergonomics and filtered
     transforms, not as a blanket rewrite.

6. **Code structure target**
   - Keep public compatibility facades stable.
   - Move implementation ownership gradually toward:
     `ntx.core`, `ntx.geometry`, `ntx.io`, `ntx.workflows`, and
     `ntx.validation`.
   - Do not move files only for aesthetics. Move a module when the move creates
     a clearer test surface, public API boundary, or benchmark owner.
   - Keep `docs/source-map.md` synchronized with internal ownership splits;
     `tests/test_source_map.py` now fails when a split module is missing from
     the architecture map.
   - Add docstrings to public APIs and short comments only where they explain
     non-obvious physics normalization, numerical conditioning, or AD behavior.

7. **Docs, examples, manuscript, and release**
   - Keep examples short and runnable; move long reproduction campaigns to
     scripts/artifacts and link them from docs.
   - Add docs pages or sections for test lanes, source layout, benchmark
     reproduction, derivative validation, performance profiling, and release
     workflows.
   - Before PyPI publication, remove Git direct references from published extras
     or replace them with index-published packages when available.
   - Keep the Trusted Publishing release job tag-gated and tied to the protected
     `pypi` environment; the remaining external step is PyPI project/trusted
     publisher setup.
   - Current status: Git direct references have been removed from package
     metadata, geometry-coupled workflows are documented as optional external
     installs, local wheel/sdist plus clean-venv smoke checks pass, and the
     repository-side PyPI Trusted Publishing workflow is present.

## 2026 Hardening Program

This is the next concrete code program. It is scoped to keep NTX scientifically
defensible while making the source tree easier to maintain, easier to test, and
stronger for differentiable research workflows.

### Exit Criteria

Do not declare this program complete until all of the following are true:

1. repository-owned line coverage for `src/ntx` is measured in CI and is
   `>= 95%`, with module floors so the headline number cannot hide weak files:
   - `solver.py`, `profiles.py`, `autodiff.py`, `neopax.py`, `inputfiles.py`,
     and `geometry.py` equivalent modules must each be `>= 90%`
   - new or refactored modules should target `>= 95%`
2. fast pull-request CI stays within roughly `5-10` minutes while preserving
   the same solver/workflow quality gates
3. all literature-anchored benchmark families below are reproducible from
   version-controlled scripts and committed artifacts
4. autodiff workflows are validated for:
   - local sensitivities
   - inverse-design recovery
   - uncertainty propagation
   - optimization loops
   - low-dimensional boundary-to-output sensitivities on imported `vmec_jax`
     and `booz_xform_jax` workflows
5. the public documentation explains:
   - the source-tree structure
   - the testing pyramid
   - the benchmark hierarchy
   - the physics-gate interpretation
6. the manuscript figure set is generated directly from the benchmark/test
   scripts tracked in the repository

### Current Gaps To Close

- measured coverage is now closed for the current split CI lanes:
  - full split-lane CI coverage is `99.0%`
  - `_neopax_field.py` is `98.1%`
  - `neopax.py` is `100.0%`
  - `vmec_jax_backend.py` is `100.0%`
  - the next coverage work should be opportunistic and physics-driven, not
    broad low-value branch chasing
- the device-parallel profiling smoke lane is now bounded for CI:
  - `scripts/profile_parallel_runtime.py` keeps the full default profiling
    workload for local measurements
  - CI passes `--num-cases 2 --grid 5,5,4` so serial/device-parallel numerical
    agreement is checked without making the parallel shard a timing benchmark
- the manuscript artifact builder now pulls the already-gated monoenergetic
  validation summary and fixed-field precise-QS benchmark into
  `manuscript_claims.md` and `manuscript_tables.md`, including the scoped
  `NTX+NEOPAX` stress metric
- the differentiable bootstrap-current optimization science figure is now tied
  to the benchmark matrix and a physics-gate artifact check: the optimized
  weighted-current proxy must remain at least as large as the committed
  baseline before the manuscript cites the gain
- the manuscript figure inventory now carries the file-backed geometry-control
  derivative audit in the supplement bundle, and the boundary-forward
  derivative audit is explicitly marked as a manuscript figure in the benchmark
  matrix metadata
- several core modules remain too large for stable review and targeted testing:
  - `profiles.py` (`73` lines after Phase 1 split; types/eval/controls/transport moved to internal modules)
  - `solver.py` (`51` lines after Phase 1 split; types/core/scan moved to internal modules)
  - `autodiff.py` (`92` lines after Phase 1 split; types/workflows/bootstrap moved to internal modules)
  - `inputfiles.py` (`59` lines after Phase 1 split; parsing/reporting/run-output moved to internal modules)
  - `neopax.py` (`47` lines after Phase 1 split; bridge/types/I/O/field/flux/scan helpers moved to internal modules)
  - `geometry.py` (`19` lines after Phase 1 split; types/evaluation moved to internal modules)
- public docstrings and internal comments are inconsistent across the source
  tree, particularly in workflow-heavy files
- the current test suite mixes:
  - unit tests,
  - example smoke/regression tests,
  - literature benchmark tests,
  - and long-running research audits
  in a way that makes coverage measurement too expensive for the default
  developer loop
- the benchmark hierarchy is scientifically sound, but it needs a stronger
  mapping from each literature figure and code-comparison result to:
  - a script,
  - a test,
  - an artifact,
  - and a manuscript figure

### Phase 0: Baseline And Instrumentation

1. Replace the static coverage badge workflow with measured coverage artifacts.
2. Split test execution into three lanes:
   - fast PR lane:
     - unit tests
     - cheap regressions
     - packaging
     - docs
     - selected CPU-only examples
   - benchmark lane:
     - literature reproductions
     - long-running fixed-field / W7-X audits
     - scaling and profiling studies
   - optional hardware lane:
     - GPU smoke
     - office workstation profiling
3. Generate and commit a machine-readable coverage summary by module.
4. Track run time and memory for:
   - `pytest`
   - docs build
   - benchmark scripts
   so regressions in developer velocity are visible.

Acceptance gates for this phase:

- CI publishes measured coverage, not a static claim
- PR lane remains within the current runtime budget
- benchmark lane can be run locally and in scheduled/manual CI without
  affecting PR speed

Current status:

- [x] shard-combined measured coverage is published in CI with machine-readable
  per-module artifacts
- [x] the first no-behavior-change workflow split is complete:
  - `inputfiles.py` now remains the compatibility surface
  - TOML parsing and config dataclasses live in `src/ntx/_inputfiles_model.py`
  - Rich tables and metadata/source helpers live in
    `src/ntx/_inputfiles_reporting.py`
  - TOML execution and `.npz` artifact writing live in
    `src/ntx/_inputfiles_run.py`
- [x] the second no-behavior-change workflow split is complete:
  - `neopax.py` now remains the compatibility surface
  - scan dataclasses/constants live in `src/ntx/_neopax_types.py`
  - HDF5/reference-scan I/O lives in `src/ntx/_neopax_io.py`
  - differentiable array/database mapping lives in `src/ntx/_neopax_bridge.py`
- [x] the third no-behavior-change workflow split is complete:
  - `geometry.py` now remains the compatibility surface
  - Boozer/VMEC/geometry dataclasses live in `src/ntx/_geometry_types.py`
  - Fourier evaluation and grid projection live in `src/ntx/_geometry_eval.py`
- [x] the fourth no-behavior-change workflow split is complete:
  - `autodiff.py` now remains the compatibility surface and local NEOPAX import fallback
  - result dataclasses live in `src/ntx/_autodiff_types.py`
  - inverse/profile/derivative/optimization workflows live in
    `src/ntx/_autodiff_workflows.py`
  - bootstrap-current deterministic and robust optimization workflows live in
    `src/ntx/_autodiff_bootstrap.py`
- [x] the fifth no-behavior-change workflow split is complete:
  - `solver.py` now remains the compatibility surface
  - case/result/prepared-system types live in `src/ntx/_solver_types.py`
  - prepared solve and custom-VJP core live in `src/ntx/_solver_core.py`
  - prepared custom-VJP adjoint helper algebra now lives in
    `src/ntx/_solver_adjoint.py`
  - block-tridiagonal factorization, low-mode back-substitution, residual
    checks, and factorized adjoint solves live in
    `src/ntx/_solver_factorization.py`
  - scan/device helpers live in `src/ntx/_solver_scan.py`
- [x] the sixth no-behavior-change workflow split is complete:
  - `profiles.py` now remains the compatibility surface
  - workflow dataclasses live in `src/ntx/_profiles_types.py`
  - scan interpolation, ambipolar solves, and primitive profile construction
    were first split into `src/ntx/_profiles_eval.py`
  - scalar and basis-control helpers live in `src/ntx/_profiles_controls.py`
  - profile-transport closure losses and explicit update algebra live in
    `src/ntx/_profiles_transport_closure.py`
  - explicit transport-loop runners live in `src/ntx/_profiles_transport.py`
- [x] the seventh no-behavior-change workflow split is complete:
  - `src/ntx/_neopax_field.py` now owns the direct raw-array NEOPAX field mirror
  - `src/ntx/_neopax_vmec_jax_field.py` owns the `vmec_jax` / `booz_xform_jax`
    backed field builders
  - `src/ntx/_neopax_field_utils.py` owns shared safe numerical helpers used by
    both paths
- [x] the VMEC-JAX Boozer helper split is complete:
  - `src/ntx/_vmec_jax_boozer.py` now owns optional checkout imports,
    in-memory Boozer transform bundling, profile `gmnc` reconstruction, and the
    right-handed Boozer sign convention
  - `src/ntx/vmec_jax_backend.py` now focuses on boundary contexts, VMEC state
    construction, relaxation, and conversion to `BoozerSurface`
  - the added physics gate checks scalar and profile handedness through
    `B_\zeta + \iota B_\theta >= 0`, matching the documented Boozer Jacobian
    convention before any transport solve is run
- [x] the VMEC-JAX backend split is now narrower:
  - `src/ntx/_vmec_jax_boundary.py` owns boundary contexts, boundary-projected
    initial states, implicit solves, and explicit relaxation
  - `src/ntx/_vmec_jax_surfaces.py` owns VMEC-JAX state/wout conversion to
    `BoozerSurface` objects
  - `src/ntx/vmec_jax_backend.py` remains the public compatibility facade
  - the boundary-edge transfer gate now checks that traced edge arrays reach
    both implicit and explicit fixed-boundary solve paths
- [x] the VMEC-JAX to NEOPAX imported-field split is now narrower:
  - `src/ntx/_neopax_vmec_jax_profiles.py` owns VMEC scalar/profile transfer
    into the differentiable field builder, including `rho = sqrt(s)`, toroidal
    flux, edge `R_{00}`, and volume-profile extraction
  - `src/ntx/_neopax_vmec_jax_boozer.py` owns the Boozer `gmnc` bundle helper
    used by imported NEOPAX field construction
  - `src/ntx/_neopax_vmec_jax_field.py` now focuses on assembling
    `DifferentiableNeopaxField` objects from VMEC-JAX state data
  - the radial-metric transfer gate now checks the axis-safe imported-field
    construction before boundary-to-current derivative workflows use it
- [x] the eighth no-behavior-change workflow split is complete:
  - `src/ntx/_neopax_scan.py` now owns NTX-to-NEOPAX scan assembly from
    callbacks, in-memory surfaces, and imported VMEC-JAX states
- [x] the NTX-to-NEOPAX scan split is now narrower:
  - `src/ntx/_neopax_scan_fields.py` owns radial, collisionality, `drds`, and
    `E_s`/`E_r` field-channel validation
  - `src/ntx/_neopax_scan_coefficients.py` owns monoenergetic coefficient
    solves and reference-normalization bridge blocks
  - `src/ntx/_neopax_scan.py` remains the compatibility builder for callback,
    explicit-surface, VMEC-JAX state, and VMEC-JAX boundary workflows
  - the field-channel normalization gate now checks
    `E_r = E_s * transport_psi_scale` before any imported closure workflow
    consumes the scan
- [x] the ninth no-behavior-change workflow split is complete:
  - `src/ntx/_solver_core.py` now owns prepared solve entry points, coefficient
    assembly, and the custom-VJP contract
  - `src/ntx/_solver_factorization.py` owns the dense block-tridiagonal Schur
    recursion, reusable LU factors, low-order back-substitution, residual
    checks, and the factorized adjoint solve
  - `src/ntx/_solver_scan.py` imports the low-level solve directly from the new
    factorization module while public `solver.py` compatibility exports remain
    unchanged
- [x] the tenth no-behavior-change workflow split is complete:
  - the autodiff workflow lane was first moved behind
    `src/ntx/_autodiff_workflows.py`; it is now further split into
    `_autodiff_inverse.py`, `_autodiff_derivatives.py`, and
    `_autodiff_profile.py`
  - `src/ntx/_autodiff_bootstrap.py` owns deterministic and robust
    bootstrap-current optimization loops and their geometry-control objective
    assembly
  - public `ntx.autodiff` and `ntx.workflows` imports remain unchanged
- [x] the eleventh no-behavior-change workflow split is complete:
  - `src/ntx/_profiles_transport_closure.py` now owns profile-transport losses,
    mismatch normalization, positivity-preserving primitive updates, and
    transport-relaxation scaling
  - `src/ntx/_profiles_transport.py` now owns only the iterative transport-loop
    runners while preserving the public `ntx.profiles` compatibility exports
- [x] the twelfth no-behavior-change profile split is complete:
  - `src/ntx/_profiles_radial.py` owns shared radial broadcast, smoothing, and
    single-radius helper algebra
  - `src/ntx/_profiles_channels.py` owns NTX scan-channel interpolation,
    species particle/current response proxies, and the charge-weighted
    ambipolar residual
  - `src/ntx/_profiles_primitives.py` owns primitive density/temperature to
    thermodynamic-force reconstruction
  - `src/ntx/_profiles_eval.py` now focuses on ambipolar profile solves and the
    bootstrap-current objective while preserving the old internal import
    surface for compatibility
- [x] the thirteenth no-behavior-change profile split is complete:
  - `src/ntx/_profiles_transport_terms.py` owns transport mismatch algebra,
    update normalization/clipping, primitive density/temperature mismatch
    terms, and closure-relaxation scaling
  - `src/ntx/_profiles_transport_closure.py` now focuses on loss assembly and
    explicit profile/primitive update application while preserving the old
    internal helper import surface for compatibility
- [x] the profile-control ownership split is complete:
  - `src/ntx/_profiles_control_scalar.py` owns scalar control application and
    scalar profile-control optimization
  - `src/ntx/_profiles_control_basis.py` owns radial-basis control application,
    basis optimization, and basis modifier algebra
  - `src/ntx/_profiles_controls.py` remains a compatibility facade for public
    and internal imports
- [x] the next no-behavior-change autodiff split is complete:
  - `src/ntx/_autodiff_inverse.py` owns the synthetic inverse-problem workflow
  - `src/ntx/_autodiff_derivatives.py` owns the finite-difference derivative
    audit workflow
  - `src/ntx/_autodiff_profile.py` owns profile electric-field fitting and
    linearized-versus-Monte-Carlo uncertainty propagation
  - `src/ntx/_autodiff_workflows.py` is now a compatibility export facade,
    preserving the previous public and private import surface
- [x] the prepared-solver derivative path is now easier to audit:
  - `src/ntx/_solver_adjoint.py` owns the prepared custom-VJP primal cache,
    coefficient pullback, and adjoint parameter-gradient accumulation
  - `src/ntx/_solver_core.py` remains focused on public prepared solves,
    coefficient-vector entry points, and transport-result assembly
- [x] the physics-gate registry is now split by ownership:
  - `src/ntx/validation/_physics_gate_analytical.py` owns test-backed
    analytical and normalization gates
  - `src/ntx/validation/_physics_gate_artifact_registry.py` owns
    artifact-backed and monitored stress-gate definitions
  - `src/ntx/validation/_physics_gate_registry.py` remains the compatibility
    facade used by public validation imports and artifact evaluators
- [ ] next restructuring target should be chosen from the remaining largest
  internal modules, with the next likely focus being documentation/testing
  structure or a split of any new workflow module that grows past reviewable
  size
- [x] profile tests are now split by lane:
  - `tests/test_profiles_unit.py` for cheap helper and failure-path coverage
  - `tests/test_profiles_workflows.py` for solve/control/transport workflows
- [x] first literature-anchored profile validation artifact added:
  - `examples/profile_force_reconstruction_audit.py`
  - archived precise-QS QA/QH primitive-to-force reconstruction figure and JSON
  - treated as a monitored benchmark-family stress test for the current
    primitive-profile builder, not a parity gate
- [x] first fast ambipolarity profile identity gate added:
  - charge-symmetric species with equal particle-flux response must give
    `sum_s Z_s Gamma_s = 0` exactly
  - this anchors the local residual algebra used by the ambipolar `E_r(r)`
    solve before broader transport or optimization examples are interpreted
- [x] first primitive-transport state-space gate added:
  - explicit primitive density/temperature updates must remain finite and
    positive even under deliberately extreme normalized transport mismatches
  - this keeps the current proxy transport lane physically bounded while the
    stronger self-consistent transport model remains planned work
- [x] first profile-interpolant derivative gate added:
  - `D33` sensitivities through the imported profile interpolation layer and
    electric-field basis must match centered finite differences on a controlled
    coefficient table
  - this protects the interpolation path used by inverse design and
    uncertainty propagation before broader geometry-family derivative claims
    are promoted
- [x] Boozer-coordinate normalization gate added:
  - owned Boozer surfaces must satisfy
    `J B^2 = B_zeta + iota B_theta`, `B^theta J = iota`, and `B^zeta J = 1`
  - this anchors the geometry normalization consumed by parallel streaming,
    magnetic-drift source terms, and imported geometry workflows
- [x] profile-control linear-response gate added:
  - scalar and radial-basis controls are identity maps at zero control
  - finite controls exactly follow their prescribed species response matrices
    and radial basis functions
  - this protects differentiable profile optimization and uncertainty
    workflows from hidden nonlinear control-map changes
- [x] VMEC-JAX boundary-edge transfer gate added:
  - traced `Rcos`, `Rsin`, `Zcos`, and `Zsin` edge arrays are forwarded to the
    implicit residual solve
  - the same traced arrays are forwarded to the explicit relaxed solve
  - this protects boundary-to-current derivative workflows from accidentally
    using stale non-differentiated boundary data
- [x] VMEC-JAX to NEOPAX radial-metric transfer gate added:
  - the imported field builder keeps `rho = sqrt(s)` and axis regularization
    explicit
  - enclosed-volume, `V'`, edge-radius, and toroidal-flux scales are tested
    with fake VMEC-JAX modules before field objects feed current workflows
  - this protects differentiable bootstrap-current examples from hidden
    normalization drift in the VMEC-to-NEOPAX bridge
- [x] NTX-to-NEOPAX field-channel normalization gate added:
  - radial `rho`, `drds`, and electric-field table shape validation is owned by
    a small helper module
  - missing `E_s` or `E_r` channels are reconstructed through the surface
    transport normalization
  - this protects bootstrap-current and profile workflows from silent
    electric-field convention drift in the database handoff
- [x] first artifact-backed autodiff uncertainty benchmark added:
  - `examples/autodiff_profile_uncertainty.py`
  - linearized covariance propagation against a Monte Carlo ensemble on the
    differentiable NEOPAX-style profile fit under a prescribed Gaussian
    parameter perturbation
  - publication-ready PNG/PDF plus JSON metrics
  - treated as the current repository-owned uncertainty-propagation stress
    benchmark for the autodiff lane, not yet a parity gate
- [x] first differentiable robust-design benchmark added:
  - `examples/bootstrap_current_robust_optimization.py`
  - deterministic versus robust optimization under a prescribed Gaussian
    control uncertainty
  - publication-ready PNG/PDF plus JSON metrics
  - treated as a tracked robust-design stress benchmark, not a literature-grade
    parity claim

### Benchmark Maturity And Open Lanes

Research-grade / positive validation surface:

- Escoto-style monoenergetic collisionality/convergence benchmark
- integrated W7-X transfer gate
- precise-QS Redl versus archived SFINCS audit
- derivative audit against centered finite differences
- prepared derivative benchmark

Tracked stress benchmarks / open lanes:

- fixed-field `NTX+NEOPAX` closure gap
- primitive-to-force profile reconstruction audit
- autodiff profile uncertainty benchmark
- robust bootstrap-current optimization benchmark
- boundary-to-output forward-mode derivative benchmark on boundary-projected
  `vmec_jax` geometry

These open lanes stay in the repository on purpose. They are useful research
and methods surfaces, but they should not be promoted to parity or literature
claims until they are anchored to stronger external baselines.
- [x] targeted branch-coverage tightening landed on the refactored workflow
  modules without adding a new heavy benchmark lane:
  - `_autodiff_workflows.py` now closes fully in the current fast coverage
    subset
  - `_profiles_transport.py` now closes fully in the current fast coverage
    subset
  - `_profiles_eval.py` and `_profiles_controls.py` are now both above `98%`
    in the current fast coverage subset
  - these gains come from narrow unit/workflow tests, not new physics paths
- [x] the next cheap-coverage hardening slice now closes the main wrapper and
  scan helpers in a targeted coverage lane:
  - `neopax.py` reaches `100%` in the targeted facade/scan subset
  - `_solver_scan.py` reaches `98.6%`
  - `parallel.py` reaches `99.0%`
  - `autodiff.py` reaches `96.2%`
  - these gains come from direct branch tests on callback normalization,
    scan-sharding/error handling, worker environment setup, and import-fallback
    behavior
- [x] the next fast-lane utility cluster is now effectively closed through
  cheap unit tests and existing lightweight module tests:
  - `cli.py` reaches `100%`
  - `io.py` reaches `100%`
  - `database.py` reaches `100%`
  - `_checkout_paths.py` reaches `98.9%`
  - these gains come from direct helper/entrypoint tests, not from subprocess
    benchmark expansion
- [x] the monoenergetic validation summary is now being promoted from a loose
  example to an artifact-backed research benchmark:
  - the figure remains the same core methods panel
  - it now writes `validation_summary.json` with machine-readable transport
    curves, Onsager residuals, low-collisionality tail slopes, and `N_xi`
    convergence metrics
  - docs now classify it as the literature-anchored Escoto/Helander numerical
    benchmark lane rather than just a convenience figure
- [x] the next core-shard remeasurement now shows that the fast CI lane is
  already beyond the program headline target while staying inside the intended
  runtime envelope:
  - `207 passed`, `2 deselected` in about `4m31s`
  - `98.24%` overall coverage on the 3.11 core shard
  - `physics_gates.py` improved to `97.1%`
  - `booz.py` is no longer one of the dominant weak modules
  - the follow-on cheap closure slice then lifted:
    - `_geometry_eval.py` to `96.8%`
    - `_neopax_io.py` to `100%`
  - the next VMEC helper slice then lifted:
    - `vmec.py` to `99.5%`
    - `vmec_jax_vmec.py` to `98.3%`
  - the next cheap targets are now mostly `vmec_jax_backend.py` and any
    remaining low-signal facade modules, so the next step should be chosen only
    if it keeps the fast-lane runtime stable
- [x] the literature/testing plan now has a maintained benchmark matrix:
  - `src/ntx/validation/benchmark_matrix.py` is the source of truth
  - lane-owned benchmark-entry modules now split monoenergetic, bootstrap,
    integrated-workflow, autodiff, profile, performance, and geometry-breadth
    metadata while preserving the public `benchmark_matrix()` facade
  - the autodiff benchmark registry is now split again into derivative-path
    and design/optimization ownership modules while preserving benchmark order
    and the public autodiff registry facade
  - `scripts/build_benchmark_matrix.py` writes
    `docs/_static/benchmark_matrix.json`
  - `docs/benchmark-matrix.md` documents positive gates, stress gates,
    software gates, and planned lanes
  - `tests/test_benchmark_matrix.py` requires every active lane to declare
    existing scripts, tests, artifacts, and docs
  - the same test file now requires every benchmark claim to carry literature
    anchors, docs, and the right active-versus-planned artifact contract
  - planned lanes now explicitly keep the full Escoto-style geometry-family
    reproduction and larger geometry-control autodiff validation visible until
    they are ready for artifacts
- [x] the first package-structure namespace step is in place without moving
  implementation files:
  - `ntx.core` re-exports solver, scan, and transport helpers
  - `ntx.workflows` re-exports autodiff, profile, and imported database helpers
  - `ntx.validation` owns benchmark/validation registries
  - flat public imports remain supported, so this is a no-behavior-change
    restructuring step
  - the top-level public export list is now checked for duplicates in the
    namespace-import tests, so future facade edits cannot silently accumulate
    repeated names
- [x] the first larger-geometry-control autodiff slice is artifact-backed:
  - `examples/geometry_control_derivative_benchmark.py`
  - controls three independent Boozer-harmonic amplitudes on the owned analytic
    surface
  - writes `docs/_static/geometry_control_derivative_benchmark.{png,pdf,json}`
  - current default max AD/centered-finite-difference mismatch is about
    `1.35e-4`
  - kept as a stress benchmark until transferred to reusable VMEC/Boozer
    geometry-control families
- [x] the next geometry-control autodiff slice now reaches repository-owned
  file-backed surfaces:
  - `examples/file_backed_geometry_control_derivative_benchmark.py`
  - runs the same AD versus centered-finite-difference audit on the sample
    Boozer file and the sample VMEC-backed surface
  - writes
    `docs/_static/file_backed_geometry_control_derivative_benchmark.{png,pdf,json}`
  - current default max AD/centered-finite-difference mismatch is about
    `2.1e-4`
  - remaining open work is now broader reusable geometry families plus a
    prepared implicit-adjoint geometry pullback, not the basic transfer from
    analytic to file-backed geometry
- [x] manuscript artifact hardening now includes the geometry-control stress
  benchmarks:
  - `scripts/build_manuscript_artifacts.py` records the owned-surface and
    file-backed grids, control modes, coefficient sets, and AD/FD mismatch
    metrics
- [x] the first imported boundary-to-output autodiff slice is now
  artifact-backed:
  - `examples/boundary_forward_mode_current_derivative_benchmark.py`
  - uses low-dimensional `vmec_jax` boundary controls, a boundary-projected
    VMEC state, `booz_xform_jax`, NTX, and NEOPAX
  - writes
    `docs/_static/boundary_forward_mode_current_derivative_benchmark.{png,pdf,json}`
  - current default max AD/centered-finite-difference mismatch is below
    `1e-4`
  - this closes the fast forward-mode boundary-control lane on the repository
    sample input, but it does not yet claim self-consistent equilibrium
    sensitivity for bootstrap current
- [x] the first self-consistent forward-mode equilibrium slice is now
  artifact-backed on committed QA and QH family cases:
  - `examples/explicit_relaxed_boundary_current_derivative_benchmark.py`
  - uses an explicitly relaxed fixed-boundary `vmec_jax` solve,
    `booz_xform_jax`, NTX, and NEOPAX
  - writes
    `docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.{png,pdf,json}`
  - the JSON artifact records ordinary-versus-explicit primal-volume agreement
    in addition to the AD/centered-finite-difference mismatch metrics on both
    committed cases
  - this closes the first self-consistent forward-mode boundary-to-current lane
    on committed QA/QH inputs while leaving the implicit-equilibrium and
    reverse-mode lanes open
  - the matching implicit QA Boozer-scalar probe still returns an all-zero
    reverse-mode gradient against a clearly nonzero centered finite difference,
    so that lane is constrained as broken rather than merely unvalidated
- [x] the implicit-equilibrium lane now has a maintained diagnostic on the
  committed QA case:
  - `examples/implicit_equilibrium_forward_mode_derivative_benchmark.py`
  - uses the implicit fixed-boundary `vmec_jax` residual solve with
    `residual_tangent_mode="auto"`
  - writes
    `docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.{png,pdf,json}`
  - current default objectives are equilibrium volume, a Boozer scalar, and a
    representative NTX monoenergetic `D33` observable
  - the current artifact is asymmetric: equilibrium volume matches centered
    finite differences, but the Boozer and NTX transport observables remain
    open on the implicit lane
  - the JSON artifact also records the still-broken reverse-mode Boozer-scalar
    diagnostic, so the remaining gap is parity through the implicit Boozer and
    transport path, then integrated current, plus reverse mode
  - `docs/_static/manuscript_claims.md` reports the max and median mismatch
    directly from the JSON artifacts
  - the figure-set metadata now covers every generated main-text and supplement
    figure, including uncertainty, profile-reconstruction, and robust-design
    stress figures
- [x] the literature roadmap has been refreshed around the remaining
  research-grade validation lanes:
  - adjoint neoclassical optimization
  - differentiable-programming verification
  - direct neoclassical ion-transport optimization
  - quasi-isodynamic and omnigenous geometry-breadth benchmarks

### Phase 1: Source-Tree Restructuring Without Physics Changes

Refactor for maintainability first. Do not mix new physics with file splitting.

Target package structure:

- `src/ntx/core/`
  - solver assembly
  - prepared solve
  - scans
  - transport post-processing
- `src/ntx/geometry/`
  - dataclasses
  - Fourier evaluation
  - Boozer loaders
  - VMEC loaders
  - radial-coordinate helpers
- `src/ntx/workflows/`
  - autodiff
  - optimization
  - uncertainty
  - profiles
  - NEOPAX coupling
- `src/ntx/io/`
  - TOML parsing
  - NPZ/HDF5 writers
  - CLI-facing config normalization
- `src/ntx/validation/`
  - physics-gate registry
  - artifact readers
  - benchmark summaries
  - current split: physics-gate types, registry definitions, and artifact-gate
    evaluation now live in separate internal modules behind the stable
    `ntx.validation.physics_gates` facade
  - current split: benchmark-matrix dataclasses and literal lane types now live
    separately from the maintained benchmark-entry registry

Concrete file splits to prioritize:

1. `solver.py`
   - case types
   - operator assembly entry points
   - prepared solve
   - scan helpers
   - custom-VJP / implicit-adjoint path
2. `profiles.py`
   - ambipolar root finding
   - profile parameterizations
   - profile transport loop
   - profile optimization helpers
   - bootstrap-current proxy / reporting
3. `autodiff.py`
   - local sensitivity helpers
   - inverse problem examples
   - optimization objectives
   - uncertainty propagation helpers
4. `inputfiles.py`
   - schema and defaults
   - parsing / validation
   - runtime dispatch
5. `neopax.py`
   - scan builder
   - array mapping
   - HDF5 I/O
   - benchmark-specific helpers
6. `geometry.py`
   - common surface dataclasses
   - Fourier/Boozer evaluation
   - VMEC mapping
   - grid projection

Rules for this phase:

- preserve public API via compatibility re-exports in `ntx.__init__`
- functionality unchanged except for bug fixes directly exposed by new tests
- add docstrings to all public functions/classes touched
- add short orienting comments only around non-obvious physics or linear
  algebra blocks

### Phase 2: Test Pyramid And Coverage Program

#### A. Unit Tests

Target:

- every pure function and dataclass validator in core modules gets direct tests
- every normalization/helper branch gets an explicit test
- every public API function gets at least one success-path and one failure-path
  test

Required unit-test groups:

1. geometry and coordinates
   - Boozer Fourier evaluation
   - VMEC radial mapping
   - Jacobian and drift source terms
2. operator assembly
   - Legendre block structure
   - nullspace enforcement
   - electric-field normalization
3. solver
   - direct vs prepared solve equality
   - scan batching equivalence
   - CPU vs multiprocess equivalence on owned fixtures
4. transport post-processing
   - `D11`, `D13`, `D31`, `D33`, `D33_spitzer`
   - Onsager residual
5. I/O and CLI
   - config parsing
   - schema failures
   - output-file integrity
6. workflow helpers
   - profile interpolants
   - NEOPAX array mapping
   - autodiff helper argument validation

#### B. Regression Tests

Target:

- lock down every repository-owned example and every committed manuscript
  artifact through JSON/NPZ summaries rather than brittle full-image diffs

Required regression surfaces:

- example outputs
- validation summaries
- benchmark manifests
- physics gate report
- bootstrap-current and profile example JSON summaries

#### C. Physics And Literature Anchored Tests

These are the non-negotiable science tests.

1. Escoto / 2024 monoenergetic convergence and benchmarking
   - reproduce the convergence studies for:
     - W7-X EIM
     - W7-X KJM
     - CIEMAT-QI
   - reproduce benchmark families for:
     - `D11`
     - `D31`
     - `D33`
   - include zero and finite `E_r` cases where the paper does
   - carry the DKES normalization appendix logic as explicit tests
2. precise-QS bootstrap-current benchmark from Landreman et al. 2022
   - fixed-field QA and QH current-profile comparison
   - Redl vs archived SFINCS gate
   - `NTX+NEOPAX` documented stress-test metric
   - quasi-symmetry-specific `E_r` sensitivity check on the fixed-field family
3. integrated W7-X workflow transfer
   - rebuilt raw-branch database gate
   - resolution ladder
   - exact loader normalization regression
4. solver-side identities from the monoenergetic literature
   - Onsager symmetry
   - stellarator-symmetry relation for the low-order coefficients where
     applicable
   - exact generated `P=2` Sonine/Hankel recovery of the current closure

#### D. Autodiff, JAX, And Optimization Tests

Required gates:

1. local sensitivities
   - direct autodiff vs centred finite differences
   - prepared implicit-adjoint vs direct reverse mode
   - Jacobian consistency under `jit` and batched execution
2. inverse design
   - recover known synthetic geometry or profile parameters from generated
     target coefficients
   - verify optimizer convergence and parameter recovery tolerance
3. uncertainty quantification
   - linearized covariance propagation using Jacobians
   - compare linearized uncertainty bands against small Monte Carlo ensembles on
     low-dimensional examples
   - verify Fisher / Hessian-vector products against finite-difference probes
4. stellarator optimization
   - bootstrap-current optimization example remains monotone under fixed seed
   - profile-control optimization improves the frozen objective
   - basis-control optimization remains stable under autodiff and `jit`

### Phase 3: Literature Benchmark Matrix

This is the benchmark matrix the code should own once hardened.

#### Benchmark family A: Monoenergetic coefficient validation

Anchor:

- Escoto et al. 2024 and thesis convergence/benchmark figures

Deliverables:

- `D11`, `D31`, `D33` parity plots against external references
- convergence ladders in `N_xi`, `N_theta`, and `N_zeta`
- appendix-style normalization audit plots

#### Benchmark family B: Bootstrap-current formula and closure validation

Anchors:

- Landreman et al. 2022
- archived QA/QH fixed-field benchmark data

Deliverables:

- Redl vs archived SFINCS figure on the interior benchmark window
- `NTX+NEOPAX` status figure on the same surfaces
- explicit documentation that this is a closure stress test, not a solver gate

#### Benchmark family C: Integrated workflow transfer

Anchor:

- frozen W7-X imported workflow reference profile

Deliverables:

- grid-convergence figure
- raw-branch transfer gate
- normalization round-trip test

#### Benchmark family D: Differentiable workflow validation

Anchors:

- Paul et al. 2019 adjoint optimization framing
- McGreivy 2024 differentiable programming framing
- TORAX 2024 style of differentiable-transport validation

Deliverables:

- derivative parity figure
- inverse-design recovery figure
- uncertainty-propagation figure
- optimization-history figure

#### Stretch benchmark family E: Additional physics-strengthening cases

Add if inputs are available without creating a new physics project:

1. low-bootstrap-current quasi-isodynamic / piecewise-omnigenous example from
   recent literature
2. trajectory-approximation stress sweep motivated by Landreman 2011
3. additional W7-X experimental-profile-inspired scans for robustness

### Phase 4: Documentation And Commenting

Add or expand:

1. testing architecture page
   - unit vs regression vs benchmark vs hardware lanes
   - expected runtime for each lane
2. source-tree architecture page
   - module ownership
   - public API boundary
   - compatibility re-export policy
3. benchmark reproducibility page
   - literature source
   - local inputs
   - script name
   - expected artifact
4. autodiff methods page
   - sensitivity
   - inverse design
   - uncertainty propagation
   - optimization

Docstring policy:

- every public dataclass, function, and workflow entry point gets:
  - purpose
  - key inputs
  - returned quantities
  - normalization or coordinate caveats where relevant

### Phase 5: Manuscript Figure Plan

The manuscript should ultimately add or refresh the following figure families
from repository-owned scripts:

1. Escoto-style monoenergetic convergence panels
   - representative `N_xi` convergence curves
   - representative DKES/SFINCS parity panels
2. W7-X integrated transfer figure
   - the positive end-to-end validation result
3. precise-QS Redl / SFINCS / `NTX+NEOPAX` figure
   - presented as:
     - Redl parity result
     - reduced-closure stress result
4. derivative validation figure
   - direct AD vs finite differences
   - prepared adjoint vs direct reverse mode
5. differentiable application figures
   - bootstrap-current optimization
   - inverse design recovery
   - uncertainty propagation / sensitivity bars
6. code-quality supplement figure/table
   - coverage by module
   - test pyramid summary
   - benchmark matrix

### Additional Literature Requirements To Carry Into The Plan

Beyond the current gates, the literature motivates these explicit requirements:

1. retain symmetry/Onsager structure as hard acceptance gates
   - Escoto 2024
   - Sugama \& Horton 1996
2. separate monoenergetic-kernel validation from closure validation
   - Escoto thesis
   - Landreman et al. 2022
   - Maa{\ss}berg et al. 2009
3. keep the monoenergetic approximation limits visible in benchmarks
   - Landreman 2011
4. validate differentiable optimization with explicit gradient checks before
   claiming design capability
   - Paul et al. 2019
   - McGreivy 2024
5. document compile-vs-steady-state performance separately for JAX workflows
   - current NTX profiling
   - TORAX 2024 style differentiable transport framing
6. keep strong-gradient limitations explicit when presenting profile and
   bootstrap-current proxy workflows
   - Trinczek, Parra, Catto 2025
7. add at least one zero-bootstrap-current or near-zero-bootstrap-current
   benchmark family from optimized omnigenous / piecewise-omnigenous literature
   before broadening the claim surface for optimization workflows
   - Calvo et al. 2025
   - Liu et al. 2026
8. keep optimization/UQ demonstrations tied to robust-design use cases rather
   than only synthetic curve fitting
   - Gil et al. 2026
   - Lee et al. 2024

## Start Here

The first implementation block after this planning pass should be:

1. measure and publish real coverage by module in CI
2. split the oversized modules without changing behavior
3. reorganize tests into fast/unit, regression, and literature-benchmark lanes
4. lock the Escoto and W7-X benchmark families into artifact-backed tests
   and the maintained benchmark matrix
5. only then push toward the 95% target and the expanded autodiff/UQ program

## Open Code Lanes

### 1. Optimization-Grade Derivatives

- [x] direct autodiff validation against centered finite differences
- [x] prepared differentiable solve interface
- [x] implicit-adjoint backward rule for the prepared dense solve
- [ ] reduce memory and factorization overhead in the adjoint path
- [ ] extend derivative benchmarks from scalar case parameters to larger
  geometry-control families
- [ ] add larger optimization loops that stress differentiability under real
  scan/database workloads

### 2. Profile-Grade Transport Workflows

- [x] ambipolar `E_r(r)` solve on scan data
- [x] profile-control and basis-control workflows
- [x] explicit source-target transport-relaxation loop
- [x] primitive density/temperature transport closure with positivity and radial
  smoothing
- [ ] move from proxy transport iteration to a more predictive self-consistent
  profile transport workflow
- [ ] improve closure expressiveness beyond the current simple source/target
  parameterization
- [ ] tighten profile-level physical interpretability and diagnostics for
  long-radius studies

### 3. Geometry Breadth

- [ ] organize stronger research workflows for hidden-symmetry, omnigenous, and
  piecewise-omnigenous studies
- [ ] support larger in-memory geometry perturbation campaigns cleanly
- [ ] expand VMEC/Boozer family examples beyond the current W7-X-centered set

### 4. Throughput And Parallelism

- [x] serial batched scan path
- [x] multiprocess parallel scan path
- [x] CPU/GPU crossover characterization on repository-owned cases
- [ ] improve prepared-geometry reuse across larger scan campaigns
- [ ] characterize production-grid crossover points more systematically
- [ ] pursue stronger multi-device throughput only where the measured workload
  justifies the complexity

### 5. Physics Expansion

- [ ] add higher-level transport closures only after the current profile lane is
  technically stable
- [ ] stage momentum-restoring or broader transport models without weakening the
  current monoenergetic core
- [ ] develop the next closure model as an arbitrary-order moment-equation
  extension rather than a benchmark fit:
  - treat the present three-moment system as the `P=2` truncation
  - make Sonine truncation order configurable
  - generate projected closure matrices programmatically for arbitrary order
  - replace hard-coded reduced collisional blocks with momentum-conserving
    arbitrary-order blocks
  - require transfer to the integrated W7-X workflow before promoting any new
    closure as a default path

### 6. Fixed-Field And Integrated Validation

- [ ] keep the validation surface split explicit:
  - fixed-field QA/QH reference family for Redl vs SFINCS vs `NTX+NEOPAX`
  - finite-beta QA/QH and W7-X for integrated workflow relevance
- [x] add a fixed-radius transport-matrix audit against SFINCS-JAX on the
  fixed-field QA/QH reference cases, focused on `D13`, `D31`, and `D33`
- [ ] isolate the outer-radius amplitude failure by auditing:
  - VMEC file loading and radial mapping
  - SFINCS-JAX transport-matrix normalization
  - NTX `D13` / `D31` / `D33` channel conventions
  - NTX to NEOPAX handoff conventions
- [x] reproduce the Boozer-based Redl path from the Zenodo bundle robustly on
  the fixed-field QA/QH reference family
- [x] add frozen local-only regression tests for the fixed-field audit helpers
  and benchmark discovery
- [x] add a curated `NTX+NEOPAX` vs SFINCS bootstrap-current validation figure
  to the README, with the benchmark status stated honestly
- [x] codify the physics-gate hierarchy in shipped docs and a lightweight gate
  report script
- [ ] keep the gate thresholds literature-anchored and synchronized with the
  benchmark artifacts

### 7. Throughput, Profiling, And Memory

- [ ] profile the prepared solve, monoenergetic scan, and `NTX+NEOPAX`
  workflow end to end on representative QA/QH/W7-X studies
- [ ] identify the dominant NTX bottlenecks before changing solver
  infrastructure:
  - operator assembly
  - prepared solve reuse
  - scan batching / vectorization
  - NTX to NEOPAX database handoff
- [ ] evaluate JAX-first optimization paths only where profiling justifies
  them:
  - stronger `jit`/`vmap` staging
  - lower-overhead scan kernels
  - prepared-geometry reuse
  - selective use of `lineax` / `equinox` if they reduce runtime or memory
- [ ] keep memory pressure and differentiability as explicit gates for any
  performance work

### 8. QA And Maintenance

- [ ] keep documentation synchronized with the actual shipped algorithms
- [ ] continue closing coverage on optional/error-path code
- [ ] keep synthetic fixtures minimal and readable
- [ ] keep NEOPAX coupling aligned with the active upstream interface

## Research References

- Escoto thesis:
  [arXiv:2510.27513](https://arxiv.org/abs/2510.27513)
- adjoint neoclassical optimization:
  [arXiv:1904.06430](https://arxiv.org/abs/1904.06430)
- differentiable programming for plasma workflows:
  [arXiv:2410.11161](https://arxiv.org/abs/2410.11161)
- hidden-symmetry optimization:
  [arXiv:2502.09350](https://arxiv.org/abs/2502.09350)
- zero-bootstrap-current piecewise omnigenity:
  [arXiv:2505.02546](https://arxiv.org/abs/2505.02546)
- combined omnigenity and piecewise omnigenity:
  [arXiv:2603.12139](https://arxiv.org/abs/2603.12139)
- reactor-relevant low-bootstrap-current stellarator context:
  [arXiv:2512.08825](https://arxiv.org/abs/2512.08825)

## Nearby Software Context

These projects matter for technical comparison and planning, not as design
templates:

- [NEOPAX](https://github.com/uwplasma/NEOPAX): imported profile and database
  workflows
- [sfincs_jax](https://github.com/uwplasma/sfincs_jax): JAX transport and scan
  infrastructure
- [gyaradax](https://github.com/gerkone/gyaradax): differentiable research
  workflow examples
- [GX](https://bitbucket.org/gyrokinetics/gx/src/gx/): production parallelism
  and throughput mindset

## Next Concrete Code Steps

1. Derive the exact NTX-to-SFINCS transport-matrix normalization bridge for the
   fixed-field QA/QH reference cases:
   - keep full transport-matrix parity focused on `L13`, `L31`, and `L33`
   - treat the fixed-field zero-`E_r` bootstrap-current mismatch itself as a
     `D13/L31` closure-path problem, since the active NEOPAX no-momentum
     closure has `A3 = 0` on that benchmark
   - keep `L33` as the main unresolved channel for full matrix parity rather
     than as the sole explanation for the fixed-field current gap
2. Keep finite-beta QA/QH and W7-X bootstrap-current validation in the
   `NTX+NEOPAX` lane, separate from the fixed-field coefficient audit.
3. Profile the prepared solve and `NTX+NEOPAX` workflow to identify the real
   runtime and memory bottlenecks before changing solver internals.
4. Continue the profile-transport and derivative work only after the
   fixed-field audit and profiling picture are technically clear.
5. Move the next closure-model push onto an arbitrary-order moment-equation
   lane:
   - no fitted remaps
   - keep `U_parallel = n c_0` fixed
   - benchmark convergence with truncation order on precise-QS QA/QH
   - require no regression in integrated W7-X workflows

## Active Code Log

- NTX now includes:
  - implicit-adjoint prepared derivatives
  - ambipolar profile solves and profile-family workflows
  - scalar and basis-control optimization layers
  - explicit profile transport-relaxation loops
  - primitive density/temperature transport closure
  - CPU/GPU/multiprocess scaling workflows
- The profile workflows were tightened after a full visual audit:
  - accepted-step transport updates are in place
  - radial smoothing was added to force-proxy and primitive-profile updates
  - primitive transport now includes explicit density/temperature source-target
    closure terms
  - the NTX-only bootstrap-current example now uses analytic radial gradients
    and an interior radial window to avoid boundary artifacts
- Current interpretation:
  - the monoenergetic and differentiable lanes are strong enough for serious
    research use now
  - the main remaining technical gap is the transition from current proxy-based
    profile transport workflows to a stronger self-consistent transport layer
- Bootstrap-current scope is now explicit again:
  - NTX owns monoenergetic coefficients and flux channels
  - NEOPAX owns bootstrap-current closure and higher-level transport workflows
  - the native-bootstrap-current experiment was reverted on purpose
  - bootstrap-current truth in validation plots should be labeled `NTX+NEOPAX`
    when that path is used
- The next closure-model lane is now constrained by explicit physics gates:
  - keep `U_parallel = n c_0`
  - treat the current closure as the `P=2` truncation
  - preserve Onsager/ambipolar structure at finite order
  - preserve intrinsic ambipolarity in symmetric limits at each truncation
  - preserve particle, momentum, and energy invariants of the projected
    collisional operator
  - preserve weighted self-adjointness of the finite-order collisional form
  - preserve a momentum-conserving common-flow null mode in the collisional
    blocks
  - preserve non-negative entropy production of the symmetric collisional form
  - recover the active low-order momentum-conserving collision blocks from the
    standard moment equations, with only the runtime heat-flow basis sign
    convention differing from the canonical notation
  - require transfer from precise-QS QA/QH to integrated W7-X
  - require controlled `Pmax` convergence on the precise-QS QA/QH stress family
  - the first generated-basis scaffold is now in place in the imported closure
    stack:
    - Sonine normalization is generated
    - source and conductivity-side `P=2` projections are generated from
      polynomial/Hankel moment identities
    - exact recovery of the current three-moment closure and the shipped W7-X
      momentum-correction regression is now a closed gate
    - the remaining implementation step is to generalize the moment system
      beyond `P=2`, not to keep rewriting the same `P=2` algebra
  - the shipped repo now exposes a lightweight physics-gate registry and report
    script:
    - `src/ntx/physics_gates.py`
    - `scripts/check_physics_gates.py`
    - the current hard artifact-backed gates are:
      - rebuilt W7-X raw-branch imported workflow `<= 2e-2`
      - precise-QS Redl vs archived SFINCS `<= 1e-1`
    - the fixed-field `NTX+NEOPAX` current benchmark remains a monitored stress
      metric rather than a solver-side acceptance gate
    - the additional literature-driven analytical requirements are now logged in
      the registry even before higher-order runtime artifacts exist:
      - intrinsic ambipolarity in symmetric limits
      - momentum-conservation null modes
      - non-negative entropy production
      - `Pmax` convergence / transfer monitoring
  - the imported closure stack now has the first arbitrary-order implementation
    scaffold in place locally:
    - configurable Sonine truncation order in the grid object
    - generated raw D13 source moments and D33 Hankel moment sequences for
      arbitrary order
    - exact `P=2` recovery and shipped momentum-correction regression both pass
    - the low-order momentum-conserving collision blocks are now also
      reconstructed from the standard low-order moment equations instead of
      remaining opaque handwritten constants; the present runtime differs from
      that canonical notation only by the sign convention of the heat-flow
      basis moment
    - a first higher-order runtime branch has now been tested:
      - current low-order closure preserved
      - diagonal Laguerre-tail damping added on the extra moments
      - committed artifact written to `docs/_static/closure_pmax_convergence.*`
      - `P=4` result:
        - precise-QS stress changes only weakly
          (`~1.16e+0` -> `~1.15e+0`)
        - imported W7-X transfer regresses immediately
          (`1.17e-12` -> `4.94e-1`)
      - conclusion:
        - this tail model is numerically stable but physically rejected
        - the next derivation has to change the collision model itself, not
          just the asymptotic damping on the same tail
    - the current model-family validation surface is now also frozen as one
      tracked report:
      - `scripts/build_closure_validation_report.py`
      - writes `docs/_static/closure_validation_report.{json,txt,png,pdf}`
      - combines:
        - precise-QS Redl vs archived SFINCS
        - fixed-field `NTX+NEOPAX` closure stress
        - rebuilt W7-X raw-branch transfer
        - rejected `Pmax>2` tail stress/transfer result
- The Zenodo `20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current`
  bundle is now available locally under the NTX repo and should be used as the
  primary fixed-field Redl/SFINCS audit source, while staying ignored by git
- The fixed-field bootstrap-current audit has now uncovered and closed three
  concrete implementation bugs:
  - the VMEC to NEOPAX bridge in `src/ntx/neopax.py` was using contravariant
    `b^theta` / `b^zeta` zero modes instead of the covariant Boozer `I/G`
    flux functions needed by the SFINCS/DKES bridge
  - the active no-momentum thermal closure in the local NEOPAX checkout was
    missing a factor of `2` in the `D13` and `D33` energy-convolution
    prefactors relative to the Legendre-formulation reference
  - the local NEOPAX momentum-correction block assembly had a broken matrix
    flattening path under the installed `lineax`, so that branch was not even
    solving the intended linear system
- A first fixed-field transport-matrix audit is now in-tree:
  - it runs SFINCS-JAX in `RHSMode=3` on the archive-backed Landreman-Paul
    QA/QH fixed-field reference equilibria at `rho = [0.25, 0.50, 0.75]`
  - it compares `L13`, `L31`, and `L33` against NTX candidate channels
    derived from `D13`, `D31`, and `D33`
  - current result: the exact `RHSMode=3` `nu_n` overwrite plus the
    archive-backed Landreman/H. Smith bridge factors tighten `L13/L31`
    substantially
  - present measured fixed-field `L13/L31` relative errors are about
    `0.12–0.29` on QA and `0.027–0.15` on QH
  - the `RHSMode=3` monoenergetic audit remains useful, but it does not probe
    the full zero-`E_r` bootstrap-current closure because it omits the
    temperature-gradient (`A2`) drive entirely
  - so the remaining open problem is no longer a generic sign or
    benchmark-family bug; it has narrowed to the full parallel-flow closure,
    especially the `RHSMode=2` row-3 (`L31/L32`) thermal channel and the final
    current observable map
- The archive-backed precise-QS current comparison is also now separated
  cleanly from the coefficient audit:
  - Redl remains close to archived SFINCS on the precise-QS family once the
    correct benchmark set is used
  - the fixed-field benchmark-side VMEC solve input also had to be corrected:
    NTX must receive `E_\psi = E_r / transport_psi_scale`, while the
    `dr/ds` factor belongs only to the DKES/SFINCS bridge metadata
  - a local NEOPAX closure patch that doubled the `D13/D33` prefactors turned
    out to be wrong: it broke the shipped W7-X no-momentum and
    momentum-correction reference tests, so those prefactors were restored to
    the validated W7-X values while keeping the lineax matrix-assembly and
    non-finite-boundary fixes
  - with that correction, the local NEOPAX W7-X reference tests pass again,
    so the remaining fixed-field QA/QH current mismatch is no longer explained
    by a generally broken local NEOPAX closure
  - the fixed-field benchmark path also had one large observable bug:
    the momentum-correction return from
    `get_Neoclassical_Fluxes_With_Momentum_Correction` is already the
    corrected `Upar`, not a separate `ΔUpar`, so the benchmark must form
    `J·B` directly from that corrected parallel flow
  - with the archived `E_r` normalization fixed and that corrected-`Upar`
    interpretation applied, `NTX+NEOPAX`
    improves substantially on the precise-QS fixed-field family:
    - interior max relative error is now about `0.319` on QA
    - interior max relative error is now about `0.101` on QH
  - a direct attempt to inject the archive-backed `reference_to_sfincs`
    factors into the NTX-to-NEOPAX database mapping over-amplified the current,
    so that is not the correct bridge
  - the paper-side benchmark now uses the exact archived fixed-field profile
    values together with archive-driven Hermite reconstruction in `rho` and an
    adaptive `nu_v` support chosen from the actual NEOPAX collisionality range
  - the previous narrow `nu_v` axis was a real setup bug, but correcting it
    does not materially reduce the fixed-field current error
  - the remaining blocker is therefore not Redl, not the benchmark family, not
    the `nu_v` support, and not the NTX VMEC solve-input normalization; it is
    the NTX-to-NEOPAX thermal/current closure for fixed-field current, now
    centered on the full parallel-flow closure rather than on the raw
    monoenergetic database handoff alone
  - the local W7-X momentum-correction reference test now passes again after
    restoring the validated prefactors, so the next blocker is no longer a
    generic lineax failure on the local NEOPAX branch; it has narrowed back to
    the fixed-field thermal/current closure itself
- The precise-QS Redl benchmark from the Zenodo bundle is now reproduced
  directly in-tree:
  - both the VMEC-based and Boozer-based Redl paths match the archived SFINCS
    profiles on the fixed-field reference family within the archived 10%
    interior-window gate
  - current measured interior max relative errors are about `9.3%` for QA
    through the VMEC path, `9.5%` for QA through the Boozer path, `4.2%` for
    QH through the VMEC path, and `4.1%` for QH through the Boozer path
  - the earlier large Redl discrepancy came from mixing benchmark families
    rather than from a failure of the Redl closure on the precise-QS reference
    cases
- The next gating audit is now explicit:
  - build an archive-backed `RHSMode=2` fixed-field parallel-flow audit
  - compare the NEOPAX row-3 `L31/L32` closure directly against SFINCS-JAX on
    the same QA/QH surfaces and profiles
  - the first `RHSMode=2` audit scaffold is now in-tree, but the full
    two-species SFINCS-JAX transport-matrix solve is still too heavy on this
    workstation in its current form, so the next implementation step is to run
    that audit in smaller slices or on a larger machine rather than to keep
    inferring row-3 mismatches indirectly from the final current profile
  - one older audit assumption has now been retired: the row-3 thermal columns
    should not be compared against raw `L31/L32` combinations directly.
    Instead, the audit must reconstruct the physical closure response under the
    exact SFINCS `whichRHS` source gradients and then convert that flow back to
    SFINCS row-3 normalization
  - the audit scaffold now supports one-species probes (`ion` or `electron`)
    plus reduced SFINCS resolution overrides, so the next direct target is the
    electron branch on the precise-QS QA/QH family rather than the full
    two-species transport matrix all at once
  - the first cached QA electron probe now closes most of that bridge:
    applying the exact `whichRHS` source gradients together with the common
    flow-normalization factor `2 B0OverBBar / sqrt(pi)` reduces the thermal
    row-3 mismatch at `rho = 0.5` to about `2.2%` for column 1 and `1.4%` for
    column 2
  - that means the dominant remaining row-3 ambiguity is no longer the
    thermal-source basis on QA; it is now:
    - extending the same bridged audit across cached QA/QH points
    - and separating the electric-field column from the thermal audit, since
      the current closure does not expose an exact `RHSMode=2` column-3 source
      channel
  - the refreshed branch-level diagnostics now make the remaining blocker much
    narrower:
    - QH total current is already near the target band, with an interior
      least-squares scale of about `0.95`
    - QA no-momentum current is already materially better than QA with
      momentum correction (`~0.24` interior max relative error versus
      `~0.32`), while QH improves strongly once momentum correction is
      included (`~0.43` down to `~0.10`)
    - that means the remaining blocker is not the raw no-momentum
      `L31/L32` current assembly itself; it is the QA momentum-correction
      branch, especially on the electron side
    - a branch-isolation check now makes that even sharper:
      - on QA, adding either the electron correction or the ion correction by
        itself makes the total current much worse than the no-momentum result,
        so the QA momentum-correction path is still not physically consistent
      - on QH, the ion correction is the part that brings the total current
        close to SFINCS, while the electron correction still moves it in the
        wrong direction
    - QA remains limited by the electron branch, not by the ion branch or by a
      global sign convention
    - the QA electron current still flips sign against archived SFINCS on `12`
      interior sample points, roughly over `rho ≈ 0.47–0.71`
    - at `rho ≈ 0.5`, the QA electron no-momentum current is about
  - one concrete implementation contradiction is now explicit in the local
    thermal/current closure: the corrected particle and heat fluxes are
    evaluated as `base + correction`, while the corrected parallel flow is not
    using the analogous row-3 correction term. That closure asymmetry is now a
    first-order audit target.
  - the first narrow `RHSMode=2` electron audit also exposed an operational
    issue: stale `sfincs_jax transport-matrix-v3` child processes can survive
    interrupted wrapper runs and silently hold CPU and memory for a long time.
    The next audit loop must therefore assume explicit child-process cleanup
    and keep each physics probe to one case, one species, one radius.
  - to make those small closure probes practical, the fixed-field QA/QH audit
    scripts now need to reuse cached NTX scan databases on disk instead of
    rebuilding the full NTX scan for every closure experiment.
      `-2.19e6`, the momentum correction contributes about `+1.92e6`, and the
      resulting total current remains slightly negative, while archived SFINCS
      expects a positive electron current of about `+3.62e6`
    - the closure-fit diagnostics show that the remaining QA mismatch is still
      dominated by the thermal/current branch magnitude, especially on the
      electron side, not by another benchmark-family or VMEC-input bug
  - the paper-side fixed-field benchmark also had two comparison bugs on the
    archived SFINCS side:
    - archived species flows were not being loaded at all because `h5py` was
      missing from the comparison script
    - the archived `FSABFlow` channels were being compared as if they were
      already current contributions, but the physically relevant observable is
      the charge-weighted species current, so the archived benchmark now uses
      `Z_a * FSABFlow_a`
  - with that correction, the archived precise-QS SFINCS decomposition now
    reconstructs `FSABjHat` to machine precision, and the fixed-field
    `NTX+NEOPAX` mismatch is now clearly species-resolved:
    - the electron current contribution is the most obviously wrong branch,
      including the sign on QA
    - the ion contribution is also too small, but its sign is less pathological
    - this further narrows the remaining blocker to the thermal/current closure
      itself, especially the row-3 electron response, rather than the raw
      monoenergetic database handoff
  - the refreshed fixed-field current diagnostics now show that the remaining
    mismatch is amplitude-dominated rather than sign-dominated:
    - interior least-squares scale factors on the current local benchmark lane
      are about `2.56` on QA and `2.63` on QH for the total
      `NTX+NEOPAX -> SFINCS` current
    - species-wise, the electron branch is far worse than the ion branch
      (`~7.8` on QA and `~1.5` on QH for electrons, versus `~4.4` on QA and
      `~3.2` on QH for ions)
    - fitting only the raw thermal `L32` contribution in the no-momentum
      current decomposition gives a best scale of about `2.76` on QA and
      `2.64` on QH, which is strong evidence that the dominant remaining error
      sits in the thermal/current closure magnitude, not in another global
      current sign or benchmark-family mismatch
  - only after that closure is tight should the fixed-field `NTX+NEOPAX`
    current figure move into the public README or main validation claims
  - the first cached one-case QA electron `RHSMode=2` probe is now complete:
    - reusing the paper-side `ntx_scan.h5` cache reduced the NTX side to a
      negligible cost and kept the total probe within about `4m 43s`
    - the resulting raw row-3 comparison at `rho = 0.5` is not yet physically
      comparable without an explicit normalization bridge:
      - `NTX+NEOPAX` electron row 3 is about
        `[4.23e4, 7.07e4, 9.21e10]`
      - `SFINCS-JAX` row 3 is about `[2.37, 2.90e2, 1.24e3]`
    - that is far too large to be a small closure-term bug; the remaining
      `RHSMode=2` audit must therefore derive and apply the exact SFINCS-to-
      thermal-coefficient bridge before using the row-3 matrix as a parity test
  - the attempted local `Upar = base + C[2]` patch in the external NEOPAX
    checkout is now explicitly rejected:
    - it worsened the cached QA fixed-field benchmark from about `0.319` to
      about `0.619`
    - it doubled the QA electron interior sign mismatches from `12` to `26`
    - it also broke the local NEOPAX regression
      `tests/test_Fluxes_with_Momentum_Correction.py`
    - the local checkout is back on the last regression-tested `Upar =
      correction * density` semantics
  - the thermodynamic-force definitions are no longer a leading suspect:
    - the local NEOPAX `A1/A2/A3` definitions match the Escoto thesis and the
      archived monoenergetic paper exactly:
      - `A1 = d ln n / dψ - 1.5 d ln T / dψ - e_a E_ψ / T_a`
      - `A2 = d ln T / dψ`
  - the fixed-field momentum-correction diagnostic is now cache-aware and
    species-resolved in-tree:
    - it reuses the archived `ntx_scan.h5` cache instead of rebuilding the NTX
      scan for every closure probe
    - it dumps the archived species-current reference together with the
      no-momentum current, the active correction current, and candidate
      reconstructions from the solved Sonine coefficients (`c0`, weighted, and
      `c2`)
    - on the current local closure lane, that diagnostic makes the main branch
      tradeoff explicit:
      - the weighted Sonine reconstruction improves QA through cancellation
        (`electron ≈ +3.06e6`, `ion ≈ -4.24e6` at `rho = 0.5`) but still
        leaves QH too loose and breaks the shipped W7-X momentum-correction
        regression
      - the regression-consistent `c0` reconstruction keeps the local W7-X test
        passing and is therefore still the baseline, even though it leaves the
        QA electron branch wrong in sign and too small on the fixed-field
        benchmark
    - the next parity target is therefore not “weighted vs `c0`” in the
      abstract; it is the missing physics/normalization that makes the QA
      electron correction branch disagree with SFINCS while the QH branch is
      already close to target
  - the new cross-benchmark mapping audit now closes one remaining ambiguity:
    - a reusable example in `examples/momentum_correction_mapping_audit.py`
      fits and evaluates simple species-specific linear reconstructions of the
      solved Sonine vector against both:
      - the precise-QS fixed-field QA/QH species-current benchmark
      - the shipped W7-X momentum-correction regression
    - that audit rejects the “simple linear reconstruction” hypothesis:
      - weights fitted on fixed-field reduce the fixed-field species error to
        about `1.25e-1` / `3.55e-2` (electron / ion), but explode on W7-X
        (`~2.17e2` / `~9.08e1`)
      - weights fitted on W7-X improve W7-X relative to the naive branches, but
        still leave W7-X at order `1e1` and do not close fixed-field either
      - the combined least-squares fit is also poor on both families
    - therefore the remaining mismatch is not fixable by swapping `c0` for a
      different constant-weight linear combination of the solved correction
      vector; the next step must derive the missing closure term or
      normalization from the momentum-restoring equations themselves
    - a further live-closure check narrows this again:
      - the only simple universal branch rule that improves both fixed-field
        QA and QH simultaneously is `electron = weighted`, `ion = c0`
      - but that branch rule fails the shipped W7-X momentum-correction
        regression outright
      - so even the best fixed-field-only branch swap is still not a valid
        production closure
      - `A3 = e_a <E·B> / (T_a <B^2>)`
    - so the next physics target remains the thermal/current closure bridge,
      not a wholesale redefinition of `A1/A2`
  - operational cleanup:
    - interrupted profiling had left huge untracked XLA/trace trees under
      `examples/outputs/`
    - those dumps were cleaned once the useful diagnostics were extracted,
      reducing the worst output roots from about `913 MB` and `1.7 GB` down to
      about `61 MB` and `0 B`
  - user-facing bootstrap-current workflows are now in-tree:
    - `examples/bootstrap_current_with_neopax.py` provides the streamlined
      NTX scan -> NEOPAX closure -> radial `j·B` profile example
    - `examples/bootstrap_current_fixed_field_validation.py` carries the
      archive-backed fixed-field QA/QH comparison into NTX itself and writes
      the README figure artifacts under `docs/_static/`
  - the benchmark-side momentum-correction semantics are now corrected:
    - `get_Neoclassical_Fluxes_With_Momentum_Correction` already returns the
      corrected `Upar` branch, not a `ΔUpar` that should be added on top of
      the no-momentum solution
    - the fixed-field scripts now also default to the exact precise-QS profile
      family from the archived benchmark and rebuild stale scan caches that are
      missing `D33_spitzer`
    - on that corrected benchmark state, the fixed-field precise-QS current
      comparison improves materially but is still not at parity:
      - QA interior max relative error is about `1.66e-1`
      - QH interior max relative error is about `3.53e-1`
    - Redl remains close on the same archive-backed family, so the remaining
      gap is again isolated to the `NTX+NEOPAX` closure
  - the fixed-field interpolation audit is now explicit:
    - the benchmark defaults to the exact literature precise-QS profile family
      rather than reconstructing that profile from archived samples
    - the final radial remap from the 17-point `NTX+NEOPAX` grid back to the
      archived SFINCS radii keeps monotone `PCHIP` as the default
      (`NTX_FIXED_FIELD_POSTPROCESS_INTERP=pchip`)
    - switching that last remap to linear does not improve the benchmark, and
      forcing NEOPAX's generic `interpax` kernels from cubic to linear moves
      the fixed-field current negligibly
    - a direct coefficient-path audit now shows the same thing internally:
      default NTSS-style `get_Dij`, direct 3D cubic interpolation, and direct
      3D linear interpolation all reproduce the same cached QA/QH errors
    - therefore interpolation is now documented and bounded, and the remaining
      open lane is still the momentum-correction closure equations
    - a cached channel-sensitivity probe sharpens that closure result:
      perturbing `D13` away from the current bridge worsens QA/QH rapidly,
      while perturbing `D33` moves the fixed-field current comparison strongly
    - the next closure-side work should therefore focus on the `D33` /
      row-3 Sonine branch, not on further `D13` bridge or interpolation churn
  - the Sonine-output mapping audit has been rerun on the corrected semantics:
    - the baseline `c0` map is still the least-bad simple universal rule
    - weighted and fitted linear remaps do not transfer across the fixed-field
      archive and the shipped W7-X regression
    - therefore the remaining open lane is not an output remap; it is the
      momentum-correction closure equations themselves
  - two more candidate explanations are now closed:
    - switching the NTX-to-NEOPAX handoff back from `D33_spitzer` to raw `D33`
      worsens QA materially (`~1.66e-1 -> ~2.93e-1`) and leaves QH effectively
      unchanged (`~3.53e-1 -> ~3.55e-1`)
    - scaling the `Eij` `D33` Sonine sub-block by a single global factor does
      not improve both precise-QS families at once:
      - QA prefers the current baseline
      - QH only improves when that block is amplified
    - so the remaining parity blocker is no longer compatible with:
      - a raw-vs-Spitzer `D33` choice
      - a simple observable remap
      - a single missing scalar on the `D33` collision-weighted block
    - the next patch has to target the detailed `Eij` closure formulas
      themselves and still preserve the shipped W7-X regression
  - latest closure-audit result:
    - the fitted higher-order `Lij` bridge that closed the precise-QS archive is
      not defensible as production physics
    - theory/source audit:
      - `D33_spitzer` is a conductivity-side monoenergetic coefficient
      - momentum restoration in the literature is a moment-equation closure, not
        a set of benchmark-fit mixing constants on higher-order `Lij` entries
    - transfer audit:
      - on a reduced W7-X workflow using the shipped inputs, the fitted bridge
        substantially worsens the current profile relative to the same NTX-built
        baseline
    - action taken:
      - removed the fitted bridge from the shipped NTX and NTX_paper benchmark
        paths
      - restored the fixed-field figure to a status benchmark on the baseline
        closure
    - current archive-backed fixed-field baseline errors vs SFINCS:
      - QA `1.66e-1`
      - QH `3.53e-1`
    - Redl remains close on the same family:
      - QA `6.86e-2`
      - QH `4.06e-2`
    - interpolation remains bounded out of the dominant error budget on this
      benchmark
    - the active open lane is again the momentum-correction closure equations
      themselves
  - physically motivated `D33` audit result:
    - Escoto's DKES-comparison appendix implies that the conductivity-side
      coefficient should be compared through the deviation from the Spitzer
      problem rather than through raw `D33` alone
    - NTX now carries an explicit `d33_mode="conductivity_difference"` path
      for NEOPAX handoff tests, defined as `D33_spitzer - D33`
    - the momentum-correction audit now shows that this conductivity-side
      branch must enter the full higher-order row-3/4/5 hierarchy
      consistently; mixed `Lij`/`Eij` choices are numerically worse and do not
      make sense physically
    - on the regenerated precise-QS fixed-field benchmark this materially
      improves the closure without any fitted mixing constants:
      - QA improves to `1.01e-1`
      - QH improves to `2.32e-1`
    - a dedicated NTX rebuild audit for the shipped W7-X workflow is now
      in-tree in `examples/bootstrap_current_w7x_rebuild_audit.py`
    - that audit rebuilds a NEOPAX-format W7-X database with `D33_spitzer`
      and tests both `spitzer` and `conductivity_difference` against the
      frozen shipped W7-X momentum-correction reference
    - transfer currently fails badly:
      - shipped external database: `1.18e-12`
      - NTX-rebuilt W7-X, `raw`: `3.66e+0`
      - NTX-rebuilt W7-X, `spitzer`: `4.18e+0`
      - NTX-rebuilt W7-X, `conductivity_difference`: `1.07e+1`
    - one integrated-workflow bridge bug is now closed:
      - legacy external NEOPAX HDF5 files use a different `D13`
        sign convention from the NTX-generated in-memory bridge
      - the NTX bridge now preserves that historical convention when loading
        such files, so round-tripping the shipped W7-X external database no
        longer flips the bootstrap-current sign
    - but the rebuilt W7-X lane still fails upstream of the closure:
      - a reduced `13x17x17` coefficient comparison against the shipped
        external W7-X database already shows order-large monoenergetic table
        differences:
        - `D11`: `9.32e+1`
        - `D13`: `2.76e+3`
        - `D33`: `1.31e+1`
    - the correct interpretation is therefore:
      - this is the right NTX-generated fixed-field closure branch for the
        precise-QS archive
      - not a universal external-database default
      - and not yet the end of the broader closure-model lane, especially on
        W7-X integrated workflows where the NTX-generated coefficient tables
        themselves are still the first blocker
    - the current W7-X integrated result is therefore:
      - the in-repo full-resolution point and subset coefficient tests still
        pass against the shipped external database
      - but the full integrated workflow remains poor on every tested
        higher-order branch
      - and `raw` is currently the least-bad W7-X branch, though still far
        from parity
  - latest W7-X database-normalization result:
    - the actual NEOPAX database loader path uses
      - `D11 -> D11 * drds^2`
      - `D13 -> D13 * drds`
      - `D33 -> nu * D33`
    - NTX now uses that same direct database normalization in the active
      `scan_to_neopax_arrays(...)` / `to_neopax_monoenergetic(...)` path
    - a new local W7-X one-point regression locks the transformed database
      arrays at a previously bad point against the shipped external database
    - on the shipped integrated W7-X workflow this closes the rebuilt raw
      branch:
      - shipped external database: `1.18e-12`
      - NTX-rebuilt W7-X, `raw`: `6.58e-6`
      - NTX-rebuilt W7-X, `spitzer`: `5.77e-1`
      - NTX-rebuilt W7-X, `conductivity_difference`: `2.67e+0`
    - direct worst-point convergence ladders now show:
      - the direct solver and the scan builder both reproduce the shipped
        W7-X coefficient table on the reference grid `25x25x63` to about
        `1e-6` relative error at the previously worst `D11`, `D13`, and `D33`
        points
      - lower-resolution grids are under-resolved
      - and blindly increasing the grid does not reproduce the frozen
        benchmark monotonically on every point, so the W7-X audit is now
        anchored to the actual benchmark grid rather than to a naive
        monotone-refinement assumption
    - the precise-QS fixed-field picture changes under the same physical
      normalization:
      - the previous stronger agreement there depended on a non-transferable
        compensating `D13` bridge and is no longer treated as production
        physics
      - with the physically consistent database normalization, the fixed-field
        benchmark becomes a closure stress test with interior errors of about
        `1.16e+0` on both QA and QH
      - Redl remains closer on that archive (`6.86e-2` on QA and `4.06e-2`
        on QH, interior window)
    - the open parity lane is therefore now narrower and cleaner:
      - integrated W7-X database parity is closed on the raw branch
      - the remaining open lane is the precise-QS closure/model discrepancy,
        not interpolation, not the W7-X handoff, and not a stale-cache issue
  - latest closure and performance follow-up on the corrected raw branch:
    - the fixed-field cached momentum-correction diagnostic was rerun on the
      physically consistent `raw` database branch at `rho = 0.5` for both QA
      and QH
    - the resulting species-level picture is sharper than the old mixed-branch
      diagnostics:
      - QA:
        - electron no-momentum relative error is about `8.15e-1`
        - electron corrected-current relative error is about `1.16e+0`
        - ion no-momentum relative error is about `1.08e+0`
        - ion corrected-current relative error is about `1.16e+0`
      - QH:
        - electron no-momentum relative error is about `1.52e+0`
        - electron corrected-current relative error is about `1.09e+0`
        - the weighted Sonine observable would reduce that one electron point
          to about `8.19e-1`, but it still fails as a transferable rule
        - ion no-momentum relative error is about `1.16e+0`
        - ion corrected-current relative error is about `1.18e+0`
    - so the corrected fixed-field lane remains a closure stress test:
      - the active raw-branch physics does not yet recover the archived
        species currents
      - and the residual gap is still species- and branch-dependent rather
        than a simple global normalization issue
    - the integrated W7-X speed lane is now profiled on the corrected raw
      branch with a dedicated workflow profiler:
      - cached rebuilt scan preparation is negligible (`~2.9e-4 s`)
      - field/species setup costs about `1.97 s`
      - database construction costs about `2.55e-1 s`
      - first-call no-momentum closure costs about `8.64 s`
      - first-call momentum correction costs about `8.81 s`
      - steady-state no-momentum closure drops to about `2.63e-2 s`
      - steady-state momentum correction drops to about `1.58e-2 s`
      - maximum resident set is about `1.85 GB`
    - the first-call profile is dominated by XLA compilation (`~15 s` in
      `backend_compile_and_load` out of `~20 s` total Python runtime), so the
      next performance work should focus on:
      - reducing recompiles
      - stabilizing shapes and dtypes
      - hoisting compiled closure calls
      - and only then revisiting deeper kernel/vectorization work
    - two immediate follow-up probes now narrow the remaining open lanes
      further:
      - the cache-aware raw-branch QA diagnostic now dumps the explicit
        additive correction terms from the moment-equation assembly, and those
        terms are too small by orders of magnitude to explain the
        `O(10^6)` A/m$^2$ species-current mismatch on the precise-QS archive
      - a persistent JAX compilation-cache experiment on the corrected W7-X
        integrated workflow leaves first-call latency essentially unchanged
        across fresh processes (`~11.7 s` / `~12.3 s`), so the current speed
        blocker is not a missing cache toggle; it remains compile-key
        stability and staged JIT reuse
    - this sharpens the remaining physics and performance tasks:
      - fixed-field QA/QH remains a closure-model stress test because the gap
        now sits in the solved Sonine closure itself rather than in omitted
        explicit additive terms
      - integrated W7-X optimization should now target retracing and shape
        instability before any deeper kernel-side work
    - a first-principles theory pass now closes the normalization lane more
      decisively:
      - the NTX monoenergetic coefficient formulas match the upstream
        monoenergetic definitions for `D11`, `D13`, `D31`, `D33`, and
        `D33_spitzer`
      - the active NTX-to-database mapping now matches the actual consumer
        convention used downstream:
        `D11 -> D11 * drds^2`, `D13 -> D13 * drds`, `D33 -> nu * D33`
      - the Sonine-to-current map in the present momentum-restoring closure is
        also physically consistent: for the chosen basis the corrected
        parallel flow depends on the lowest solved Sonine coefficient only,
        so `Upar = n * c0` is not a bookkeeping bug
      - therefore the remaining precise-QS fixed-field gap is not a residual
        coefficient/db-normalization mistake; it is the reduced
        momentum-restoring closure model itself relative to a fuller
        collisional treatment
    - the current CI baseline was also checked directly against GitHub Actions:
      - the open `tests` failures were not from the higher-order scaffold
      - they included stale `ruff` line-length failures and one real runtime
        bug in the publication-figure pipeline
      - that runtime bug came from a non-traceable VMEC reference-bridge path
        in `src/ntx/neopax.py` (`TracerBoolConversionError` under the
        differentiable bootstrap-current optimization example)
      - the bridge now avoids Python `bool`/`int` conversions on traced JAX
        arrays, and a regression test covers the traced VMEC-surface path
    - hardening-program phase 0 is now underway:
      - CI is being switched from a static coverage claim to measured,
        shard-combined coverage artifacts
      - the testing docs now distinguish fast PR, benchmark, and hardware
        lanes explicitly
      - manuscript-figure traceability now covers the fixed-field closure
        report: the publication bundle, benchmark matrix, and generated
        manuscript artifacts all point to the same Redl gate and monitored
        `NTX+NEOPAX` stress metric
      - the next implementation step after this instrumentation is to use the
        module-wise coverage report to choose the first no-behavior-change file
        splits in `solver.py`, `profiles.py`, and `autodiff.py`
