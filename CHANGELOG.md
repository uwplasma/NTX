# Changelog

## Unreleased

- condensed the README into a decision-oriented quickstart, corrected the
  normalized drift-kinetic equation on the documentation entry page, added a
  generated public API reference and normalization glossary, and separated the
  release checklist from the authoritative research plan
- added a reusable angular-oversampling audit with coefficient error, compile
  and warm-runtime timing, and XLA temporary-memory reporting; the committed
  finite-beta QA/NCSX/HSX artifact supports a warning-only `2.25`-times-Nyquist
  starting-grid recommendation while preserving successive-grid acceptance
- exposed the production algebraic diagnostic as
  `TransportResult.schur_residual_l2` while preserving `residual_l2` as a
  compatibility alias, and added independent full-system and dense-operator
  residual oracles through `N_xi=63`
- refreshed geometry-family discovery for current and legacy `vmec_jax`
  layouts, added the finite-beta NFP2 QA example, and retained the near-zero-
  transform vacuum QA input as a visible diagnostic-only case
- added a public prepared-derivative audit that requires independent full
  primal and transpose residual convergence and compares direct reverse mode,
  forward mode, the factor-reusing adjoint, and centered finite differences
- measured selective recomputation against saved-factor adjoints; the committed
  CPU artifact keeps rematerialization disabled because it does not reduce
  temporary memory for the prepared block solve
- added opt-in SOLVAX chunked Jacobians to profile sensitivity and uncertainty
  workflows, preserving native JAX reverse mode as the small-control default
- added reusable fixed-bucket prepared scan solvers with CPU-sequential and
  accelerator-vectorized execution, explicit warmup timing, and executable
  memory diagnostics
- replaced per-call scan JIT factories with reusable module-level kernels and
  synchronized CPU/GPU benchmark timing
- added optional persistent JAX compilation-cache configuration and cache-miss
  diagnostics
- limited branch CI to pull-request runs while retaining post-merge `main`
  validation, avoiding duplicate workflow execution
- moved generated truncated solves, reusable Schur factors, primal factor reuse,
  exact transpose reuse, and tail-aware residuals to `solvax>=0.7.3,<0.8`
  while retaining all physics
  assembly and observable definitions in NTX
- corrected the production residual to include the reconstructed mode-3
  coupling and added an opt-in full Legendre residual audit
- added strict Fourier geometry sampling reports and adaptive two-successive-step
  angular/Legendre convergence gates
- migrated WOUT loading to the current root-level `vmec_jax.read_wout` API
- replaced removed WOUT-to-state reconstruction with separate finalized-WOUT
  and traceable `SpectralState`/`SolverRuntime` Boozer paths
- migrated fixed-boundary reverse-mode workflows to
  `vmec_jax.implicit.solve_implicit`
- restored physical VMEC radial-flux normalization in the direct WOUT builder
- made the memory-intensive boundary-to-current reverse-mode integration gate
  opt-in with `NTX_RUN_BOUNDARY_AUTODIFF=1`

## 0.2.4

Repository and package-size hardening after the `0.2.3` output release.

Highlights:

- rewrote repository history to remove old NetCDF fixture blobs larger than
  `2 MiB` and repeated generated `docs/_static` artifact history
- restored the current documentation artifacts once on top of the rewritten
  history so README/docs figures remain available without carrying repeated
  generated-asset history
- added CI guardrails that reject tracked files larger than `2 MiB`, cap the
  tracked tree size, and cap committed docs artifacts
- excluded `docs/_static` from built distributions so PyPI wheel/sdist
  downloads stay focused on installable source code
- documented clone-reset guidance for collaborators after the public history
  rewrite

## 0.2.3

Highlights:

- TOML/CLI runs now write NetCDF by default and can select NetCDF, NPZ, or
  HDF5 by output filename
- `ntx input.toml --plot` writes a PDF summary panel for the saved payload
- file-backed runs reuse one prepared geometry/operator system for reporting,
  solving, and output writing
- NEOPAX HDF5 scan writing now uses direct uncompressed datasets with HDF5
  timestamps disabled
- docs now introduce the drift-kinetic equation before the Legendre block
  system and document the new output/plotting workflow

## 0.2.2

CI and release hardening after the `0.2.1` validation release.

Highlights:

- robust bootstrap-current optimization artifacts now separate robust-objective
  improvement from the signed weighted-current ratio
- the robust example smoke test now checks artifact/schema and finite
  uncertainty metrics instead of requiring improvement from a two-step toy run
- refreshed the robust-design figure artifact and JSON metadata for the docs

## 0.2.1

Validation, documentation, and performance-artifact hardening after the first
PyPI release.

Highlights:

- fixed-field QA/QH bootstrap-current validation figure is now an overlay-only
  SFINCS, Redl, and `NTX+NEOPAX` current comparison, with the `<1e-1`
  interior-window gates retained in JSON and tests
- `examples/bootstrap_current_with_neopax.py` keeps the corrected
  no-momentum/default current assembly and records the selected `D33` branch in
  the generated summary
- fixed-field closure diagnostics, benchmark matrix, manuscript artifacts, and
  physics-gate docs were refreshed from committed artifacts
- solver, profile, and bootstrap-autodiff internals were split into smaller
  ownership modules without changing public APIs
- production and fixed-workload strong-scaling performance artifacts were added
  for CPU/GPU and multiprocess/device-parallel lanes
- README and example documentation were tightened around the promoted claims,
  monitored stress gates, and remaining research lanes

## 0.2.0

Release-candidate update for the first PyPI publication.

Highlights:

- artifact-backed W7-X integrated transfer and fixed-field Redl/SFINCS release
  gates
- fixed-field `NTX+NEOPAX` retained as a reduced-closure stress diagnostic,
  not a parity claim
- geometry-family `D11/D31/D33` convergence stress figure across local public
  VMEC example families
- prepared-geometry and compiled-solver reuse performance profile for repeated
  fixed-geometry scans
- explicit-relaxed `vmec_jax -> booz_xform_jax -> NTX` derivative stress lane
  retained as the promoted differentiable equilibrium path
- implicit-equilibrium derivative path closed as a non-shipping diagnostic
- refreshed benchmark matrix, physics-gate registry, manuscript artifact
  bundle, and publication figure manifest
- split-lane CI, package workflow, docs build, wheel/sdist smoke checks, and
  tag-gated release workflow

## 0.1.0

Initial NTX release for research use.

Highlights:

- Escoto-style monoenergetic Legendre-space solver in JAX
- DKES and VMEC surface support
- direct `ntx input.toml` execution path with Rich terminal output
- `.npz` output bundles with coefficients, metadata, diagnostics, and modes
- imported differentiable solve and scan APIs
- NEOPAX mapping helpers and reference-database workflows
- artifact-backed monoenergetic, fixed-field Redl/SFINCS, W7-X transfer,
  closure-stress, geometry-control, boundary-derivative, and performance
  validation reports
- maintained benchmark matrix, physics-gate registry, manuscript artifact
  bundle, and publication figure manifest
- geometry-family breadth summary that keeps unresolved implicit and broader
  W7-X/QI/omnigenous lanes scoped as future work
- CPU CI, docs CI, and packaging/release validation workflows
