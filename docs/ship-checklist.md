# Ship Checklist

For each NTX release, open research and software lanes should either be closed
or explicitly moved to documented future work with a clear reason.

## Blocking Lanes

| Lane | Current Status | Required Before Merge |
| --- | --- | --- |
| Code refactoring | Partly closed; continue | Public facades remain stable while solver, bootstrap-autodiff, profile dataclass, validation artifact-gate, and input/output ownership have been split into smaller internal modules; keep shrinking only with tests and docs. |
| Repository hygiene | Current worktree audited | Split the dirty worktree into reviewable commit batches; all current untracked source/artifact files are tied to scripts/tests/docs, and local caches should stay removed. |
| CI runtime and coverage | Closed; monitor | Maintain `>=95%` repository-owned coverage, module floors, and a normal CI wall time near `5-10` minutes. |
| Literature-anchored physics gates | In progress | Add or preserve fast gates for convergence, Onsager residuals, exact low-order recovery, coefficient sign/normalization, and artifact-backed literature comparisons. |
| Fixed-field `NTX+NEOPAX` closure | Scoped total-current stress gate closed | Keep the passing QA/QH total-current stress gate artifact-backed; do not claim species-current parity or promote a broader default closure unless a physics-derived model preserves this gate and the integrated W7-X raw-branch transfer. No fitted bridge constants. |
| Multi-CPU and multi-GPU algorithms | Production and strong-scaling maps artifact-backed; monitor | CPU/GPU/device-parallel/multiprocess crossover maps, fixed-workload strong-scaling maps, and prepared-geometry reuse profiles are artifact-backed; promote only algorithms that beat serial batched JAX on the target production workload. Additional healthy GPU nodes and device-memory timelines remain future work. |
| `vmec_jax` and `booz_xform_jax` integration | Closed for first release; broader sensitivities planned | Keep projected-boundary and explicit-relaxed lanes as the promoted differentiable paths; the implicit-equilibrium diagnostic is documented as non-shipping. |
| SFINCS comparisons | Partly closed | Add more artifact-backed comparisons with aligned physics settings and normalizations; distinguish promoted agreement gates from monitored stress gates. |
| Documentation | In progress | Keep docs synchronized with source layout, benchmark matrix, test lanes, performance guidance, examples, and release path. |
| Implicit-equilibrium derivative lane | Closed as non-shipping diagnostic | Do not promote this path for optimization claims; restore only after Boozer and NTX transport observables match centered finite differences, not just equilibrium volume. |
| Broader W7-X/QI/omnigenous families | Production stress artifact added | Keep the new VMEC family convergence artifact as reduced NTX stress evidence; add independent-code parity, owned W7-X KJM input coverage, and radial/electric-field ladders before promotion. |
| PyPI/release automation | Closed for `v0.2.0`; monitor | PyPI Trusted Publishing published `v0.2.0` from `release.yml`; future releases should keep the same green-CI, tag, artifact-provenance, and PyPI smoke-test path. |

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
  bridge and NTX-to-NEOPAX scan owner without rerunning expensive
  boundary/equilibrium artifacts in every shard.
- A fast owned-surface physics gate now checks `D11`, `D31`, `D33`, Onsager
  residual, and coarse-to-fine angular-grid transfer on the analytic Boozer
  surface.
- The committed monoenergetic validation-summary artifact is now a physics
  gate: the maximum finest plotted `N_\xi` convergence error over the DKES-style
  and VMEC surfaces must stay below `2.5e-1`, and the example test checks the
  plotted convergence ladder.
- A symmetric-limit physics gate now checks zero radial transport and the
  inverse-collisionality Spitzer parallel-conductivity normalization on a
  constant-field Boozer surface.
- A Boozer-coordinate normalization gate now checks
  `J B^2 = B_zeta + iota B_theta`, `B^theta J = iota`, and
  `B^zeta J = 1` on the owned analytic surface before solver coefficients are
  interpreted.
- A finite Legendre source-projection gate now checks that the magnetic-drift
  source occupies only the expected `k=0` and `k=2` rows and that the
  parallel-conductivity source occupies only the physical-`B` `k=1` row.
- A small owned-surface monoenergetic ladder now checks that
  `D11/D31/D13/D33/D33_spitzer` move toward the finest small-grid reference as
  angular and Legendre resolution are increased.
- A derivative-consistency gate now checks that the hand-coded
  `dD_k/dnu_hat` and `dD_k/depsi_hat` blocks used by the implicit-adjoint path
  match JAX differentiation of the assembled Legendre-space operator.
- A profile-control linear-response gate now checks that scalar and radial-basis
  controls preserve the zero-control identity and exactly follow their
  prescribed response matrices before profile optimization and UQ workflows use
  them.
- A VMEC-JAX boundary-edge transfer gate now checks that traced fixed-boundary
  Fourier edge arrays are passed into both implicit and explicit equilibrium
  paths before any boundary-to-output derivative claim is interpreted.
- A VMEC-JAX to NEOPAX radial-metric transfer gate checks the imported
  `rho = sqrt(s)` mesh, axis regularization, volume scale, edge-radius scale,
  and toroidal-flux normalization before bootstrap-current derivative workflows
  consume the field.
- The prepared derivative-path artifact is now a physics gate: the committed
  custom-VJP derivative path must stay within `1e-4` relative mismatch of
  direct reverse-mode, while speedup stays a reported performance metric.
- The committed derivative-geometry artifacts are now physics gates: owned
  analytic geometry controls, file-backed Boozer/VMEC controls,
  boundary-projected current derivatives, and explicit-relaxed QA/QH
  boundary-to-current derivatives must stay within their artifact thresholds.
  The implicit-equilibrium derivative artifact remains a monitored
  non-shipping diagnostic because residual contraction and Boozer-space/NTX
  transport tangent parity still fail.
- `python scripts/test_lane_manifest.py --check` passes with split core lanes,
  8 integration examples, and 18 opt-in heavy example tests.
- CI test-shard jobs are bounded by a ten-minute timeout, and subprocess-based
  parallel smoke tests have explicit subprocess timeouts to prevent silent
  workflow stalls.
- The device-parallel profiling script now exposes explicit smoke-profile
  controls, and CI uses `--num-cases 2 --grid 5,5,4` to check
  serial/device-parallel agreement without rerunning the full profiling
  workload.
- `python scripts/build_benchmark_matrix.py` reports every active benchmark
  gate complete and keeps the broader geometry/autodiff breadth lanes planned.
- `python -m ruff check .`, `python -m mypy src/ntx`, documentation build,
  manuscript-artifact build, package build, and `twine check` pass locally.
- A clean-venv wheel smoke test passes for `ntx --help`, `python -m ntx --help`,
  and importing `GridSpec`.
- Public package metadata no longer exposes Git direct references; optional
  geometry-coupled workflows document direct upstream installs until those
  packages are available from standard package indexes.
- The flat public API remains supported, and namespace-import tests now also
  require the top-level export list to stay duplicate-free.
- The repository-side PyPI Trusted Publishing job is present and tag-gated, the
  GitHub `pypi` environment exists, and PyPI has a trusted-publisher entry for
  `uwplasma/NTX` using `release.yml`. The `v0.2.0` tag release published to
  PyPI successfully on 2026-04-24.
- The fixed-field `NTX+NEOPAX` lane is explicitly scoped as a total-current
  stress gate. The release claim is the positive W7-X integrated transfer, the
  fixed-field Redl/SFINCS gate, and the fixed-field `NTX+NEOPAX` total-current
  stress result below `1e-1`; species-current parity and broader closure
  transfer remain future work.
- Manuscript claim artifacts now include the monoenergetic validation-summary
  convergence gate, the fixed-field Redl/SFINCS gate, and the scoped
  fixed-field `NTX+NEOPAX` total-current stress gate, so paper-facing claims
  match the active physics gates.
- The finite-beta QA closure lane now has explicit physics-gate separation:
  same-grid coefficient normalization is a passing gate, while the
  profile-current observable and species-cancellation scale remain monitored
  stress diagnostics until same-grid profile-current closure comparisons pass.
- The finite-beta current-conditioning sidecar now records the stricter
  coefficient precision needed by the cancellation-dominated net-current
  observable. The current smoke ladder is still looser than the `1e-1`
  current-conditioned target. The first production stress-radius rerun and
  tight-harmonic probe leave the coefficient floor near `2.05e-2`; the
  production six-point radius/collisionality ladder keeps all coefficient
  differences below `2.07e-2`. The corrected Boozer-field path evaluates
  `B00` on normalized radius and converts `dB00/d rho` with the VMEC minor
  radius. The closure-quadrature sidecar has zero accepted current-gate passes.
  The source-channel sidecar reconstructs the same corrected current to
  roundoff from one-channel solves and localizes the high-order response to
  mixed density/electric and temperature-gradient drives under the current
  profile contract. The profile-response sidecar extends that measurement over
  all committed profile radii and records the response multiplier against Redl
  collisionality and geometry drivers. The closure-target sidecar now ranks
  those drivers and records `epsilon` as the strongest single profile response
  driver without applying any runtime correction. The radial-interpolation
  sidecar rebuilds the same database on the exact field radii but the
  full-profile maximum remains above the `1e-1` current gate. The next required
  step is a quadrature-converged profile-current/source-response closure on the
  same finite-beta contract before any finite-beta parity promotion.
- The field-radius-matched source-channel sidecar now repeats the physical RHS
  decomposition after removing the sparse radial interpolation layer. It
  reconstructs the corrected current to roundoff and keeps the quadrature-stable
  `X=18, P=18` result as a reduced-closure stress diagnostic.
- The differentiable bootstrap-current optimization figure is now represented
  in the benchmark matrix and physics-gate registry as a monitored stress gate:
  the committed weighted-current gain must stay above the baseline before the
  manuscript cites that science-case number.
- The publication figure manifest now includes the file-backed
  geometry-control derivative audit, and the boundary-forward derivative audit
  is explicitly marked as a manuscript figure in the benchmark matrix.
- The geometry-family breadth summary now reads committed analytic,
  file-backed, boundary-projected, explicit-relaxed, and implicit-equilibrium
  derivative artifacts into a single publication-ready stress figure; broad
  W7-X/QI/omnigenous validation remains scoped as future work.
- The fixed-field closure report is now included in the publication figure
  manifest and traced back to the fixed-field Redl gate plus the scoped
  `NTX+NEOPAX` total-current stress gate in the benchmark matrix.
- The source-code map is now guarded by a focused test, so future internal
  ownership splits must update the architecture documentation and CI lane
  manifest in the same change.
- The current refactoring pass split bootstrap-current autodiff workflows into
  common, deterministic, and robust modules, and split profile dataclasses into
  species, ambipolar-result, control, and transport-result ownership modules
  while preserving the flat public API and compatibility facades.
- The current source-ownership pass also split artifact-gate scalar evaluation
  from finite-beta artifact-gate ownership, finite-beta artifact-gate
  definitions from the general artifact registry, finite-beta geometry-breadth
  metadata from the general geometry benchmark matrix, and TOML-run
  orchestration from suffix-selected NetCDF/NPZ/HDF5 output writing while
  preserving the public `ntx.inputfiles` and top-level exports.
- The current repo-hygiene pass verified every untracked source/artifact file
  against docs, tests, or benchmark metadata and removed local cache
  directories.
- The next valuable coverage work should be opportunistic and physics-driven;
  coverage is no longer a blocking lane.
- The expensive boundary/equilibrium artifact reruns remain opt-in through
  `NTX_RUN_HEAVY_BOUNDARY_EXAMPLES=1`.
- The prepared-geometry reuse profile is now a performance software gate: the
  compiled prepared steady path must remain coefficient-consistent with direct
  solves, while speedup stays a measured performance claim.

## Immediate Next Order

1. Keep the CI lane manifest and benchmark matrix locked as new tests are added.
2. Expand owned geometry-family benchmark artifacts only from committed
   scripts/tests/docs.
3. Keep release documentation synchronized with the published package state and
   require green `tests`, `package`, and `release` workflows before the next tag.
