# NTX Authoritative Plan

Last reviewed: 2026-07-12

This is the single authoritative implementation plan for NTX and its numerical
dependency SOLVAX. Historical work belongs in git history, release notes, and
generated benchmark artifacts, not in this file. When priorities or acceptance
criteria change, update this file in the same pull request.

## Mission And Scope

NTX will be a small, research-grade, JAX-native solver for the radially local,
monoenergetic drift-kinetic equation on stellarator flux surfaces. It will:

- compute converged `D11`, `D31`, `D13`, `D33`, and `D33_spitzer` coefficients;
- support file-backed and in-memory VMEC/Boozer geometry;
- provide bounded-memory CPU/GPU scans and prepared repeated solves;
- provide verified derivatives for sensitivity, UQ, inverse design, and
  optimization;
- export owned, normalization-explicit data for NEOPAX and other profile tools;
- make every promoted physics claim traceable from literature to code, tests,
  artifacts, figures, and documentation.

NTX is not a full multi-species, full-speed Fokker-Planck solver. Full
linearized collisions, tangential magnetic drifts, finite-orbit-width physics,
surface-potential variation, and radially global physics remain comparison or
research lanes unless a separate model-expansion proposal is accepted.

SOLVAX owns physics-independent numerical algorithms. NTX owns drift-kinetic
equations, geometry conventions, sources, nullspaces, normalizations,
observables, convergence policy, and physics validation.

## Rules For Development

1. Never commit implementation work directly to `main`; use reviewed PRs.
2. Preserve the public NTX API unless a deprecation cycle is documented.
3. Do not introduce fitted bridge constants or case-specific thresholds into
   runtime physics.
4. Separate numerical convergence, normalization parity, and model discrepancy.
5. A promoted claim requires:
   `reference -> equation/contract -> script -> test -> artifact -> docs figure`.
6. Fast CI contains deterministic unit and small physics tests. Production
   benchmarks run manually or on scheduled/dedicated hardware and are checked
   in CI through artifact schemas and thresholds.
7. Add comments for invariants and non-obvious numerical decisions, not for
   line-by-line narration. Every exported function and result type gets a
   useful docstring with units, shapes, assumptions, and failure behavior.
8. External source code is for audit and independent reimplementation. Do not
   copy restrictively licensed source into the MIT-licensed NTX repository.

## Validated Baseline

The current baseline is NTX `0.2.4` at commit `02fa393` and SOLVAX `0.7.0` at
commit `f827cc8`.

Already established:

- Fourier collocation in the two surface angles and a Legendre pitch-angle
  expansion;
- block-tridiagonal elimination over Legendre mode;
- low-mode source and observable truncation with the high-order tail eliminated;
- analytic, DKES-style, VMEC, and Boozer geometry paths;
- NetCDF, NPZ, and HDF5 output plus PDF plotting;
- prepared solves, scans, custom-VJP coefficient derivatives, and finite-
  difference derivative audits;
- NEOPAX-compatible scans and scoped bootstrap-current workflows;
- CI lane manifest, package/release workflows, and repository-owned coverage
  above 95%;
- finite-beta same-grid coefficient comparisons and a closed `1.32e-1`
  reduced-closure stress result. This is not a full-physics parity claim.

Measured during the 2026-07 solver audit:

- SOLVAX reproduces NTX low-mode solutions within `2e-14` to `2e-13` on
  analytic, finite-beta W7-X, finite-beta QH, and NCSX geometries;
- the current generated-block SOLVAX path lowers executable temporary memory
  by about 6%, but is 5-12% slower on representative production CPU grids
  because low blocks are assembled more than once;
- at `25 x 31 x 24`, NTX uses about `73.9 MiB` executable temporary memory and
  runs in about `0.196 s` warm on the audited CPU;
- at `25 x 25 x 63`, NTX uses about `48.1 MiB` and runs in about `0.280 s`;
- vmapped `17 x 25 x 16` CPU scans use about `22 MiB`, `99 MiB`, and `364 MiB`
  at batch sizes 1, 8, and 32, with nearly linear runtime growth;
- the `17 x 25 x 16` custom VJP compiles in about `1.5 s`, runs in about
  `44 ms`, and uses about `69 MiB` of executable temporary storage;
- repeated public scan calls spend substantial time rebuilding prepared
  geometry and derivative matrices.

These values are machine-specific evidence, not permanent performance gates.

## Literature And Reference-Code Contract

### Monoenergetic spectral reference

The Escoto et al. formulation and implementation establish the principal model
and algorithmic reference:

- the Legendre projection couples only neighboring pitch modes;
- source modes vanish above order 2, while the eliminated high-order tail still
  changes the low-order Schur complements;
- convergence must be scanned independently in `N_theta`, `N_zeta`, and `N_xi`;
- low-collisionality `D31` can require Legendre references as high as
  `N_xi=380`;
- near-zero QI current needs both relative and physically meaningful absolute
  convergence criteria;
- W7-X EIM, W7-X KJM, and QI cases at zero and finite radial electric field are
  core geometry/regime benchmarks;
- `D31 = -D13` is a required Onsager diagnostic under the aligned convention.

NTX will reproduce these convergence methods and owned benchmark families. It
will not assume that `N_xi=32`, 63, or any other fixed order is universally
converged.

### Higher-fidelity drift-kinetic reference

SFINCS supplies the model-hierarchy and independent-solver reference:

- pitch-angle scattering and full linearized momentum-conserving collision
  operators are distinct contracts;
- reduced trajectories, full trajectories, magnetic drifts, radial electric
  field, and `Phi1` options are distinct physics models;
- the distribution depends on speed as well as pitch angle and surface angles;
- multi-species flows, fluxes, bootstrap current, density variation, and the
  distribution function provide independent observables;
- PETSc KSP/preconditioner diagnostics, direct sparse factorization choices,
  convergence scans, radial scans, HDF5 output, and explicit memory estimates
  are useful software patterns.

NTX comparisons must first align the monoenergetic equation, trajectory model,
collision operator, electric field, geometry, normalization, interpolation,
resolution, and observable. Full-physics runs are discrepancy bounds, not
parity failures.

### Numerical literature

The implementation follows established numerical requirements:

- block elimination is the direct reference while the structure is exact;
- Schur-complement growth and true residuals must be monitored because block LU
  is not unconditionally stable for general nonsymmetric systems;
- FGMRES is the admissible Krylov method for the nonsymmetric streaming
  operator; PCG is not admissible for the full NTX operator;
- iterative refinement is accepted only when working-precision residuals
  contract and coefficient/gradient accuracy is recovered;
- Fourier sampling must resolve retained geometry harmonics, and variable-
  coefficient products require oversampling or explicit aliasing evidence;
- implicit derivatives are valid only for a converged primal and converged
  transpose/tangent solve.

## Current Correctness Findings

These are the first implementation blockers:

- [ ] Replace `TransportResult.residual_l2`. The current implementation checks
  only modes 0-2 and omits the eliminated-tail coupling. It reports order
  `1e-2` for solutions whose full-system residual is order `1e-15`.
- [ ] Distinguish a cheap reduced-Schur residual from an optional full-system
  residual. Never label the reduced diagnostic as the full DKE residual.
- [ ] Add direct tests for both residual definitions, including `N_xi=2`, 16,
  32, and 63.
- [ ] Enforce a geometry-spectrum sampling floor before solving:
  `N_theta >= 2*m_max+1` and `N_zeta >= 2*n_max+1`, with documented reduced
  toroidal mode convention.
- [ ] Add an oversampling/dealiasing study for the variable-coefficient
  collocation operator and choose a policy from evidence.
- [ ] Add two-step convergence checks. One coarse/fine comparison is not
  sufficient because spatial and Legendre convergence can be nonmonotonic.

Observed examples motivating these gates:

- W7-X at `nu_hat=1e-4` still changes by 3-9% between `N_xi=48` and 63;
- NCSX contains `m_max=10`; `N_theta=17` violates Nyquist and produced a D31
  error near 80%, while `N_theta>=21` removed that failure;
- finite-beta QH spatial convergence is nonmonotonic across the audited ladder.

## Ordered Pull Requests

The order below is mandatory unless this file is revised with a reason.

### PR 0: Plan Consolidation (NTX)

Status: in progress.

- [x] Replace the accumulated historical plan with this authoritative plan.
- [ ] Link this file from contributor/development documentation.
- [ ] Mark `docs/research-roadmap.md` as a user-facing summary generated from,
  or subordinate to, this plan.
- [ ] Close or supersede stale NTX PR #3 after its useful changes are recovered.

Acceptance:

- one authoritative plan;
- no competing implementation order in README, ship checklist, research
  roadmap, or benchmark matrix;
- documentation build and link check pass.

### PR 1: SOLVAX Foundation For NTX

Repository: `uwplasma/SOLVAX`.

- [ ] Ensure `pytest` always imports `src/solvax`, not a globally installed
  older release.
- [ ] Remove unused `lineax`, or actually use and document it. The current
  README says SOLVAX builds on the Lineax operator interface, but the source
  does not import Lineax.
- [ ] Add a fused generated-block truncated solver that:
  - assembles each required block no more than once per solve;
  - accepts multiple right-hand sides;
  - supports `keep_lowest == n_blocks` and the NTX `N_xi=2` boundary;
  - supports factor reuse and transposed solves;
  - keeps tail storage independent of total Legendre order;
  - remains compatible with `jit`, `vmap`, `grad`, float32, and float64.
- [ ] Add optional diagnostics for true residual action, Schur-complement growth,
  refinement history, convergence, and breakdown.
- [ ] Add NTX-shaped tests at block sizes and orders representative of
  `13 x 15 x 32`, `17 x 25 x 32`, and `25 x 25 x 63` without importing NTX.
- [ ] Benchmark compile time, warm runtime, temporary memory, and accuracy on
  CPU and GPU.

SOLVAX documentation changes:

- [ ] Correct the Lineax relationship in README and docs.
- [ ] Put matrix assumptions before solver selection, especially PCG.
- [ ] State the `custom_linear_solve` convergence invariant prominently.
- [ ] Add a generated-block kinetic example and an example index with expected
  runtime, device suitability, output, and mathematical assumptions.
- [ ] Add reproducible CPU/GPU benchmark tables and versioned JSON results.
- [ ] Document callback assembly count, factor lifetime, transpose convention,
  complex adjoint convention, and mixed-precision failure behavior.
- [ ] Fix minor documentation defects, including duplicated arguments and
  claims that exceed the implemented algorithmic variant.

Acceptance:

- all SOLVAX tests pass against the checkout;
- coverage remains at least 95%;
- generated-block parity is `<=1e-11` float64 and appropriately gated float32;
- no production-grid memory regression relative to NTX;
- warm CPU runtime is no worse than NTX by more than 3%, and at least one
  accelerator or large-order lane shows a measured benefit;
- wheel and dependency footprint are measured and documented.

### PR 2: NTX Residuals, Resolution, And Convergence

Repository: `uwplasma/NTX`.

- [ ] Fix residual semantics and tests before changing solver ownership.
- [ ] Add `GeometryResolutionReport` containing retained harmonic extrema,
  Nyquist floor, selected grid, oversampling ratio, and warnings/errors.
- [ ] Add an adaptive convergence API that refines angular and Legendre orders
  independently and records every coefficient at every step.
- [ ] Use both relative and scale-aware absolute tolerances for coefficients
  close to zero, especially QI/QH D31.
- [ ] Require two successive accepted refinements for research-grade status.
- [ ] Return `converged`, `unresolved`, or `model-out-of-scope`; do not silently
  promote the last available grid.
- [ ] Add small analytic gates and artifact-backed high-resolution ladders.

Default internal numerical target:

- coefficient changes `<=1e-2` for `D11`, `D31`, and `D33` on two successive
  refinements;
- tighter tolerances may be required by cancellation-conditioned downstream
  current observables;
- independent-code model differences use separate documented tolerances.

Acceptance:

- full residual agrees with a materialized small-system oracle;
- undersampled geometry fails clearly before solve;
- adaptive ladders reproduce known low- and high-collisionality behavior;
- no existing public solve result changes beyond corrected diagnostics.

### PR 3: NTX Uses SOLVAX

- [x] Add a bounded dependency such as `solvax>=0.7,<0.8`, adjusted to the
  actual release containing the fused API.
- [x] Replace generic factorization, truncated solve, factor reuse, transpose
  solve, and refinement code with SOLVAX calls.
- [x] Keep operator assembly, nullspace row, source modes, parameter derivatives,
  and transport moments in NTX.
- [x] Reduce `_solver_factorization.py` to a thin physics adapter.
- [x] Preserve all public result pytrees and signatures.
- [x] Supersede rather than merge the current stale PR #3 unchanged.

Merged as NTX PR #8 against SOLVAX `0.7.3`. Float64 coefficient audits cover
`N_xi=2,16,32,63,140`; CPU and GPU memory/runtime results are recorded in
`docs/performance.md`. The stale implementation PR was closed as superseded.

Acceptance:

- coefficient and mode parity on analytic, W7-X, QH, NCSX, QI, and tokamak
  cases;
- direct, JIT, vmap, custom-VJP, and finite-difference derivative parity;
- `N_xi=2`, 16, 32, 63, and at least one `N_xi>=140` artifact;
- no CPU/GPU memory regression and no material runtime regression;
- NTX wheel remains small and installation uses PyPI only.

### PR 4: Prepared Scans, JIT, Runtime, And Memory

- [x] Add `compile_prepared_scan_solver(...)` and a reusable result object.
- [x] Reuse geometry, derivative matrices, compiled executables, and fixed batch
  shapes across calls.
- [x] Do not create new jitted function objects in hot public APIs.
- [x] Use bounded sequential `lax.map` or very small chunks on CPU unless a
  measured crossover justifies vmap.
- [x] Tune bounded vmap/device shards on GPU from memory and throughput maps.
- [x] Standardize a small set of batch buckets and pad the final batch.
- [x] Add optional persistent compilation cache configuration, cache-miss
  diagnostics, and ahead-of-time warmup without making cache state necessary
  for acceptable runtime.
- [ ] Report preparation, tracing/lowering, compilation, first execution, warm
  execution, peak RSS, executable temporary memory, and device memory
  separately.
- [x] Evaluate buffer donation only where an input can legally back an output.
- [x] Measure custom-VJP saved-state memory before using rematerialization;
  retain saved-factor adjoints because whole-solve rematerialization did not
  reduce temporary memory on the committed workload.
- [ ] Keep full XLA/Perfetto/XProf capture opt-in and targeted.

The synchronized CPU and two-GPU maps are recorded in `docs/performance.md`.
Sequential bucket `8` is the parity-preserving default on every backend.
Explicit GPU vectorization is faster at production scan widths but remains a
non-default research mode because JAX `0.6.2` batched factorization misses the
`1e-10` coefficient gate by up to `3.1e-7` on the larger low-collisionality
grid. The default path is bitwise identical to the compiled scalar reference.

Required maps:

- CPU: scan sizes 1, 8, 32, 128 and production grids;
- GPU: batch-size/memory crossover on at least one healthy GPU, then both office
  GPUs if stable;
- grids: `17 x 25 x 16`, `25 x 31 x 24`, `25 x 25 x 63`, and one larger
  low-collisionality case;
- compare eager, prepared, compiled prepared, bounded scan, vmap, and device
  sharding.

Acceptance:

- repeated fixed-geometry calls do not repeat geometry preparation;
- CPU default stays within a documented memory budget;
- GPU OOM produces actionable guidance rather than process failure where
  feasible;
- coefficient parity is maintained at `<=1e-10` float64;
- CLI displays preparation, compile, solve, and output timings separately.

### PR 5: Autodiff And Implicit Solves

- [x] Reuse SOLVAX transpose factors in the prepared custom VJP.
- [x] Compare storing full factors/state against selective recomputation.
- [x] Use SOLVAX chunked Jacobian utilities for large geometry/profile control
  sets where they reduce measured memory.
- [x] Require primal and transpose residual convergence before returning an
  implicit derivative as valid.
- [x] Add derivative status and residual metadata to result artifacts.
- [x] Keep direct AD, forward mode, prepared adjoint, and centered finite
  differences side by side.

Acceptance:

- scalar and multi-control gradients pass centered finite differences;
- gradients are stable under resolution refinement;
- derivative memory and runtime are reported separately from primal work;
- no derivative claim is based on an unconverged nonlinear or linear solve.

### PR 6: Reference Benchmark Matrix

Adopt a model-aligned benchmark hierarchy.

Tier A, exact/analytic:

- [ ] uniform-field zero radial transport;
- [ ] Spitzer branch and inverse-collisionality scaling;
- [ ] Fourier derivative and quadrature identities;
- [ ] Boozer coordinate/Jacobian identities;
- [ ] source Legendre support, nullspace row, block coupling, and Onsager
  convention;
- [ ] dense small-system and full-residual oracle.

Tier B, same monoenergetic physics:

- [ ] W7-X EIM and KJM, QI, QA, QH, NCSX/HSX, tokamak, and generic stellarator;
- [ ] zero and finite radial electric field;
- [ ] collisionality ladders spanning plateau, `1/nu`, and low-collisionality
  transition behavior within model validity;
- [ ] independent `N_theta`, `N_zeta`, and `N_xi` convergence;
- [ ] D11, D31, D13, and D33, not D11 alone;
- [ ] relative and absolute D31 criteria for low-current configurations;
- [ ] selected distribution/moment comparisons, not only final coefficients.

Tier C, SFINCS aligned reduced physics:

- [ ] same VMEC surface, radial coordinate, electric field, pitch-angle
  scattering, reduced trajectory, source, and normalization;
- [ ] owned generation scripts and raw outputs;
- [ ] coefficient and moment comparison before bootstrap closure;
- [ ] interpolation and radial-grid audit.

Tier D, deliberate model discrepancy:

- [ ] full linearized collisions versus reduced momentum restoration;
- [ ] full versus reduced trajectories and magnetic drifts;
- [ ] `Phi1` and multi-species effects;
- [ ] document sign, ordering, asymptotic trend, and discrepancy envelope rather
  than demand numerical parity.

Tier E, integrated workflows:

- [ ] Redl, NTX+NEOPAX, and SFINCS use the same geometry, profiles, radii,
  interpolation, electric field, and current normalization;
- [ ] species currents and cancellation conditioning remain visible;
- [ ] W7-X transfer and finite-beta QA/QH stress gates must not regress.

Every production benchmark stores input provenance, software versions, commit
SHAs, hardware, dtype, geometry spectrum, grid, solver status, timings, memory,
and tolerances.

### PR 7: Fixed-Field And Profile Closure

- [ ] Keep the accepted `1.32e-1` result closed as a scoped reduced-closure
  stress benchmark.
- [ ] Preserve external-dataset regressions, but prefer owned same-contract
  datasets for new claims.
- [ ] Add species-resolved current, source-channel, interpolation, velocity-
  quadrature, and cancellation diagnostics to reusable APIs rather than
  thousand-line scripts.
- [ ] No fitted correction enters runtime.
- [ ] Any closure change must derive from documented moment equations and pass
  fixed-field QA/QH plus integrated W7-X simultaneously.

Acceptance for a broader default closure:

- coefficient normalization/convergence passes first;
- profile-current maximum relative discrepancy is `<=1e-1` on owned QA/QH;
- integrated W7-X does not regress;
- species-current behavior and limiting cases are physically consistent;
- derivation, implementation, tests, and figures are complete.

### PR 8: User Experience, README, Docs, And Examples

README target: concise entry page, approximately 150-220 lines.

- [ ] Keep: one-sentence purpose, `pip install ntx`, one CLI quickstart, one
  Python quickstart, outputs, capability/scope table, two highest-ROI figures,
  current validation statement, and links to docs.
- [ ] Remove from README: long closure history, detailed stress-radius numbers,
  stale-command warnings, full research-lane lists, and large command catalogs.
- [ ] Fix the displayed DKE in `docs/index.md`; the current rendered expression
  is missing plus signs between operator terms.
- [ ] Clearly distinguish “solves directly”, “provided by downstream closure”,
  “validated comparison”, and “planned research”.
- [ ] Add a short “choose your workflow” table: single coefficient, scan,
  VMEC/Boozer input, NEOPAX export, bootstrap profile, autodiff, validation, and
  performance profiling.

Adopt SOLVAX's decision-oriented documentation pattern:

1. Getting started and installation.
2. Choosing a workflow and resolution.
3. Physics model, assumptions, and validity boundaries.
4. Numerical method, convergence, residuals, and failure modes.
5. Geometry and normalization contracts.
6. Scans, outputs, and plotting.
7. Autodiff and optimization.
8. NEOPAX and external integrations.
9. Validation and benchmark matrix.
10. Performance and hardware guidance.
11. API reference and contributor architecture.

Documentation cleanup:

- [ ] Make this file authoritative; reduce `docs/research-roadmap.md` to a
  readable summary linked here.
- [ ] Make `docs/ship-checklist.md` release-only, not another roadmap.
- [ ] Consolidate repeated finite-beta narrative into one validation case study
  plus machine-readable artifacts.
- [ ] Correct duplicated example numbering and stale file references.
- [ ] Generate API documentation from docstrings.
- [ ] Add cross-links from equations to source and tests.
- [ ] Add a glossary for coordinates, normalizations, coefficients, grids, and
  closure terminology.
- [ ] State expected runtime/memory and optional dependencies for every tutorial.

Example cleanup:

- [ ] Reclassify the current 61 scripts and roughly 26,600 lines into:
  `examples/quickstart`, `examples/workflows`, `benchmarks`, and
  `validation/cases`.
- [ ] Keep user-facing examples short, ideally below 200 lines.
- [ ] Move reusable calculation, parsing, plotting, and artifact logic into
  tested package or validation modules.
- [ ] Replace thousand-line scripts with thin CLIs over reusable functions.
- [ ] Add an example manifest with category, runtime class, optional packages,
  hardware, inputs, outputs, owning tests, artifact, and documentation page.
- [ ] Provide one canonical example each for CLI, Python solve, scan, adaptive
  convergence, VMEC/Boozer geometry, NEOPAX export, bootstrap profile,
  derivative audit, UQ, and optimization.

Acceptance:

- a new user reaches a plotted result from installation in under five minutes;
- README and docs contain no contradictory status claims;
- all commands are tested or artifact-backed;
- docs build with warnings treated as errors;
- user examples are readable without understanding validation internals.

### PR 9: Source Ownership, Dependencies, And Maintainability

Target ownership:

- `ntx.solver`: public preparation, single solve, compiled solve, scan;
- `ntx.operators`: drift-kinetic block/source assembly only;
- `ntx.geometry`: surface types and evaluation;
- `ntx.io`: stable input/output contracts;
- `ntx.neopax`: downstream data bridge;
- `ntx.profiles`: profile/current workflows;
- `ntx.autodiff`: derivative-facing public workflows;
- `ntx.validation`: benchmark contracts and artifact schemas;
- SOLVAX: all generic direct, iterative, refinement, fixed-point, operator,
  Jacobian-chunking, and implicit-solve algorithms.

- [ ] Keep flat compatibility exports while documenting owned namespaces.
- [ ] Remove unused NTX runtime dependencies after clean-wheel testing:
  `scipy` moves to dev/examples, `typing-extensions` is removed if unused, and
  direct `jaxlib` is removed if `jax` provides the correct installation
  contract. Keep `netCDF4` as a supported runtime output/input dependency.
- [ ] Measure wheel and clean-environment install size in CI. Current wheels are
  about 153 KiB for NTX and 53 KiB for SOLVAX.
- [ ] Document all exported functions/classes. The audit found 70 undocumented
  top-level public definitions; internal callbacks do not need ceremonial
  docstrings.
- [ ] Require docstrings to include units, array shapes, coordinate convention,
  differentiability, static arguments, and raised errors where relevant.
- [ ] Keep modules below roughly 400 lines unless cohesion justifies more.

Acceptance:

- no duplicate generic numerical implementation remains in NTX;
- import time and clean install size do not regress materially;
- API and source-map tests pass;
- no circular ownership or undocumented public symbol remains.

### PR 10: Release Hardening

- [ ] Full lint, typing, tests, docs, benchmark-artifact validation, package
  build, twine check, and clean-wheel smoke install.
- [ ] CI remains normally within 5-10 minutes; production benchmarks stay out of
  normal PR jobs.
- [ ] Coverage remains at least 95%, with physics value prioritized over
  low-value line coverage.
- [ ] Repository-size guard rejects files over 2 MiB unless explicitly owned.
- [ ] Release notes distinguish numerical changes, physics changes, API changes,
  performance changes, and remaining non-shipping lanes.
- [ ] Tag, GitHub release, PyPI trusted publishing, and post-publish install
  smoke test only after all active gates pass.

## Non-Blocking Research Lanes

These remain visible but do not block the next NTX release.

### Physics expansion

- implicit-equilibrium derivatives through converged VMEC and Boozer transforms;
- tangential magnetic drifts and near-omnigenous low-collisionality layers;
- `Phi1` and surface-potential variation;
- full-speed and full linearized collision models;
- finite-orbit-width and radially global formulations;
- nonlinear and multi-species kinetic extensions.

Each requires a model proposal, literature derivation, ownership decision, and
new validation matrix. Do not grow the existing monoenergetic core by accident.

### Geometry and design breadth

- owned W7-X KJM, EIM, QI, QA, QH, HSX, NCSX, LHD, tokamak, omnigenous,
  hidden-symmetry, and piecewise-omnigenous families;
- radial, electric-field, collisionality, and resolution ladders;
- robust profile/UQ/design examples;
- boundary-to-bootstrap optimization with verified derivatives.

### Iterative and distributed numerics

- matrix-free FGMRES with coarse angular or reduced-physics direct
  preconditioning;
- p-multigrid and line smoothers;
- Krylov recycling across smooth collisionality/electric-field continuation;
- mixed-precision block factorization with refinement;
- distributed multi-GPU and multi-node database generation.

Promotion requires bounded iteration counts under refinement, true residual
convergence, coefficient/gradient parity, and demonstrated memory/runtime
benefit. Generic block Jacobi, unpreconditioned FGMRES, PCG, and implicit
midpoint are not production candidates based on current mathematics and tests.

## Test And Artifact Policy

Fast tests:

- equations, signs, units, source support, nullspace, block structure;
- small dense oracle and residuals;
- geometry spectrum and sampling floor;
- JIT/vmap/grad behavior;
- prepared and direct parity;
- finite-difference derivative checks;
- I/O schemas and CLI smoke tests.

Artifact-backed tests:

- production resolution and collisionality ladders;
- independent-code comparisons;
- finite-beta profile/current closure;
- CPU/GPU performance and memory;
- equilibrium and boundary derivatives;
- publication figures.

Artifacts must be small, owned, reproducible, schema-versioned, and linked to
their generating script and source inputs. Raw large outputs live in an
external archival dataset with checksums, not git.

## Completion Definition

NTX is ready for the next research-grade release when:

- all active PRs above their release boundary are merged;
- residuals and convergence status are truthful;
- SOLVAX owns the generic solver without performance or memory regression;
- adaptive resolution gates pass the owned geometry matrix;
- promoted same-physics comparisons pass and model discrepancies are scoped;
- prepared scans and derivatives meet CPU/GPU memory and runtime targets;
- README, docs, examples, API, and source map are consistent and usable;
- no fitted runtime physics or stale proxy remains;
- CI, docs, package, release, and post-install checks pass;
- every remaining item appears under Non-Blocking Research Lanes and is excluded
  from release and manuscript claims.

## Primary References

- Escoto et al., [*A fast neoclassical code for the evaluation of
  monoenergetic transport coefficients*](https://arxiv.org/abs/2312.12248),
  Nuclear Fusion 64 (2024) 076030.
- Escoto, [*Fast and accurate calculation of the bootstrap current and radial
  neoclassical transport in low collisionality stellarator
  plasmas*](https://arxiv.org/abs/2510.27513), PhD thesis.
- Beidler et al., [*Benchmarking of the mono-energetic transport
  coefficients*](https://doi.org/10.1088/0029-5515/51/7/076001), Nuclear
  Fusion 51 (2011) 076001.
- Landreman et al., [*Comparison of particle trajectories and collision
  operators for collisional transport in nonaxisymmetric
  plasmas*](https://arxiv.org/abs/1312.6058), Physics of Plasmas 21 (2014)
  042503.
- Landreman and Ernst, *New velocity-space discretization for continuum kinetic
  calculations and Fokker-Planck collisions*, JCP 243 (2013) 130-150.
- Saad and Schultz, GMRES; Saad, flexible GMRES; Parks et al., Krylov recycling.
- Demmel, Higham, and Schreiber, stability of block LU factorization.
- Carson and Higham, mixed-precision iterative refinement.
- Orszag and standard spectral-method references for Fourier aliasing.
- [JAX documentation](https://docs.jax.dev/) for JIT caching, profiling,
  memory, custom linear solves, buffer donation, and rematerialization.
