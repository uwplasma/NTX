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
  - do not promote QA/QH bootstrap-current figures to the README until the
    fixed-field coefficient audit is tighter

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
- [ ] only after the fixed-field audit is tighter, add a curated
  `NTX+NEOPAX` vs SFINCS bootstrap-current validation figure to the README

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
  - `NTX+NEOPAX` is still not close on the precise-QS fixed-field family,
    with interior max relative errors now around `0.79` on QA and `0.81` on
    QH in the current sampled-radial comparison
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
  - the audit scaffold now supports one-species probes (`ion` or `electron`)
    plus reduced SFINCS resolution overrides, so the next direct target is the
    electron branch on the precise-QS QA/QH family rather than the full
    two-species transport matrix all at once
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
  - only after that closure is tight should the fixed-field `NTX+NEOPAX`
    current figure move into the public README or main validation claims
