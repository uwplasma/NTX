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
- [ ] add a fixed-radius transport-matrix audit against SFINCS-JAX on the
  fixed-field QA/QH reference cases, focused on `D13`, `D31`, and `D33`
- [ ] isolate the outer-radius amplitude failure by auditing:
  - VMEC file loading and radial mapping
  - SFINCS-JAX transport-matrix normalization
  - NTX `D13` / `D31` / `D33` channel conventions
  - NTX to NEOPAX handoff conventions
- [ ] reproduce the Boozer-based Redl path from the Zenodo bundle robustly on
  the fixed-field QA/QH reference family
- [ ] add frozen local-only regression tests for the fixed-field audit helpers
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

1. Land the fixed-field QA/QH transport-matrix audit against SFINCS-JAX,
   centered on `D13`, `D31`, and `D33`.
2. Reproduce the Boozer-based Redl path from the Zenodo benchmark cleanly and
   compare it against the VMEC-based Redl path on the same fixed-field cases.
3. Keep finite-beta QA/QH and W7-X bootstrap-current validation in the
   `NTX+NEOPAX` lane, separate from the fixed-field coefficient audit.
4. Profile the prepared solve and `NTX+NEOPAX` workflow to identify the real
   runtime and memory bottlenecks before changing solver internals.
5. Continue the profile-transport and derivative work only after the
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
- A first fixed-field transport-matrix audit is now in-tree:
  - it runs SFINCS-JAX in `RHSMode=3` on the Landreman-Paul QA/QH reference
    equilibria at `rho = [0.25, 0.50, 0.75]`
  - it compares `L13`, `L31`, and `L33` against NTX candidate channels
    derived from `D13`, `D31`, and `D33`
  - first result: simple `-D13 dr/ds` and `nu D33` mappings are not enough to
    close the gap, especially in `L33`, so the remaining fixed-field mismatch
    is not just the earlier sign bug
