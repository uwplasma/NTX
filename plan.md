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

### 6. Native Bootstrap-Current Workflow

- [x] add a native NTX neoclassical closure layer that uses NTX monoenergetic
  coefficients directly instead of routing through NEOPAX
- [x] implement the no-momentum-correction bootstrap-current path first:
  - species inputs
  - thermal-speed grid and quadrature
  - collisionality and thermodynamic forces
  - energy convolution
  - `Lij` assembly
  - `U_parallel` / `j·B` / bootstrap-current outputs
- [ ] keep the first native implementation physically equivalent to the current
  NTX+NEOPAX no-momentum path before adding any new model features
- [x] add a native database-to-current API that can evaluate:
  - one radius / one species set
  - one radial profile
  - one VMEC/Boozer family scan
- [ ] add explicit diagnostics for:
  - species-resolved current contributions
  - `Lij` entries
  - energy-integrand breakdown
  - sensitivity of the current to `D13`, `D31`, and `D33`
- [ ] only after the no-momentum branch is stable, decide whether
  momentum-correction belongs natively in NTX or remains a higher-level
  transport feature

### 7. Validation, Benchmarks, And Gates

- [ ] add unit tests for the native bootstrap-current closure:
  - quadrature normalization
  - thermodynamic-force construction
  - collisionality helpers
  - `Lij` assembly signs and symmetries
  - current sign under controlled `D13` / `D31` inputs
- [ ] add regression tests tying NTX-native bootstrap current to the current
  NTX+NEOPAX no-momentum result on small frozen fixtures
- [ ] add benchmark tests against SFINCS-JAX / SFINCS for the finite-beta QA and
  QH cases already used in the paper-side audit
- [ ] add fixed-radius transport-matrix audits against SFINCS-JAX for the
  helical VMEC path, with explicit checks on `D13`, `D31`, and `D33`
- [ ] add a W7-X native bootstrap-current regression on the imported workflow
- [ ] define hard gates for native bootstrap-current work:
  - Gate 1: NTX-native reproduces NTX+NEOPAX no-momentum current on frozen test
    cases to tight tolerance
  - Gate 2: QA/QH sign and radial trend agree with SFINCS
  - Gate 3: QA/QH max relative error improves materially over the current audit
  - Gate 4: W7-X imported bootstrap-current workflow remains stable
  - Gate 5: all new APIs have unit tests and regression tests

### 8. Examples, Docs, And Release Surface

- [ ] add a native NTX example for bootstrap-current calculation from VMEC /
  Boozer without routing through NEOPAX
- [ ] keep the example simple and user-facing:
  - load VMEC or Boozer geometry
  - define species / profiles
  - compute bootstrap-current profile
  - write a polished figure and structured output
- [ ] add a separate validation example that compares NTX-native against
  SFINCS-JAX / SFINCS on curated QA/QH/W7-X cases
- [ ] add polished documentation pages for the native bootstrap-current model:
  - equations
  - normalization
  - energy convolution
  - species inputs / outputs
  - examples and benchmark interpretation
- [ ] add a curated bootstrap-current validation figure to the README showing
  NTX vs SFINCS profile agreement on a benchmark case
- [ ] ensure the README stays concise while the full model details live in the
  docs

### 9. QA And Maintenance

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

1. Implement native no-momentum-correction bootstrap current in NTX and match
   the present NTX+NEOPAX path on frozen fixtures.
2. Audit native QA/QH bootstrap-current profiles against SFINCS-JAX and SFINCS
   until the remaining amplitude gap is understood quantitatively.
3. Add the native bootstrap-current example, validation example, docs, and
   README validation figure.
4. Strengthen the current profile transport loop from proxy closure toward a
   more predictive self-consistent transport workflow.
5. Reduce adjoint memory/factorization cost for prepared dense solves on larger
   optimization scans.

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
- The current QA/QH bootstrap-current audit established:
  - `vmec_jax.read_wout()` is not the raw sign/load problem
  - NTX had two convention bugs that are now fixed:
    - file-backed VMEC loader signs for `iota`, helical `n`, and `jacobian`
    - NTX-to-NEOPAX `D13` sign in the monoenergetic handoff
  - after those fixes, the remaining QA/QH mismatch is an amplitude/model gap,
    not a sign/normalization bug
- The next NTX development batch is therefore centered on native
  bootstrap-current support, tighter SFINCS-facing validation gates, and a
  cleaner public workflow for bootstrap-current calculations directly in NTX
- The first native bootstrap-current slice is now in-tree:
  - `src/ntx/bootstrap.py` provides species inputs, primitive-to-force
    construction, no-momentum `Lij` assembly, and native `j·B` evaluation
  - the first regression gates are in place for zero-force closure, primitive
    profile construction, and a frozen current-profile value test
  - CI failures seen in GitHub Actions at this point are real lint failures,
    not exhausted Actions minutes
