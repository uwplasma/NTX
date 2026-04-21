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
- [x] add a curated `NTX+NEOPAX` vs SFINCS bootstrap-current validation figure
  to the README, with the benchmark status stated honestly

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
      - legacy MONKES-style NEOPAX HDF5 files use a different `D13`
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
