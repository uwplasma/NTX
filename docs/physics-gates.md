# Physics Gates

NTX uses explicit physics gates so that validation claims are tied to the
actual model scope.

These gates separate five different questions:

1. is the monoenergetic solver algebra correct?
2. is the database handoff to the closure workflow correct?
3. does the imported integrated workflow transfer cleanly?
4. do differentiable geometry and boundary-control workflows match centered
   finite differences on their claimed scope?
5. where does the reduced closure model stop matching fuller collisional tools?

That separation matters. Without it, a closure-model gap can be mistaken for a
solver bug, or a benchmark-specific fit can be mistaken for production physics.

## Gate Families

### 1. Analytical Identities

These are hard structural checks:

- **Monoenergetic validation summary:** the committed
  `validation_summary.json` artifact is now a release gate. The maximum of the
  DKES-style and VMEC finest plotted `N_\xi` convergence errors must stay below
  `2.5e-1` against the finest plotted reference. This keeps the promoted
  methods figure tied to a machine-checked convergence metric.
- **Onsager symmetry:** `|D13 + D31|` must remain small on converged solves.
- **Owned-surface coefficient convergence:** on the repository-owned analytic
  Boozer surface, the fast test suite now checks that the `D11`, `D31`, and
  `D33` Legendre-resolution errors decrease from `N_\xi=6` to `N_\xi=8`
  relative to the `N_\xi=10` reference, that the finest fast-lane error remains
  bounded, and that the same coefficients transfer between the `5x5` and `7x7`
  angular grids within the current release tolerance.
- **Constant-field symmetric limit:** when `B` is constant on the Boozer
  surface, the magnetic-drift source vanishes. The fast test suite therefore
  requires `D11`, `D31`, and `D13` to vanish while the parallel-conductivity
  channel remains positive and equal to the Spitzer branch reported as
  `D33_spitzer`.
- **Spitzer inverse-collisionality normalization:** in the same constant-field
  limit, the fast test suite checks that `D33_spitzer` scales as `1 / nu_hat`.
  This catches drift-kinetic normalization regressions without needing an
  external benchmark file.
- **Constant-field radial-electric-field invariance:** in the constant-field
  limit, sweeping `er_hat` must not create radial transport or change the
  parallel-conductivity branch. This guards the radial-electric-field advection
  term against producing unphysical transport when the magnetic-drift drive is
  absent.
- **Finite Legendre source projection:** the magnetic-drift drive must populate
  only the `k=0` and `k=2` source rows with the runtime `2/3` and `1/3`
  weights, and the parallel-conductivity source must populate only the `k=1`
  row as the physical `B`. This protects the equation-to-code map before any
  solve or closure post-processing is involved.
- **Imported Boozer handedness:** VMEC-to-Boozer helper paths must choose the
  same right-handed Boozer convention as the file-backed loader, so
  `B_\zeta + \iota B_\theta >= 0` before the geometry Jacobian
  `\mathcal J = |B_\zeta + \iota B_\theta| / B^2` is consumed by the solver
  or imported closure workflow. This protects the sign convention without
  fitting any transport coefficient.
- **Operator parameter-derivative consistency:** the hand-coded
  `dD_k/dnu_hat` and `dD_k/depsi_hat` blocks used by the implicit-adjoint path
  must match JAX differentiation of the assembled Legendre-space operator.
  This catches collisionality and radial-electric-field normalization
  regressions before they can contaminate sensitivity, inverse-design, or
  uncertainty-quantification workflows.
- **Primitive profile force reconstruction:** the profile workflow must recover
  `A3 = d ln T / dr` and
  `A1 = d ln n / dr - 3 d ln T / (2 dr) + C_E Z E_r` before those forces are
  used in particle-flux or bootstrap-current proxy calculations.
- **Prepared derivative-path consistency:** the committed derivative-path
  benchmark must keep the prepared custom-VJP electric-field derivative within
  `1e-4` relative mismatch of direct reverse-mode on the same prepared surface.
  The speedup remains reported, but agreement is the release gate.
- **Exact `P=2` recovery:** the generated Sonine/Hankel projection must recover
  the current three-moment closure exactly at `P=2`.
- **Low-order collision-block recovery:** the active low-order
  momentum-conserving collisional blocks must be reproducible from the
  standard low-order moment equations, with only the runtime heat-flow basis
  sign convention differing from the canonical notation.
- **Fixed observable map:** for the present Sonine basis, the corrected
  parallel-flow observable remains `U_parallel = n c_0`.
- **Intrinsic ambipolarity in symmetric limits:** finite-order closure work
  must preserve the symmetric-limit ambipolar structure emphasized by the
  Sugama–Nishimura formulation.
- **Momentum-conservation null mode:** the collisional blocks must retain a
  common-flow null mode so that total parallel momentum is conserved.
- **Particle conservation:** the projected collisional operator must preserve
  the density invariant and must not generate a spurious source term.
- **Energy conservation:** the projected collisional operator must preserve the
  energy invariant in the same finite basis used for the closure.
- **Weighted self-adjointness:** the finite-order collision operator should
  preserve the self-adjoint structure of the linearized Coulomb problem under
  the appropriate weighted inner product.
- **Non-negative entropy production:** the symmetric collisional form must
  remain positive semidefinite, following the finite-order constraints
  emphasized by Sugama–Horton.

These are not benchmark fits. They come directly from the model derivation and
from the present closure basis.

## 2. Differentiability Artifact Gates

These gates protect the end-to-end JAX workflows without overstating their
geometry breadth:

- **Owned analytic geometry-control derivatives:** direct AD must match
  centered finite differences below `2e-4` on the committed three-harmonic
  analytic-surface audit.
- **File-backed geometry-control derivatives:** the same direct AD/finite
  difference comparison must stay below `5e-4` on the repository-owned Boozer
  and VMEC sample surfaces.
- **Boundary-projected current derivatives:** forward-mode derivatives through
  the optional JAX geometry backends, NTX coefficients, and the integrated
  current objective must stay below `1e-5` on the committed sample input.
- **Explicit-relaxed boundary-to-current derivatives:** the self-consistent
  forward-mode QA/QH family must stay below `1e-4`; the artifact also reports
  the ordinary-vs-explicit-relaxed volume agreement so the derivative check is
  not hiding a branch mismatch.
- **Implicit-equilibrium derivatives:** this remains a monitored stress lane,
  not a passing gate. The current artifact validates the equilibrium-volume
  derivative but leaves the Boozer-space and NTX transport observables open.
- **Bootstrap-current optimization gain:** the committed science/application
  artifact must keep the optimized weighted-current proxy at least equal to the
  baseline before the manuscript cites the gain. This is a stress gate, not a
  broad optimization-design claim.

## 3. Independent-Code Comparison Gates

These are trust-building comparisons against independent workflows:

- **Precise-QS Redl vs archived SFINCS:** on the interior benchmark window, the
  Redl reconstruction should stay below `1e-1` maximum relative error.
- **Fixed-field transport-matrix audits:** the archive-backed `RHSMode=3` and
  `RHSMode=2` comparisons against SFINCS-JAX are used to localize
  normalization and closure differences.

These comparisons are useful because they check the physical bridge to
well-established neoclassical calculations without redefining NTX as “whatever
 matches another code.”

### 4. Integrated-Workflow Transfer Gate

The strongest imported-workflow gate is the rebuilt W7-X bootstrap-current
workflow.

The active acceptance target is:

- **Monoenergetic validation summary:** committed validation-summary finest
  plotted coefficient error `<= 2.5e-1` on both the DKES-style and VMEC surfaces.
- **Prepared derivative path:** maximum prepared-vs-direct derivative mismatch
  `<= 1e-4` on the committed derivative-path benchmark.
- **Differentiable geometry path:** the promoted finite-difference agreement
  gates above must pass for the analytic, file-backed, boundary-projected, and
  explicit-relaxed artifacts; the implicit-equilibrium artifact is monitored
  separately until it closes.
- **W7-X rebuilt raw branch:** best observed maximum relative error
  `<= 2e-2` against the frozen reference profile.

This gate is important because it validates the full path:

`NTX monoenergetic tables -> database normalization -> imported closure workflow`.

### 5. Closure Stress Tests

The fixed-field precise-QS `NTX+NEOPAX` current comparison is retained as a
stress test, not as a release gate for the monoenergetic solver.

The current interior QA/QH errors are tracked continuously, but they are
interpreted as a **reduced momentum-restoring closure-model gap** rather than
as a remaining normalization bug in NTX.

## Current Policy

The gate registry is exposed in the public API:

- `ntx.physics_gates.physics_gate_registry()`
- `ntx.physics_gates.evaluate_artifact_gates(...)`
- `ntx.validation.physics_gate_registry()`
- `ntx.validation.evaluate_artifact_gates(...)`

To inspect the tracked artifact-backed gates locally:

```bash
python scripts/check_physics_gates.py
```

The script reads the tracked benchmark artifacts in `docs/_static/` and reports
which gates are:

- pass/fail acceptance gates,
- test-backed analytical gates,
- or monitored stress metrics.

A compact companion report is built by:

```bash
python scripts/build_closure_validation_report.py
```

This report packages the same tracked artifacts into one summary figure and
JSON/Markdown set. It is useful when reviewing the current model-family status
without rereading the individual benchmark outputs one by one.

## Acceptance Rules For Closure Work

Any higher-order closure change must satisfy all of the following:

1. keep the monoenergetic coefficient-side invariants unchanged
2. keep `U_parallel = n c_0`
3. recover the present three-moment system exactly at `P=2`
4. preserve finite-order symmetry structure as far as the projected model
   allows
5. improve the fixed-field precise-QS closure stress test only if it also
   preserves the integrated W7-X workflow
6. show controlled convergence in `Pmax` on the precise-QS QA/QH family
7. avoid any regression in the integrated W7-X workflow when `Pmax` changes

That is the standard for physically defensible closure work in this repository.

## Current Higher-Order Scaffold

The first higher-order implementation stage is now in place in the imported
closure stack:

- configurable Sonine truncation order in the closure grid
- generated raw D13 source-moment sequences for arbitrary order
- generated raw D33 Hankel moment sequences for arbitrary order
- exact recovery of the present `P=2` closure
- exact recovery of the active low-order momentum-conserving collision blocks
  from the standard moment equations

That stage is intentionally incomplete. The production runtime still stops at
`P=2` because the arbitrary-order momentum-conserving collision blocks have not
yet been derived and validated. This is a physics boundary, not an
implementation oversight.

The first higher-order tail experiment has now also been run against the
committed gate set. It keeps the current low-order closure unchanged and adds
a diagonal Laguerre-tail damping model on the extra moments. That branch
is numerically stable, but it is not physically acceptable: the first
`Pmax=4` run barely moves the precise-QS stress metric while regressing the
imported W7-X closure error from `1.17e-12` at `P=2` to about `4.94e-1`.
That result is now pinned in `docs/_static/closure_pmax_convergence.json` and
is treated as a rejected higher-order branch rather than as production physics.

## Additional Literature Requirements

Beyond the existing acceptance targets, the literature imposes a few stronger
requirements on any generalized closure:

- the finite-order system should preserve Onsager symmetry rather than recover
  it only asymptotically
- intrinsic ambipolarity should remain exact in symmetric limits at each
  truncation order
- particle and energy conservation should remain exact collisional invariants
  of the projected system
- the collisional operator must conserve momentum exactly and should not break
  the common-flow null space
- the projected collisional form should remain self-adjoint under the weighted
  inner product used by the finite-order derivation
- the symmetric collisional form should not generate negative entropy
  production
- convergence in `Pmax` should be demonstrated on a stress-test family, not
  only on an integrated workflow that already closes at `P=2`

These are now treated as first-class design requirements for the higher-order
closure lane.
