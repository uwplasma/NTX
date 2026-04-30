# Physics Gates

NTX uses explicit physics gates so that validation claims are tied to the
actual model scope.

These gates separate five different questions:

1. is the monoenergetic solver algebra correct?
2. is the database handoff to the closure workflow correct?
3. does the imported integrated workflow transfer cleanly?
4. do differentiable geometry and boundary-control workflows match centered
   finite differences on their claimed scope?
5. where does the reduced closure model pass as a total-current stress
   diagnostic, and where does it still stop matching fuller collisional tools?

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
- **Boozer Jacobian identity:** the Boozer geometry projection must preserve
  `\mathcal J B^2 = B_\zeta + \iota B_\theta` together with
  `B^\theta \mathcal J = \iota` and `B^\zeta \mathcal J = 1` on owned
  surfaces. This anchors the field normalization used by the drift source,
  parallel streaming, and imported geometry handoff before any benchmark
  coefficient is interpreted.
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
- **Profile interpolant derivative consistency:** the imported profile
  interpolants must give `D33` sensitivities with respect to electric-field
  basis parameters that agree with centered finite differences on a controlled
  coefficient table. This protects the interpolation layer used by profile
  inverse design and uncertainty propagation.
- **Profile-control linear response:** scalar and radial-basis profile controls
  must be identity maps at zero control and exactly linear in their prescribed
  response matrices. This protects the profile optimization, sensitivity, and
  uncertainty workflows from hidden nonlinearities in the control-to-force map.
- **VMEC-JAX boundary edge transfer:** the traced fixed-boundary Fourier edge
  arrays must be forwarded to both the implicit VMEC residual solve and the
  explicit relaxation solve. This protects boundary-to-output derivatives from
  accidentally following stale non-differentiated boundary data.
- **VMEC-JAX to NEOPAX radial metric transfer:** the imported field builder
  must preserve the `rho = sqrt(s)` radial mapping, axis regularization,
  enclosed-volume scale, edge major-radius scale, and toroidal-flux
  normalization before any bootstrap-current or boundary-derivative workflow
  consumes the field.
- **NTX-to-NEOPAX field-channel normalization:** scan assembly must preserve
  `E_r = E_s * transport_psi_scale` while checking the `rho`, `drds`, and
  electric-field table shapes before the monoenergetic coefficients are handed
  to the imported closure workflow.
- **Primitive profile force reconstruction:** the profile workflow must recover
  `A3 = d ln T / dr` and
  `A1 = d ln n / dr - 3 d ln T / (2 dr) + C_E Z E_r` before those forces are
  used in particle-flux or reduced bootstrap-current response calculations.
- **Charge-symmetric ambipolar cancellation:** the profile workflow must return
  zero charge-weighted particle-flux residual for equal particle-flux responses
  with opposite charges. This is the local implementation gate for the
  quasineutral ambipolarity condition `sum_s Z_s Gamma_s(E_r) = 0`, which sets
  the stellarator radial electric field outside intrinsically ambipolar
  symmetry limits.
- **Primitive transport positivity floor:** explicit primitive-profile updates
  must keep density and temperature finite and positive. The update uses
  exponential relaxation plus a small floor, so large transport mismatches
  cannot create unphysical negative thermodynamic state variables.
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
- **Source-channel superposition:** for a frozen finite-beta stress-radius
  momentum-restoring matrix, the density/electric, effective temperature, and
  parallel-electric one-channel solves must reconstruct the full corrected
  current to roundoff before the dominant channel is interpreted physically.
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
- **Same-coordinate Boozer-file round trip:** generated `boozmn` surfaces must
  reload on the VMEC half grid and reproduce the in-memory
  `vmec_jax -> booz_xform_jax -> NTX` transport coefficients below `1e-6`.
  This protects the radial-coordinate convention used by packed Boozer spectra.
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
  artifact must keep the optimized weighted-current response at least equal to the
  baseline before the manuscript cites the gain. This is a stress gate, not a
  broad optimization-design claim.

## 3. Independent-Code Comparison Gates

These are trust-building comparisons against independent workflows:

- **Precise-QS Redl vs archived SFINCS:** on the interior benchmark window, the
  Redl reconstruction should stay below `1e-1` maximum relative error.
- **Fixed-field transport-matrix audits:** the archive-backed `RHSMode=3` and
  `RHSMode=2` comparisons against SFINCS-JAX are used to localize
  normalization and closure differences.
- **Owned finite-beta same-grid coefficients:** the finite-beta QA ladder now
  has a passing normalization-side gate. The maximum same-grid
  `L13/L31/L33` relative difference is required to stay below `1e-1` before
  any finite-beta profile-current result is interpreted. This is not by itself
  a current-parity gate when the net current is cancellation-conditioned.
- **Owned finite-beta source-channel reconstruction:** the finite-beta source
  decomposition is a stress gate on the linear momentum-restoring system:
  one-channel solves must reconstruct the full corrected current below `1e-8`.
  The resulting dominant-channel statement is diagnostic, not a fitted current
  correction.
- **Owned finite-beta temperature-source response:** the source-channel sidecar
  also stores the Redl density and temperature target terms on the same current
  observable. The high-order Redl/NTX effective-temperature response ratio is a
  monitored stress metric, not an acceptance gate and not a runtime fit.
- **Owned finite-beta profile source-response:** the source-response sidecar
  extends the effective-temperature channel measurement over the finite-beta
  profile. Its radial response span and correlations with Redl collisionality,
  trapped fraction, epsilon, and `L32` are monitored diagnostics for a future
  physics-derived closure, not fitted correction factors.
- **Owned finite-beta closure-target driver audit:** the closure-target sidecar
  ranks local neoclassical drivers for the measured temperature-source response
  before any runtime model is proposed. The current artifact selects the Redl
  geometry factor `epsilon` as the strongest single driver and records that
  diagnostic regressions are not applied as production corrections.
- **Owned finite-beta profile-current observable:** the finite-beta
  bootstrap-current profile remains a monitored stress diagnostic, not a
  parity gate. The current artifacts keep the net-current residual, the
  applied-versus-needed momentum correction, and the species-correction
  cancellation scale visible because the stress radius is dominated by
  electron/ion cancellation.
- **Owned finite-beta current conditioning:** the coefficient ladder is also
  compared with the species-current L1 scale. The current artifact reports that
  the stress-radius net current needs about `1.3e-3` coefficient precision for
  a `1e-1` current gate, while the completed smoke ladder is about `2.1e-2`.
- **Owned finite-beta resolution floor:** the stress-radius point has also
  been rerun at `35 x 43 x 48` and with a tighter VMEC harmonic cutoff. The
  coefficient floor stays near `2.05e-2`, so angular resolution and harmonic
  truncation are not treated as the closure fix.
- **Owned finite-beta production ladder:** the six production same-grid
  radius/collisionality points all stay below `2.07e-2` coefficient difference.
  The largest current-conditioned precision gap remains the inner stress point,
  so the coefficient-resolution lane is localized rather than a broad
  whole-profile failure.
- **Owned finite-beta closure quadrature:** higher Sonine order is now
  monitored together with velocity quadrature. The only stress-radius
  current-gate pass occurs at `P=14, X=10`, where `X < Pmax`, and does not
  transfer to `X=14` or `X=18`. The accepted quadrature-stable current-gate
  pass count is zero, so this is treated as quadrature aliasing, not a valid
  physics closure.
  This keeps the next physics step honest: profile-current diagnostics must
  tighten the conditioned uncertainty before a new reduced-closure term is
  promoted.

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
- **Geometry-family transport convergence:** the public VMEC example-family
  `D11/D31/D33` ladder is monitored as a stress diagnostic; it must stay finite
  and visible before any broad geometry-family parity claim is promoted.
- **W7-X rebuilt raw branch:** best observed maximum relative error
  `<= 2e-2` against the frozen reference profile.

This gate is important because it validates the full path:

`NTX monoenergetic tables -> database normalization -> imported closure workflow`.

### 5. Closure Stress Tests

The fixed-field precise-QS `NTX+NEOPAX` current comparison is a scoped
total-current stress gate, not a release gate for the monoenergetic solver and
not an independent species-current parity claim.

The current interior QA/QH total-current errors now pass the documented
`1e-1` stress tolerance after applying only physics-normalization choices that
are already part of the closure model:

- the archived SFINCS observable is compared through its
  flux-surface-averaged parallel-flow convention,
- the reduced fixed-field branch uses the explicit low-order
  Spitzer-conductivity block,
- the closure grid uses the `P=2` low-order moment system.

No fitted bridge constants, benchmark-specific scale factors, or hard
thresholds are used in the current fixed-field branch.

The current compact closure report records:

- Redl fixed-field QA/QH maximum interior errors of `6.86e-2` and `4.06e-2`
- NTX+NEOPAX fixed-field QA/QH total-current stress errors of `8.30e-2`
  and `9.95e-2`
- rebuilt W7-X raw-branch integrated transfer error of `1.83e-2`

Those numbers are intentionally interpreted together. The fixed-field stress
case compares a reduced momentum-restoring closure built from monoenergetic
coefficients against a fuller drift-kinetic reference, while the integrated
W7-X transfer checks the actual database normalization and imported workflow
used by NTX. The low-order Spitzer-conductivity fixed-field branch does not
transfer to the imported W7-X database convention, so the W7-X release gate
remains the validated raw branch. A broader closure default is promotable only
if it preserves both the precise-QS total-current gate and the integrated W7-X
raw-branch transfer gate.

The owned finite-beta closure lane is tracked by additional artifact
gates:

- same-grid finite-beta coefficient normalization: passing, with current
  maximum relative difference about `2.1e-2`;
- finite-beta profile-current observable: monitored, with current stress-radius
  total-current relative difference about `3.1e-1`;
- finite-beta species-cancellation scale: monitored, with current stress-radius
  residual about `4e-3` of the species momentum-correction L1 scale.
- finite-beta production coefficient floor: monitored, with the current
  production stress probe still about `15.9x` looser than the
  cancellation-conditioned coefficient target.
- finite-beta production radial/collisionality ladder: monitored, with all
  production coefficient differences below `2.07e-2` but the maximum
  current-conditioned precision gap still about `15.9x`.
- finite-beta closure quadrature: monitored, with one under-integrated
  current-gate pass and a highest-quadrature largest-order stress difference
  near `4e-1`.
- finite-beta profile source response: monitored, with the current high-order
  temperature response multiplier spanning `0.717` to `1.317` over the profile.
- finite-beta closure-target driver ranking: monitored, with the current best
  single driver `epsilon`, absolute Pearson correlation `0.975`, and no runtime
  correction applied.

This is the current physics interpretation: the coefficient-side bridge is no
longer the leading suspect for the finite-beta QA stress case, and the remaining
gap lives in a quadrature-converged reduced profile-current/source-response
observable under strong species-current cancellation. Future closure work must
improve that observable without changing device-specific scale factors and
without regressing the fixed-field precise-QS or integrated W7-X gates.

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
5. preserve the fixed-field precise-QS total-current closure stress gate
6. improve species-resolved fixed-field closure parity only if it also
   preserves the integrated W7-X workflow
7. show controlled convergence in `Pmax` on the precise-QS QA/QH family
8. avoid any regression in the integrated W7-X workflow when `Pmax` changes

That is the standard for physically defensible closure work in this repository.

## Current Higher-Order Closure Status

The first higher-order validation stage is now in place in the imported
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
