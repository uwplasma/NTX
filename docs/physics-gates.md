# Physics Gates

NTX uses explicit physics gates so that validation claims are tied to the
actual model scope.

These gates separate four different questions:

1. is the monoenergetic solver algebra correct?
2. is the database handoff to the closure workflow correct?
3. does the imported integrated workflow transfer cleanly?
4. where does the reduced closure model stop matching fuller collisional tools?

That separation matters. Without it, a closure-model gap can be mistaken for a
solver bug, or a benchmark-specific fit can be mistaken for production physics.

## Gate Families

### 1. Analytical Identities

These are hard structural checks:

- **Onsager symmetry:** `|D13 + D31|` must remain small on converged solves.
- **Owned-surface coefficient convergence:** on the repository-owned analytic
  Boozer surface, the fast test suite now checks that the `D11`, `D31`, and
  `D33` Legendre-resolution errors decrease from `N_\xi=6` to `N_\xi=8`
  relative to the `N_\xi=10` reference, that the finest fast-lane error remains
  bounded, and that the same coefficients transfer between the `5x5` and `7x7`
  angular grids within the current release tolerance.
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

## 2. Independent-Code Comparison Gates

These are trust-building comparisons against independent workflows:

- **Precise-QS Redl vs archived SFINCS:** on the interior benchmark window, the
  Redl reconstruction should stay below `1e-1` maximum relative error.
- **Fixed-field transport-matrix audits:** the archive-backed `RHSMode=3` and
  `RHSMode=2` comparisons against SFINCS-JAX are used to localize
  normalization and closure differences.

These comparisons are useful because they check the physical bridge to
well-established neoclassical calculations without redefining NTX as “whatever
 matches another code.”

### 3. Integrated-Workflow Transfer Gate

The strongest imported-workflow gate is the rebuilt W7-X bootstrap-current
workflow.

The active acceptance target is:

- **W7-X rebuilt raw branch:** best observed maximum relative error
  `<= 2e-2` against the frozen reference profile.

This gate is important because it validates the full path:

`NTX monoenergetic tables -> database normalization -> imported closure workflow`.

### 4. Closure Stress Tests

The fixed-field precise-QS `NTX+NEOPAX` current comparison is retained as a
stress test, not as a release gate for the monoenergetic solver.

The current interior QA/QH errors are tracked continuously, but they are
interpreted as a **reduced momentum-restoring closure-model gap** rather than
as a remaining normalization bug in NTX.

## Current Policy

The gate registry is exposed in the public API:

- `ntx.physics_gates.physics_gate_registry()`
- `ntx.physics_gates.evaluate_artifact_gates(...)`

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
