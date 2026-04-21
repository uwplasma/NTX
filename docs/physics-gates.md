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
- **Exact `P=2` recovery:** the generated Sonine/Hankel projection must recover
  the current three-moment closure exactly at `P=2`.
- **Fixed observable map:** for the present Sonine basis, the corrected
  parallel-flow observable remains `U_parallel = n c_0`.

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

## Acceptance Rules For Closure Work

Any higher-order closure change must satisfy all of the following:

1. keep the monoenergetic coefficient-side invariants unchanged
2. keep `U_parallel = n c_0`
3. recover the present three-moment system exactly at `P=2`
4. preserve finite-order symmetry structure as far as the projected model
   allows
5. improve the fixed-field precise-QS closure stress test only if it also
   preserves the integrated W7-X workflow

That is the standard for physically defensible closure work in this repository.
