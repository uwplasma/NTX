# Validation

NTX validation is organized around four layers:

1. unit and regression tests in `tests/`
2. workflow and convergence examples in `examples/`
3. CPU/GPU runtime checks
4. downstream database and profile checks through NEOPAX

The gate hierarchy behind those layers is now documented explicitly in
[`physics-gates.md`](physics-gates.md). In short:

- analytical identities and exact `P=2` recovery are hard gates,
- independent-code comparisons are trust-building physics gates,
- the rebuilt integrated W7-X raw branch is the main transfer gate,
- the precise-QS fixed-field `NTX+NEOPAX` current benchmark is a closure stress
  test rather than a monoenergetic parity requirement.

## Validation Philosophy

NTX is validated as a standalone solver. The repository therefore emphasizes:

- internal numerical consistency
- convergence behavior
- trustworthy geometry loading
- end-to-end workflow checks
- imported JAX workflows
- CPU/GPU execution stability

Independent comparisons are useful, but they are treated as trust-building
studies rather than as the definition of NTX itself.

## What Is Covered

The maintained suite covers:

- Fourier-series evaluation and flux-surface averages
- operator assembly and nullspace handling
- dense block-tridiagonal solves
- scan helpers and prepared-solver reuse
- autodiff inverse and profile-analysis workflows
- DKES-style, VMEC, and Boozer file loaders
- TOML input parsing and `.npz` output writing
- imported NEOPAX-array and HDF5 mapping helpers
- `vmec_jax` and `booz_xform_jax` integration points
- serial versus parallel-scan equivalence
- example and publication-figure regeneration

## Core Physics Checks

### Onsager Closure

Every solve reports:

```{math}
|D_{13} + D_{31}|
```

This is the main scalar physics sanity check exposed directly by NTX.

The current tracked artifact-backed gates can be summarized locally with:

```bash
python scripts/check_physics_gates.py
```

### Resolution Convergence

The W7-X imported workflow is audited with:

```bash
python examples/bootstrap_current_reference_audit_w7x.py
```

This script rebuilds a reduced W7-X scan at several NTX resolutions, evaluates
the resulting bootstrap-current profile, and writes a publication-ready
convergence figure.

### Precise-QS Redl Benchmark

The archived Landreman--Paul precise-QS fixed-field benchmark can be reproduced
locally with:

```bash
python examples/precise_qs_redl_sfincs_audit.py
```

This archive-backed audit reads the original SFINCS profiles from the Zenodo
bundle, reconstructs the Redl current through both:

- the VMEC-side trapped-fraction path
- the Boozer-side trapped-fraction path

and checks both against the archived SFINCS profile on the same surfaces. On
the current local stack, both Redl paths recover the archived interior-window
benchmark gate on the fixed-field QA/QH references:

- QA interior max relative error: about `9.3%` for the VMEC path and `9.5%`
  for the Boozer path
- QH interior max relative error: about `4.2%` for the VMEC path and `4.1%`
  for the Boozer path

This is a fixed-field Redl/SFINCS consistency study. It is intentionally kept
separate from the finite-beta and imported `NTX+NEOPAX` bootstrap-current
workflow checks.

### Fixed-Field Transport-Matrix Audit

The remaining fixed-field coefficient-side gap is audited with:

```bash
python examples/fixed_field_transport_matrix_audit.py
```

This script runs SFINCS-JAX in `RHSMode=3` on the same QA/QH fixed-field
reference family and compares `L13`, `L31`, and `L33` against NTX candidate
channels built from `D13`, `D31`, and `D33`.

The present conclusion is narrow but important:

- the benchmark family is now correct
- the SFINCS `RHSMode=3` overwrite must be matched exactly through
  `nu_n = nuPrime * B0OverBBar / (GHat + iota IHat)`
- archive-backed Landreman/H. Smith bridge factors tighten `L13` and `L31`
  substantially once the correct `nu_n` is used
- current fixed-field `L13/L31` relative errors are about `0.12–0.29` on QA
  and `0.027–0.15` on QH
- the largest unresolved normalization/model gap is now the `L33` bridge, with
  current fixed-field relative errors of about `0.14–0.16`
- this is only an `RHSMode=3` monoenergetic statement; for the zero-`E_r`
  fixed-field bootstrap-current comparison itself, the active no-momentum
  closure also depends on the temperature-gradient (`A2`) channel, so the next
  gating audit is the full `RHSMode=2` row-3 thermal closure rather than more
  retuning of the old `L13/L31/L33` bridge plot alone
- the first cached `RHSMode=2` QA electron probe now confirms the thermal
  row-3 bridge itself: reconstructing the closure response from the exact
  SFINCS `whichRHS` source gradients and converting it back with the common
  factor `2 B0OverBBar / sqrt(pi)` brings the density- and thermal-source
  row-3 columns down to about `2.2%` and `1.4%` relative error at
  `rho = 0.5`
- that narrows the remaining fixed-field blocker further: the thermal-source
  normalization is no longer the leading uncertainty on QA, while the
  electric-field column and the uncached QH species-resolved probes are still
  open
- README-level `NTX+NEOPAX` bootstrap-current promotion should wait until this
  fixed-field transport-matrix bridge is tighter

### Fixed-Field Current Benchmark Status

The archive-backed precise-QS fixed-field bootstrap-current comparison now uses:

- the correct archived QA/QH benchmark family,
- the exact literature profile family used in the archived benchmark,
- fresh NTX-to-NEOPAX scan caches that carry `D33_spitzer`,
- and an adaptive `nu_v` support chosen from the actual NEOPAX collisionality
  range.

The default profile family is now the exact literature benchmark used in the
archived precise-QS Redl/SFINCS study:
`n(rho) = 4.13 (1 - rho^{10})` and `T(rho) = 12 (1 - rho^2)` in the archived
normalized units.

Interpolation matters here, so the benchmark now fixes the interpolation story
explicitly:

- SFINCS geometry uses linear interpolation in `s = r_N^2` between neighboring
  VMEC surfaces.
- the fixed-field NTX/NEOPAX comparison now uses the exact literature profile
  family by default instead of reconstructing those profiles from archived
  sampled values, which removes one unnecessary interpolation ambiguity.
- the postprocessing map from the 17-point NTX+NEOPAX radial grid back to the
  archived SFINCS radii is kept as monotone `PCHIP` by default
  (`NTX_FIXED_FIELD_POSTPROCESS_INTERP=pchip`), with a `linear` override kept
  for audit runs.

On the cached fixed-field audit, switching that final postprocessing step from
`PCHIP` to `linear` changes QH negligibly and slightly worsens QA, so `PCHIP`
remains the default. By contrast, forcing NEOPAX's generic `interpax`
interpolators from cubic to linear produces negligible movement in the current
benchmark. A direct coefficient-path audit closes that loop further: the default
NTSS-style `get_Dij` path, direct 3D cubic interpolation, and direct 3D linear
interpolation all reproduce the same cached QA/QH current errors to numerical
precision. The remaining mismatch is therefore not dominated by interpolation
kernel choice; it remains a momentum-correction closure problem.

An additional cached sensitivity probe now narrows that closure problem
further. Scaling the NTX-to-NEOPAX `D13` channel away from the baseline quickly
worsens both QA and QH, while scaling the effective `D33` channel moves the
fixed-field current comparison strongly. That is not a production fix by
itself, but it is a useful diagnostic: the active mismatch is now centered on
how `D33` enters the momentum-correction Sonine system rather than on the
`D13/L31` bridge or on interpolation.

Those corrections remove the main setup ambiguities and fix the VMEC-bridge
bug. They also exposed one wrong local closure change: doubling the
`D13/D33` convolution prefactors broke the shipped W7-X NEOPAX reference
tests, so those prefactors were restored while keeping the lineax
matrix-assembly and non-finite-boundary fixes. With that restoration in place,
the local W7-X no-momentum and momentum-correction reference tests pass again.

A further benchmark-side fix was also required: the NTX VMEC solve must
receive `E_\psi = E_r / transport_psi_scale`, not the DKES/SFINCS bridge
factor `E_r dr/ds`. The active NEOPAX closure returns the corrected parallel
flow itself, not a separate `\Delta U_\parallel`, so the benchmark must form
the current directly from that corrected `U_\parallel`. With that corrected
interpretation, the present archive-backed fixed-field benchmark writes:

- `docs/_static/bootstrap_current_fixed_field_validation.png`
- `docs/_static/bootstrap_current_fixed_field_validation.pdf`
- `docs/_static/bootstrap_current_fixed_field_validation.json`

and now gives current interior max relative errors of:

- QA: `1.66e-1`
- QH: `3.53e-1`

So the archive-backed precise-QS figure remains a status benchmark, not a
closed transferable parity claim. The earlier fitted bridge that closed QA/QH
below `1e-1` on this archive did not transfer to the shipped W7-X workflow and
has been removed from the shipped code path.

The in-tree fixed-field momentum-correction diagnostic now makes that closure
tradeoff explicit on cached QA/QH probes. It records the archived species
currents together with three candidate reconstructions from the solved Sonine
system:

- the regression-consistent `c0` reconstruction,
- the weighted Sonine reconstruction `[1, 0.4, 8/35] \cdot c`,
- and a `c2`-only probe used only for debugging.

With the corrected total-`U_\parallel` semantics, the mapping audit becomes
stricter:

- the baseline `c0` reconstruction still behaves best among the simple
  universal rules, but it leaves fixed-field errors of about `7.6e-1` and
  drives the shipped W7-X ion branch to order `1e1`
- the weighted Sonine reconstruction is worse on both the fixed-field archive
  and the W7-X regression
- least-squares fits trained on the fixed-field archive do not transfer to
  W7-X, and fits trained on W7-X do not close QA/QH

So the remaining mismatch is not fixable by promoting another constant Sonine
weight vector to production. The open lane is now the momentum-correction
closure equations themselves.

The cache-aware raw-branch diagnostic now also dumps the explicit additive
terms returned by the moment-equation correction assembly. On the cached QA
probe at `\rho = 0.5`, those additive terms project to current contributions
that are orders of magnitude smaller than the `O(10^6)` A/m$^2$ species-current
mismatch. So the remaining fixed-field gap is not being driven by a small
missing `add1..add4`-style explicit term. Under the physically consistent raw
normalization, the dominant discrepancy sits in the solved Sonine closure
itself.

Two further closure-side checks now rule out the next obvious shortcuts:

- on the pre-bridge baseline, replacing `D33_spitzer` with raw `D33` in the
  NTX-to-NEOPAX handoff made QA materially worse (`1.66e-1 -> 2.93e-1`) and
  did not improve QH (`3.53e-1 -> 3.55e-1`), so the remaining gap was not
  caused by using the Spitzer-corrected `D33` branch.
- scaling the `E_{ij}` `D33`-driven Sonine sub-block by a single global factor
  also fails as a universal fix: QA prefers the present baseline, while QH only
  improves if that block is amplified. That means the remaining mismatch is not
  a missing scalar prefactor on the `D33` collision-weighted block either.

The closure-side audit still isolates the remaining dominant contribution to the
higher-order `D33` Sonine moments in `L_{43}/L_{34}`, `L_{45}/L_{54}`, and
`L_{55}` rather than to the lower-order `L_{33}` term, interpolation, or a
simple observable remap. However, the literature and source audit matters here:

- Escoto's monoenergetic solver and the upstream source implementation expose
  `D33_spitzer` as a Spitzer-conductivity coefficient, not as a complete
  higher-order momentum-correction closure by itself.
- Taguchi and the Sugama-Nishimura moment-equation papers derive momentum
  restoration through coupled Laguerre/Sonine moment equations, not through
  benchmark-fitted blends of monoenergetic `D33` moments.
- Maaßberg's momentum-correction benchmarks show that energy weighting in the
  source and closure model can matter physically, but that is still a
  closure-model statement, not a license to insert geometry-family-specific
  mixing constants into the production observable path.

For that reason, the previously fitted higher-order `Lij` bridge is now treated
as a rejected audit clue rather than as production physics. A reduced W7-X
transfer audit showed that it substantially worsens the current profile
relative to the same NTX-built baseline, so it has been removed from the public
benchmark path. The open lane remains the momentum-correction closure
equations themselves.

The database-facing normalization is now anchored to the actual consumer path
in NEOPAX's database loader:

- `D11 -> D11 * drds^2`
- `D13 -> D13 * drds`
- `D33 -> nu * D33`

That closes the integrated W7-X handoff, but it also shows that the stronger
precise-QS agreement obtained earlier from a custom `D13` bridge was not
production physics. Under the physically consistent database normalization, the
precise-QS fixed-field benchmark is a closure stress test, not a parity claim.
The regenerated interior maximum relative errors versus archived SFINCS are now:

- QA: `1.16e+0`
- QH: `1.16e+0`

Redl remains at `6.86e-2` on QA and `4.06e-2` on QH on the same family.

## Higher-Order Closure Development Gates

The next closure model is now constrained enough that it should be treated as a
physics implementation project rather than another benchmark-fitting exercise.

The active gates are:

- keep the coefficient-side path fixed:
  - monoenergetic Onsager checks stay closed
  - NTX-to-database normalization stays identical to the validated consumer
    path
- keep the observable map fixed:
  - for the current Sonine basis, `U_parallel = n c_0`
- recover the current three-moment closure exactly as the `P=2` truncation of
  any generalized implementation
- preserve Onsager/ambipolar structure at finite truncation order
- require transfer:
  - improve the precise-QS QA/QH fixed-field benchmark
  - do not regress the integrated W7-X workflow

The first implementation stage of that generalized closure is now in place in
the imported closure stack: the truncation order is configurable, the raw
`D13` source moments and `D33` Hankel sequences are generated for arbitrary
order, and the resulting machinery still recovers the shipped `P=2`
momentum-correction workflow exactly. The remaining missing physics is the
arbitrary-order momentum-conserving collision block, so production runs remain
at `P=2`.

The first implementation step on that lane is now in place in the imported
closure stack: the Sonine basis normalization and source-projection algebra are
generated programmatically and tested against the current three-moment formulas.
That scaffold has now been tightened further: the runtime `P=2` closure can be
reconstructed from generated Sonine coefficients and Hankel moment sequences,
and still passes the shipped W7-X momentum-correction regression. The same is
now true for the low-order momentum-conserving collisional blocks: they can be
generated directly from the standard low-order moment equations, with only the
heat-flow basis sign convention differing from the canonical notation used in
that derivation. So the remaining work is no longer about recovering the
existing algebra. It is about adding physically justified higher-order moments
and collisional couplings on top of an exact and tested `P=2` base.

A dedicated rebuild audit now tests transfer directly:

- `python examples/bootstrap_current_w7x_rebuild_audit.py`

That script rebuilds a NEOPAX-format W7-X database from NTX, then compares:

- the shipped external W7-X database,
- an NTX-rebuilt W7-X database using `d33_mode="raw"`,
- an NTX-rebuilt W7-X database using `d33_mode="spitzer"`,
- an NTX-rebuilt W7-X database using
  `d33_mode="conductivity_difference"`.

On that shipped W7-X momentum-corrected workflow the transfer now closes on the
raw database branch:

- shipped external database: `1.18e-12`
- NTX-rebuilt W7-X, `raw`: `6.58e-6`
- NTX-rebuilt W7-X, `spitzer`: `5.77e-1`
- NTX-rebuilt W7-X, `conductivity_difference`: `2.67e+0`

The sharper reading is now:

- the integrated W7-X mismatch was dominated by the `D13` database handoff, not
  by the direct monoenergetic solve
- the rebuilt W7-X raw branch now reproduces the frozen reference workflow
  tightly
- the conductivity-side `D33_spitzer - D33` interpretation remains a useful
  audit clue on the precise-QS fixed-field archive, but it is not the active
  database-normalization path for the integrated workflow
- the remaining open lane is therefore the precise-QS closure/model gap, not
  the W7-X database handoff or interpolation

The W7-X picture is now more specific than before:

- the full-resolution in-repo W7-X point and subset coefficient tests still
  pass against the shipped external database
- direct solver checks at previously worst coefficient points show that both
  the single-point solve and the scan builder reproduce the frozen benchmark
  table on the reference grid `25x25x63` to about `1e-6` relative error
- the shipped W7-X integrated workflow is now closed on the rebuilt raw branch
- lower-resolution scans are under-resolved, and blindly increasing the grid
  does not reproduce the frozen reference monotonically on every point, so the
  audit is anchored to the reference resolution rather than to a naive
  monotone-refinement assumption

The next closure step has now been tested explicitly as well. A local
`Pmax > 2` branch was built by preserving the present low-order closure and
adding a diagonal Laguerre-tail damping model on the extra moments. That
branch is stable, but it fails the transfer gate:

- `P=2`: imported W7-X closure error `1.17e-12`
- `P=4`: imported W7-X closure error `4.94e-1`
- the same `P=4` run only shifts the precise-QS stress metric from about
  `1.16e+0` to about `1.15e+0`

So the current higher-order tail is not an acceptable production extension. It
does not close the precise-QS closure gap, and it immediately regresses the
already-validated imported W7-X workflow. The committed artifact for that
negative result is:

- `docs/_static/closure_pmax_convergence.json`
- `docs/_static/closure_pmax_convergence.png`
- `docs/_static/closure_pmax_convergence.pdf`

To keep the current closure-model status reproducible as one tracked artifact,
the repository now also builds:

- `docs/_static/closure_validation_report.json`
- `docs/_static/closure_validation_report.txt`
- `docs/_static/closure_validation_report.png`
- `docs/_static/closure_validation_report.pdf`

from:

```bash
python scripts/build_closure_validation_report.py
```

That summary freezes the present interpretation in one place:

- precise-QS Redl vs archived SFINCS passes the independent-code gate
- rebuilt W7-X raw-branch transfer passes the integrated-workflow gate
- fixed-field `NTX+NEOPAX` remains a monitored closure stress test
- the first `Pmax>2` tail model remains rejected because it regresses W7-X

![Fixed-field precise-QS bootstrap-current benchmark](_static/bootstrap_current_fixed_field_validation.png)

### End-To-End Bootstrap-Current Workflow

The pure NTX radial-profile workflow is:

```bash
python examples/bootstrap_current_from_vmec_or_boozmn.py
```

That script demonstrates the direct path from VMEC or Boozer input to radial
profiles of:

- `D11`
- `D13`
- `nu_hat * D33`
- a compact bootstrap-current proxy

The shortest `NTX + NEOPAX` radial-profile workflow is:

```bash
python examples/bootstrap_current_with_neopax.py
```

It writes:

- `docs/_static/bootstrap_current_with_neopax.png`
- `docs/_static/bootstrap_current_with_neopax.pdf`
- `docs/_static/bootstrap_current_with_neopax.json`

![NTX + NEOPAX bootstrap-current profile](_static/bootstrap_current_with_neopax.png)

## CPU And GPU Validation

Run the GPU smoke checks with:

```bash
python -m pytest -m gpu -q
python scripts/run_gpu_regression.py --output-json gpu-smoke-results.json
```

Profile runtime with:

```bash
python scripts/profile_runtime.py --backend cpu --output-json runtime-profile.json
python scripts/profile_runtime.py --backend gpu --output-json runtime-profile-gpu.json
```

Profile the two parallel execution layers with:

```bash
python scripts/profile_parallel_runtime.py --output-json parallel-runtime.json
python scripts/profile_multiprocess_runtime.py --backend cpu --workers 2
python scripts/profile_multiprocess_runtime.py --backend gpu --workers 2
```

For shared GPU systems, use:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Practical Performance Conclusion

The current measured guidance is:

- use serial batched JAX for small and medium studies
- use the multiprocess lane for larger throughput-oriented runs

Details and figures are in [Performance](performance.md).

## NEOPAX Compatibility

NTX-to-NEOPAX compatibility is exercised through:

- `tests/test_neopax_adapter.py`
- `tests/test_neopax_arrays.py`
- `tests/test_neopax_qi.py`

These tests cover:

- HDF5 loading and writing
- pure-array scan mapping
- imported surface scans mapped into NEOPAX normalization
- round-trips through `write_neopax_scan_hdf5(...)`

## Optional External Consistency Studies

When an independent transport workflow such as
[SFINCS-JAX](https://github.com/uwplasma/sfincs_jax) is available in the local
research environment, NTX studies can also be checked against it. Those
comparisons are useful for confidence, but they are not required to run NTX or
to understand the code.
