# Validation

NTX validation is organized around four layers:

1. unit and regression tests in `tests/`
2. workflow and convergence examples in `examples/`
3. CPU/GPU runtime checks
4. downstream database and profile checks through NEOPAX

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
- the exact archived profile values,
- archive-driven Hermite reconstruction in `rho`,
- and an adaptive `nu_v` support chosen from the actual NEOPAX collisionality
  range.

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

and currently gives current interior max relative errors of about:

- QA: `5.14e-1`
- QH: `6.15e-1`

The remaining fixed-field bootstrap-current gap is therefore no longer a
benchmark-family, Redl, `nu_v`-axis, or VMEC solve-input issue. It is now
best treated as a benchmark-specific thermal/current-closure problem.

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
