# Validation

NTX ships with three complementary validation paths:

1. unit and regression tests in `tests/`
2. GPU smoke and regression checks through `tests/test_gpu_smoke.py` and
   `scripts/run_gpu_regression.py`
3. archived cross-code comparison scripts against DKES and SFINCS benchmark
   tables

## VMEC Validation

The VMEC path is covered by:

- loader and normalization tests in `tests/test_vmec.py`
- physics checks in `tests/test_vmec_physics.py`
- QI-specific checks in `tests/test_vmec_qi.py`
- regression snapshots in `tests/test_vmec_regression.py`
- scan coverage in `tests/test_vmec_scan.py`
- `vmec_jax -> booz_xform_jax -> NTX` convergence checks in
  `tests/test_vmec_jax_backend.py`
- reference VMEC geometry, single-point transport, subset database, and script
  coverage in `tests/test_reference_vmec.py`
- standalone executable-comparison coverage through
  `scripts/compare_reference_executable.py`
- standalone `sfincs_jax` geometry comparison coverage through
  `scripts/compare_sfincs_geometry.py` and `tests/test_sfincs_vmec_geometry.py`
- imported NEOPAX array-path coverage in `tests/test_neopax_arrays.py`
- QI imported NEOPAX HDF5 round-trip coverage in `tests/test_neopax_qi.py`

The scan tests cover both:

- loop-versus-scan agreement
- `er_hat` versus explicit `epsi_hat` agreement on VMEC surfaces

The VMEC loader also distinguishes between:

- `vmec_nyquist_option = 1`: reduced VMEC spectral set
- `vmec_nyquist_option = 2`: full Nyquist mode set
- `vmec_mode_convention = "reduced"`: reduced `(xm, xn)` mode-table convention
- `vmec_mode_convention = "filtered_nyquist"`: filtered Nyquist convention used
  in SFINCS-style VMEC geometry paths

This distinction is covered in `tests/test_vmec.py` so changes in VMEC mode
selection do not silently alter the active spectral set.

The file-backed VMEC and Boozer loaders are now explicitly JAX-backed:

- `load_vmec_surface(...)` reads `wout` files through `vmec_jax`
- `load_boozmn_surface(...)` reads `boozmn` files through `booz_xform_jax`

The imported `surface_from_vmec_jax_wout(...)` builder now applies the same
right-handed Boozer handedness convention as `load_boozmn_surface(...)`. On the
local W7-X `boozmn` reference, the resulting transport coefficients now agree
within about `2%` on the `13 x 17 x 16` regression grid used in
`tests/test_vmec_jax_backend.py`. That local regression now covers two
operating points, `(\nu_hat, \epsilon_\psi) = (1e-4, 0)` and
`(1e-3, 1e-3)`, so the Boozer-transform lane is treated as its own validated
workflow rather than as an implied NEOPAX validation path.

For NEOPAX-facing W7-X VMEC scans, the intended imported JAX validation path is now
the direct VMEC-harmonic builder:

- `surface_from_vmec_jax_vmec_wout(...)`
- `surface_from_vmec_jax_vmec_wout_file(...)`

That path reads the `wout` through `vmec_jax`, uses the same half-grid
interpolation and VMEC sign conventions as the validated W7-X reference lane,
and avoids introducing an extra Boozer-transform interpretation step into the
NEOPAX subset comparison.

Direct VMEC comparison against the external validation executable is now a
parity check for the reduced-mode VMEC convention. The current W7-X
reduced-mode VMEC example closes to roundoff against that external lane.

For the W7-X NEOPAX VMEC database specifically, NTX now also has a
comparison-only reference lane aligned with the validated W7-X reference
database workflow:

- `load_vmec_surface_reference(...)`
- `vmec_reference_factors(...)`
- `build_reference_vmec_scan(...)`
- `examples/DKES_like_database/Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py`

That path uses the same VMEC sign convention, the same Boozer-side electric
field conversion factors from the `boozmn` file, and the same Legendre
resolution convention with `nl -> n_xi = nl - 1`.

On the local W7-X subset used in the regression tests, this path matches the
existing NEOPAX reference HDF5 to better than `1e-2` relative error for
`D11`, `D13`, `D31`, and `D33`.

The new direct `vmec_jax` VMEC-harmonic helper is tested separately in
`tests/test_vmec_jax_vmec.py`, where it matches the validated reference VMEC
surface to roundoff at the single-surface solve level.

NTX now also carries three additional omnigenous VMEC families from the local
`omnigenity_optimization` study set:

- QA and QH, each with both `wout` and `boozmn` fixtures
- QI, with a `wout` fixture and imported NEOPAX-array scan coverage

Those fixtures are covered in `tests/test_omnigenity_cases.py`. On the current
`13 x 17 x 16` transform regression grid, the fully JAX
`vmec_jax -> booz_xform_jax -> NTX` lane stays within about `6.3%` on QA and
about `3.5%` on QH relative to the file-backed Boozer transport reference. The
omnigenous QI `wout` fixture also closes to roundoff against the
comparison-reference VMEC-harmonic loader and produces finite imported
NEOPAX-array scans.

The imported NEOPAX mapping now also has a pure-array lane:

- `build_ntx_neopax_scan_from_surfaces(...)` for explicit in-memory surfaces
- `scan_to_neopax_arrays(...)` for JAX-friendly NEOPAX normalization

That lane is covered by differentiability tests on the scan electric field and
by a QI VMEC round-trip test that writes and reloads a NEOPAX-style HDF5 file
without depending on an external reference database.

NTX now also vendors external-reference subsets for the same omnigenous
families:

- `tests/fixtures/benchmarks/omnigenity/external_reference_qa_subset.h5`
- `tests/fixtures/benchmarks/omnigenity/external_reference_qh_subset.h5`
- `tests/fixtures/benchmarks/omnigenity/external_reference_qi_subset.h5`

Generate or refresh them with:

```bash
python scripts/generate_reference_omnigenity_subsets.py
```

These subset databases are compared directly against
`build_reference_vmec_scan(...)` in
`tests/test_reference_omnigenity.py`.

Current interpretation:

- QA: closes to within about `2e-2` relative error on the external subset
- QH: closes to within about `3e-2` relative error on the external subset
- QI: closes on the vendored `rho = 0.25, 0.5` subset after switching the
  comparison-only VMEC reference loader to the same cubic half-grid
  interpolation used by the reference workflow

For the `sfincs_jax` geometry comparison, NTX applies a toroidal-angle
convention conversion before comparing arrays:

- reverse the sampled zeta direction
- flip the sign of the Jacobian reconstructed from `sfincs_jax`

With that conversion, the snapped-surface filtered-Nyquist W7-X geometry arrays
match `sfincs_jax` to roundoff.

## Archived Cross-Code Comparisons

Run:

```bash
python scripts/compare_archived_benchmarks.py --output-json archived-benchmarks.json
```

Or restrict the report to one benchmark family:

```bash
python scripts/compare_archived_benchmarks.py --case W7X-EIM
python scripts/compare_archived_benchmarks.py --case W7X-KJM
python scripts/compare_archived_benchmarks.py --case CIEMAT-QI
```

This script defaults to `JAX_PLATFORM_NAME=cpu` so the archived comparison does
not depend on accelerator FFT behavior.

This script evaluates vendored archived thesis benchmark tables for:

- W7-X EIM
- W7-X KJM
- CIEMAT-QI

and reports:

- the NTX coefficients at the chosen benchmark grid
- archived DKES and SFINCS coefficients
- archived monoenergetic reference coefficients when an exact grid-matched
  reference is vendored
- relative errors for `D11`, `D31`, and when available `D33`

Current interpretation:

- W7-X EIM: NTX matches the archived monoenergetic reference at the
  `23 x 55 x 80` grid used by the thesis convergence study, while still showing
  the expected spread against the archived DKES and SFINCS curves.
- W7-X KJM: NTX matches the archived monoenergetic reference at the
  `19 x 79 x 180` benchmark grid, again with visible cross-code spread against
  DKES and SFINCS.
- CIEMAT-QI: NTX now matches the archived monoenergetic reference at the exact
  `47 x 215 x 160` grid to roundoff. The remaining spread is genuinely against
  the archived DKES and SFINCS tables, not against Escoto's monoenergetic
  reference solve.

## Runtime Profiling

Run:

```bash
python scripts/profile_runtime.py --backend cpu --output-json runtime-profile.json
```

Choose `--backend gpu` when you want a GPU-specific profile.

This profiles the batched scan path against a Python loop for one DKES case and
one VMEC case, and writes:

- backend and device information
- scan compile-and-run time
- steady-state scan time
- loop time
- scan-versus-loop speedup

The current scan implementation uses a jitted batched kernel, which is the main
throughput path for parameter scans on both CPU and GPU.

Final office hardware validation on 2026-04-10 used the updated GPU entrypoints
with `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid shared-device allocation
failures. The resulting office GPU checks closed with:

- `pytest -m gpu -q tests/test_gpu_smoke.py` -> `2 passed`
- `scripts/run_gpu_regression.py`:
  - DKES smoke steady time: `0.0553 s`, max relative error `9.44e-09`
  - VMEC smoke steady time: `0.0563 s`, max relative error `1.03e-12`
- `scripts/profile_runtime.py --backend cpu`:
  - DKES 8-case scan steady time: `0.979 s`
  - VMEC 8-case scan steady time: `1.130 s`
- `scripts/profile_runtime.py --backend gpu`:
  - DKES 8-case scan steady time: `1.566 s`
  - VMEC 8-case scan steady time: `1.280 s`

Interpretation:

- the GPU lane is numerically correct on real hardware
- the small smoke cases remain CPU-favorable because launch and transfer
  overheads dominate
- the batched GPU scan path is stable and useful, but the current smoke grids
  are too small to make GPU wall time beat CPU wall time

NTX also now exposes a prepared-system path that caches the geometry and
derivative blocks for repeated solves. On a local W7-X sample solve at
`9 x 11 x 8`, the cached path reduced the steady repeated-solve wall time by
about `1.34x` relative to rebuilding the geometry and derivative blocks for
every solve.

## Runtime Measurements

Run:

```bash
python scripts/benchmark_reference_executable.py --case w7x_eim_er0 --platform cpu
python scripts/benchmark_reference_executable.py --case w7x_eim_er0 --platform gpu --disable-preallocate
python scripts/benchmark_reference_executable.py --case w7x_eim_er0 --platform cpu --mode compiled --skip-reference
```

The runtime harness defaults to `--mode eager`. Use `--mode compiled` only when
you explicitly want to measure the jitted prepared-solver path on a fixed
surface and grid.

On `office` on 2026-04-08, for the W7-X EIM `23 x 55 x 80` case at
`nu_hat = 1e-5`, `er_hat = 0`:

- CPU, same host:
  - external reference wall time: about `4.29 s`
  - NTX first run: about `13.53 s`
  - NTX steady run: about `8.99 s`
  - NTX/reference steady runtime ratio: about `2.10x`
  - NTX/reference RSS ratio: about `10.66x`
- GPU, same host, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`:
  - external reference wall time: about `5.00 s`
  - NTX first run: about `10.70 s`
  - NTX steady run: about `4.37 s`
  - NTX/reference steady runtime ratio: about `0.87x`
  - sampled NTX GPU memory: about `740 MiB`

On the local macOS CPU run on 2026-04-08, the same W7-X EIM case showed the
tradeoff between the eager and compiled prepared-solver paths:

- eager mode:
  - first run: about `8.54 s`
  - steady run: about `6.44 s`
- compiled mode:
  - compile plus first run: about `6.82 s`
  - steady run: about `5.71 s`

This is a real improvement, but not a dramatic one for the heavy
`23 x 55 x 80` CPU case. On the `office` CPU host, an attempted repeated
compiled run for that same heavy case took long enough to be operationally
unattractive for the runtime workflow, so the default harness remains eager
mode for same-host comparisons.

The benchmark script test path no longer exercises the heavy W7-X EIM
production case. It now uses a dedicated `w7x_eim_smoke` case (`5 x 5 x 4`) so
CI and office validation keep the CLI coverage without spending minutes inside
the runtime checks.

One CPU tuning attempt that did **not** help:

- forcing single-thread CPU execution with
  `OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`
  made NTX slower on `office` and did not materially reduce RSS.
