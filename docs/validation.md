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
- Eduardo-REFERENCE_EXECUTABLE-reference VMEC geometry, single-point transport, subset
  database, and script coverage in `tests/test_reference_executable_reference_vmec.py`
- standalone REFERENCE_EXECUTABLE comparison coverage through `scripts/compare_reference_executable.py`
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
`tests/test_vmec_jax_backend.py`.

For NEOPAX-facing W7-X VMEC scans, the intended imported JAX parity path is now
the direct VMEC-harmonic builder:

- `surface_from_vmec_jax_vmec_wout(...)`
- `surface_from_vmec_jax_vmec_wout_file(...)`

That path reads the `wout` through `vmec_jax`, uses the same half-grid
interpolation and VMEC sign conventions as the validated W7-X reference lane,
and avoids introducing an extra Boozer-transform interpretation step into the
NEOPAX subset comparison.

Direct VMEC comparison against REFERENCE_EXECUTABLE is now a parity check for the reduced-mode
VMEC convention. The current W7-X reduced-mode VMEC example closes to roundoff
against the live REFERENCE_EXECUTABLE executable.

For the W7-X NEOPAX VMEC database specifically, NTX now also has a
comparison-only reference lane that mirrors Eduardo Neto's `vmec_neopax`
workflow:

- `load_vmec_surface_reference_executable_reference(...)`
- `reference_executable_vmec_factors(...)`
- `build_reference_executable_reference_vmec_scan(...)`
- `examples/DKES_like_database/Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py`

That path uses the same VMEC sign convention as `Field.from_vmec_s(...)`, the
same Boozer-side electric-field conversion factors from the `boozmn` file, and
the same Legendre resolution convention with `nl -> n_xi = nl - 1`.

On the local W7-X subset used in the regression tests, this path matches the
existing NEOPAX reference HDF5 to better than `1e-2` relative error for
`D11`, `D13`, `D31`, and `D33`.

The new direct `vmec_jax` VMEC-harmonic helper is tested separately in
`tests/test_vmec_jax_vmec.py`, where it matches the validated reference VMEC
surface to roundoff at the single-surface solve level.

The imported NEOPAX mapping now also has a pure-array lane:

- `build_ntx_neopax_scan_from_surfaces(...)` for explicit in-memory surfaces
- `scan_to_neopax_arrays(...)` for JAX-friendly NEOPAX normalization

That lane is covered by differentiability tests on the scan electric field and
by a QI VMEC round-trip test that writes and reloads a NEOPAX-style HDF5 file
without depending on an external reference database.

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
- CIEMAT-QI: the archived DKES, SFINCS, and monoenergetic tables are vendored
  and parsed, but the exact `47 x 215 x 160` NTX solve is substantially heavier
  than the W7-X cases, so it is best run selectively with `--case CIEMAT-QI`
  rather than treated as a default smoke check.

## Runtime Profiling

Run:

```bash
python scripts/profile_runtime.py --output-json runtime-profile.json
```

This script also defaults to `JAX_PLATFORM_NAME=cpu`. Override the environment
when you want a GPU-specific profile.

This profiles the batched scan path against a Python loop for one DKES case and
one VMEC case, and writes:

- backend and device information
- scan compile-and-run time
- steady-state scan time
- loop time
- scan-versus-loop speedup

The current scan implementation uses a jitted batched kernel, which is the main
throughput path for parameter scans on both CPU and GPU.

NTX also now exposes a prepared-system path that caches the geometry and
derivative blocks for repeated solves. On a local W7-X sample solve at
`9 x 11 x 8`, the cached path reduced the steady repeated-solve wall time by
about `1.34x` relative to rebuilding the geometry and derivative blocks for
every solve.

## Runtime Benchmarks

Run:

```bash
python scripts/benchmark_against_reference_executable.py --case w7x_eim_er0 --platform cpu
python scripts/benchmark_against_reference_executable.py --case w7x_eim_er0 --platform gpu --disable-preallocate
python scripts/benchmark_against_reference_executable.py --case w7x_eim_er0 --platform cpu --mode compiled --skip-reference_executable
```

The benchmark harness defaults to `--mode eager`. Use `--mode compiled` only
when you explicitly want to measure the jitted prepared-solver path on a fixed
surface and grid.

On `office` on 2026-04-08, for the W7-X EIM `23 x 55 x 80` case at
`nu_hat = 1e-5`, `er_hat = 0`:

- CPU, same host:
  - REFERENCE_EXECUTABLE wall time: about `4.29 s`
  - NTX first run: about `13.53 s`
  - NTX steady run: about `8.99 s`
  - NTX/REFERENCE_EXECUTABLE steady runtime ratio: about `2.10x`
  - NTX/REFERENCE_EXECUTABLE RSS ratio: about `10.66x`
- GPU, same host, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`:
  - REFERENCE_EXECUTABLE wall time: about `5.00 s`
  - NTX first run: about `10.70 s`
  - NTX steady run: about `4.37 s`
  - NTX/REFERENCE_EXECUTABLE steady runtime ratio: about `0.87x`
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
compiled benchmark run for that same heavy case took long enough to be
operationally unattractive for the benchmark workflow, so the default harness
remains eager-mode for same-host comparisons.

The benchmark script test path no longer exercises the heavy W7-X EIM production
case. It now uses a dedicated `w7x_eim_smoke` benchmark case (`5 x 5 x 4`) so
CI and office validation keep the CLI coverage without spending minutes inside a
runtime benchmark.

One CPU tuning attempt that did **not** help:

- forcing single-thread CPU execution with
  `OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`
  made NTX slower on `office` and did not materially reduce RSS.
