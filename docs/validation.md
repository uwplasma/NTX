# Validation

NTX validation is organized around three layers:

1. unit and regression tests in `tests/`
2. file-loader and CLI checks using NTX-owned synthetic fixtures
3. GPU smoke and runtime checks for CPU/GPU execution stability

## What Is Covered

The default test suite covers:

- Fourier-series evaluation and flux-surface averages
- operator assembly and nullspace handling
- dense block-tridiagonal solves
- autodiff inverse and profile-analysis helper workflows
- DKES-style, magnetic-configuration, VMEC, and Boozer file loaders
- TOML input parsing and `.npz` output writing
- imported NEOPAX-array and HDF5 mapping helpers
- `vmec_jax` and `booz_xform_jax` integration points
- serial versus device-parallel scan equivalence

## Current Local Status

Latest local suite:

- `94 passed, 2 skipped`

The two skipped tests are the GPU smoke tests on non-GPU machines.

## GPU Validation

Run:

```bash
python -m pytest -m gpu -q
python scripts/run_gpu_regression.py --output-json gpu-smoke-results.json
```

The GPU regression script reports:

- active backend and devices
- first-run compile time
- steady-state solve time
- coefficient deltas against NTX-owned smoke references

For office hardware runs, set:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The NTX GPU entrypoints already default this internally to avoid shared-device
allocation failures.

## Runtime Profiling

Run:

```bash
python scripts/profile_runtime.py --backend cpu --output-json runtime-profile.json
python scripts/profile_runtime.py --backend gpu --output-json runtime-profile-gpu.json
```

This profiles batched scans against a Python loop for one DKES-style case and
one VMEC case using the NTX-owned sample fixtures.

To profile device-parallel scans:

```bash
python scripts/profile_parallel_runtime.py --output-json parallel-runtime.json
```

On a local workstation, host-device emulation can be forced in a fresh process:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 python scripts/profile_parallel_runtime.py
```

On office hardware, the profiler reports both visible GPUs and the subset that
passes an NTX smoke solve. Under the current office stack, `cuda:1` is visible
to JAX but does not pass the NTX dense-solve smoke check, so the guarded
parallel helper excludes it automatically.

## NEOPAX Compatibility

NEOPAX compatibility is validated through:

- `tests/test_neopax_adapter.py`
- `tests/test_neopax_arrays.py`
- `tests/test_neopax_qi.py`

Those tests check:

- NEOPAX-style HDF5 loading
- pure-array scan mapping
- imported surface scans mapped into NEOPAX normalization
- HDF5 round-trips through `write_neopax_scan_hdf5(...)`
