# GPU Runs

NTX uses the same JAX solver path on CPU and GPU.

## GPU Test Targets

- `tests/test_gpu_smoke.py`
- `scripts/run_gpu_regression.py`
- `scripts/sh_office_gpu_smoke.sh`

## Typical Session

```bash
sh office
cd /path/to/NTX
python -m pip install -e ".[dev,docs,io]"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
scripts/sh_office_gpu_smoke.sh
```

## What The Regression Script Reports

- backend and visible devices
- compile-plus-first-run timing
- steady-state timing
- solved coefficients
- max relative error against NTX-owned smoke references

## Device-Parallel Scans

For larger scans, NTX also exposes a device-parallel scan path through
`solve_monoenergetic_parallel_scan(...)` and the profiling helper:

```bash
python scripts/profile_parallel_runtime.py --output-json parallel-runtime.json
```

This is intended for multi-device CPU or GPU jobs when scan throughput matters
more than single-case latency.

The helper now performs an NTX smoke check on local devices before using them.
If a visible device fails that check, it is excluded from the parallel solve
instead of silently returning bad coefficients.

## Current Hardware Interpretation

The current GPU lane is numerically stable and validated on office hardware.
For the small repository smoke cases, CPU remains faster in steady-state wall
time. That is expected: these grids are small enough that GPU launch and
transfer overheads dominate.

For the new parallel profiler on office:

- JAX sees two GPUs
- only one passes the NTX dense-solve smoke check under the current stack
- the guarded parallel path therefore runs on the healthy subset and preserves
  correct coefficients
