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
For CI or quick local smoke checks, use
`--num-cases 2 --grid 5,5,4` to keep the serial/device-parallel equality check
fast while preserving the default profiling behavior for real measurements.

The helper now performs an NTX smoke check on local devices before using them.
If a visible device fails that check, it is excluded from the parallel solve
instead of silently returning bad coefficients.

NTX also provides a separate multiprocess path:

```bash
python scripts/profile_multiprocess_runtime.py --backend gpu --workers 2
```

That path runs one Python worker per GPU with process-local
`CUDA_VISIBLE_DEVICES` pinning. It is the current robust route for office
hardware because it avoids the single-process cuSolver failure mode seen on
`cuda:1`.

## Current Hardware Interpretation

The current GPU lane is numerically stable and validated on office hardware.
For the small repository smoke cases, CPU remains faster in steady-state wall
time. That is expected: these grids are small enough that GPU launch and
transfer overheads dominate.

For the single-process profiler on office:

- JAX sees two GPUs
- only one passes the NTX dense-solve smoke check under the current stack
- the guarded parallel path therefore runs on the healthy subset and preserves
  correct coefficients

For the multiprocess profiler on office:

- both GPUs execute correctly when pinned to separate worker processes
- coefficient deltas are zero at the repository smoke-case tolerance
- wall time is still worse than the serial batched solve for the small smoke
  grids because process launch and per-worker compilation dominate

So the current guidance is:

- use the serial batched JAX scan for small and medium studies
- use the guarded single-process path only when all visible devices are healthy
- use the multiprocess path for larger multi-GPU throughput workloads or for
  platforms that need strict one-process-per-GPU isolation

The current scaling figures and JSON payloads are documented in
[`Performance`](performance.md).
