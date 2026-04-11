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

## Current Hardware Interpretation

The current GPU lane is numerically stable and validated on office hardware.
For the small repository smoke cases, CPU remains faster in steady-state wall
time. That is expected: these grids are small enough that GPU launch and
transfer overheads dominate.
