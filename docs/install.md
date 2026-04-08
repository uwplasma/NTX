# Install

## From Source

Runtime install:

```bash
python -m pip install -e .
```

Development install:

```bash
python -m pip install -e ".[dev,docs,io]"
```

## Dependencies

Core runtime:

- `jax`
- `jaxlib`
- `numpy`
- `scipy`
- `rich`

Optional extras:

- `.[dev]` for `pytest`, `ruff`, and `mypy`
- `.[docs]` for Sphinx docs
- `.[io]` for additional NetCDF / xarray tooling

## Verification

```bash
ruff check .
mypy src/ntx
pytest -q
sphinx-build -b html docs docs/_build/html
```

## CPU And GPU

NTX uses JAX arrays throughout, so the same solver path can run on CPU or GPU.

For production physics runs, keep `x64 = true` in the input file unless you are
deliberately testing reduced precision.

The repository CPU workflow runs:

```bash
pytest -m "not gpu"
```

The GitHub Actions CPU matrix covers Python `3.10`, `3.11`, and `3.12`.

GPU smoke and regression coverage is provided through:

- `tests/test_gpu_smoke.py`
- `scripts/run_gpu_regression.py`
- `scripts/sh_office_gpu_smoke.sh`

A typical GPU session in the office environment is:

```bash
sh office
cd /Users/rogeriojorge/local/.NTX
python -m pip install -e ".[dev,docs,io]"
scripts/sh_office_gpu_smoke.sh
```

This writes `gpu-smoke-results.json` in the repository root with device
information, timings, coefficient deltas, and regression summaries.

For local performance profiling on either CPU or GPU:

```bash
python scripts/profile_runtime.py --output-json runtime-profile.json
```
