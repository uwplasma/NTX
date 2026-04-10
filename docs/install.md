# Install

## From Source

Runtime install:

```bash
python -m pip install -e .
```

Installed package entrypoints:

```bash
ntx --help
python -m ntx --help
```

## From A Built Distribution

Build wheel and sdist:

```bash
python -m build
```

Install the built wheel:

```bash
python -m pip install dist/*.whl
```

Development install:

```bash
python -m pip install -e ".[dev,docs,io]"
```

Install the JAX geometry backends:

```bash
python -m pip install -e ".[geometry]"
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
- `.[io]` for HDF5-based integrations
- `.[geometry]` for `vmec_jax` and `booz_xform_jax`

## Verification

```bash
ruff check .
mypy src/ntx
pytest -q
sphinx-build -b html docs docs/_build/html
python -m build
python -m twine check dist/*
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
cd /path/to/NTX
python -m pip install -e ".[dev,docs,io]"
scripts/sh_office_gpu_smoke.sh
```

This writes `gpu-smoke-results.json` in the repository root with device
information, timings, coefficient deltas, and regression summaries.

The supported VMEC and Boozer file readers are the JAX implementations. NTX
does not depend on the original VMEC or BOOZ_XFORM executables for those file
paths.

For local performance profiling on either CPU or GPU:

```bash
python scripts/profile_runtime.py --output-json runtime-profile.json
```

This script defaults to `JAX_PLATFORM_NAME=cpu`. Set `JAX_PLATFORM_NAME=gpu`
explicitly if you want to profile the GPU backend instead.
