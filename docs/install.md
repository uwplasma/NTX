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
