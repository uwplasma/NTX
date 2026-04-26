# Installation

## Basic Install

Install NTX from PyPI:

```bash
pip install ntx
```

This is enough for:

- `ntx input.toml`
- Python imports for the core solver
- bundled analytic and DKES-style examples

## Development Install

For tests, linting, and docs:

```bash
pip install -e ".[dev,docs,io]"
```

## Geometry Extras

To use VMEC `wout` files and Boozer `boozmn` files through the JAX geometry
helpers, install the optional upstream geometry packages directly:

```bash
pip install git+https://github.com/uwplasma/vmec_jax.git
pip install git+https://github.com/uwplasma/booz_xform_jax.git
```

Those packages provide the dependencies used by:

- [`src/ntx/vmec.py`](../src/ntx/vmec.py)
- [`src/ntx/booz.py`](../src/ntx/booz.py)

## Optional Runtime Extras

### NEOPAX coupling

Install [NEOPAX](https://github.com/uwplasma/NEOPAX) in the active
environment if you want to convert NTX scan data directly into NEOPAX
monoenergetic objects.

### GPU execution

Install a JAX build compatible with your accelerator stack. NTX itself does not
pin a GPU-specific JAX wheel.

## First Run

After installation:

```bash
ntx examples/example_surface.toml --plot
```

That should:

1. print a Rich run summary
2. solve one sample monoenergetic problem
3. write `examples/outputs/example_surface.nc`
4. write `examples/outputs/example_surface.pdf`

Then inspect the result:

```bash
python examples/plot_output_file.py examples/outputs/example_surface.nc
```

## Local Quality Checks

Run the standard local checks:

```bash
python -m ruff check .
python -m mypy src/ntx
python -m pytest -q
python -m sphinx -b html docs docs/_build/html
```

## Build A Distribution

```bash
python -m build
python -m twine check dist/*
```

The release workflow is documented in [Release](release.md).
