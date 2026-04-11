# NTX

NTX is a JAX-native monoenergetic neoclassical transport solver for stellarator
flux surfaces. It implements the Legendre-space block-tridiagonal formulation
described in Javier Escoto's PhD thesis:
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

NTX has two intended usage lanes:

- a fast command-line workflow for file-driven runs
- an imported Python workflow for batched scans, NEOPAX coupling, and
  differentiable JAX pipelines

## Install

Basic install:

```bash
python -m pip install ntx
```

Editable local install:

```bash
python -m pip install -e ".[dev,docs,io]"
```

To use VMEC and Boozer JAX-backed geometry helpers:

```bash
python -m pip install -e ".[geometry]"
```

## Quick Start

CLI:

```bash
ntx examples/example_surface.toml
ntx examples/sample_dkes.toml
ntx examples/sample_vmec.toml
```

Python:

```python
from ntx import GridSpec, MonoenergeticCase, example_surface, solve_monoenergetic

surface = example_surface()
grid = GridSpec(n_theta=9, n_zeta=11, n_xi=12)
case = MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0)
result = solve_monoenergetic(surface, grid, case)

print(result.D11, result.D13, result.D33)
```

## Supported Inputs

- built-in analytic sample surface
- DKES-style `ddkes2.data` Boozer harmonics
- text magnetic-configuration Fourier tables
- VMEC `wout` files through `vmec_jax`
- Boozer `boozmn` files through `booz_xform_jax`

## Outputs

`ntx input.toml` prints a Rich terminal summary and writes a compressed `.npz`
payload containing:

- solved transport coefficients
- residual and Onsager diagnostics
- resolved electric-field normalization
- surface metadata
- geometry metadata
- optional low-order Legendre modes

## NEOPAX

NTX includes direct mapping helpers for NEOPAX-style monoenergetic databases:

- `build_ntx_neopax_scan(...)`
- `build_ntx_neopax_scan_from_surfaces(...)`
- `scan_to_neopax_arrays(...)`
- `to_neopax_monoenergetic(...)`
- `write_neopax_scan_hdf5(...)`

The example script:

```bash
python examples/neopax_with_ntx.py
```

builds a small VMEC scan and maps it into NEOPAX-style arrays.

Autodiff examples:

```bash
python examples/autodiff_inverse_problem.py
python examples/neopax_autodiff_profiles.py
```

These write polished figures into `docs/_static/` and demonstrate:

- parameter recovery from synthetic transport data
- profile sensitivity and inversion on NEOPAX-style monoenergetic arrays
- both PNG and PDF figure export for manuscript-ready workflows

## Validation

The repository validation now stays entirely inside NTX-owned fixtures plus
optional NEOPAX HDF5 compatibility checks.

Current local status:

- `101 passed, 2 skipped`
- GPU smoke tests are skipped on non-GPU machines
- office GPU hardware validation is documented in `docs/gpu.md`

## Documentation

- [Install](docs/install.md)
- [Input File](docs/input-file.md)
- [Algorithm](docs/algorithm.md)
- [Autodiff](docs/autodiff.md)
- [Examples](docs/examples.md)
- [Validation](docs/validation.md)
- [NEOPAX](docs/neopax.md)
- [GPU](docs/gpu.md)
- [Release](docs/release.md)
