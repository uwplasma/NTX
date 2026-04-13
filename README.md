# NTX

NTX is a JAX-native monoenergetic neoclassical transport solver for stellarator
flux surfaces. It implements the Legendre-space formulation in Javier Escoto's
PhD thesis: [arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

NTX has two main lanes:

- a file-driven CLI for fast production runs
- an imported Python/JAX lane for scans, autodiff, NEOPAX coupling, and
  parallel throughput workflows

## Fastest Start

Install the package:

```bash
python -m pip install ntx
```

Run the smallest bundled case:

```bash
ntx examples/example_surface.toml
```

That command:

- prints a Rich summary to the terminal
- solves one monoenergetic transport case
- writes a compressed `.npz` output file next to the input TOML

Run a slightly more realistic DKES-style example:

```bash
ntx examples/sample_dkes.toml
```

Inspect the `.npz` output with the plotting example:

```bash
python examples/plot_output_npz.py examples/outputs/sample_dkes.npz
```

This writes:

- `docs/_static/output_file_summary.png`
- `docs/_static/output_file_summary.pdf`

## Installation

Basic install:

```bash
python -m pip install ntx
```

Local editable install with test and docs tools:

```bash
python -m pip install -e ".[dev,docs,io]"
```

Install the JAX VMEC and Boozer geometry helpers:

```bash
python -m pip install -e ".[geometry]"
```

## Main Inputs

The primary user entrypoint is:

```bash
ntx input.toml
```

Each input file defines:

- a surface source
- a spectral/angular grid
- one monoenergetic case
- output and logging options

Bundled surface sources:

- built-in analytic example
- DKES-style `ddkes2.data`
- VMEC `wout_*.nc`
- Boozer `boozmn` files through the Python/JAX geometry helpers

See [docs/input-file.md](docs/input-file.md) for the complete schema.

## Main Outputs

Every CLI run writes a compressed `.npz` file containing:

- `D11`, `D31`, `D13`, `D33`, `D33_spitzer`
- residual and Onsager diagnostics
- resolved electric-field normalization
- angular geometry arrays such as `theta_grid`, `zeta_grid`, `b`,
  `radial_drift_spatial`
- optional low-order Legendre modes
- serialized run metadata and input text

The plotting helper:

```bash
python examples/plot_output_npz.py path/to/output.npz
```

turns that payload into a four-panel publication-style summary figure.

## Ways To Run NTX

### 1. CLI solve from TOML

```bash
ntx examples/sample_dkes.toml
ntx examples/sample_vmec.toml
```

### 2. Python single-case solve

```python
from ntx import GridSpec, MonoenergeticCase, example_surface, solve_monoenergetic

surface = example_surface()
grid = GridSpec(n_theta=9, n_zeta=11, n_xi=12)
case = MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0)
result = solve_monoenergetic(surface, grid, case)

print(result.D11, result.D13, result.D33)
```

### 3. Batched JAX scans

```python
import jax.numpy as jnp
from ntx import GridSpec, example_surface, solve_monoenergetic_scan

surface = example_surface()
grid = GridSpec(9, 11, 8)
nu_hat = jnp.logspace(-4, -2, 8)
epsi_hat = jnp.zeros_like(nu_hat)
coefficients = solve_monoenergetic_scan(surface, grid, nu_hat, epsi_hat=epsi_hat)
```

### 4. Multiprocess throughput scans

```python
import jax.numpy as jnp
from ntx import GridSpec, example_surface, solve_monoenergetic_multiprocess_scan

surface = example_surface()
grid = GridSpec(9, 11, 8)
nu_hat = jnp.logspace(-4, -2, 32)
epsi_hat = jnp.zeros_like(nu_hat)
coefficients = solve_monoenergetic_multiprocess_scan(
    surface,
    grid,
    nu_hat,
    epsi_hat=epsi_hat,
    backend="cpu",
    workers=4,
)
```

### 5. NEOPAX coupling

NTX exposes direct monoenergetic scan builders and mapping helpers for
NEOPAX-style databases:

- `build_ntx_neopax_scan(...)`
- `build_ntx_neopax_scan_from_surfaces(...)`
- `scan_to_neopax_arrays(...)`
- `to_neopax_monoenergetic(...)`
- `write_neopax_scan_hdf5(...)`

Minimal example:

```bash
python examples/neopax_with_ntx.py
```

Bootstrap-current example from VMEC or direct Boozer input:

```bash
python examples/bootstrap_current_from_vmec_or_boozmn.py
```

When both `wout` and `boozmn` are available, the example uses the direct
Boozer path by default because that is currently the tighter W7-X
bootstrap-current comparison lane.

The example writes:

- `docs/_static/bootstrap_current_from_vmec_or_boozmn.png`
- `docs/_static/bootstrap_current_from_vmec_or_boozmn.pdf`
- `docs/_static/bootstrap_current_from_vmec_or_boozmn.json`

![W7-X bootstrap-current comparison](docs/_static/bootstrap_current_from_vmec_or_boozmn.png)

### 6. Differentiable workflows

Inverse and sensitivity examples:

```bash
python examples/autodiff_inverse_problem.py
python examples/neopax_autodiff_profiles.py
python examples/bootstrap_current_optimization.py
```

These examples generate manuscript-quality figures in `docs/_static/`.

## Parallelization

NTX supports:

- serial batched JAX scans for small and medium studies
- multiprocess multi-device scans for throughput-oriented workloads

The current measured guidance is:

- use serial batched JAX as the default
- use the multiprocess lane when scan sizes are large enough to amortize
  process-launch and compilation costs

See:

- [docs/gpu.md](docs/gpu.md)
- [docs/performance.md](docs/performance.md)

## Autodiff And NEOPAX

NTX is designed to be usable inside differentiable JAX pipelines. The imported
lane supports:

- `jit`
- `vmap`
- differentiation with respect to transport inputs and geometry coefficients
- NEOPAX-style database generation entirely in Python

The most relevant examples are:

- `examples/bootstrap_current_from_vmec_or_boozmn.py`
- `examples/neopax_autodiff_profiles.py`
- `examples/bootstrap_current_optimization.py`
- `examples/neopax_with_ntx.py`

## Publication Figures

Generate the full manuscript-ready figure bundle:

```bash
python examples/make_publication_figures.py
```

This regenerates:

- validation figures
- autodiff inverse and profile figures
- bootstrap-current optimization science figure
- CPU/GPU performance scaling figures

For the W7-X bootstrap-current validation figure:

```bash
python examples/bootstrap_current_from_vmec_or_boozmn.py
```

and writes a manifest to `docs/_static/publication_figure_manifest.json`.

## Validation

Current local status:

- `110 passed, 2 skipped`
- full validation remains local-first because hosted CI is billing-blocked
- office GPU validation is documented in [docs/gpu.md](docs/gpu.md)

## Documentation

- [Install](docs/install.md)
- [Input File](docs/input-file.md)
- [Algorithm](docs/algorithm.md)
- [Autodiff](docs/autodiff.md)
- [Examples](docs/examples.md)
- [Validation](docs/validation.md)
- [NEOPAX](docs/neopax.md)
- [GPU](docs/gpu.md)
- [Performance](docs/performance.md)
- [Manuscript Figures](docs/manuscript.md)
- [Release](docs/release.md)
