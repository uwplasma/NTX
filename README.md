[![Release](https://img.shields.io/github/v/release/uwplasma/NTX)](https://github.com/uwplasma/NTX/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/uwplasma/NTX/blob/main/LICENSE)
[![Tests](https://github.com/uwplasma/NTX/actions/workflows/tests.yml/badge.svg)](https://github.com/uwplasma/NTX/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/ntx/badge/?version=latest)](https://ntx.readthedocs.io/en/latest/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-blue)](https://ntx.readthedocs.io/en/latest/testing.html)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/ntx/)

# NTX

NTX is a JAX-native monoenergetic neoclassical transport solver for stellarator
flux surfaces. It implements the Legendre-space formulation described in Javier
Escoto's PhD thesis, [Fast monoenergetic neoclassical transport coefficients in
stellarators](https://arxiv.org/abs/2510.27513).

NTX is intended to be useful in two modes:

- a straightforward command-line workflow for production solves
- an imported Python/JAX workflow for scans, autodiff, optimization, NEOPAX
  coupling, and throughput-oriented parallel execution

## Fastest Start

Install:

```bash
python -m pip install ntx
```

Run the smallest bundled case:

```bash
ntx examples/example_surface.toml
```

That command:

- prints a Rich summary to the terminal
- solves one monoenergetic case
- writes `examples/example_surface.npz`

Inspect the output graphically:

```bash
python examples/plot_output_npz.py examples/example_surface.npz
```

## What NTX Computes

For a single flux surface and monoenergetic case, NTX computes:

- `D11`
- `D31`
- `D13`
- `D33`
- `D33_spitzer`

plus diagnostics and, optionally, the low-order Legendre modes used in the
coefficient calculation.

## Installation

Basic install:

```bash
python -m pip install ntx
```

Editable install with test and docs tools:

```bash
python -m pip install -e ".[dev,docs,io]"
```

Install the VMEC/Boozer geometry helpers:

```bash
python -m pip install -e ".[geometry]"
```

## Simplest Way To Run The Code

The main command is:

```bash
ntx input.toml
```

Minimal input file:

```toml
[surface]
type = "example"

[grid]
n_theta = 9
n_zeta = 9
n_xi = 8

[case]
nu_hat = 1e-2
epsi_hat = 0.0
```

The CLI writes a compressed `.npz` file containing:

- transport coefficients
- residual and Onsager diagnostics
- resolved electric-field normalization
- geometry arrays on the angular grid
- run metadata and the original input text

## Main Inputs

The CLI supports:

- the built-in analytic sample surface
- DKES-style Boozer harmonic files
- VMEC `wout` files through `vmec_jax`

Imported Python workflows additionally support direct Boozer-file loading and
in-memory surface construction.

Important numerical knobs:

- `n_theta`, `n_zeta`, `n_xi`
- `nu_hat`
- `epsi_hat` or `er_hat`
- VMEC radial and mode-selection options

Full schema:

- [docs/input-file.md](docs/input-file.md)

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
grid = GridSpec(n_theta=17, n_zeta=25, n_xi=32)
case = MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0)
result = solve_monoenergetic(surface, grid, case)

print(result.D11, result.D31, result.D13, result.D33)
```

### 3. Batched differentiable JAX scans

```python
import jax.numpy as jnp
from ntx import GridSpec, example_surface, solve_monoenergetic_scan

surface = example_surface()
grid = GridSpec(17, 25, 16)
nu_hat = jnp.logspace(-5, -2, 8)
epsi_hat = jnp.zeros_like(nu_hat)
coefficients = solve_monoenergetic_scan(surface, grid, nu_hat, epsi_hat=epsi_hat)
```

### 4. Throughput-oriented multiprocess scans

```python
import jax.numpy as jnp
from ntx import GridSpec, example_surface, solve_monoenergetic_multiprocess_scan

surface = example_surface()
grid = GridSpec(17, 25, 16)
nu_hat = jnp.logspace(-5, -2, 64)
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

NTX exposes direct scan builders and mapping helpers for NEOPAX-style
monoenergetic databases:

- `build_ntx_neopax_scan(...)`
- `build_ntx_neopax_scan_from_surfaces(...)`
- `scan_to_neopax_arrays(...)`
- `to_neopax_monoenergetic(...)`
- `write_neopax_scan_hdf5(...)`

Run:

```bash
python examples/neopax_with_ntx.py
```

### 6. Autodiff and optimization workflows

Run:

```bash
python examples/autodiff_inverse_problem.py
python examples/neopax_autodiff_profiles.py
python examples/derivative_audit.py
python examples/derivative_path_benchmark.py
python examples/ambipolar_profile.py
python examples/ambipolar_profile_family.py
python examples/profile_control_optimization.py
python examples/profile_basis_optimization.py
python examples/profile_transport_loop.py
python examples/primitive_profile_transport.py
python examples/bootstrap_current_optimization.py
python examples/bootstrap_current_with_neopax.py
```

These generate figures for:

- inverse problems
- sensitivity analysis
- direct autodiff versus finite-difference derivative audits
- direct reverse-mode versus prepared custom-VJP derivative timing
- ambipolar residual landscapes and bootstrap-current-proxy profile solves
- controlled families of ambipolar and bootstrap-current-proxy profiles
- differentiable optimization of scalar profile controls
- low-dimensional radial-basis optimization of profile controls
- accepted-step transport iteration of the radial profile closure with radial smoothing
- primitive density/temperature transport workflows with explicit source-target closure
- differentiable bootstrap-current optimization
- radial bootstrap-current profiles from NTX + NEOPAX

## Bootstrap-Current Validation

Fixed-field precise-QS benchmark against archived SFINCS, SFINCS-JAX reruns,
Redl, and `NTX+NEOPAX`:

```bash
python examples/bootstrap_current_fixed_field_validation.py
```

This uses the local precise-QS Zenodo archive and writes:

- `docs/_static/bootstrap_current_fixed_field_validation.png`
- `docs/_static/bootstrap_current_fixed_field_validation.pdf`
- `docs/_static/bootstrap_current_fixed_field_validation.json`

This benchmark now uses the physically motivated conductivity-side branch
available from NTX-generated scans:

- `D33_cond = D33_spitzer - D33`

That choice is not a fitted bridge. It follows the DKES-style conductivity
normalization discussed in Escoto's thesis and related source material, where
the parallel-conductivity channel is compared relative to the Spitzer problem
rather than through a raw conductivity-like coefficient alone. The closure
audit showed that this conductivity-difference kernel must enter the full
higher-order row-3/4/5 hierarchy consistently. Mixed choices for the
no-momentum `Lij` branch and the momentum-correction `Eij` branch are both
numerically worse and physically inconsistent.

On the precise-QS fixed-field archive, the regenerated interior max relative
errors versus archived SFINCS are now:

- QA: `1.01e-1`
- QH: `2.32e-1`

Redl remains close on the same family (`6.86e-2` on QA and `4.06e-2` on QH).
The benchmark keeps a monotone `PCHIP` radial postprocessing map by default,
since the interpolation audit showed that neither the final radial remap nor
the internal NTSS-style versus direct 3D interpolation choice is the dominant
error source here.

This is still not a universal parity claim. A dedicated W7-X rebuild audit now
exists in:

- `examples/bootstrap_current_w7x_rebuild_audit.py`

That script rebuilds a NEOPAX-format W7-X database with NTX, including
`D33_spitzer`, and compares it against the shipped external database on the
same momentum-corrected workflow. The transfer result is currently negative:

- shipped external database: `1.18e-12` max relative error against the frozen
  W7-X reference current
- NTX-rebuilt W7-X, `d33_mode="spitzer"`: `4.18e+0`
- NTX-rebuilt W7-X, `d33_mode="conductivity_difference"`: `1.07e+1`

So the conductivity-difference closure is a physically motivated improvement
for the precise-QS fixed-field archive, but it does not yet transfer to the
shipped W7-X integrated workflow. The remaining open lane is therefore still
the broader momentum-correction model, not interpolation or the raw
monoenergetic handoff.

![Fixed-field precise-QS bootstrap-current benchmark](docs/_static/bootstrap_current_fixed_field_validation.png)

Streamlined radial-profile example with NEOPAX:

```bash
python examples/bootstrap_current_with_neopax.py
```

This shows the direct NTX scan -> NEOPAX closure workflow and writes:

- `docs/_static/bootstrap_current_with_neopax.png`
- `docs/_static/bootstrap_current_with_neopax.pdf`
- `docs/_static/bootstrap_current_with_neopax.json`

![NTX + NEOPAX bootstrap-current profile](docs/_static/bootstrap_current_with_neopax.png)

W7-X bootstrap-current convergence audit:

```bash
python examples/bootstrap_current_reference_audit_w7x.py
```

This rebuilds a reduced W7-X scan at several NTX resolutions and writes a
two-panel convergence figure showing the bootstrap-current profile and the
maximum relative error versus grid resolution:

![W7-X bootstrap-current convergence](docs/_static/bootstrap_current_reference_audit_w7x.png)

## Parallelization

NTX supports:

- serial batched JAX scans
- guarded single-process device-parallel scans
- multiprocess CPU/GPU scans for larger throughput jobs

Current practical guidance:

- use serial batched JAX for small and medium studies
- use the multiprocess lane for larger throughput-oriented jobs

See:

- [docs/performance.md](docs/performance.md)
- [docs/gpu.md](docs/gpu.md)

## NEOPAX Connection

NTX is designed to provide monoenergetic coefficient tables to
[NEOPAX](https://github.com/uwplasma/NEOPAX). The dedicated interface is
documented in:

- [docs/neopax.md](docs/neopax.md)

## Documentation

Full documentation:

- [ntx.readthedocs.io/en/latest/](https://ntx.readthedocs.io/en/latest/)

## Local Quality Checks

```bash
python -m ruff check .
python -m mypy src/ntx
python -m pytest -q
python -m sphinx -b html docs docs/_build/html
```
