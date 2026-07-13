[![Release](https://img.shields.io/github/v/release/uwplasma/NTX)](https://github.com/uwplasma/NTX/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/uwplasma/NTX/blob/main/LICENSE)
[![Tests](https://github.com/uwplasma/NTX/actions/workflows/tests.yml/badge.svg)](https://github.com/uwplasma/NTX/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/ntx/badge/?version=latest)](https://ntx.readthedocs.io/en/latest/)
[![Coverage](https://img.shields.io/badge/coverage-measured%20in%20CI-blue)](https://ntx.readthedocs.io/en/latest/testing.html)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/ntx/)

# NTX

NTX is a JAX-native monoenergetic neoclassical transport solver for stellarator
flux surfaces.

## Install

```bash
pip install ntx
```

Optional VMEC and Boozer geometry workflows use the upstream JAX geometry
packages:

```bash
pip install git+https://github.com/uwplasma/vmec_jax.git
pip install git+https://github.com/uwplasma/booz_xform_jax.git
```

## Quick Start

Run the bundled surface and create NetCDF and PDF output:

```bash
ntx examples/example_surface.toml --plot
```

Choose NetCDF, NPZ, or HDF5 by the output suffix and plot an existing result:

```bash
ntx examples/example_surface.toml --output result.h5 --plot
python examples/plot_output_file.py result.h5
```

Use the solver directly from Python:

```python
from ntx import GridSpec, MonoenergeticCase, example_surface, solve_monoenergetic

surface = example_surface()
grid = GridSpec(n_theta=17, n_zeta=25, n_xi=32)
case = MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0)

result = solve_monoenergetic(surface, grid, case)
print(result.D11, result.D31, result.D13, result.D33)
```

For repeated scans, prepare the geometry and compile once:

```python
import jax.numpy as jnp
from ntx import GridSpec, compile_prepared_scan_solver, example_surface
from ntx import prepare_monoenergetic_system

prepared = prepare_monoenergetic_system(example_surface(), GridSpec(17, 25, 16))
scan = compile_prepared_scan_solver(prepared)
scan.warmup()
nu_hat = jnp.logspace(-5, -2, 8)
coefficients = scan(nu_hat, epsi_hat=jnp.zeros_like(nu_hat))
```

See the [performance guide](docs/performance.md) before choosing scan batching,
CPU parallelism, or GPU execution.

## Physics And Scope

NTX solves the local monoenergetic drift-kinetic equation on one flux surface at
fixed speed. The retained terms are parallel streaming, mirror force,
radial-electric-field precession, and Lorentz pitch-angle scattering. A finite
Legendre expansion in pitch angle produces the block-tridiagonal system solved
for radial-transport and parallel-flow source terms.

| Scope | What NTX provides |
| --- | --- |
| Solved directly | Monoenergetic `D11`, `D31`, `D13`, `D33`, and `D33_spitzer`, with residual and Onsager diagnostics |
| Downstream closure | Species/profile integration, ambipolar electric field, and bootstrap-current workflows through NTX profile tools and NEOPAX |
| Validated comparisons | Analytical limits, convergence ladders, independent fixed-field comparisons, geometry-family convergence, and derivative checks |
| Research scope | Broader full-collision closure, implicit-equilibrium sensitivities, and additional stellarator-family promotion remain outside shipping claims |

The [physics model](docs/physics.md) gives the equation, ordering,
normalizations, source projections, and coefficient definitions. The
[convergence guide](docs/convergence.md) explains residual semantics and the
required angular and Legendre refinement studies.

## Choose A Workflow

| Goal | Start here |
| --- | --- |
| Solve one coefficient set | `ntx examples/example_surface.toml --plot` |
| Run a prepared collisionality/electric-field scan | [Python API and performance](docs/performance.md) |
| Load VMEC or Boozer geometry | [Geometry and inputs](docs/geometry.md) |
| Export a NEOPAX database | `python examples/build_neopax_scan_from_ertilde.py --help` |
| Calculate a bootstrap-current profile | [Profile workflows](docs/profiles.md) |
| Differentiate or optimize | [Autodiff](docs/autodiff.md) |
| Check resolution and validation | [Physics gates](docs/physics-gates.md) and [benchmark matrix](docs/benchmark-matrix.md) |
| Profile CPU/GPU execution | [Performance](docs/performance.md) and [GPU notes](docs/gpu.md) |

The complete runnable catalog, including expected optional dependencies and
outputs, is in [docs/examples.md](docs/examples.md).

## Validation

Each promoted claim maps to a script, test, committed artifact, acceptance
threshold, and documentation entry in the maintained
[benchmark matrix](docs/benchmark-matrix.md). Runtime code does not use fitted
bridge constants to force agreement with a benchmark.

| Monoenergetic convergence and identities | Fixed-field current comparison |
| --- | --- |
| ![Monoenergetic validation summary](docs/_static/validation_summary.png) | ![Fixed-field SFINCS, Redl, and NTX plus NEOPAX bootstrap-current comparison](docs/_static/bootstrap_current_fixed_field_validation.png) |

The fixed-field current result is a scoped reduced-closure stress comparison,
not species-resolved or full-collision parity. Detailed assumptions, current
normalizations, finite-beta diagnostics, and independent-reference provenance
are in [docs/validation.md](docs/validation.md). Run the active gate summary with:

```bash
python scripts/check_physics_gates.py
```

## Outputs

NetCDF, NPZ, and HDF5 outputs contain transport coefficients, diagnostics,
resolved electric-field normalization, geometry arrays, and run metadata. The
format is selected by filename suffix. See [docs/input-file.md](docs/input-file.md)
for the TOML schema, CLI options, and output variables.

## Documentation

- [Getting started](https://ntx.readthedocs.io/en/latest/)
- [Physics and normalizations](docs/physics.md)
- [Numerics and convergence](docs/numerics.md)
- [API reference](docs/api.rst)
- [Glossary](docs/glossary.md)
- [Examples](docs/examples.md)
- [Validation](docs/validation.md)
- [Source map](docs/source-map.md)
- [Authoritative development plan](plan.md)

## Development

```bash
pip install -e ".[dev,docs,io]"
python -m ruff check .
python -m mypy src/ntx
python scripts/test_lane_manifest.py --check
python -m sphinx -W -b html docs docs/_build/html
```
