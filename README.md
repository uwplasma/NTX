[![Release](https://img.shields.io/github/v/release/uwplasma/NTX)](https://github.com/uwplasma/NTX/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/uwplasma/NTX/blob/main/LICENSE)
[![Tests](https://github.com/uwplasma/NTX/actions/workflows/tests.yml/badge.svg)](https://github.com/uwplasma/NTX/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/ntx/badge/?version=latest)](https://ntx.readthedocs.io/en/latest/)
[![Coverage](https://img.shields.io/badge/coverage-measured%20in%20CI-blue)](https://ntx.readthedocs.io/en/latest/testing.html)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/ntx/)

# NTX

NTX is a JAX-native monoenergetic neoclassical transport solver for stellarator
flux surfaces. It solves the Legendre-space formulation described in Javier
Escoto's PhD thesis, [Fast monoenergetic neoclassical transport coefficients in
stellarators](https://arxiv.org/abs/2510.27513).

Use NTX as:

- a command-line solver for file-backed transport calculations,
- a Python/JAX library for scans, autodiff, uncertainty propagation, and
  optimization,
- a NEOPAX-compatible monoenergetic database builder for bootstrap-current
  workflows.

## Install

```bash
pip install ntx
```

For local development:

```bash
pip install -e ".[dev,docs,io]"
```

Optional geometry-coupled examples use upstream JAX geometry tools:

```bash
pip install git+https://github.com/uwplasma/vmec_jax.git
pip install git+https://github.com/uwplasma/booz_xform_jax.git
```

## Quick Start

Run the smallest bundled case:

```bash
ntx examples/example_surface.toml --plot
```

This writes `examples/outputs/example_surface.nc` plus a PDF summary panel.
Choose the output format by filename:

```bash
ntx examples/example_surface.toml --output examples/outputs/example_surface.npz --plot
ntx examples/example_surface.toml --output examples/outputs/example_surface.h5 --plot
python examples/plot_output_file.py examples/outputs/example_surface.nc
```

Use NTX from Python:

```python
from ntx import GridSpec, MonoenergeticCase, example_surface, solve_monoenergetic

surface = example_surface()
grid = GridSpec(n_theta=17, n_zeta=25, n_xi=32)
case = MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0)

result = solve_monoenergetic(surface, grid, case)
print(result.D11, result.D31, result.D13, result.D33)
```

For JAX scans:

```python
import jax.numpy as jnp
from ntx import GridSpec, example_surface, solve_monoenergetic_scan

surface = example_surface()
grid = GridSpec(17, 25, 16)
nu_hat = jnp.logspace(-5, -2, 8)

coefficients = solve_monoenergetic_scan(
    surface,
    grid,
    nu_hat,
    epsi_hat=jnp.zeros_like(nu_hat),
)
```

## Outputs

For each monoenergetic case, NTX computes:

- `D11`, `D31`, `D13`, `D33`, and `D33_spitzer`,
- residual and Onsager diagnostics,
- resolved electric-field normalization,
- geometry arrays and run metadata in NetCDF, NPZ, or HDF5 outputs.

The input schema is documented in [docs/input-file.md](docs/input-file.md).

## Physics In One Paragraph

NTX solves the local monoenergetic drift-kinetic equation on one flux surface,
keeping parallel streaming, mirror force, radial-electric-field precession, and
Lorentz pitch-angle scattering at fixed speed. The unknown non-adiabatic
response is projected onto Legendre polynomials in pitch angle, giving the
block-tridiagonal system solved by the code. The two right-hand sides are the
radial-transport drive and the parallel-flow/bootstrap-current drive; NTX
returns the monoenergetic coefficients consumed by profile and NEOPAX workflows.
The full equation, ordering, normalizations, and coefficient definitions are in
[docs/physics.md](docs/physics.md).

## Validation Snapshot

Validation claims are tracked in the maintained
[benchmark matrix](docs/benchmark-matrix.md) and
[physics-gate summary](docs/physics-gates.md). The README keeps only the
highest-signal artifacts:

| Solver validation | Fixed-field current comparison |
| --- | --- |
| ![Monoenergetic validation summary](docs/_static/validation_summary.png) | ![Fixed-field SFINCS, Redl, and NTX + NEOPAX bootstrap-current comparison](docs/_static/bootstrap_current_fixed_field_validation.png) |

| Owned finite-beta bootstrap stress | Differentiable geometry/current path |
| --- | --- |
| ![Owned finite-beta Redl and NTX + NEOPAX bootstrap-current stress audit](docs/_static/owned_finite_beta_bootstrap_comparison.png) | ![Explicit-relaxed boundary current derivative benchmark](docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.png) |

| Owned finite-beta geometry | Same-grid SFINCS-JAX input generation |
| --- | --- |
| ![Owned finite-beta NTX and NEOPAX geometry dataset provenance](docs/_static/owned_geometry_neopax_dataset.png) | ![Owned finite-beta SFINCS-JAX generation contract](docs/_static/owned_finite_beta_sfincs_jax_inputs.png) |

Current promoted validation includes monoenergetic convergence and identities,
the fixed-field Redl comparison on the precise-QS benchmark family, the
fixed-field `NTX+NEOPAX` total-current stress gate, the integrated W7-X
workflow transfer, and prepared derivative agreement against direct
reverse-mode differentiation. The fixed-field current result uses documented
normalization and moment-closure conventions, not fitted bridge constants.
External W7-X transfer checks are not labeled as SFINCS parity; promoted
SFINCS/Redl/NTX+NEOPAX figures must use one owned geometry, profile,
collisionality, electric-field, interpolation, and normalization contract.
The owned finite-beta lane now uses local stellarator input/wout pairs,
passes the physical VMEC edge-flux scale into the Boozer NTX path, generates
same-grid SFINCS-JAX input decks, and records completed transport-matrix
output with the `nuPrime -> nu_n` bridge when the local checkout is available.
The finite-beta bootstrap-current stress audit runs Redl and `NTX+NEOPAX` on
the same VMEC wout, Boozer transform, profiles, radial grid, adaptive `nu/v`
support, and current normalization. The kept artifact uses the production QA
radial/collisionality ladder and Pmax 12; its JSON sidecar tracks the remaining
inner-radius reduced-closure gap and the Sonine-order convergence trend rather
than promoting a hidden parity curve. The same-grid finite-beta coefficient
sidecar now includes that inner stress radius and separates a `2.1e-2`
coefficient-level difference from the `3.1e-1` profile-current stress gap.
The profile-current observable sidecar shows the remaining stress is not a
correction-sign failure: the correction has the right sign but applies about
`0.80` of the correction needed at the stress radius. The same sidecar records
that the net current is cancellation-dominated there, so the remaining residual
is a sub-percent species-correction imbalance rather than an order-unity
coefficient failure.
Optimized finite-beta QH/QI Boozer
reconstruction remains an explicit geometry-backend blocker.
Species-resolved fixed-field closure parity, broader geometry-family studies,
and large-optimization studies remain tracked as stress diagnostics or planned
research lanes in the docs.

Run the local gate summary with:

```bash
python scripts/check_physics_gates.py
```

## Common Workflows

CLI solves:

```bash
ntx examples/sample_dkes.toml
ntx examples/sample_vmec.toml
```

NEOPAX database and bootstrap-current examples:

```bash
python examples/neopax_with_ntx.py
python examples/owned_geometry_neopax_dataset.py
python examples/owned_finite_beta_sfincs_jax_inputs.py
python examples/owned_finite_beta_bootstrap_comparison.py
python examples/bootstrap_current_with_neopax.py
python examples/bootstrap_current_from_vmec_or_boozmn.py
```

Autodiff and optimization examples:

```bash
python examples/derivative_audit.py
python examples/explicit_relaxed_boundary_current_derivative_benchmark.py
python examples/bootstrap_current_optimization.py
```

Performance examples:

```bash
python examples/prepared_geometry_reuse_profile.py --preset smoke
python scripts/benchmark_scaling.py --help
```

Full example coverage is in [docs/examples.md](docs/examples.md).

## Current Open Research Lanes

The major open lanes are:

- full geometry-family reproduction on paper-resolution W7-X, QI, QA/QH, and
  additional stellarator-family inputs,
- reusable hidden-symmetry and omnigenous benchmark families,
- broader geometry-control autodiff with direct AD, prepared adjoints, and
  finite-difference agreement on reusable geometry families,
- restoration of implicit-equilibrium sensitivities only after residual
  contraction and Boozer/NTX transport finite-difference agreement pass,
- additional dedicated GPU nodes with healthy multi-GPU execution and
  device-memory timelines,
- broader fixed-field NTX+NEOPAX closure transfer, including species-resolved
  current decomposition and any future default closure, without regressing the
  integrated W7-X workflow,
- broader profile, uncertainty, and robust-design studies before promoting
  stellarator-design claims.

The live roadmap is in [docs/research-roadmap.md](docs/research-roadmap.md).

## Documentation

- Documentation: [ntx.readthedocs.io/en/latest/](https://ntx.readthedocs.io/en/latest/)
- Validation: [docs/validation.md](docs/validation.md)
- Performance: [docs/performance.md](docs/performance.md)
- GPU notes: [docs/gpu.md](docs/gpu.md)
- NEOPAX bridge: [docs/neopax.md](docs/neopax.md)
- Source map: [docs/source-map.md](docs/source-map.md)

## Local Checks

```bash
python -m ruff check .
python -m mypy src/ntx
python -m pytest -q
python -m sphinx -b html docs docs/_build/html
```
