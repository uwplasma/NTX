# NTX

NTX is a JAX-native solver for monoenergetic neoclassical transport on
stellarator flux surfaces.

The first solver implements the Legendre-space block-tridiagonal drift-kinetic
equation formulation from Francisco Javier Escoto López's PhD thesis,
[`arXiv:2510.27513`](https://arxiv.org/abs/2510.27513).

## Equation

For each monoenergetic case, NTX solves the truncated Legendre system

```text
L_k f^(k-1) + D_k f^(k) + U_k f^(k+1) = s^(k)
```

for modes needed to evaluate the geometric monoenergetic coefficients
`D11`, `D31`, `D13`, and `D33`.

## Example

```python
from ntx import (
    GridSpec,
    MonoenergeticCase,
    example_surface,
    load_dkes_surface,
    solve_monoenergetic,
)

surface = example_surface()
grid = GridSpec(n_theta=5, n_zeta=5, n_xi=6)
case = MonoenergeticCase(nu_hat=1e-3, er_hat=0.0)
result = solve_monoenergetic(surface, grid, case)
print(result.as_dict())

surface = load_dkes_surface("/path/to/ddkes2.data")
case = MonoenergeticCase(nu_hat=1e-5, er_hat=1e-3)
result = solve_monoenergetic(surface, grid, case)
```

## Input Files

The installed executable accepts a single TOML input file:

```bash
ntx input.toml
```

```toml
[surface]
type = "dkes"
path = "/path/to/ddkes2.data"

[grid]
n_theta = 19
n_zeta = 79
n_xi = 180

[case]
nu_hat = 1e-5
er_hat = 1e-3

[output]
npz = "run_outputs/w7x_eim.npz"
include_modes = true

[benchmark]
reference_table = "/path/to/reference_executable_Monoenergetic_Database.dat"

[logging]
verbose = true
```

NTX prints detailed terminal output using Rich and stores the solve results,
metadata, and optional comparison deltas in the requested `.npz` file.

## Benchmarks

NTX can read archived REFERENCE_EXECUTABLE-style monoenergetic tables for regression and local
benchmark comparisons:

```bash
ntx benchmark \
  --dkes /path/to/ddkes2.data \
  /path/to/reference_executable_Monoenergetic_Database.dat \
  --nu-hat 1e-5 \
  --er-hat 1e-3
```

Large benchmark artifacts should still stay outside the main package unless they
are intentionally reduced to compact regression fixtures.
