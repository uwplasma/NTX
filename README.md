# NTX

NTX is a JAX-native monoenergetic neoclassical transport code for stellarator
flux surfaces. It implements the Legendre-space block-tridiagonal formulation
developed in Javier Escoto's PhD thesis
([arXiv:2510.27513](https://arxiv.org/abs/2510.27513)).

The current release solves for the monoenergetic geometric coefficients
`D11`, `D31`, `D13`, `D33`, and the Spitzer `D33` normalization on DKES-style
Boozer surfaces and on VMEC `wout` equilibria.

## Install

From a local checkout:

```bash
python -m pip install -e ".[dev,docs,io]"
```

Minimal runtime install:

```bash
python -m pip install -e .
```

## Quick Start

The main entrypoint is:

```bash
ntx input.toml
```

Repository examples:

```bash
ntx examples/example_surface.toml
ntx examples/w7x_dkes.toml
ntx examples/w7x_vmec.toml
```

`ntx` prints a detailed Rich run summary and writes a compressed `.npz` with:

- the resolved inputs
- surface metadata
- geometry arrays on the angular grid
- transport coefficients
- residual diagnostics
- optional low-order Legendre modes

## Input Styles

NTX accepts three surface families:

- `type = "example"` for the built-in analytic surface
- `type = "dkes"` for DKES-style `ddkes2.data`
- `type = "vmec"` for VMEC `wout_*.nc`

Example DKES input:

```toml
[surface]
type = "dkes"
path = "../tests/fixtures/w7x_eim_sample.ddkes2.data"

[grid]
n_theta = 9
n_zeta = 9
n_xi = 8
dtype = "float64"
x64 = true

[case]
nu_hat = 1e-5
er_hat = 1e-3

[output]
npz = "outputs/w7x_dkes.npz"
include_modes = true

[logging]
verbose = true
```

Example VMEC input:

```toml
[surface]
type = "vmec"
path = "../tests/fixtures/wout_w7x_standardConfig.nc"
psi_n = 0.25
vmec_radial_option = 0
vmec_nyquist_option = 1
min_bmn_to_load = 0.0

[grid]
n_theta = 9
n_zeta = 11
n_xi = 8
dtype = "float64"
x64 = true

[case]
nu_hat = 1e-3
epsi_hat = 1e-3
```

For VMEC inputs, use `epsi_hat`. `er_hat` is only supported for surfaces that
provide a `psi_p` normalization directly.

## Other Ways To Run

Direct CLI solve:

```bash
python -m ntx.cli solve --dkes tests/fixtures/w7x_eim_sample.ddkes2.data \
  --nu-hat 1e-5 --er-hat 1e-3 --n-theta 9 --n-zeta 9 --n-xi 8
```

Direct VMEC solve:

```bash
python -m ntx.cli solve --vmec tests/fixtures/wout_w7x_standardConfig.nc \
  --psi-n 0.25 --nu-hat 1e-3 --epsi-hat 1e-3 --n-theta 9 --n-zeta 11 --n-xi 8
```

Programmatic API:

```python
from ntx import GridSpec, MonoenergeticCase, load_dkes_surface, load_vmec_surface, solve_monoenergetic

grid = GridSpec(n_theta=9, n_zeta=11, n_xi=8)

dkes_surface = load_dkes_surface("tests/fixtures/w7x_eim_sample.ddkes2.data")
dkes_result = solve_monoenergetic(dkes_surface, grid, MonoenergeticCase(nu_hat=1e-5, er_hat=1e-3))

vmec_surface = load_vmec_surface("tests/fixtures/wout_w7x_standardConfig.nc", psi_n=0.25)
vmec_result = solve_monoenergetic(vmec_surface, grid, MonoenergeticCase(nu_hat=1e-3, epsi_hat=1e-3))
```

## Algorithm

For each monoenergetic case, NTX solves

```text
L_k f^(k-1) + D_k f^(k) + U_k f^(k+1) = s^(k)
```

with:

- dense spatial blocks on the `(theta, zeta)` grid
- forward Schur-complement elimination in Legendre index
- backward substitution for the retained low-order modes
- a nullspace constraint `f^(0)(0,0) = 0`
- JAX `jit`, `vmap`, and x64 support for physics runs

The current implementation stores the low-order Legendre modes needed for the
transport coefficients and writes those modes to the output file when requested.

## Outputs

The `.npz` payload includes:

- run configuration and raw input text
- surface metadata and geometry metadata as JSON
- algorithm metadata
- angular grids and geometry arrays
- surface Fourier harmonics
- transport coefficients and residuals
- optional `f1_modes` and `f3_modes`

For VMEC runs, the output also includes:

- selected and requested `psi_n`
- `ns`, `mpol`, `ntor`
- total and loaded Fourier mode counts
- `phi_edge`, `psi_a_hat`, and `aminor_p` when available

## Examples And Docs

Repository examples:

- [examples/example_surface.toml](/Users/rogeriojorge/local/.NTX/examples/example_surface.toml)
- [examples/w7x_dkes.toml](/Users/rogeriojorge/local/.NTX/examples/w7x_dkes.toml)
- [examples/w7x_vmec.toml](/Users/rogeriojorge/local/.NTX/examples/w7x_vmec.toml)
- [examples/w7x_vmec_filtered.toml](/Users/rogeriojorge/local/.NTX/examples/w7x_vmec_filtered.toml)

Primary docs:

- [docs/install.md](/Users/rogeriojorge/local/.NTX/docs/install.md)
- [docs/input-file.md](/Users/rogeriojorge/local/.NTX/docs/input-file.md)
- [docs/algorithm.md](/Users/rogeriojorge/local/.NTX/docs/algorithm.md)
- [docs/examples.md](/Users/rogeriojorge/local/.NTX/docs/examples.md)

## Development Checks

```bash
ruff check .
mypy src/ntx
pytest -q
sphinx-build -b html docs docs/_build/html
```
