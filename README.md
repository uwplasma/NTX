# NTX

NTX is a JAX-native prototype solver for monoenergetic neoclassical transport in
stellarator flux surfaces. It solves the Legendre-space block-tridiagonal form of
the monoenergetic drift-kinetic equation described in Escoto's PhD thesis
([arXiv:2510.27513](https://arxiv.org/abs/2510.27513)).

The first implementation focuses on the monoenergetic geometric coefficients
`D11`, `D31`, `D13`, `D33`, and the Spitzer `D33` normalization. It is written as
pure JAX functions so that the same numerical path can run on CPU or GPU.

```bash
python -m pip install -e ".[dev,docs,io]"
pytest -m "not gpu"
ntx input.toml
```

Example `input.toml`:

```toml
[surface]
type = "dkes"
path = "/path/to/ddkes2.data"

[grid]
n_theta = 19
n_zeta = 79
n_xi = 180
dtype = "float64"
x64 = true

[case]
nu_hat = 1e-5
er_hat = 1e-3

[output]
npz = "w7x_eim_run.npz"
include_modes = true

[benchmark]
reference_table = "/path/to/reference_executable_Monoenergetic_Database.dat"

[logging]
verbose = true
```

This entrypoint prints a verbose Rich summary to the terminal and writes a
compressed `.npz` file with coefficients, low-order Legendre modes, input
metadata, and optional benchmark deltas.
