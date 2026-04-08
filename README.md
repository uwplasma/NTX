# NTX

NTX is a JAX-native solver for monoenergetic neoclassical transport in
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

[logging]
verbose = true
```

VMEC input uses the same entrypoint:

```toml
[surface]
type = "vmec"
path = "/path/to/wout_vmec.nc"
psi_n = 0.25
vmec_radial_option = 0
vmec_nyquist_option = 1
min_bmn_to_load = 0.0

[grid]
n_theta = 19
n_zeta = 79
n_xi = 180

[case]
nu_hat = 1e-5
epsi_hat = 1e-3
```

`ntx input.toml` prints a verbose Rich summary to the terminal and writes a
compressed `.npz` file with the resolved inputs, geometry arrays, transport
coefficients, residuals, and optionally the stored low-order Legendre modes.

The installed CLI does not run comparisons against external tools. Local
REFERENCE_EXECUTABLE comparisons live in the standalone script
[`scripts/compare_reference_executable.py`](/Users/rogeriojorge/local/.NTX/scripts/compare_reference_executable.py).

Full input and output documentation is in
[`docs/input-file.md`](/Users/rogeriojorge/local/.NTX/docs/input-file.md).
