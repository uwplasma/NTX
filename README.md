# NTX

NTX is a JAX-native monoenergetic neoclassical transport code for stellarator
flux surfaces. It implements the Legendre-space block-tridiagonal formulation
developed in Javier Escoto's PhD thesis
([arXiv:2510.27513](https://arxiv.org/abs/2510.27513)).

The current release solves for the monoenergetic geometric coefficients
`D11`, `D31`, `D13`, `D33`, and the Spitzer `D33` normalization on DKES-style
Boozer surfaces and on VMEC equilibria. For imported JAX workflows, NTX now
includes an explicit `vmec_jax -> booz_xform_jax -> NTX` path and a direct
NTX-to-NEOPAX mapping layer.

## Install

From a local checkout:

```bash
python -m pip install -e ".[dev,docs,io]"
```

Minimal runtime install:

```bash
python -m pip install -e .
```

Local JAX geometry / transport stack:

```bash
python -m pip install -e /Users/rogeriojorge/local/vmec_jax
python -m pip install -e /Users/rogeriojorge/local/booz_xform_jax
python -m pip install -e /Users/rogeriojorge/local/tests/NEOPAX
python -m pip install -e /Users/rogeriojorge/local/.NTX"[dev,docs,io]"
```

Primary checks:

```bash
ruff check .
mypy src/ntx
pytest -q
sphinx-build -b html docs docs/_build/html
```

GitHub Actions runs the CPU suite across Python `3.10`, `3.11`, and `3.12`.

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
ntx examples/qi_vmec_erhat.toml
python examples/vmec_jax_booz_xform_jax_ntx.py
python examples/neopax_with_ntx.py
```

`ntx` prints a detailed Rich run summary and writes a compressed `.npz` with:

- the resolved inputs
- source-file metadata
- source-file checksums
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
er_hat = 1e-3
```

For DKES and Boozer surfaces, NTX resolves `epsi_hat = er_hat / psi_p`.

For VMEC surfaces, NTX derives a radial normalization from the selected surface:

- `r_n = sqrt(psi_n)`
- `r_hat = Aminor_p * r_n`
- `dpsi_hat/dr_hat = 2 * psi_a_hat * r_n / Aminor_p`
- `epsi_hat = er_hat / (dpsi_hat/dr_hat)`

This keeps the VMEC `er_hat` path tied to explicit surface metadata rather than a
placeholder scale. The stored output distinguishes this electric-field
conversion scale from the coefficient normalization scale:

- `surface_transport_psi_scale = dpsi_hat/dr_hat`
- `surface_coefficient_psi_scale = 1` for Escoto-style VMEC outputs

For VMEC mode selection:

- `vmec_nyquist_option = 1` uses a reduced VMEC spectral set
- `vmec_nyquist_option = 2` uses the full Nyquist mode set (`xm_nyq`, `xn_nyq`)
- `vmec_mode_convention = "reduced"` keeps the reduced `(xm, xn)` mode table
  and truncates the VMEC coefficient arrays to the same length
- `vmec_mode_convention = "filtered_nyquist"` keeps the filtered Nyquist subset
  with `|m| < mpol` and `|n| <= ntor` in field-period units

The default is `vmec_mode_convention = "reduced"`, which follows the reduced
mode-table convention used in Escoto-style VMEC workflows. Use
`"filtered_nyquist"` when comparing against SFINCS-style VMEC geometry paths.

## Other Ways To Run

Direct CLI solve:

```bash
python -m ntx.cli solve --dkes tests/fixtures/w7x_eim_sample.ddkes2.data \
  --nu-hat 1e-5 --er-hat 1e-3 --n-theta 9 --n-zeta 9 --n-xi 8
```

Direct VMEC solve:

```bash
python -m ntx.cli solve --vmec tests/fixtures/wout_w7x_standardConfig.nc \
  --psi-n 0.25 --nu-hat 1e-3 --er-hat 1e-3 --n-theta 9 --n-zeta 11 --n-xi 8
```

Programmatic API:

```python
from ntx import (
    GridSpec,
    MonoenergeticCase,
    compile_prepared_solver,
    load_dkes_surface,
    load_vmec_surface,
    prepare_monoenergetic_system,
    solve_monoenergetic,
)

grid = GridSpec(n_theta=9, n_zeta=11, n_xi=8)

dkes_surface = load_dkes_surface("tests/fixtures/w7x_eim_sample.ddkes2.data")
dkes_result = solve_monoenergetic(dkes_surface, grid, MonoenergeticCase(nu_hat=1e-5, er_hat=1e-3))

vmec_surface = load_vmec_surface("tests/fixtures/wout_w7x_standardConfig.nc", psi_n=0.25)
vmec_result = solve_monoenergetic(vmec_surface, grid, MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3))

prepared = prepare_monoenergetic_system(dkes_surface, grid)
compiled_solver = compile_prepared_solver(prepared)
compiled_result = compiled_solver(MonoenergeticCase(nu_hat=1e-5, er_hat=1e-3))
```

Use `compile_prepared_solver()` when you want repeated solves on one fixed
surface and grid. The standard `solve_monoenergetic()` and `solve_prepared()`
paths remain the right default for single solves and for heavy CPU benchmark
cases where XLA compile time may not amortize cleanly.

Imported JAX VMEC/Boozer path:

```python
import vmec_jax as vj

from ntx import GridSpec, MonoenergeticCase, solve_monoenergetic, surface_from_vmec_jax_state

run = vj.run_fixed_boundary(
    "/Users/rogeriojorge/local/vmec_jax/examples/data/input.circular_tokamak",
    max_iter=1,
    use_initial_guess=True,
    vmec_project=False,
    verbose=True,
)
geom = vj.eval_geom(run.state, run.static)
signgs = vj.signgs_from_sqrtg(geom.sqrtg, axis_index=1)
surface = surface_from_vmec_jax_state(
    state=run.state,
    static=run.static,
    indata=run.indata,
    signgs=int(signgs),
    s=0.25,
    mboz=6,
    nboz=0,
)
result = solve_monoenergetic(
    surface,
    GridSpec(n_theta=17, n_zeta=17, n_xi=40),
    MonoenergeticCase(nu_hat=1e-4, epsi_hat=0.0),
)
```

NTX-to-NEOPAX mapping:

```python
from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    load_vmec_surface,
    to_neopax_monoenergetic,
)

reference = load_neopax_reference_scan(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/Dij_NEOPAX_FULL_S_NEW_W7X.h5"
)

def surface_loader(rho_value: float):
    return load_vmec_surface(
        "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/wout_W7-X_standard_configuration.nc",
        psi_n=rho_value**2,
        vmec_radial_option=1,
        vmec_nyquist_option=2,
        vmec_mode_convention="filtered_nyquist",
    )

scan = build_ntx_neopax_scan(
    surface_loader,
    rho=reference.rho[:2],
    nu_v=reference.nu_v[2:5],
    Es=reference.Es[:2, :3],
    Er=reference.Er[:2, :3],
    drds=reference.drds[:2],
    grid=GridSpec(n_theta=17, n_zeta=33, n_xi=60),
)
database = to_neopax_monoenergetic(scan, a_b=1.0)
```

## Differentiable Core

NTX now has an explicit imported differentiable lane separate from the CLI/file
lane.

The CLI entrypoint `ntx input.toml` remains optimized for file-based runs,
verbose terminal output, and `.npz` export. That path is not intended to be
used under autodiff.

The imported solver path is designed to stay inside JAX once the surface object
has been constructed:

```python
from ntx import GridSpec, MonoenergeticCase, example_surface, solve_monoenergetic_internal

surface = example_surface()
grid = GridSpec(5, 5, 4)
case = MonoenergeticCase(nu_hat=1e-2, er_hat=1e-3)

Dij, f, s = solve_monoenergetic_internal(surface, grid, case)
```

Here:

- `Dij` has shape `(3, 3)`
- `f` and `s` contain the retained low-order mode systems with shape
  `(3, 3, n_theta * n_zeta)`

This API is the intended starting point for NTX-driven monoenergetic database
generation inside higher-level JAX workflows such as NEOPAX.

For scan-based imported workflows, NTX also provides an in-memory database
builder:

```python
from ntx import GridSpec, build_monoenergetic_database_arrays, example_surface
import jax.numpy as jnp

database = build_monoenergetic_database_arrays(
    example_surface(),
    GridSpec(5, 5, 4),
    nu_hat=jnp.asarray([1e-2, 2e-2]),
    er_hat=jnp.asarray([0.0, 1e-3]),
)
```

This returns a `MonoenergeticDatabaseArrays` object with tensor-product scan
arrays such as `D11`, `D13`, and `D33` over `(nu_hat, scan_field)` for one
surface, entirely in memory.

Current differentiability scope:

- gradients through `nu_hat`
- gradients through `er_hat`
- gradients through Boozer Fourier coefficients and other surface arrays
- `jit` over surface arguments in the imported solver path

Current non-differentiable scope:

- text and NetCDF file parsing
- CLI/config loading
- Rich terminal output and `.npz` serialization
- legacy file-driven VMEC helpers used for existing regression fixtures

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

For imported differentiable use, NTX also exposes the low-order internal solve
output `(Dij, f, s)` directly. This mirrors the shape of the low-level
monoenergetic solver interface used by other JAX transport workflows while
keeping the terminal/file interface separate.

On VMEC surfaces, NTX evaluates `B`, the Jacobian, covariant and contravariant
field components, and the radial transport normalization on the requested
angular grid before assembling the Legendre blocks.

## Outputs

The `.npz` payload includes:

- run configuration and raw input text
- source filename, file size, modification time, and SHA-256 checksum
- surface metadata and geometry metadata as JSON
- algorithm metadata
- angular grids and geometry arrays
- surface Fourier harmonics
- transport coefficients and residuals
- optional `f1_modes` and `f3_modes`

For text-based surface inputs such as DKES `ddkes2.data`, the raw source text is
also stored in the output payload.

For VMEC runs, the output also includes:

- selected and requested `psi_n`
- `ns`, `mpol`, `ntor`
- total and loaded Fourier mode counts
- `phi_edge`, `psi_a_hat`, `aminor_p`, `r_n`, `r_hat`
- `dpsi_hat/dr_hat` and `dr_hat/dpsi_hat`

## Examples And Docs

Repository examples:

- [examples/example_surface.toml](/Users/rogeriojorge/local/.NTX/examples/example_surface.toml)
- [examples/w7x_dkes.toml](/Users/rogeriojorge/local/.NTX/examples/w7x_dkes.toml)
- [examples/w7x_vmec.toml](/Users/rogeriojorge/local/.NTX/examples/w7x_vmec.toml)
- [examples/w7x_vmec_filtered.toml](/Users/rogeriojorge/local/.NTX/examples/w7x_vmec_filtered.toml)
- [examples/qi_vmec_erhat.toml](/Users/rogeriojorge/local/.NTX/examples/qi_vmec_erhat.toml)

Validation and benchmark scripts:

- [scripts/compare_archived_benchmarks.py](/Users/rogeriojorge/local/.NTX/scripts/compare_archived_benchmarks.py)
- [scripts/compare_reference_executable.py](/Users/rogeriojorge/local/.NTX/scripts/compare_reference_executable.py)
- [scripts/benchmark_against_reference_executable.py](/Users/rogeriojorge/local/.NTX/scripts/benchmark_against_reference_executable.py)

Primary docs:

- [docs/install.md](/Users/rogeriojorge/local/.NTX/docs/install.md)
- [docs/input-file.md](/Users/rogeriojorge/local/.NTX/docs/input-file.md)
- [docs/algorithm.md](/Users/rogeriojorge/local/.NTX/docs/algorithm.md)
- [docs/examples.md](/Users/rogeriojorge/local/.NTX/docs/examples.md)
- [docs/validation.md](/Users/rogeriojorge/local/.NTX/docs/validation.md)
- [docs/gpu.md](/Users/rogeriojorge/local/.NTX/docs/gpu.md)

Useful analysis scripts:

- `python scripts/compare_archived_benchmarks.py --output-json archived-benchmarks.json`
- `python scripts/compare_archived_benchmarks.py --case W7X-EIM`
- `python scripts/compare_archived_benchmarks.py --case W7X-KJM`
- `python scripts/compare_archived_benchmarks.py --case CIEMAT-QI`
- `python scripts/profile_runtime.py --output-json runtime-profile.json`

Those two scripts default to `JAX_PLATFORM_NAME=cpu` so they are stable on any
machine. Override the platform explicitly when you want to force GPU execution.

The archived comparison script now covers the three thesis benchmark families:
W7-X EIM, W7-X KJM, and CIEMAT-QI. For W7-X EIM and W7-X KJM, NTX matches the
vendored archived monoenergetic reference tables at the benchmark grids used by
the thesis, while the DKES and SFINCS comparisons remain useful cross-code
validation reports rather than equality gates.

## GPU Runs

CPU CI keeps running `pytest -m "not gpu"`.

For a GPU check in your office environment:

```bash
sh office
cd /Users/rogeriojorge/local/.NTX
python -m pip install -e ".[dev,docs,io]"
scripts/sh_office_gpu_smoke.sh
```
