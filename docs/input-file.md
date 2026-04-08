# Input File

The installed executable runs one solve from one TOML file:

```bash
ntx input.toml
```

## Structure

Required tables:

- `[surface]`
- `[grid]`
- `[case]`

Optional tables:

- `[output]`
- `[logging]`

## Surface Inputs

### `[surface]`

Common keys:

- `type`
  - Required.
  - Allowed values: `"example"`, `"dkes"`, `"vmec"`.
- `path`
  - Required for `"dkes"` and `"vmec"`.
  - Path to `ddkes2.data` for `"dkes"`.
  - Path to `wout_*.nc` for `"vmec"`.

DKES/example-specific behavior:

- `type = "example"`
  - Uses the built-in analytic test surface.
- `type = "dkes"`
  - Loads Boozer harmonics and scalar flux-surface data from the DKES-style
    input file.

VMEC-specific keys:

- `psi_n`
  - Required when `type = "vmec"`.
  - Normalized toroidal-flux label in `[0, 1]`.
- `vmec_radial_option`
  - Optional, default `0`.
  - `0`: use the requested `psi_n`.
  - `1`: snap to the nearest interior VMEC radial surface.
  - `2`: snap to the nearest VMEC radial surface including endpoints.
- `vmec_nyquist_option`
  - Optional, default `1`.
  - `1`: drop explicit VMEC Nyquist modes.
  - `2`: keep Nyquist modes.
- `min_bmn_to_load`
  - Optional, default `0.0`.
  - Drops VMEC modes with `|B_mn / B00|` below the threshold.

## Grid Inputs

### `[grid]`

- `n_theta`
  - Required integer.
  - Number of poloidal grid points.
- `n_zeta`
  - Required integer.
  - Number of toroidal grid points on one field period.
- `n_xi`
  - Required integer.
  - Highest Legendre index retained in the block-tridiagonal recursion.
  - Must be at least `2`.
- `dtype`
  - Optional string, default `"float64"`.
  - JAX dtype name for real arrays.
- `x64`
  - Optional boolean, default `true`.
  - Enables JAX x64 mode for the solve.

## Case Inputs

### `[case]`

- `nu_hat`
  - Required float.
  - Monoenergetic collisionality.
- `epsi_hat`
  - Optional float.
  - Direct normalized radial-electric-field input used internally by the
    Legendre-space operator.
- `er_hat`
  - Optional float.
  - Alternative electric-field input.
  - Internally converted to `epsi_hat = er_hat / psi_p`.
  - Only valid when the selected surface provides `psi_p`.

Exactly one of `epsi_hat` and `er_hat` may be set.

For VMEC inputs, use `epsi_hat`. The current VMEC path does not infer a
`psi_p` normalization for `er_hat`.

## Output Inputs

### `[output]`

- `npz`
  - Optional path, default `input_file.with_suffix(".npz")`.
  - Output file written by `numpy.savez_compressed`.
- `include_modes`
  - Optional boolean, default `true`.
  - When true, stores the solved low-order `f1_modes` and `f3_modes`.

## Logging Inputs

### `[logging]`

- `verbose`
  - Optional boolean, default `true`.
  - When true, prints Rich tables for the resolved surface, case, and result.

## Examples

### DKES

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

### VMEC

```toml
[surface]
type = "vmec"
path = "/path/to/wout_w7x_standardConfig.nc"
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

## Terminal Output

`ntx input.toml` prints:

- the input file path
- the resolved surface summary
- the resolved solve parameters
- the transport coefficients
- the output `.npz` path

## NPZ Outputs

Always written:

- `input_path`
- `surface_type`
- `surface_path`
- `surface_psi_n`
- `surface_vmec_radial_option`
- `surface_vmec_nyquist_option`
- `surface_min_bmn_to_load`
- `n_theta`
- `n_zeta`
- `n_xi`
- `dtype`
- `x64`
- `nu_hat`
- `epsi_hat_input`
- `er_hat_input`
- `epsi_hat_resolved`
- `surface_nfp`
- `surface_iota`
- `surface_psi_p`
- `surface_b0`
- `theta_grid`
- `zeta_grid`
- `b`
- `d_b_dtheta`
- `d_b_dzeta`
- `jacobian`
- `b_sub_theta`
- `b_sub_zeta`
- `b_sup_theta`
- `b_sup_zeta`
- `radial_drift_spatial`
- `volume_prime`
- `b2_mean`
- `D11`
- `D31`
- `D13`
- `D33`
- `D33_spitzer`
- `residual_l2`
- `onsager_residual`
- `result_json`

Conditionally written for Boozer/DKES surfaces:

- `surface_b_theta`
- `surface_b_zeta`
- `surface_modes_m`
- `surface_modes_n`
- `surface_modes_b_cos`

Conditionally written for VMEC surfaces:

- `surface_modes_m`
- `surface_modes_n`
- `surface_modes_b_cos`
- `surface_modes_jacobian_cos`
- `surface_modes_b_sub_theta_cos`
- `surface_modes_b_sub_zeta_cos`
- `surface_modes_b_sup_theta_cos`
- `surface_modes_b_sup_zeta_cos`

Only written when `output.include_modes = true`:

- `f1_modes`
- `f3_modes`

## External Comparison

The installed `ntx` executable does not compare against external tables or
external solvers.

Use the standalone script instead:

```bash
python scripts/compare_reference_executable.py input.toml
```

That script is intentionally outside the installed CLI so the primary endpoint
remains a pure solver entrypoint.
