# Input File

The primary NTX user interface is:

```bash
ntx input.toml
```

One TOML file defines one solve.

## Required Tables

- `[surface]`
- `[grid]`
- `[case]`

## Optional Tables

- `[output]`
- `[logging]`

## `[surface]`

### Common Keys

- `type`
  - Required.
  - Allowed values: `"example"`, `"dkes"`, `"vmec"`.
- `path`
  - Required for `"dkes"` and `"vmec"`.
  - Path is resolved relative to the TOML file location.

### `type = "example"`

Uses the built-in analytic surface. No `path` is needed.

### `type = "dkes"`

Reads a DKES-style `ddkes2.data` file and extracts:

- `nfp`
- `psi_p`
- `chi_p`
- `iota`
- `B_theta`
- `B_zeta`
- Boozer Fourier harmonics

### `type = "vmec"`

Reads a VMEC `wout_*.nc` file and extracts:

- Fourier harmonics for `B`
- Fourier harmonics for the Jacobian
- covariant and contravariant field components
- radial-grid metadata
- selected flux-surface metadata
- `psi_a_hat`
- `Aminor_p`
- VMEC radial transport normalization

Additional VMEC keys:

- `psi_n`
  - Required.
  - Requested normalized toroidal-flux label in `[0, 1]`.
- `vmec_radial_option`
  - Optional, default `0`.
  - `0`: use the requested `psi_n`
  - `1`: snap to the nearest interior VMEC surface
  - `2`: snap to the nearest VMEC surface including endpoints
- `vmec_nyquist_option`
  - Optional, default `1`.
  - `1`: drop Nyquist modes
  - `2`: keep Nyquist modes
- `min_bmn_to_load`
  - Optional, default `0.0`.
  - Drop modes whose `|B_mn / B00|` is below the threshold

## `[grid]`

- `n_theta`
  - Required integer
- `n_zeta`
  - Required integer
- `n_xi`
  - Required integer
  - Must be at least `2`
- `dtype`
  - Optional string, default `"float64"`
- `x64`
  - Optional boolean, default `true`

## `[case]`

- `nu_hat`
  - Required float
- `epsi_hat`
  - Optional float
- `er_hat`
  - Optional float
  - Converted internally through a surface-specific transport scale

Exactly one of `epsi_hat` and `er_hat` may be set.

Resolved electric-field normalization:

- DKES / Boozer:
  - `epsi_hat = er_hat / psi_p`
- VMEC:
  - `r_n = sqrt(psi_n)`
  - `r_hat = Aminor_p * r_n`
  - `dpsi_hat/dr_hat = 2 * psi_a_hat * r_n / Aminor_p`
  - `epsi_hat = er_hat / (dpsi_hat/dr_hat)`

VMEC inputs require `Aminor_p` in the `wout` file and `surface.psi_n > 0` so
that this normalization is well defined.

## `[output]`

- `npz`
  - Optional path
  - Default: `input.toml` with the suffix changed to `.npz`
- `include_modes`
  - Optional boolean, default `true`
  - When true, write `f1_modes` and `f3_modes`

## `[logging]`

- `verbose`
  - Optional boolean, default `true`
  - When true, NTX prints detailed Rich tables describing the solve

## Verbose Terminal Output

Verbose runs print:

- the input file path
- the surface summary
- surface metadata from the loaded file
- geometry statistics on the angular grid
- the resolved solve parameters
- the solver/algorithm summary
- the transport coefficients and residuals
- the output payload summary

## Example Inputs

### Built-In Surface

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

### DKES

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

### VMEC

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

[output]
npz = "outputs/w7x_vmec.npz"
include_modes = true

[logging]
verbose = true
```

## NPZ Contents

Every output file includes the run configuration, raw input text, scalar
metadata, and resolved transport results.

### Core Run Metadata

- `input_path`
- `input_toml_text`
- `run_config_json`
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

### Surface Metadata

- `surface_nfp`
- `surface_iota`
- `surface_psi_p`
- `surface_transport_psi_scale`
- `surface_b0`
- `surface_mode_count`
- `surface_stellarator_symmetric`
- `surface_source_name`
- `surface_source_size_bytes`
- `surface_source_mtime`
- `surface_metadata_json`

Additional DKES / Boozer keys:

- `surface_b_theta`
- `surface_b_zeta`
- `surface_chi_p`
- `surface_modes_m`
- `surface_modes_n`
- `surface_modes_b_cos`

Additional VMEC keys:

- `vmec_requested_psi_n`
- `vmec_selected_psi_n`
- `vmec_ns`
- `vmec_mpol`
- `vmec_ntor`
- `vmec_total_mode_count`
- `vmec_loaded_mode_count`
- `vmec_psi_a_hat`
- `vmec_phi_edge`
- `vmec_aminor_p`
- `vmec_r_n`
- `vmec_r_hat`
- `vmec_dpsi_hat_dr_hat`
- `vmec_dr_hat_dpsi_hat`
- `surface_modes_m`
- `surface_modes_n`
- `surface_modes_b_cos`
- `surface_modes_jacobian_cos`
- `surface_modes_b_sub_theta_cos`
- `surface_modes_b_sub_zeta_cos`
- `surface_modes_b_sup_theta_cos`
- `surface_modes_b_sup_zeta_cos`

### Geometry Arrays

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
- `geometry_metadata_json`

### Solver And Result Metadata

- `D11`
- `D31`
- `D13`
- `D33`
- `D33_spitzer`
- `residual_l2`
- `onsager_residual`
- `algorithm_metadata_json`
- `result_json`

### Optional Mode Outputs

Only written when `output.include_modes = true`:

- `f1_modes`
- `f3_modes`
