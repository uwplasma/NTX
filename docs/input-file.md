# Input File

The primary NTX command-line interface is:

```bash
ntx input.toml
```

One TOML file defines one monoenergetic solve.

## Minimal Example

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

Running

```bash
ntx input.toml
```

prints a Rich summary and writes `input.nc` unless another output file is
configured. Current examples default to NetCDF, so a typical file-backed run is:

```bash
ntx examples/example_surface.toml --plot
```

This writes `examples/outputs/example_surface.nc` and
`examples/outputs/example_surface.pdf`.

## Top-Level Tables

Required tables:

- `[surface]`
- `[grid]`
- `[case]`

Optional tables:

- `[output]`
- `[logging]`

## `[surface]`

### Common Keys

| Key | Required | Meaning |
| --- | --- | --- |
| `type` | yes | one of `"example"`, `"dkes"`, `"vmec"` |
| `path` | for `"dkes"` and `"vmec"` | input file path, resolved relative to the TOML file |

### `type = "example"`

Uses `example_surface()` from [`src/ntx/geometry.py`](../src/ntx/geometry.py).

### `type = "dkes"`

Loads a DKES-style Boozer harmonic file through
`load_dkes_surface(...)` in [`src/ntx/io.py`](../src/ntx/io.py).

The file supplies:

- `nfp`
- `psi_p`
- `chi_p`
- `iota`
- `B_theta`
- `B_zeta`
- Boozer Fourier harmonics for `B`

### `type = "vmec"`

Loads one VMEC flux surface through `load_vmec_surface(...)` in
[`src/ntx/vmec.py`](../src/ntx/vmec.py).

Additional keys:

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `psi_n` | yes | none | requested normalized toroidal flux |
| `vmec_radial_option` | no | `0` | radial interpolation / snapping mode |
| `vmec_nyquist_option` | no | `1` | reduced or Nyquist mode set |
| `vmec_mode_convention` | no | `"reduced"` | reduced versus filtered-Nyquist mode convention |
| `min_bmn_to_load` | no | `0.0` | drop small `B_mn` harmonics below this threshold |

#### `vmec_radial_option`

- `0`: interpolate to the requested `psi_n`
- `1`: snap to nearest interior VMEC surface
- `2`: snap to nearest VMEC surface including endpoints

#### `vmec_nyquist_option`

- `1`: reduced mode set
- `2`: keep Nyquist modes

#### `vmec_mode_convention`

- `"reduced"`
- `"filtered_nyquist"`

## `[grid]`

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `n_theta` | yes | none | number of poloidal grid points |
| `n_zeta` | yes | none | number of toroidal grid points in one field period |
| `n_xi` | yes | none | highest Legendre mode index retained |
| `dtype` | no | `"float64"` | working scalar dtype (`float64` or `float32`) |
| `x64` | no | `true` | enable JAX x64 mode |

Constraints from [`src/ntx/grids.py`](../src/ntx/grids.py):

- `n_theta >= 3`
- `n_zeta >= 3`
- `n_xi >= 2`

## `[case]`

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `nu_hat` | yes | none | monoenergetic collisionality |
| `epsi_hat` | no | `null` | normalized electric field in the solver normalization |
| `er_hat` | no | `null` | radial electric field in the input normalization |

Exactly one of `epsi_hat` and `er_hat` may be set.

### Electric-Field Resolution

If `epsi_hat` is given, NTX uses it directly.

If `er_hat` is given, NTX converts it internally:

- Boozer / DKES:
  ```{math}
  \hat E_\psi = \hat E_r / \psi_p
  ```
- VMEC:
  ```{math}
  r_n = \sqrt{\psi_n},\quad
  \hat r = a_\mathrm{minor} r_n,\quad
  \frac{d\hat\psi}{d\hat r} = \frac{2\hat\psi_a r_n}{a_\mathrm{minor}},\quad
  \hat E_\psi = \hat E_r \left(\frac{d\hat\psi}{d\hat r}\right)^{-1}
  ```

## `[output]`

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `path` | no | `input.nc` | output file path; suffix selects NetCDF, NPZ, or HDF5 |
| `npz` | no | none | legacy alias for `path = "...npz"` |
| `netcdf`, `nc` | no | none | alias for NetCDF output path |
| `hdf5`, `h5` | no | none | alias for HDF5 output path |
| `include_modes` | no | `true` | include `f1_modes` and `f3_modes` |

Set only one output path key. Supported suffixes are:

- `.nc` or `.netcdf`: uncompressed NetCDF4, the default file-backed format
- `.npz`: compressed NumPy archive, useful for compact Python-only exchange
- `.h5` or `.hdf5`: uncompressed HDF5, useful for fast array interchange

The CLI can override the TOML path:

```bash
ntx input.toml --output outputs/run.npz
ntx input.toml --output outputs/run.nc --plot
```

Python callers can use the same suffix selection:

```python
from ntx import run_from_input_file

payload = run_from_input_file("input.toml", output_path="outputs/run.nc", plot=True)
print(payload["output_path"], payload["plot_pdf"])
```

## `[logging]`

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `verbose` | no | `true` | print Rich run summaries and metadata tables |

## Verbose Terminal Output

Verbose runs print:

- input file path
- surface summary
- source-file metadata
- geometry statistics
- resolved electric-field normalization
- algorithm summary
- transport coefficients and diagnostics
- prepare/solve/write/plot runtime timings
- output-file summary
- optional PDF plot path when `--plot` is used

## Output Payload

The CLI writes the same data to NetCDF, NPZ, and HDF5. NetCDF and HDF5 store
string metadata as file attributes and numeric arrays as variables/datasets.
NPZ stores every entry as a NumPy array. The payload has four classes of data.

### 1. Input And Run Metadata

- `input_path`
- `input_toml_text`
- `run_config_json`
- `surface_type`
- `surface_path`
- `surface_source_name`
- `surface_source_size_bytes`
- `surface_source_mtime`
- `surface_source_sha256`

### 2. Solver Setup

- `n_theta`
- `n_zeta`
- `n_xi`
- `dtype`
- `x64`
- `nu_hat`
- `epsi_hat_input`
- `er_hat_input`
- `epsi_hat_resolved`

### 3. Geometry Arrays

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

### 4. Transport Results

- `D11`
- `D31`
- `D13`
- `D33`
- `D33_spitzer`
- `residual_l2`
- `onsager_residual`
- optionally `f1_modes`
- optionally `f3_modes`

Metadata is also stored as JSON strings:

- `surface_metadata_json`
- `geometry_metadata_json`
- `algorithm_metadata_json`

All of this is written in `save_run_output(...)` in
[`src/ntx/_inputfiles_output.py`](../src/ntx/_inputfiles_output.py), with
`save_run_output(...)` selecting the concrete writer from the filename suffix.

## Example Inputs

### Analytic Sample

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

### DKES-Style Boozer Input

```toml
[surface]
type = "dkes"
path = "sample.ddkes2.data"

[grid]
n_theta = 17
n_zeta = 25
n_xi = 32

[case]
nu_hat = 1e-5
er_hat = 1e-3

[output]
path = "outputs/sample_dkes.nc"
include_modes = true
```

### VMEC Input

```toml
[surface]
type = "vmec"
path = "wout_example.nc"
psi_n = 0.25
vmec_radial_option = 0
vmec_nyquist_option = 1
vmec_mode_convention = "reduced"
min_bmn_to_load = 0.0

[grid]
n_theta = 17
n_zeta = 25
n_xi = 32

[case]
nu_hat = 1e-5
er_hat = 1e-3

[output]
path = "outputs/sample_vmec.nc"
include_modes = true
```

## Post-Processing

Inspect an output file graphically with:

```bash
python examples/plot_output_file.py path/to/output.nc
```

That script turns the saved payload into a publication-style multi-panel figure.
The legacy script name `examples/plot_output_npz.py` remains available and now
accepts `.nc`, `.npz`, and `.h5` files.
