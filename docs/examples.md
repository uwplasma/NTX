# Examples

The repository ships with ready-to-run TOML examples in `examples/`.

## Built-In Surface

```bash
ntx examples/example_surface.toml
```

This is the smallest self-contained run and is useful for smoke tests and quick
inspection of the output payload.

## DKES Surface

```bash
ntx examples/w7x_dkes.toml
```

This run uses the repository DKES fixture:

- `tests/fixtures/w7x_eim_sample.ddkes2.data`

## VMEC Surface

```bash
ntx examples/w7x_vmec.toml
```

This run uses the repository VMEC fixture:

- `tests/fixtures/wout_w7x_standardConfig.nc`

The example uses:

- `psi_n = 0.25`
- `vmec_radial_option = 0`
- `vmec_nyquist_option = 1`
- `min_bmn_to_load = 0.0`
- `er_hat = 1e-3`

## Filtered VMEC Surface

```bash
ntx examples/w7x_vmec_filtered.toml
```

This variant demonstrates:

- radial snapping with `vmec_radial_option = 1`
- keeping Nyquist modes with `vmec_nyquist_option = 2`
- mode filtering through `min_bmn_to_load = 1e-3`

## QI VMEC Surface

```bash
ntx examples/qi_vmec_erhat.toml
```

This example uses the repository QI VMEC fixture:

- `tests/fixtures/wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc`

The example demonstrates:

- a second VMEC family beyond the W7-X regression surface
- direct VMEC `er_hat` input
- surface-normalized conversion through `dpsi_hat/dr_hat`
- a lower-radius surface with `psi_n = 0.12247^2`

## Programmatic Example

```python
from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic

surface = load_vmec_surface("tests/fixtures/wout_w7x_standardConfig.nc", psi_n=0.25)
grid = GridSpec(n_theta=9, n_zeta=11, n_xi=6)
case = MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3)
result = solve_monoenergetic(surface, grid, case)
print(result.as_dict())
```

## VMEC-JAX To NTX

```bash
python examples/vmec_jax_booz_xform_jax_ntx.py \
  --input /Users/rogeriojorge/local/vmec_jax/examples/data/input.circular_tokamak \
  --s 0.25 --mboz 6 --nboz 0
```

This example:

- runs a small `vmec_jax` equilibrium solve
- transforms the selected surface with `booz_xform_jax`
- solves Escoto's monoenergetic system directly in NTX
- exercises the Boozer-transform workflow rather than the W7-X NEOPAX parity lane

## NTX To NEOPAX

```bash
python examples/neopax_with_ntx.py
```

This example:

- loads the W7-X NEOPAX reference scan layout
- rebuilds a small subset with NTX through
  `surface_from_vmec_jax_vmec_wout_file(...)`
- maps the result directly into `NEOPAX.Monoenergetic`
- closes the local W7-X subset to better than `1e-2` relative error on
  `D11`, `D13`, `D31`, and `D33`

The NEOPAX adapter is intended for imported workflows. The CLI `ntx input.toml`
remains the right entrypoint for standalone file-driven runs.

## QI NTX To NEOPAX

```bash
python examples/qi_neopax_with_ntx.py
```

This example:

- builds a small QI VMEC scan from explicit in-memory NTX surfaces
- maps the result into pure NEOPAX-style arrays through
  `scan_to_neopax_arrays(...)`
- writes a NEOPAX-style HDF5 file without depending on an external reference

Use this example when you want a second imported VMEC family beyond the W7-X
reference subset.

## W7-X Reference Database

```bash
python examples/DKES_like_database/Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py \
  --rho 0.25,0.5 --nu-v 1e-4,1e-3 --er-tilde 0.0,1e-3
```

This example mirrors the W7-X VMEC database workflow used by Eduardo Neto's
`vmec_neopax` REFERENCE_EXECUTABLE branch:

- VMEC harmonics come from the `wout` file
- Boozer-side conversion factors come from the `boozmn` file
- the output is a NEOPAX-style HDF5 table with `D11`, `D13`, `D31`, `D33`, and
  the stored electric-field conversion factors

Use this example when you want direct parity against the existing W7-X NEOPAX
reference database. Use the separate `vmec_jax -> booz_xform_jax` example when
you specifically want the Boozer-transform JAX lane rather than the W7-X NEOPAX
parity lane.
