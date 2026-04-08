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

## Filtered VMEC Surface

```bash
ntx examples/w7x_vmec_filtered.toml
```

This variant demonstrates:

- radial snapping with `vmec_radial_option = 1`
- keeping Nyquist modes with `vmec_nyquist_option = 2`
- mode filtering through `min_bmn_to_load = 1e-3`

## Programmatic Example

```python
from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic

surface = load_vmec_surface("tests/fixtures/wout_w7x_standardConfig.nc", psi_n=0.25)
grid = GridSpec(n_theta=9, n_zeta=11, n_xi=6)
case = MonoenergeticCase(nu_hat=1e-3, epsi_hat=1e-3)
result = solve_monoenergetic(surface, grid, case)
print(result.as_dict())
```
