# NEOPAX

NTX can be used as the monoenergetic coefficient provider for NEOPAX-style
workflows.

## Main Helpers

- `build_ntx_neopax_scan(...)`
- `build_ntx_neopax_scan_from_surfaces(...)`
- `scan_to_neopax_arrays(...)`
- `to_neopax_monoenergetic(...)`
- `write_neopax_scan_hdf5(...)`

## Imported Workflow

Typical pattern:

```python
from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    surface_from_vmec_jax_vmec_wout_file,
    scan_to_neopax_arrays,
)

rho = ...
nu_v = ...
Es = ...
Er = ...
drds = ...

def surface_loader(rho_value: float):
    return surface_from_vmec_jax_vmec_wout_file("wout.nc", s=float(rho_value**2))

scan = build_ntx_neopax_scan(
    surface_loader,
    rho=rho,
    nu_v=nu_v,
    Es=Es,
    Er=Er,
    drds=drds,
    grid=GridSpec(n_theta=9, n_zeta=11, n_xi=12),
)

arrays = scan_to_neopax_arrays(scan, a_b=1.0)
```

## HDF5 Workflow

Load an existing NEOPAX-style table:

```python
from ntx import load_neopax_reference_scan

scan = load_neopax_reference_scan("monoenergetic.h5")
```

Write one:

```python
from ntx import write_neopax_scan_hdf5

write_neopax_scan_hdf5(scan, "monoenergetic_out.h5")
```

## Notes

- `scan_to_neopax_arrays(...)` is the JAX-friendly path.
- `to_neopax_monoenergetic(...)` is the convenience path when the `NEOPAX`
  Python package is installed in the active environment.
