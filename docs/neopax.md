# NEOPAX

NTX can act as the monoenergetic coefficient provider for
[NEOPAX](https://github.com/uwplasma/NEOPAX) workflows.

This page describes the interfaces exposed by [`src/ntx/neopax.py`](../src/ntx/neopax.py)
and when to use each of them.

## Main Objects

### `NeopaxScan`

Structured scan payload storing:

- `rho`
- `nu_v`
- `Er`
- `Es`
- `drds`
- `D11`
- `D13`
- `D33`

plus optional metadata and normalization arrays.

### `NeopaxMonoenergeticArrays`

Pure-array payload designed for imported JAX workflows. This is the preferred
object when the caller wants to stay in array land and avoid a hard dependency
on the NEOPAX Python package.

## Main Helpers

- `build_ntx_neopax_scan(...)`
- `build_ntx_neopax_scan_from_surfaces(...)`
- `scan_to_neopax_arrays(...)`
- `to_neopax_monoenergetic(...)`
- `write_neopax_scan_hdf5(...)`
- `load_neopax_reference_scan(...)`

The imported profile layer in [`src/ntx/profiles.py`](../src/ntx/profiles.py)
builds directly on `NeopaxScan` when the next step is an ambipolar or
bootstrap-current-proxy solve instead of immediate export into the external
package object.

For end-to-end examples, see:

- [`examples/neopax_with_ntx.py`](../examples/neopax_with_ntx.py) for the
  smallest scan-to-array workflow
- [`examples/bootstrap_current_with_neopax.py`](../examples/bootstrap_current_with_neopax.py)
  for a radial bootstrap-current profile built from an NTX scan and evaluated
  through NEOPAX
- [`examples/bootstrap_current_fixed_field_validation.py`](../examples/bootstrap_current_fixed_field_validation.py)
  for the local precise-QS fixed-field comparison against SFINCS, SFINCS-JAX,
  and Redl

## Typical Imported Workflow

```python
import jax.numpy as jnp
from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    scan_to_neopax_arrays,
    surface_from_vmec_jax_vmec_wout_file,
)

rho = jnp.linspace(0.2, 0.8, 5)
nu_v = jnp.logspace(-5, -2, 8)
Es = jnp.zeros((rho.size, 6))
Er = jnp.zeros_like(Es)
drds = jnp.ones_like(rho)

def surface_loader(rho_value: float):
    return surface_from_vmec_jax_vmec_wout_file("wout.nc", s=float(rho_value**2))

scan = build_ntx_neopax_scan(
    surface_loader,
    rho=rho,
    nu_v=nu_v,
    Es=Es,
    Er=Er,
    drds=drds,
    grid=GridSpec(n_theta=17, n_zeta=25, n_xi=32),
)

arrays = scan_to_neopax_arrays(scan, a_b=1.0)
```

By default, the conversion keeps the raw `D33` database convention used by the
integrated workflow:

```python
arrays = scan_to_neopax_arrays(scan, a_b=1.0, d33_mode="raw")
```

The `d33_mode="spitzer"` and `d33_mode="conductivity_difference"` branches are
explicit audit choices. They are useful for fixed-field closure stress tests,
but they are not the public default because they do not satisfy the integrated
W7-X transfer gate.

## Explicit-Surface Workflow

If the surfaces are already in memory, avoid the callback boundary:

```python
from ntx import build_ntx_neopax_scan_from_surfaces

scan = build_ntx_neopax_scan_from_surfaces(
    surfaces,
    rho=rho,
    nu_v=nu_v,
    Es=Es,
    Er=Er,
    drds=drds,
    grid=grid,
)
```

This is the cleaner choice for JAX-native surface-generation pipelines.

## HDF5 Workflow

Load a NEOPAX-style monoenergetic table:

```python
from ntx import load_neopax_reference_scan

scan = load_neopax_reference_scan("monoenergetic.h5")
```

Write one:

```python
from ntx import write_neopax_scan_hdf5

write_neopax_scan_hdf5(scan, "monoenergetic_out.h5")
```

## Conversion Layers

### JAX-friendly path

`scan_to_neopax_arrays(...)`

Use this when:

- the next step is still JAX-based
- gradients matter
- the caller wants direct access to the normalized arrays

### Convenience object path

`to_neopax_monoenergetic(...)`

Use this when the NEOPAX package is installed and the goal is to hand the data
to NEOPAX directly as a Python object.

## What NTX Supplies To NEOPAX

NTX supplies the monoenergetic geometric coefficients on the requested radial,
collisionality, and electric-field grid. NEOPAX then uses those tables in its
own higher-level transport workflow.

That separation of responsibility is deliberate:

- NTX owns the monoenergetic solve
- NEOPAX owns the radial multi-species transport layer

## Profile-Grade Imported Workflows

When the next step is still inside NTX, use the profile helpers on top of the
scan payload:

- `evaluate_scan_channel(...)`
- `evaluate_species_particle_flux(...)`
- `evaluate_species_current_response(...)`
- `ambipolar_residual_profile(...)`
- `solve_ambipolar_er_profile(...)`
- `solve_ambipolar_profile_family(...)`
- `bootstrap_current_objective(...)`
- `apply_profile_control(...)`
- `optimize_profile_control(...)`
- `apply_profile_basis_control(...)`
- `optimize_profile_basis_control(...)`
- `advance_profile_transport(...)`
- `profile_transport_loss(...)`
- `solve_profile_transport_loop(...)`

Those helpers are documented on the [Profiles](profiles.md) page.
