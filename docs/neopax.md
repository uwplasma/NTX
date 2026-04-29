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
- [`examples/owned_geometry_neopax_dataset.py`](../examples/owned_geometry_neopax_dataset.py)
  for an owned finite-beta `vmec_jax -> booz_xform_jax -> NTX -> NEOPAX`
  dataset, direct wout-harmonic stress cases, and interpolation-path audit
- [`examples/owned_finite_beta_sfincs_jax_inputs.py`](../examples/owned_finite_beta_sfincs_jax_inputs.py)
  for same-grid SFINCS-JAX input generation, completed-output ingestion, and
  coefficient-level NTX comparison before any finite-beta `SFINCS`/Redl/
  `NTX+NEOPAX` parity promotion
- [`examples/owned_finite_beta_sfincs_jax_resolution_audit.py`](../examples/owned_finite_beta_sfincs_jax_resolution_audit.py)
  for the production stress-radius coefficient-resolution and harmonic-cutoff
  audit that keeps the finite-beta current gap out of hidden numerical knobs
- [`examples/owned_finite_beta_sfincs_jax_production_ladder_audit.py`](../examples/owned_finite_beta_sfincs_jax_production_ladder_audit.py)
  for the production radius/collisionality coefficient ladder that localizes
  the remaining finite-beta stress to the profile-current closure layer
- [`examples/owned_finite_beta_bootstrap_comparison.py`](../examples/owned_finite_beta_bootstrap_comparison.py)
  for an owned finite-beta Redl and `NTX+NEOPAX` bootstrap-current stress
  audit on the same VMEC wout, Boozer transform, profiles, radial grid, and
  current normalization
- [`examples/owned_finite_beta_closure_localization.py`](../examples/owned_finite_beta_closure_localization.py)
  for the sidecar that separates same-grid coefficient error from the remaining
  finite-beta profile-current closure gap
- [`examples/owned_finite_beta_profile_current_observable_audit.py`](../examples/owned_finite_beta_profile_current_observable_audit.py)
  for the finite-beta stress-radius observable decomposition into no-momentum
  current, momentum correction, correction needed to match Redl, species-current
  cancellation scale, and Pmax trend
- [`examples/owned_finite_beta_current_conditioning_audit.py`](../examples/owned_finite_beta_current_conditioning_audit.py)
  for the cancellation-conditioned coefficient-precision requirement that must
  be met before the finite-beta net-current residual is assigned to a reduced
  closure change
- [`examples/owned_finite_beta_closure_quadrature_audit.py`](../examples/owned_finite_beta_closure_quadrature_audit.py)
  for the Sonine-order versus velocity-quadrature audit that rejects
  under-integrated apparent finite-beta current-gate passes
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

When converting NEOPAX parallel-flow output into current, use one charge
conversion only. If the workflow uses `species.charge`, that array already
contains the signed physical charge in Coulombs. If the workflow uses
`species.charge_qp`, multiply by one elementary-charge factor:

```python
current = elementary_charge * np.sum(species.charge_qp[:, None] * upar, axis=0)
```

## JAX-Native Geometry Workflows

Use the matching VMEC input and `wout` when the geometry should stay inside the
JAX geometry stack and the Boozer transform should be owned by the same run:

```python
import jax.numpy as jnp
from ntx import GridSpec, build_ntx_neopax_scan_from_surfaces, surface_from_vmec_jax_wout

rho = jnp.asarray([0.35, 0.65])
psi_p = 0.013346299916410087  # abs(phi_edge)/(2*pi) from the matching wout
surfaces = tuple(
    surface_from_vmec_jax_wout(
        input_path="input.LandremanPaul2021_QA_lowres_pressure_current",
        wout_path="wout_LandremanPaul2021_QA_lowres_pressure_current.nc",
        s=float(rho_value**2),
        mboz=4,
        nboz=4,
        psi_p=psi_p,
    )
    for rho_value in rho
)

scan = build_ntx_neopax_scan_from_surfaces(
    surfaces,
    rho=rho,
    nu_v=jnp.asarray([1.0e-3, 1.0e-2]),
    Es=jnp.zeros((rho.size, 1)),
    drds=jnp.ones_like(rho),
    grid=GridSpec(7, 7, 6),
    source_name="owned-finite-beta-qa",
)
```

`surface_from_vmec_jax_vmec_wout_file(...)` is still useful when only a `wout`
file is available. It reads the VMEC harmonic tables through `vmec_jax` and
uses NTX's radial interpolation of those tables. That path is not identical to
the Boozer-transform path above, so the two should be compared only as an
interpolation/geometry-loader audit on the same owned input family.

For physical-current or NEOPAX database workflows, do not rely on the
low-level Boozer helper's default `psi_p=1`. Pass the VMEC edge toroidal flux
divided by `2*pi` explicitly. The owned finite-beta diagnostic records this
value in its JSON sidecar and uses it to keep the Boozer-coordinate and direct
VMEC-harmonic transport paths on the same flux normalization.

For in-memory differentiable studies, avoid file-backed geometry loops and use
`build_ntx_neopax_scan_from_vmec_jax_state(...)` or
`build_ntx_neopax_scan_from_vmec_jax_boundary_params(...)`. Those helpers keep
the VMEC state, Boozer transform, NTX scan, and NEOPAX-style arrays on the
JAX-facing path used by the derivative examples.

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

The writer stores uncompressed numeric datasets with HDF5 object timestamps
disabled. That keeps repeated database regeneration fast and avoids needless
binary churn from file metadata.

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
