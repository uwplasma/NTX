# Geometry And Inputs

NTX can solve the monoenergetic problem from four geometry families:

1. the built-in analytic sample surface
2. DKES-style Boozer harmonic files
3. VMEC `wout` files through [`vmec_jax`](https://github.com/uwplasma/vmec_jax)
4. Boozer `boozmn` files through
   [`booz_xform_jax`](https://github.com/uwplasma/booz_xform_jax)

This page explains how those inputs are turned into the common internal
representation used by the solver.

## Internal Surface Objects

NTX has two main surface dataclasses in
[`src/ntx/geometry.py`](../src/ntx/geometry.py):

- `BoozerSurface`
- `VmecSurface`

Both are JAX pytrees, so they can be passed through `jit`, `vmap`, and `grad`
in the imported lane.

## Boozer Harmonic Surfaces

`BoozerSurface` stores:

- integer mode arrays `m`, `n`
- Fourier coefficients `b_cos` and optional `b_sin`
- field-period count `nfp`
- rotational transform `iota`
- flux normalization `psi_p`
- covariant Boozer field components `B_theta`, `B_zeta`
- optional `B0`

The Fourier evaluator in
[`src/ntx/geometry.py`](../src/ntx/geometry.py) computes

```{math}
\phi_{mn} = m\theta + n N_\mathrm{fp}\zeta
```

and returns `B`, `\partial_\theta B`, and `\partial_\zeta B` directly from the
trigonometric series.

### DKES-Style Files

`load_dkes_surface(...)` in [`src/ntx/io.py`](../src/ntx/io.py) parses
`ddkes2.data`-style inputs and fills `BoozerSurface`.

Use this path when:

- a Boozer harmonic surface is already available
- the goal is a direct monoenergetic solve without VMEC preprocessing

## VMEC `wout` Files

`load_vmec_surface(...)` in [`src/ntx/vmec.py`](../src/ntx/vmec.py) reads
VMEC equilibria through `vmec_jax` and extracts a single surface.

The loader resolves:

- `nfp`, `ns`, `mpol`, `ntor`
- `iota(\psi_n)`
- Fourier coefficients for:
  - `B`
  - Jacobian
  - `B_\theta`, `B_\zeta`
  - `B^\theta`, `B^\zeta`
- `psi_a_hat`
- `Aminor_p`
- the electric-field transport normalization

### VMEC Knobs

The VMEC loader exposes four important knobs.

#### `psi_n`

The requested normalized toroidal-flux label.

#### `vmec_radial_option`

- `0`: interpolate to the requested `psi_n`
- `1`: snap to the nearest interior VMEC surface
- `2`: snap to the nearest VMEC surface including endpoints

This affects whether NTX uses the exact requested radial location or a discrete
surface from the VMEC radial grid.

#### `vmec_nyquist_option`

- `1`: reduced mode set
- `2`: keep Nyquist modes

#### `vmec_mode_convention`

- `"reduced"`: use the reduced VMEC `(xm, xn)` table
- `"filtered_nyquist"`: retain the Nyquist subset satisfying the mode filters

#### `min_bmn_to_load`

Discard modes whose relative amplitude `|B_mn/B_00|` is below the threshold.

This is a practical performance control. It reduces harmonic count, memory
traffic, and transform cost when very small modes are not needed.

## Boozer `boozmn` Files

NTX can also load Boozer harmonic files directly through the Python/JAX Boozer
helpers in [`src/ntx/booz.py`](../src/ntx/booz.py).

Use this path when:

- the equilibrium has already been transformed to Boozer coordinates
- the workflow is based on Boozer harmonics instead of VMEC harmonics

In that case, NTX bypasses VMEC interpolation and goes directly to
`BoozerSurface`.

The direct `boozmn` loader treats Boozer spectra and Boozer radial profiles as
VMEC half-grid quantities. In modern `boozmn` files this information is exposed
through `s_in`, `s_b`, and the packed-surface `jlist` convention, where
`jlist = compute_surfs + 2`. The full-grid toroidal-flux profile `phi_b` is
metadata and is not the radial coordinate for selecting packed `B_{mn}`
surfaces. This matters because a full-grid shift changes the selected surface
and therefore the magnetic-drift source before any closure or current-profile
normalization is reached.

The maintained same-coordinate round-trip audit is:

```bash
python examples/boozmn_same_coordinate_roundtrip_audit.py
```

It generates a `boozmn` file from a VMEC `wout`, reloads the same VMEC
half-grid surfaces through `load_boozmn_surface(...)`, and compares the loaded
geometry and `D11/D31/D13/D33` against the in-memory
`vmec_jax -> booz_xform_jax -> NTX` path. That audit validates the direct
loader on the same coordinate representation. It is separate from audits that
compare VMEC-harmonic and Boozer-coordinate representations, since those expose
different geometry channels.

![Same-coordinate Boozer-file round-trip audit](_static/boozmn_same_coordinate_roundtrip_audit.png)

## Geometry Evaluation On The Angular Grid

All surface families are converted to `GeometryOnGrid` by
`geometry_on_grid(...)` in [`src/ntx/geometry.py`](../src/ntx/geometry.py).

That object stores:

- `theta_2d`, `zeta_2d`
- `B`
- `\partial_\theta B`, `\partial_\zeta B`
- `\mathcal J`
- `B_\theta`, `B_\zeta`
- `B^\theta`, `B^\zeta`
- `\langle B^2 \rangle`
- `V'`
- the radial-drift source factor
- the normalization scales used later by the transport coefficients

This is the canonical internal geometry representation for the solver.

## Angular Domain

NTX discretizes one field period:

```{math}
\theta \in [0, 2\pi),
\qquad
\zeta \in [0, 2\pi/N_\mathrm{fp}).
```

This is implemented in `periodic_grid(...)` in
[`src/ntx/grids.py`](../src/ntx/grids.py).

## Coordinate Flattening Convention

Flux-surface fields are flattened with theta as the fastest index:

- `flatten_fs(...)`
- `unflatten_fs(...)`

in [`src/ntx/grids.py`](../src/ntx/grids.py).

This matters for:

- dense operator assembly
- mode-vector interpretation
- file-backed output arrays

## Geometry In The CLI Output

The `ntx input.toml` workflow writes the evaluated geometry into the output
NetCDF, NPZ, or HDF5 file:

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

These are written in `save_run_output(...)` in
[`src/ntx/_inputfiles_output.py`](../src/ntx/_inputfiles_output.py).

## Recommended Input Strategy

- Use `example_surface()` for fast development and testing.
- Use DKES-style or Boozer inputs when the Boozer harmonics are already
  available and fixed.
- Use VMEC inputs when the workflow begins from equilibrium data and radial
  interpolation matters.
- Use the imported `vmec_jax` lane when NTX needs to sit inside a larger JAX
  analysis or optimization loop.
