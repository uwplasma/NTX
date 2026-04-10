# NEOPAX

NTX now has an explicit mapping layer for NEOPAX-style monoenergetic databases.

The intended workflow is:

1. generate monoenergetic coefficients with NTX
2. map them into `NEOPAX.Monoenergetic`
3. evaluate fluxes or solve transport equations in NEOPAX without going through
   an intermediate HDF5 file

For the W7-X VMEC database used in current NEOPAX tests, NTX now has two
distinct imported geometry lanes:

1. a direct `vmec_jax` VMEC-harmonic lane, which is the current validation gate
   for NEOPAX-facing W7-X scans
2. a `vmec_jax -> booz_xform_jax -> NTX` Boozer-transform lane, which remains
   useful for end-to-end JAX Boozer workflows, is validated locally against the
   file-backed `boozmn` transport reference, but is not the primary W7-X NEOPAX
   reference gate

NTX also keeps a comparison-only path aligned with the external W7-X validation
workflow.

## Local Install

For the current local JAX toolchain, install the geometry and transport stack
from the local checkouts:

```bash
python -m pip install -e <vmec_jax-checkout>
python -m pip install -e <booz_xform_jax-checkout>
python -m pip install -e <neopax-checkout>
python -m pip install -e ".[dev,docs,io]"
```

## NTX Scan To NEOPAX

```python
from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
    to_neopax_monoenergetic,
)

reference = load_neopax_reference_scan(
    "<neopax-checkout>/tests/inputs/Dij_NEOPAX_FULL_S_NEW_W7X.h5"
)
def surface_loader(rho_value: float):
    return surface_from_vmec_jax_vmec_wout_file(
        "<neopax-checkout>/tests/inputs/wout_W7-X_standard_configuration.nc",
        s=rho_value**2,
    )

scan = build_ntx_neopax_scan(
    surface_loader,
    rho=reference.rho[[1, 3]],
    nu_v=reference.nu_v[[5, 7, 9]],
    Es=reference.Es[[1, 3]][:, [0, 7, 9]],
    Er=reference.Er[[1, 3]][:, [0, 7, 9]],
    drds=reference.drds[[1, 3]],
    grid=GridSpec(n_theta=25, n_zeta=25, n_xi=63),
)

database = to_neopax_monoenergetic(scan, a_b=1.0)
```

The resulting `database` object is a `NEOPAX.Monoenergetic` instance and can be
passed directly into NEOPAX flux and transport solvers.

This direct `vmec_jax` VMEC-harmonic path matches the local W7-X validation
subset to better than `1e-2` relative error on `D11`, `D13`, `D31`, and
`D33`.

## Imported Array Path

When the surfaces are already in memory, NTX can build the scan without a
Python callback and keep the NEOPAX mapping in pure arrays:

```python
from ntx import (
    GridSpec,
    build_ntx_neopax_scan_from_surfaces,
    load_vmec_surface,
    scan_to_neopax_arrays,
)

rho = [0.12247, 0.25]
surfaces = tuple(
    load_vmec_surface(
        "tests/fixtures/wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc",
        psi_n=rho_value**2,
    )
    for rho_value in rho
)
scan = build_ntx_neopax_scan_from_surfaces(
    surfaces,
    rho=rho,
    nu_v=[1e-4, 1e-3],
    Es=[[0.0, 5e-4], [0.0, 5e-4]],
    Er=[[0.0, 5e-4], [0.0, 5e-4]],
    drds=[1.0, 1.0],
    grid=GridSpec(n_theta=9, n_zeta=11, n_xi=16),
)
arrays = scan_to_neopax_arrays(scan, a_b=1.0)
```

This path is intended for imported, differentiable workflows. It keeps the NTX
scan and NEOPAX normalization in JAX arrays until the caller explicitly asks
for `NEOPAX.Monoenergetic`.

## W7-X Reference Database Path

To reproduce the layout and normalization used by the existing W7-X NEOPAX
reference database workflow:

```python
from ntx import (
    build_reference_vmec_scan,
    to_neopax_monoenergetic,
    write_neopax_scan_hdf5,
)

scan = build_reference_vmec_scan(
    "<neopax-checkout>/tests/inputs/wout_W7-X_standard_configuration.nc",
    "<neopax-checkout>/tests/inputs/boozmn_wout_W7-X_standard_configuration.nc",
    rho=[0.25, 0.5],
    nu_v=[1e-4, 1e-3, 1e-2],
    er_tilde=[0.0, 1e-3, 1e-2],
    nt=25,
    nz=25,
    nl=64,
)

database = to_neopax_monoenergetic(scan, a_b=float(scan.a_b))
write_neopax_scan_hdf5(scan, "Dij_NEOPAX_subset_ntx.h5")
```

Important conventions in this path:

- VMEC surfaces are loaded from the `wout` file with the same conventions as
  the validated reference workflow
- Boozer-side electric-field conversion factors come from the `boozmn` file
- `nl` maps to `GridSpec.n_xi = nl - 1`
- the stored HDF5 datasets follow the reference NEOPAX layout:
  `rho`, `nu_v`, `Er`, `Er_tilde`, `Es`, `drds`, `D11`, `D13`, `D31`, `D33`,
  and the Boozer-side conversion factors

The repository example script wraps this directly:

```bash
python examples/DKES_like_database/Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py \
  --rho 0.25,0.5 --nu-v 1e-4,1e-3 --er-tilde 0.0,1e-3
```

`surface_from_vmec_jax_vmec_wout_file(...)` is the practical WOUT-backed helper
for the imported W7-X NEOPAX path. It reads the `wout` through `vmec_jax` and
builds the VMEC harmonic surface directly, matching the validation sign,
interpolation, and harmonic conventions without going through a Boozer
transform.

`surface_from_vmec_jax_wout(...)` remains available for imported
`vmec_jax -> booz_xform_jax -> NTX` Boozer workflows.

## VMEC-JAX To NTX

For an imported JAX workflow, NTX can build Boozer harmonics directly from
in-memory `vmec_jax` state:

```python
import vmec_jax as vj

from ntx import GridSpec, MonoenergeticCase, solve_monoenergetic, surface_from_vmec_jax_state

run = vj.run_fixed_boundary(
    "<vmec_jax-checkout>/examples/data/input.circular_tokamak",
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

This path stays inside Python and JAX after the VMEC solve, which is the right
foundation for end-to-end differentiable NEOPAX workflows.

## Current Validation State

What is closed:

- the NTX-to-NEOPAX constructor path reproduces the existing NEOPAX HDF5
  mapping exactly when given the same coefficient tables
- the `vmec_jax -> booz_xform_jax -> NTX` example path runs locally
- the Boozer `boozmn` loader now interpolates the radial profiles and matches
  the validated external Boozer geometry convention on the shared W7-X test case
- the `vmec_jax -> booz_xform_jax -> NTX` Boozer-transform lane now matches the
  file-backed `boozmn` transport reference within about `2%` at two local W7-X
  operating points covered by `tests/test_vmec_jax_backend.py`
- the comparison-only W7-X VMEC validation path now matches the existing NEOPAX
  W7-X subset to better than `1e-2` relative error on `D11`, `D13`, `D31`, and
  `D33`
- the NTX replacement script for the W7-X VMEC database writes the same HDF5
  layout used by the NEOPAX reference tests

What is still open:

- a QI external reference database parity target is still open, but NTX now has
  a repository-backed QI imported scan and HDF5 round-trip path
