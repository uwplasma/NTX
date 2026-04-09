# NEOPAX

NTX now has an explicit mapping layer for NEOPAX-style monoenergetic databases.

The intended workflow is:

1. generate monoenergetic coefficients with NTX
2. map them into `NEOPAX.Monoenergetic`
3. evaluate fluxes or solve transport equations in NEOPAX without going through
   an intermediate HDF5 file

For the W7-X VMEC database used in current NEOPAX tests, NTX also has a
comparison-only path that mirrors Eduardo Neto's `vmec_neopax` REFERENCE_EXECUTABLE branch
closely enough to serve as a parity gate.

## Local Install

For the current local JAX toolchain, install the geometry and transport stack
from the local checkouts:

```bash
python -m pip install -e /Users/rogeriojorge/local/vmec_jax
python -m pip install -e /Users/rogeriojorge/local/booz_xform_jax
python -m pip install -e /Users/rogeriojorge/local/tests/NEOPAX
python -m pip install -e "/Users/rogeriojorge/local/.NTX[dev,docs,io]"
```

## NTX Scan To NEOPAX

```python
from pathlib import Path

from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    surface_from_vmec_jax_wout,
    to_neopax_monoenergetic,
)

reference = load_neopax_reference_scan(
    Path("/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/Dij_NEOPAX_FULL_S_NEW_W7X.h5")
)
input_path = Path("/Users/rogeriojorge/local/tests/simsopt/tests/test_files/input.W7-X_standard_configuration")

def surface_loader(rho_value: float):
    return surface_from_vmec_jax_wout(
        input_path=input_path,
        wout_path="/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/wout_W7-X_standard_configuration.nc",
        s=rho_value**2,
        mboz=12,
        nboz=12,
    )

scan = build_ntx_neopax_scan(
    surface_loader,
    rho=reference.rho[:2],
    nu_v=reference.nu_v[2:5],
    Es=reference.Es[:2, :3],
    Er=reference.Er[:2, :3],
    drds=reference.drds[:2],
    grid=GridSpec(n_theta=17, n_zeta=33, n_xi=60),
)

database = to_neopax_monoenergetic(scan, a_b=1.0)
```

The resulting `database` object is a `NEOPAX.Monoenergetic` instance and can be
passed directly into NEOPAX flux and transport solvers.

## W7-X Reference Database Path

To reproduce the layout and normalization used by
`Examples/DKES_like_database/Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py`
from Eduardo Neto's REFERENCE_EXECUTABLE fork:

```python
from ntx import (
    build_reference_executable_reference_vmec_scan,
    to_neopax_monoenergetic,
    write_neopax_scan_hdf5,
)

scan = build_reference_executable_reference_vmec_scan(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/wout_W7-X_standard_configuration.nc",
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/boozmn_wout_W7-X_standard_configuration.nc",
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
  `Field.from_vmec_s(...)`
- Boozer-side electric-field conversion factors come from the `boozmn` file
- `nl` maps to `GridSpec.n_xi = nl - 1`
- the stored HDF5 datasets follow the REFERENCE_EXECUTABLE/NEOPAX layout:
  `rho`, `nu_v`, `Er`, `Er_tilde`, `Es`, `drds`, `D11`, `D13`, `D31`, `D33`,
  and the Boozer-side conversion factors

The repository example script wraps this directly:

```bash
python examples/DKES_like_database/Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py \
  --rho 0.25,0.5 --nu-v 1e-4,1e-3 --er-tilde 0.0,1e-3
```

`surface_from_vmec_jax_wout(...)` is the practical WOUT-backed helper for this
workflow. It keeps the geometry lane inside `vmec_jax` and `booz_xform_jax`,
and it rebuilds the VMEC static configuration with `ns = wout.ns` when the
reference `wout` carries a finer radial mesh than the original VMEC input file.

## VMEC-JAX To NTX

For an imported JAX workflow, NTX can build Boozer harmonics directly from
in-memory `vmec_jax` state:

```python
import vmec_jax as vj

from ntx import GridSpec, MonoenergeticCase, solve_monoenergetic, surface_from_vmec_jax_state

run = vj.run_fixed_boundary(
    "/Users/rogeriojorge/local/vmec_jax/examples/data/input.circular_tokamak",
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

- the NTX-to-NEOPAX constructor path reproduces the existing NEOPAX/REFERENCE_EXECUTABLE HDF5
  mapping exactly when given the same coefficient tables
- the `vmec_jax -> booz_xform_jax -> NTX` example path runs locally
- the Boozer `boozmn` loader now interpolates the radial profiles and matches
  the JAX REFERENCE_EXECUTABLE Boozer geometry convention on the shared W7-X test case
- the comparison-only W7-X VMEC reference path now matches the existing NEOPAX
  W7-X subset to better than `1e-2` relative error on `D11`, `D13`, `D31`, and
  `D33`
- the NTX replacement script for the W7-X VMEC database writes the same HDF5
  layout used by the NEOPAX reference tests

What is still open:

- the fully JAX `vmec_jax -> booz_xform_jax -> NTX` W7-X NEOPAX lane still
  needs to be closed to the same level as the comparison-only reference path
- QI VMEC NEOPAX database parity is still open
