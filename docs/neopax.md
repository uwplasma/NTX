# NEOPAX

NTX now has an explicit mapping layer for NEOPAX-style monoenergetic databases.

The intended workflow is:

1. generate monoenergetic coefficients with NTX
2. map them into `NEOPAX.Monoenergetic`
3. evaluate fluxes or solve transport equations in NEOPAX without going through
   an intermediate HDF5 file

## Local Install

For the current local JAX toolchain, install the geometry and transport stack
from the local checkouts:

```bash
python -m pip install -e /Users/rogeriojorge/local/vmec_jax
python -m pip install -e /Users/rogeriojorge/local/booz_xform_jax
python -m pip install -e /Users/rogeriojorge/local/tests/NEOPAX
python -m pip install -e /Users/rogeriojorge/local/.NTX"[dev,docs,io]"
```

## NTX Scan To NEOPAX

```python
from pathlib import Path

from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    load_vmec_surface,
    to_neopax_monoenergetic,
)

reference = load_neopax_reference_scan(
    Path("/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/Dij_NEOPAX_FULL_S_NEW_W7X.h5")
)

def surface_loader(rho_value: float):
    return load_vmec_surface(
        "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/wout_W7-X_standard_configuration.nc",
        psi_n=rho_value**2,
        vmec_radial_option=1,
        vmec_nyquist_option=2,
        vmec_mode_convention="filtered_nyquist",
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
- the NTX-generated W7-X subset stays in the same regime as the existing NEOPAX
  reference subset and gives comparable `D33`

What is still open:

- full W7-X and QI VMEC parity between NTX-generated databases and the existing
  NEOPAX reference databases is not closed yet
- the remaining mismatch is in the VMEC/Boozer normalization and interpretation
  path, not in the NEOPAX adapter itself
