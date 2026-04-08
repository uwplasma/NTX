from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic
from ntx.geometry import VmecSurface, geometry_on_grid

VMEC_FIXTURE = Path(
    "/Users/rogeriojorge/local/tests/sfincs_jax/tests/ref/wout_w7x_standardConfig.nc"
)


@pytest.mark.skipif(not VMEC_FIXTURE.exists(), reason="local VMEC fixture not available")
def test_load_vmec_surface_and_geometry():
    surface = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25)
    assert isinstance(surface, VmecSurface)
    geom = geometry_on_grid(surface, GridSpec(7, 9, 4))
    assert geom.surface_type == "vmec"
    assert geom.b.shape == (7, 9)
    assert jnp.all(jnp.isfinite(geom.b))
    assert jnp.all(jnp.isfinite(geom.jacobian))
    assert float(geom.b0) > 0.0


@pytest.mark.skipif(not VMEC_FIXTURE.exists(), reason="local VMEC fixture not available")
def test_vmec_surface_solves_with_epsi_hat():
    surface = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25)
    result = solve_monoenergetic(
        surface,
        GridSpec(7, 9, 4),
        MonoenergeticCase(nu_hat=1e-3, epsi_hat=1e-3),
    )
    values = jnp.asarray([result.D11, result.D31, result.D13, result.D33, result.D33_spitzer])
    assert jnp.all(jnp.isfinite(values))
