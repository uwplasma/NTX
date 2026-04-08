from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic
from ntx.geometry import VmecSurface, geometry_on_grid

VMEC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wout_w7x_standardConfig.nc"


def test_load_vmec_surface_and_geometry():
    surface = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25)
    assert isinstance(surface, VmecSurface)
    assert surface.path == VMEC_FIXTURE.resolve()
    assert surface.nfp == 5
    assert surface.ns > 1
    assert surface.loaded_mode_count > 0
    assert surface.total_mode_count >= surface.loaded_mode_count
    assert surface.psi_p is None
    geom = geometry_on_grid(surface, GridSpec(7, 9, 4))
    assert geom.surface_type == "vmec"
    assert geom.b.shape == (7, 9)
    assert jnp.all(jnp.isfinite(geom.b))
    assert jnp.all(jnp.isfinite(geom.jacobian))
    assert float(geom.b0) > 0.0


def test_vmec_radial_option_snaps_to_grid():
    requested = 0.253
    direct = load_vmec_surface(VMEC_FIXTURE, psi_n=requested, vmec_radial_option=0)
    snapped = load_vmec_surface(VMEC_FIXTURE, psi_n=requested, vmec_radial_option=1)
    assert direct.psi_n == pytest.approx(requested)
    assert snapped.requested_psi_n == pytest.approx(requested)
    assert snapped.psi_n != pytest.approx(requested)


def test_vmec_mode_filtering_reduces_mode_count():
    full = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25, min_bmn_to_load=0.0)
    filtered = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25, min_bmn_to_load=1e-2)
    assert filtered.loaded_mode_count < full.loaded_mode_count


def test_vmec_surface_requires_epsi_hat_for_er_input():
    surface = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25)
    with pytest.raises(ValueError, match="er_hat"):
        solve_monoenergetic(surface, GridSpec(7, 9, 4), MonoenergeticCase(1e-3, er_hat=1e-3))
