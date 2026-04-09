from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import GridSpec, MonoenergeticCase, load_boozmn_surface, solve_monoenergetic
from ntx.geometry import geometry_on_grid

FIXTURE = Path(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/boozmn_wout_W7-X_standard_configuration.nc"
)

if not FIXTURE.exists():
    pytest.skip("local Boozer fixture not available", allow_module_level=True)


def test_load_boozmn_surface_by_rho_and_s_are_consistent():
    by_rho = load_boozmn_surface(FIXTURE, rho=0.25)
    by_s = load_boozmn_surface(FIXTURE, s=0.0625)
    assert abs(by_rho.rho - by_s.rho) < 2.0e-2
    assert by_rho.mode_count > 0
    assert by_rho.surface.nfp == 5
    assert by_rho.surface.b0 > 0.0


def test_boozmn_surface_solves_finite_transport():
    payload = load_boozmn_surface(FIXTURE, rho=0.25)
    result = solve_monoenergetic(
        payload.surface,
        GridSpec(n_theta=9, n_zeta=13, n_xi=24, dtype=jnp.float32),
        MonoenergeticCase(nu_hat=1.0e-4, epsi_hat=0.0),
    )
    assert jnp.isfinite(result.D11)
    assert jnp.isfinite(result.D13)
    assert jnp.isfinite(result.D33)


@pytest.mark.parametrize("rho", [0.12247, 0.5])
def test_boozmn_surface_matches_reference_executable_geometry_convention(rho: float):
    import sys

    if "/Users/rogeriojorge/local/tests/reference_executable_f0" not in sys.path:
        sys.path.insert(0, "/Users/rogeriojorge/local/tests/reference_executable_f0")
    pytest.importorskip("reference_executable")
    import reference_executable  # type: ignore

    field = reference_executable.Field.from_booz_xform(str(FIXTURE), s=rho**2, ntheta=17, nzeta=33, cutoff=0.0)
    payload = load_boozmn_surface(FIXTURE, rho=rho)
    geom = geometry_on_grid(payload.surface, GridSpec(n_theta=17, n_zeta=33, n_xi=10))

    assert jnp.allclose(geom.b, field.Bmag, rtol=5.0e-5, atol=1.0e-7)
    assert jnp.allclose(geom.b_sub_theta, field.B_sub_t, rtol=5.0e-5, atol=1.0e-12)
    assert jnp.allclose(geom.b_sub_zeta, field.B_sub_z, rtol=5.0e-5, atol=1.0e-12)
    assert jnp.allclose(geom.iota, field.iota, rtol=5.0e-5, atol=1.0e-7)
