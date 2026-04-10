import jax.numpy as jnp
import pytest

from ntx import GridSpec, MonoenergeticCase, load_boozmn_surface, solve_monoenergetic
from ntx._checkout_paths import find_neopax_root
from ntx.geometry import geometry_on_grid

NEOPAX_ROOT = find_neopax_root()
FIXTURE = (
    None
    if NEOPAX_ROOT is None
    else NEOPAX_ROOT / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
)

if FIXTURE is None or not FIXTURE.exists():
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
def test_boozmn_surface_geometry_is_self_consistent(rho: float):
    payload = load_boozmn_surface(FIXTURE, rho=rho)
    geom = geometry_on_grid(payload.surface, GridSpec(n_theta=17, n_zeta=33, n_xi=10))
    assert geom.b.shape == (17, 33)
    assert jnp.all(geom.b > 0.0)
    assert jnp.all(jnp.isfinite(geom.d_b_dtheta))
    assert jnp.all(jnp.isfinite(geom.d_b_dzeta))
    assert jnp.all(jnp.isfinite(geom.b_sub_theta))
    assert jnp.all(jnp.isfinite(geom.b_sub_zeta))
    assert jnp.all(jnp.isfinite(geom.radial_drift_spatial))
    assert jnp.isfinite(geom.iota)
