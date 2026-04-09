from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import GridSpec, MonoenergeticCase, load_boozmn_surface, solve_monoenergetic

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
