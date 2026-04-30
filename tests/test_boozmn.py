from __future__ import annotations

import jax.numpy as jnp

from ntx import GridSpec, MonoenergeticCase, load_boozmn_surface, solve_monoenergetic
from ntx.geometry import geometry_on_grid

from .fixture_data import SAMPLE_BOOZMN


def test_load_boozmn_surface_by_rho_and_s_are_consistent():
    by_rho = load_boozmn_surface(SAMPLE_BOOZMN, rho=0.5)
    by_s = load_boozmn_surface(SAMPLE_BOOZMN, s=0.25)
    assert abs(by_rho.rho - by_s.rho) < 1.0e-12
    assert by_rho.mode_count > 0
    assert by_rho.surface.nfp == 2
    assert by_rho.surface.b0 > 0.0


def test_load_boozmn_surface_uses_half_grid_for_packed_modes():
    payload = load_boozmn_surface(SAMPLE_BOOZMN, surface_index=1)
    by_s = load_boozmn_surface(SAMPLE_BOOZMN, s=payload.s)

    assert abs(payload.s - 0.375) < 1.0e-12
    assert payload.surface_index == 1
    assert by_s.surface_index == payload.surface_index
    assert jnp.allclose(by_s.surface.b_cos, payload.surface.b_cos)
    assert jnp.allclose(by_s.surface.iota, payload.surface.iota)


def test_boozmn_surface_solves_finite_transport():
    payload = load_boozmn_surface(SAMPLE_BOOZMN, rho=0.5)
    result = solve_monoenergetic(
        payload.surface,
        GridSpec(n_theta=7, n_zeta=9, n_xi=10, dtype=jnp.float32),
        MonoenergeticCase(nu_hat=1.0e-3, epsi_hat=0.0),
    )
    assert jnp.isfinite(result.D11)
    assert jnp.isfinite(result.D13)
    assert jnp.isfinite(result.D33)


def test_boozmn_surface_geometry_is_self_consistent():
    payload = load_boozmn_surface(SAMPLE_BOOZMN, rho=0.5)
    geom = geometry_on_grid(payload.surface, GridSpec(n_theta=9, n_zeta=11, n_xi=6))
    assert geom.b.shape == (9, 11)
    assert jnp.all(geom.b > 0.0)
    assert jnp.all(jnp.isfinite(geom.d_b_dtheta))
    assert jnp.all(jnp.isfinite(geom.d_b_dzeta))
    assert jnp.all(jnp.isfinite(geom.b_sub_theta))
    assert jnp.all(jnp.isfinite(geom.b_sub_zeta))
    assert jnp.all(jnp.isfinite(geom.radial_drift_spatial))
    assert jnp.isfinite(geom.iota)
