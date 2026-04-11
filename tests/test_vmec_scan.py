from __future__ import annotations

import jax.numpy as jnp

from ntx import GridSpec, load_vmec_surface, solve_monoenergetic_scan

from .fixture_data import SAMPLE_WOUT


def test_vmec_scan_matches_pointwise_solve_shape():
    surface = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25)
    nu = jnp.asarray([1.0e-4, 1.0e-3, 1.0e-2])
    er = jnp.asarray([0.0, 1.0e-3, 2.0e-3])
    scan = solve_monoenergetic_scan(surface, GridSpec(7, 9, 6), nu, er_hat=er)
    assert scan["D11"].shape == (3,)
    assert scan["D13"].shape == (3,)
    assert scan["D33"].shape == (3,)
    assert jnp.all(jnp.isfinite(scan["D11"]))


def test_vmec_scan_er_hat_matches_epsi_hat():
    surface = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25)
    nu = jnp.asarray([1.0e-3, 2.0e-3])
    er = jnp.asarray([1.0e-3, 2.0e-3])
    er_scan = solve_monoenergetic_scan(surface, GridSpec(7, 9, 6), nu, er_hat=er)
    epsi = er / surface.transport_psi_scale
    epsi_scan = solve_monoenergetic_scan(surface, GridSpec(7, 9, 6), nu, epsi_hat=epsi)
    assert jnp.max(jnp.abs(er_scan["D11"] - epsi_scan["D11"])) < 1.0e-12
