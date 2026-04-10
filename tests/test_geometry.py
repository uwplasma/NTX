from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from ntx.geometry import BoozerSurface, evaluate_boozer_modes, example_surface, geometry_on_grid
from ntx.grids import GridSpec, flux_surface_average
from ntx.io import load_dkes_surface


def test_boozer_mode_evaluation_single_cosine():
    surface = BoozerSurface(
        m=jnp.asarray([0, 1]),
        n=jnp.asarray([0, 0]),
        b_cos=jnp.asarray([1.0, 0.2]),
        nfp=1,
        iota=0.5,
        psi_p=1.0,
        b_theta=0.1,
        b_zeta=1.0,
    )
    theta = jnp.asarray([0.0, jnp.pi / 2])
    zeta = jnp.asarray([0.0, 0.0])
    b, dbdt, dbdz = evaluate_boozer_modes(surface, theta, zeta)
    assert jnp.allclose(b, jnp.asarray([1.2, 1.0]))
    assert jnp.allclose(dbdt, jnp.asarray([0.0, -0.2]))
    assert jnp.allclose(dbdz, jnp.zeros_like(dbdz))


def test_flux_surface_average_of_constant_is_constant():
    geom = geometry_on_grid(example_surface(), GridSpec(5, 5, 4))
    value = flux_surface_average(
        3.0 * jnp.ones_like(geom.b),
        geom.jacobian,
        geom.grid.dtheta,
        geom.grid.dzeta,
    )
    assert jnp.allclose(value, 3.0)


def test_load_dkes_surface_matches_reference_sign_convention():
    fixture = Path(__file__).resolve().parent / "fixtures" / "w7x_eim_sample.ddkes2.data"
    surface = load_dkes_surface(fixture)
    assert surface.nfp == 5
    assert jnp.isclose(surface.psi_p, -0.5237)
    assert jnp.isclose(surface.iota, -0.8615619629558907)
    assert jnp.isclose(surface.b0, 2.4311)
    b00, _, _ = evaluate_boozer_modes(surface, jnp.asarray(0.0), jnp.asarray(0.0))
    assert jnp.isclose(b00, 2.4093327)
