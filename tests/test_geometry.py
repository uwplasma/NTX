from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from ntx.geometry import (
    BoozerSurface,
    evaluate_boozer_modes,
    evaluate_fourier_series,
    example_surface,
    geometry_on_grid,
)
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


def test_fourier_series_supports_sine_coefficients():
    theta = jnp.asarray([0.0, jnp.pi / 2])
    zeta = jnp.asarray([0.0, 0.0])
    value, d_dtheta, d_dzeta = evaluate_fourier_series(
        m=jnp.asarray([1]),
        n=jnp.asarray([0]),
        cos_coeffs=jnp.asarray([0.2]),
        theta=theta,
        zeta=zeta,
        nfp=1,
        sin_coeffs=jnp.asarray([0.3]),
    )

    assert jnp.allclose(value, jnp.asarray([0.2, 0.3]), atol=1e-12)
    assert jnp.allclose(d_dtheta, jnp.asarray([0.3, -0.2]), atol=1e-12)
    assert jnp.allclose(d_dzeta, jnp.zeros_like(d_dzeta))


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
    fixture = Path(__file__).resolve().parent / "fixtures" / "sample_surface.ddkes2.data"
    surface = load_dkes_surface(fixture)
    assert surface.nfp == 2
    assert jnp.isclose(surface.psi_p, 1.25)
    assert jnp.isclose(surface.iota, 0.4)
    assert jnp.isclose(surface.b0, 2.4)
    b00, _, _ = evaluate_boozer_modes(surface, jnp.asarray(0.0), jnp.asarray(0.0))
    assert jnp.isclose(b00, 2.57)
