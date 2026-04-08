from __future__ import annotations

import jax.numpy as jnp

from ntx import (
    GridSpec,
    MonoenergeticCase,
    example_surface,
    solve_monoenergetic,
    solve_monoenergetic_scan,
)
from ntx.geometry import BoozerSurface


def test_uniform_field_has_zero_radial_transport():
    surface = BoozerSurface(
        m=jnp.asarray([0]),
        n=jnp.asarray([0]),
        b_cos=jnp.asarray([1.0]),
        nfp=1,
        iota=0.6,
        psi_p=1.0,
        b_theta=0.1,
        b_zeta=1.0,
    )
    result = solve_monoenergetic(surface, GridSpec(5, 5, 4), MonoenergeticCase(1e-1))
    assert abs(float(result.D11)) < 1e-10
    assert abs(float(result.D31)) < 1e-10


def test_example_surface_returns_finite_coefficients():
    result = solve_monoenergetic(example_surface(), GridSpec(5, 5, 6), MonoenergeticCase(1e-2))
    values = jnp.asarray([result.D11, result.D31, result.D13, result.D33, result.D33_spitzer])
    assert jnp.all(jnp.isfinite(values))
    assert result.D33_spitzer > 0.0
    assert result.D11 >= -1e-10


def test_n_xi_two_boundary_case_runs():
    result = solve_monoenergetic(example_surface(), GridSpec(5, 5, 2), MonoenergeticCase(1e-2))
    assert jnp.isfinite(result.D33_spitzer)


def test_vmap_parameter_scan_matches_single_solve_shape():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    scan = solve_monoenergetic_scan(surface, grid, jnp.asarray([1e-2, 2e-2]))
    single = solve_monoenergetic(surface, grid, MonoenergeticCase(1e-2))
    assert scan["D11"].shape == (2,)
    assert jnp.allclose(scan["D11"][0], single.D11)


def test_spitzer_scales_inverse_with_collisionality():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    low = solve_monoenergetic(surface, grid, MonoenergeticCase(1e-2))
    high = solve_monoenergetic(surface, grid, MonoenergeticCase(2e-2))
    assert jnp.allclose(low.D33_spitzer / high.D33_spitzer, 2.0, rtol=1e-10)
