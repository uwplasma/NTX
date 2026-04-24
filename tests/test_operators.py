from __future__ import annotations

import jax
import jax.numpy as jnp

from ntx import example_surface
from ntx.geometry import BoozerSurface, geometry_on_grid
from ntx.grids import GridSpec, flatten_fs
from ntx.operators import (
    OperatorContext,
    apply_nullspace_condition,
    derivative_blocks,
    operator_blocks,
    parameter_derivative_blocks,
    source_modes,
)


def constant_surface():
    return BoozerSurface(
        m=jnp.asarray([0]),
        n=jnp.asarray([0]),
        b_cos=jnp.asarray([1.0]),
        nfp=1,
        iota=0.6,
        psi_p=1.0,
        b_theta=0.1,
        b_zeta=1.0,
    )


def test_uniform_field_source_has_no_radial_drift():
    spec = GridSpec(5, 5, 4)
    surface = constant_surface()
    geom = geometry_on_grid(surface, spec)
    ctx = OperatorContext(surface, geom, jnp.asarray(1e-2), jnp.asarray(0.0))
    s1, s3 = source_modes(ctx, spec.n_xi)
    assert jnp.allclose(s1, 0.0)
    assert jnp.allclose(s3[1], 1.0)


def test_parallel_source_uses_b_not_b_over_b0():
    spec = GridSpec(5, 5, 4)
    surface = BoozerSurface(
        m=jnp.asarray([0]),
        n=jnp.asarray([0]),
        b_cos=jnp.asarray([2.0]),
        nfp=1,
        iota=0.6,
        psi_p=1.0,
        b_theta=0.1,
        b_zeta=1.0,
        b0=2.0,
    )
    geom = geometry_on_grid(surface, spec)
    ctx = OperatorContext(surface, geom, jnp.asarray(1e-2), jnp.asarray(0.0))
    _, s3 = source_modes(ctx, spec.n_xi)
    assert jnp.allclose(s3[1], 2.0)


def test_source_modes_match_finite_legendre_projection():
    spec = GridSpec(5, 5, 4)
    surface = example_surface()
    geom = geometry_on_grid(surface, spec)
    ctx = OperatorContext(surface, geom, jnp.asarray(1e-2), jnp.asarray(0.0))
    s1, s3 = source_modes(ctx, spec.n_xi)

    vm0 = flatten_fs(geom.radial_drift_spatial * (2.0 / 3.0))
    vm2 = flatten_fs(geom.radial_drift_spatial / 3.0)

    assert s1.shape == (spec.n_xi + 1, spec.n_fs)
    assert s3.shape == (spec.n_xi + 1, spec.n_fs)
    assert jnp.allclose(s1[0, 0], 0.0)
    assert jnp.allclose(s1[0, 1:], -vm0[1:])
    assert jnp.allclose(s1[1], 0.0)
    assert jnp.allclose(s1[2], -vm2)
    assert jnp.allclose(s1[3:], 0.0)
    assert jnp.allclose(s3[0], 0.0)
    assert jnp.allclose(s3[1], flatten_fs(geom.b))
    assert jnp.allclose(s3[2:], 0.0)


def test_nullspace_condition_replaces_first_row():
    matrix = jnp.ones((4, 4))
    upper = jnp.ones((4, 4))
    fixed, upper_fixed = apply_nullspace_condition(matrix, upper)
    assert jnp.allclose(fixed[0], jnp.asarray([1.0, 0.0, 0.0, 0.0]))
    assert jnp.allclose(upper_fixed[0], jnp.zeros(4))


def test_operator_block_shapes():
    spec = GridSpec(5, 5, 4)
    surface = constant_surface()
    geom = geometry_on_grid(surface, spec)
    ctx = OperatorContext(surface, geom, jnp.asarray(1e-2), jnp.asarray(0.0))
    dtheta, dzeta = derivative_blocks(geom)
    lower, diagonal, upper = operator_blocks(ctx, 1, dtheta, dzeta)
    assert lower.shape == (25, 25)
    assert diagonal.shape == (25, 25)
    assert upper.shape == (25, 25)


def test_parameter_derivative_blocks_match_operator_autodiff():
    spec = GridSpec(5, 5, 4)
    surface = BoozerSurface(
        m=jnp.asarray([0, 1, 1, 2], dtype=jnp.int32),
        n=jnp.asarray([0, 0, 1, -1], dtype=jnp.int32),
        b_cos=jnp.asarray([1.0, 0.06, 0.025, 0.01]),
        nfp=5,
        iota=0.85,
        psi_p=1.0,
        chi_p=0.85,
        b_theta=0.05,
        b_zeta=1.0,
    )
    geom = geometry_on_grid(surface, spec)
    dtheta, dzeta = derivative_blocks(geom)
    k = 2
    nu_hat = jnp.asarray(1.0e-2, dtype=spec.jax_dtype)
    epsi_hat = jnp.asarray(2.0e-3, dtype=spec.jax_dtype)

    def diagonal_block(nu_value, epsi_value):
        ctx = OperatorContext(surface, geom, nu_value, epsi_value)
        return operator_blocks(ctx, k, dtheta, dzeta)[1]

    ctx = OperatorContext(surface, geom, nu_hat, epsi_hat)
    nu_block, epsi_block = parameter_derivative_blocks(ctx, k, dtheta, dzeta)
    autodiff_nu_block = jax.jacfwd(lambda value: diagonal_block(value, epsi_hat))(nu_hat)
    autodiff_epsi_block = jax.jacfwd(lambda value: diagonal_block(nu_hat, value))(epsi_hat)

    assert jnp.allclose(nu_block, autodiff_nu_block, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(epsi_block, autodiff_epsi_block, rtol=1.0e-12, atol=1.0e-12)
