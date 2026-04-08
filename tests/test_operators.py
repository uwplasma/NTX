from __future__ import annotations

import jax.numpy as jnp

from ntx.geometry import BoozerSurface, geometry_on_grid
from ntx.grids import GridSpec
from ntx.operators import (
    OperatorContext,
    apply_nullspace_condition,
    derivative_blocks,
    operator_blocks,
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
