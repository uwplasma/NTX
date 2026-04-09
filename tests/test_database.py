from __future__ import annotations

import jax
import jax.numpy as jnp

from ntx import (
    GridSpec,
    build_monoenergetic_database_arrays,
    example_surface,
    stack_monoenergetic_database_arrays,
)
from ntx.geometry import BoozerSurface


def test_database_builder_matches_scan_shapes_and_values():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    nu_hat = jnp.asarray([1e-2, 2e-2])
    er_hat = jnp.asarray([0.0, 1e-3])

    database = build_monoenergetic_database_arrays(
        surface,
        grid,
        nu_hat,
        er_hat=er_hat,
    )

    assert database.scan_field_name == "er_hat"
    assert database.D11.shape == (2, 2)
    assert database.D13.shape == (2, 2)
    assert database.D33.shape == (2, 2)
    assert jnp.all(jnp.isfinite(database.D11))


def test_database_builder_is_differentiable_in_scan_field():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    nu_hat = jnp.asarray([1e-2, 2e-2])

    grad = jax.grad(
        lambda er: jnp.sum(
            build_monoenergetic_database_arrays(
                surface,
                grid,
                nu_hat,
                er_hat=jnp.asarray([0.0, er]),
            ).D11
        )
    )(1e-3)
    assert jnp.isfinite(grad)


def test_database_stack_adds_surface_axis():
    base = example_surface()
    grid = GridSpec(5, 5, 4)
    nu_hat = jnp.asarray([1e-2, 2e-2])
    er_hat = jnp.asarray([0.0, 1e-3])

    varied = BoozerSurface(
        m=base.m,
        n=base.n,
        b_cos=base.b_cos.at[1].set(0.08),
        nfp=base.nfp,
        iota=base.iota,
        psi_p=base.psi_p,
        b_theta=base.b_theta,
        b_zeta=base.b_zeta,
        chi_p=base.chi_p,
    )

    first = build_monoenergetic_database_arrays(base, grid, nu_hat, er_hat=er_hat)
    second = build_monoenergetic_database_arrays(varied, grid, nu_hat, er_hat=er_hat)
    stacked = stack_monoenergetic_database_arrays((first, second))

    assert stacked.D11.shape == (2, 2, 2)
    assert stacked.D13.shape == (2, 2, 2)
    assert stacked.scan_field_name == "er_hat"
    assert jnp.allclose(stacked.D11[0], first.D11)
