"""Prepared-solver adjoint and custom-VJP helper algebra."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ._solver_factorization import (
    _factorize_prepared_modes,
    _solve_factorized_modes,
)
from ._solver_types import PreparedMonoenergeticSystem
from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .operators import OperatorContext, parameter_derivative_blocks, source_modes
from .transport import coefficients_from_modes


def _prepared_implicit_vjp_primal(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    geom = prepared.geometry
    grid = prepared.grid
    ctx = _operator_context(prepared.surface, geom, grid, nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    saved_lu, saved_piv, saved_lower, saved_upper = _factorize_prepared_modes(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
    )
    f1_full = _solve_factorized_modes(saved_lu, saved_piv, saved_lower, saved_upper, s1)
    f3_full = _solve_factorized_modes(saved_lu, saved_piv, saved_lower, saved_upper, s3)

    def coefficient_fn(modes1, modes3, nu_value):
        return jnp.stack(coefficients_from_modes(geom, modes1, modes3, nu_value))

    coefficients = coefficient_fn(f1_full[:3], f3_full[:3], ctx.nu_hat)
    return coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper


def _coefficient_mode_pullback(
    geom,
    f1_low: Array,
    f3_low: Array,
    nu_hat: Array,
    coefficient_bar: Array,
) -> tuple[Array, Array, Array]:
    def coefficient_fn(modes1, modes3, nu_value):
        return jnp.stack(coefficients_from_modes(geom, modes1, modes3, nu_value))

    _, pullback = jax.vjp(coefficient_fn, f1_low, f3_low, nu_hat)
    f1_bar, f3_bar, nu_bar = pullback(coefficient_bar)
    return f1_bar, f3_bar, nu_bar


def _parameter_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
) -> tuple[Array, Array]:
    def zero_first_row(block: Array) -> Array:
        return block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))

    nu_bar = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    epsi_bar = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = zero_first_row(diagonal_nu)
            diagonal_epsi = zero_first_row(diagonal_epsi)
        nu_bar = nu_bar - (
            jnp.vdot(lambda1[k], diagonal_nu @ f1_full[k])
            + jnp.vdot(lambda3[k], diagonal_nu @ f3_full[k])
        )
        epsi_bar = epsi_bar - (
            jnp.vdot(lambda1[k], diagonal_epsi @ f1_full[k])
            + jnp.vdot(lambda3[k], diagonal_epsi @ f3_full[k])
        )
    return nu_bar, epsi_bar


def _operator_context(
    surface: BoozerSurface | VmecSurface,
    geom,
    grid: GridSpec,
    nu_hat,
    epsi_hat,
) -> OperatorContext:
    return OperatorContext(
        surface=surface,
        geometry=geom,
        nu_hat=jnp.asarray(nu_hat, dtype=grid.jax_dtype),
        epsi_hat=jnp.asarray(epsi_hat, dtype=grid.jax_dtype),
    )
