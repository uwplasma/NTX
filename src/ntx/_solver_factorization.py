"""Block-tridiagonal solve and factorized adjoint helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from solvax import (
    BlockTridiagFactors,
    block_thomas_factor_fn,
    block_thomas_solve,
    block_thomas_truncated_fn_with_residual,
)

from .operators import OperatorContext, apply_nullspace_condition, operator_blocks


def _solve_modes(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    s1: Array,
    s3: Array,
) -> tuple[Array, Array]:
    """Return source solutions for modes 0, 1, and 2."""

    f1, f3, _ = _solve_modes_with_tail_residual(
        ctx, n_xi, d_theta, d_zeta, s1, s3
    )
    return f1, f3


def _solve_modes_with_tail_residual(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    s1: Array,
    s3: Array,
) -> tuple[Array, Array, Array]:
    """Return retained modes and the residual of the tail-eliminated system."""

    block_fn = _operator_block_fn(ctx, d_theta, d_zeta)
    keep_lowest = 3
    rhs_low = jnp.stack((s1[:keep_lowest], s3[:keep_lowest]), axis=-1)
    modes, residual = block_thomas_truncated_fn_with_residual(
        block_fn,
        n_blocks=n_xi + 1,
        rhs_low=rhs_low,
        keep_lowest=keep_lowest,
        residual_rhs_index=0,
    )
    f1_modes = modes[..., 0]
    f3_modes = modes[..., 1]
    return f1_modes, f3_modes, residual


def _factorize_prepared_modes(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array, Array]:
    factors = block_thomas_factor_fn(
        _operator_block_fn(ctx, d_theta, d_zeta), n_blocks=n_xi + 1
    )
    return factors.delta_lu, factors.delta_piv, factors.lower, factors.upper


def _solve_factorized_modes(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    source: Array,
) -> Array:
    factors = BlockTridiagFactors(saved_lu, saved_piv, saved_lower, saved_upper)
    return block_thomas_solve(factors, source)


def _solve_factorized_adjoint(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    source_bar: Array,
) -> Array:
    factors = BlockTridiagFactors(saved_lu, saved_piv, saved_lower, saved_upper)
    return block_thomas_solve(factors, source_bar, transpose=True)


def _operator_block_fn(ctx: OperatorContext, d_theta: Array, d_zeta: Array):
    def block_fn(k):
        lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)

        def fix_nullspace(args):
            diagonal_in, upper_in = args
            diagonal_fixed, upper_fixed = apply_nullspace_condition(
                diagonal_in, upper_in
            )
            assert upper_fixed is not None
            return diagonal_fixed, upper_fixed

        diagonal, upper = jax.lax.cond(
            k == 0, fix_nullspace, lambda args: args, (diagonal, upper)
        )
        return lower, diagonal, upper

    return block_fn


def _full_mode_residual_norm(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    source: Array,
    modes: Array,
) -> Array:
    """Evaluate the original block equations for every retained full mode."""
    if modes.shape[0] != n_xi + 1 or source.shape[0] != n_xi + 1:
        raise ValueError("full residual requires n_xi + 1 source and solution modes")
    residuals = []
    for k in range(n_xi + 1):
        lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)
        if k == 0:
            diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal, upper)
            assert upper_fixed is not None
            diagonal = diagonal_fixed
            upper = upper_fixed
        value = diagonal @ modes[k] - source[k]
        if k > 0:
            value = value + lower @ modes[k - 1]
        if k < n_xi:
            value = value + upper @ modes[k + 1]
        residuals.append(value)
    residual = jnp.concatenate(residuals)
    return jnp.linalg.norm(residual) / jnp.sqrt(residual.size)
