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

    f1, f3, _ = _solve_modes_with_tail_residual(ctx, n_xi, d_theta, d_zeta, s1, s3)
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
    factors = block_thomas_factor_fn(_operator_block_fn(ctx, d_theta, d_zeta), n_blocks=n_xi + 1)
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
        return _conditioned_operator_blocks(ctx, k, d_theta, d_zeta)

    return block_fn


def _conditioned_operator_blocks(
    ctx: OperatorContext,
    k: int | Array,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array]:
    lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)

    def fix_nullspace(args):
        diagonal_in, upper_in = args
        diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal_in, upper_in)
        assert upper_fixed is not None
        return diagonal_fixed, upper_fixed

    diagonal, upper = jax.lax.cond(k == 0, fix_nullspace, lambda args: args, (diagonal, upper))
    return lower, diagonal, upper


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
        lower, diagonal, upper = _conditioned_operator_blocks(ctx, k, d_theta, d_zeta)
        value = diagonal @ modes[k] - source[k]
        if k > 0:
            value = value + lower @ modes[k - 1]
        if k < n_xi:
            value = value + upper @ modes[k + 1]
        residuals.append(value)
    residual = jnp.concatenate(residuals)
    return jnp.linalg.norm(residual) / jnp.sqrt(residual.size)


def _full_mode_relative_residual_norm(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    source: Array,
    modes: Array,
) -> Array:
    residual_rms = _full_mode_residual_norm(ctx, n_xi, d_theta, d_zeta, source, modes)
    source_rms = jnp.linalg.norm(source) / jnp.sqrt(source.size)
    tiny = jnp.finfo(residual_rms.dtype).tiny
    return residual_rms / jnp.maximum(source_rms, tiny)


def _full_mode_transpose_relative_residual_norm(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    source_bar: Array,
    adjoint_modes: Array,
) -> Array:
    """Evaluate ``||A.T @ lambda - g|| / ||g||`` from physics blocks."""

    if adjoint_modes.shape[0] != n_xi + 1 or source_bar.shape[0] != n_xi + 1:
        raise ValueError("transpose residual requires n_xi + 1 source and solution modes")
    applied = [jnp.zeros_like(adjoint_modes[0]) for _ in range(n_xi + 1)]
    for k in range(n_xi + 1):
        lower, diagonal, upper = _conditioned_operator_blocks(ctx, k, d_theta, d_zeta)
        applied[k] = applied[k] + diagonal.T @ adjoint_modes[k]
        if k > 0:
            applied[k - 1] = applied[k - 1] + lower.T @ adjoint_modes[k]
        if k < n_xi:
            applied[k + 1] = applied[k + 1] + upper.T @ adjoint_modes[k]
    residual = jnp.concatenate(
        [value - source for value, source in zip(applied, source_bar, strict=True)]
    )
    source_norm = jnp.linalg.norm(source_bar)
    tiny = jnp.finfo(residual.dtype).tiny
    return jnp.linalg.norm(residual) / jnp.maximum(source_norm, tiny)
