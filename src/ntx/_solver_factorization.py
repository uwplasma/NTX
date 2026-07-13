"""Block-tridiagonal solve and factorized adjoint helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import lu_factor, lu_solve

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

    lower_terminal, delta, lower_next = _terminal_delta(ctx, n_xi, d_theta, d_zeta)
    x = lu_solve(lu_factor(delta), lower_next)

    n_fs = delta.shape[0]
    saved_delta_init = jnp.zeros((3, n_fs, n_fs), dtype=delta.dtype)
    saved_lower_init = jnp.zeros((3, n_fs, n_fs), dtype=delta.dtype)
    saved_upper_init = jnp.zeros((3, n_fs, n_fs), dtype=delta.dtype)
    if n_xi == 2:
        saved_delta_init = saved_delta_init.at[2].set(delta)
        saved_lower_init = saved_lower_init.at[2].set(lower_terminal)

    def scan_step(carry, k):
        x_prev, saved_delta, saved_lower, saved_upper = carry
        lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)

        def fix_nullspace(args):
            diagonal_in, upper_in = args
            diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal_in, upper_in)
            assert upper_fixed is not None
            return diagonal_fixed, upper_fixed

        diagonal, upper = jax.lax.cond(k == 0, fix_nullspace, lambda args: args, (diagonal, upper))
        delta_k = diagonal - upper @ x_prev

        def save_needed(args):
            saved_delta_in, saved_lower_in, saved_upper_in = args
            return (
                saved_delta_in.at[k].set(delta_k),
                saved_lower_in.at[k].set(lower),
                saved_upper_in.at[k].set(upper),
            )

        saved_delta, saved_lower, saved_upper = jax.lax.cond(
            k <= 2,
            save_needed,
            lambda args: args,
            (saved_delta, saved_lower, saved_upper),
        )
        x_next = jax.lax.cond(
            k > 0,
            lambda _: lu_solve(lu_factor(delta_k), lower),
            lambda _: x_prev,
            operand=None,
        )
        return (x_next, saved_delta, saved_lower, saved_upper), None

    ks = jnp.arange(n_xi - 1, -1, -1)
    (_, saved_delta, saved_lower, saved_upper), _ = jax.lax.scan(
        scan_step,
        (x, saved_delta_init, saved_lower_init, saved_upper_init),
        ks,
    )

    sigma1 = {2: s1[2], 1: s1[1], 0: s1[0]}
    sigma3 = {2: s3[2], 1: s3[1], 0: s3[0]}

    lu2 = lu_factor(saved_delta[2])
    y1 = lu_solve(lu2, sigma1[2])
    sigma1[1] = s1[1] - saved_upper[1] @ y1

    lu1 = lu_factor(saved_delta[1])
    y13 = lu_solve(lu1, jnp.stack((sigma1[1], sigma3[1]), axis=-1))
    y1 = y13[:, 0]
    y3 = y13[:, 1]
    sigma1[0] = s1[0] - saved_upper[0] @ y1
    sigma3[0] = s3[0] - saved_upper[0] @ y3

    f1 = []
    f3 = []
    lu0 = lu_factor(saved_delta[0])
    f03 = lu_solve(lu0, jnp.stack((sigma1[0], sigma3[0]), axis=-1))
    f1_0 = f03[:, 0]
    f3_0 = f03[:, 1]
    f1.append(f1_0)
    f3.append(f3_0)
    rhs_13 = jnp.stack(
        (
            sigma1[1] - saved_lower[1] @ f1[0],
            sigma3[1] - saved_lower[1] @ f3[0],
        ),
        axis=-1,
    )
    f13 = lu_solve(lu1, rhs_13)
    f1.append(f13[:, 0])
    f3.append(f13[:, 1])

    rhs_23 = jnp.stack(
        (
            sigma1[2] - saved_lower[2] @ f1[1],
            sigma3[2] - saved_lower[2] @ f3[1],
        ),
        axis=-1,
    )
    f23 = lu_solve(lu2, rhs_23)
    f1.append(f23[:, 0])
    f3.append(f23[:, 1])
    f1_modes = jnp.stack(f1)
    f3_modes = jnp.stack(f3)
    residuals = (
        saved_delta[0] @ f1_modes[0] - sigma1[0],
        saved_lower[1] @ f1_modes[0] + saved_delta[1] @ f1_modes[1] - sigma1[1],
        saved_lower[2] @ f1_modes[1] + saved_delta[2] @ f1_modes[2] - sigma1[2],
    )
    residual = jnp.concatenate(residuals)
    residual_l2 = jnp.linalg.norm(residual) / jnp.sqrt(residual.size)
    return f1_modes, f3_modes, residual_l2


def _factorize_prepared_modes(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array, Array]:
    lower_terminal, delta_terminal, lower_next = _terminal_delta(ctx, n_xi, d_theta, d_zeta)
    lu_terminal, piv_terminal = lu_factor(delta_terminal)
    x_prev = lu_solve((lu_terminal, piv_terminal), lower_next)

    zeros_block = jnp.zeros_like(delta_terminal)
    zeros_piv = jnp.zeros((delta_terminal.shape[0],), dtype=jnp.int32)
    saved_lu = [zeros_block] * (n_xi + 1)
    saved_piv = [zeros_piv] * (n_xi + 1)
    saved_lower = [zeros_block] * (n_xi + 1)
    saved_upper = [zeros_block] * (n_xi + 1)
    saved_lu[n_xi] = lu_terminal
    saved_piv[n_xi] = piv_terminal
    saved_lower[n_xi] = lower_terminal

    for k in range(n_xi - 1, -1, -1):
        lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)
        if k == 0:
            diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal, upper)
            assert upper_fixed is not None
            diagonal = diagonal_fixed
            upper = upper_fixed
        delta_k = diagonal - upper @ x_prev
        lu_k, piv_k = lu_factor(delta_k)
        saved_lu[k] = lu_k
        saved_piv[k] = piv_k
        saved_lower[k] = lower
        saved_upper[k] = upper
        if k > 0:
            x_prev = lu_solve((lu_k, piv_k), lower)

    return (
        jnp.stack(saved_lu),
        jnp.stack(saved_piv),
        jnp.stack(saved_lower),
        jnp.stack(saved_upper),
    )


def _solve_factorized_modes(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    source: Array,
) -> Array:
    n_xi = source.shape[0] - 1
    y = [jnp.zeros_like(source[0])] * (n_xi + 1)
    y[n_xi] = lu_solve((saved_lu[n_xi], saved_piv[n_xi]), source[n_xi])
    for k in range(n_xi - 1, -1, -1):
        rhs = source[k] - saved_upper[k] @ y[k + 1]
        y[k] = lu_solve((saved_lu[k], saved_piv[k]), rhs)

    modes = [y[0]]
    for k in range(1, n_xi + 1):
        propagated = lu_solve((saved_lu[k], saved_piv[k]), saved_lower[k] @ modes[k - 1])
        modes.append(y[k] - propagated)
    return jnp.stack(modes)


def _solve_factorized_adjoint(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    source_bar: Array,
) -> Array:
    n_xi = source_bar.shape[0] - 1
    mu = [jnp.zeros_like(source_bar[0])] * (n_xi + 1)
    mu[n_xi] = source_bar[n_xi]
    for k in range(n_xi - 1, -1, -1):
        propagated = lu_solve((saved_lu[k + 1], saved_piv[k + 1]), mu[k + 1], trans=1)
        mu[k] = source_bar[k] - saved_lower[k + 1].T @ propagated

    adjoint = [lu_solve((saved_lu[0], saved_piv[0]), mu[0], trans=1)]
    for k in range(1, n_xi + 1):
        rhs = mu[k] - saved_upper[k - 1].T @ adjoint[k - 1]
        adjoint.append(lu_solve((saved_lu[k], saved_piv[k]), rhs, trans=1))
    return jnp.stack(adjoint)


def _terminal_delta(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array]:
    lower, diagonal, _ = operator_blocks(ctx, n_xi, d_theta, d_zeta)
    return lower, diagonal, lower


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
