"""Block-tridiagonal solve and factorized adjoint helpers.

The prepared factorization/solve pair is backed by :mod:`solvax`
(``block_thomas_factor`` / ``block_thomas_solve``), which implements the same
Schur-complement (block Thomas) recursion this module previously carried
inline:

    Delta_{N} = D_{N},  Delta_k = D_k - U_k Delta_{k+1}^{-1} L_{k+1}
    sigma_k   = b_k - U_k Delta_{k+1}^{-1} sigma_{k+1}
    x_0 = Delta_0^{-1} sigma_0,  x_k = Delta_k^{-1} (sigma_k - L_k x_{k-1})

Two pieces intentionally stay local:

- ``_solve_modes`` fuses on-the-fly operator-block assembly with a truncated
  sweep so that only O(1) blocks are ever materialized. solvax's array-based
  ``block_thomas_truncated`` would require materializing all
  ``(n_xi + 1, n_fs, n_fs)`` blocks up front, which regresses peak memory for
  large grids and under the vmapped nu/epsi scans.
- ``_solve_factorized_adjoint`` solves the transposed system by reusing the
  forward LU factors with ``trans=1`` (``Delta_k^T`` are exactly the Schur
  complements of ``A^T``); solvax does not currently expose a transposed
  block-Thomas solve on precomputed factors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import lu_factor, lu_solve
from solvax import BlockTridiagFactors, block_thomas_factor, block_thomas_solve

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
    return jnp.stack(f1), jnp.stack(f3)


def _stacked_operator_blocks(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array]:
    """Materialize `(L_k, D_k, U_k)` stacks with the nullspace fix applied."""

    lower, diagonal, upper = jax.vmap(
        lambda k: operator_blocks(ctx, k, d_theta, d_zeta)
    )(jnp.arange(n_xi + 1))
    diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal[0], upper[0])
    assert upper_fixed is not None
    diagonal = diagonal.at[0].set(diagonal_fixed)
    upper = upper.at[0].set(upper_fixed)
    # The terminal super-diagonal block does not exist; keep it zero so the
    # saved factors match the previous layout exactly.
    upper = upper.at[n_xi].set(jnp.zeros_like(upper[n_xi]))
    return lower, diagonal, upper


def _factorize_prepared_modes(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array, Array]:
    lower, diagonal, upper = _stacked_operator_blocks(ctx, n_xi, d_theta, d_zeta)
    factors = block_thomas_factor(lower, diagonal, upper)
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


def _residual_norm(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    source: Array,
    modes: Array,
) -> Array:
    residuals = []
    for k in range(3):
        lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)
        if k == 0:
            diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal, upper)
            assert upper_fixed is not None
            diagonal = diagonal_fixed
            upper = upper_fixed
        value = diagonal @ modes[k] - source[k]
        if k > 0:
            value = value + lower @ modes[k - 1]
        if k < 2:
            value = value + upper @ modes[k + 1]
        residuals.append(value)
    residual = jnp.concatenate(residuals)
    _ = n_xi
    return jnp.linalg.norm(residual) / jnp.sqrt(residual.size)
