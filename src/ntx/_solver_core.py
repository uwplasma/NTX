"""Prepared monoenergetic solve path and custom-VJP core."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import lu_factor, lu_solve

from ._solver_types import (
    CompiledPreparedSolver,
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    TransportResult,
    transport_result_from_arrays,
)
from .config import enable_x64
from .geometry import BoozerSurface, VmecSurface, geometry_on_grid
from .grids import GridSpec
from .operators import (
    OperatorContext,
    apply_nullspace_condition,
    derivative_blocks,
    operator_blocks,
    parameter_derivative_blocks,
    source_modes,
)
from .transport import coefficients_from_modes, onsager_error


def prepare_monoenergetic_system(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
) -> PreparedMonoenergeticSystem:
    """Precompute geometry and derivative blocks for repeated solves."""

    enable_x64(grid.x64)
    geom = geometry_on_grid(surface, grid)
    d_theta, d_zeta = derivative_blocks(geom)
    return PreparedMonoenergeticSystem(
        surface=surface,
        grid=grid,
        geometry=geom,
        d_theta=d_theta,
        d_zeta=d_zeta,
    )


def solve_monoenergetic(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    case: MonoenergeticCase,
) -> TransportResult:
    """Solve one monoenergetic DKE case."""

    prepared = prepare_monoenergetic_system(surface, grid)
    return solve_prepared(prepared, case)


def solve_monoenergetic_internal(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    case: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Solve one monoenergetic case and return `(Dij, f, s)` low-order arrays."""

    prepared = prepare_monoenergetic_system(surface, grid)
    return solve_prepared_internal(prepared, case)


def solve_prepared(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> TransportResult:
    """Solve one monoenergetic case using precomputed geometry and derivatives."""

    return transport_result_from_arrays(_solve_prepared_arrays(prepared, case))


def solve_prepared_internal(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Solve one prepared monoenergetic case and return `(Dij, f, s)` low-order arrays."""

    values = _solve_prepared_arrays(prepared, case)
    result = transport_result_from_arrays(values)
    dij = _monoenergetic_matrix(result.D11, result.D31, result.D13, result.D33)
    return dij, values[9], values[10]


def solve_prepared_coefficient_vector(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> Array:
    """Return the coefficient vector `[D11, D31, D13, D33, D33_spitzer]`."""

    return _solve_prepared_coefficient_vector_raw(
        prepared,
        case.nu_hat,
        case.resolved_epsi_hat(prepared.geometry.transport_psi_scale),
    )


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def solve_prepared_coefficient_vector_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> Array:
    """Coefficient-vector solve with an explicit custom-VJP contract point."""

    return solve_prepared_coefficient_vector(prepared, case)


def compile_prepared_solver(
    prepared: PreparedMonoenergeticSystem,
) -> CompiledPreparedSolver:
    """Return a jitted monoenergetic solver for repeated solves on one geometry."""

    compiled = jax.jit(
        lambda nu_hat, epsi_hat: _solve_prepared_arrays_from_values(
            prepared,
            nu_hat,
            epsi_hat,
        )
    )

    def solve(case: MonoenergeticCase) -> TransportResult:
        epsi_hat = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
        return transport_result_from_arrays(compiled(case.nu_hat, epsi_hat))

    return solve


def _solve_prepared_coefficient_vector_vjp_fwd(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[
    Array,
    tuple[Array, Array, Array | None, bool, bool, Array, Array, Array, Array, Array, Array],
]:
    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper = (
        _prepared_implicit_vjp_primal(
            prepared,
            case.nu_hat,
            resolved_epsi_hat,
        )
    )
    return coefficients, (
        jnp.asarray(case.nu_hat),
        resolved_epsi_hat,
        None if transport_scale is None else jnp.asarray(transport_scale),
        case.epsi_hat is not None,
        case.er_hat is not None,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
    )


def _solve_prepared_coefficient_vector_vjp_bwd(
    prepared: PreparedMonoenergeticSystem,
    residuals: tuple[
        Array,
        Array,
        Array | None,
        bool,
        bool,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ],
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase]:
    (
        nu_hat,
        resolved_epsi_hat,
        transport_scale,
        uses_epsi_hat,
        uses_er_hat,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
    ) = residuals
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        nu_hat,
        resolved_epsi_hat,
    )
    f1_bar_low, f3_bar_low, nu_bar_direct = _coefficient_mode_pullback(
        prepared.geometry,
        f1_full[:3],
        f3_full[:3],
        ctx.nu_hat,
        coefficient_bar,
    )
    g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
    g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
    lambda1 = _solve_factorized_adjoint(saved_lu, saved_piv, saved_lower, saved_upper, g1)
    lambda3 = _solve_factorized_adjoint(saved_lu, saved_piv, saved_lower, saved_upper, g3)
    nu_bar_implicit, epsi_bar = _parameter_gradient_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
    )
    nu_bar = nu_bar_direct + nu_bar_implicit
    if uses_epsi_hat:
        return (MonoenergeticCase(nu_hat=nu_bar, epsi_hat=epsi_bar, er_hat=None),)
    if uses_er_hat:
        assert transport_scale is not None
        er_bar = epsi_bar / transport_scale
        return (MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=er_bar),)
    return (MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=None),)


solve_prepared_coefficient_vector_vjp.defvjp(
    _solve_prepared_coefficient_vector_vjp_fwd,
    _solve_prepared_coefficient_vector_vjp_bwd,
)


def _solve_prepared_coefficient_vector_raw(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> Array:
    values = _solve_prepared_arrays_from_values(prepared, nu_hat, epsi_hat)
    return jnp.stack(values[:5])


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


def _solve_prepared_arrays(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, ...]:
    return _solve_prepared_arrays_from_values(
        prepared,
        case.nu_hat,
        case.resolved_epsi_hat(prepared.geometry.transport_psi_scale),
    )


def _solve_prepared_arrays_from_values(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> tuple[Array, ...]:
    geom = prepared.geometry
    grid = prepared.grid
    ctx = _operator_context(prepared.surface, geom, grid, nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    f1_modes, f3_modes = _solve_modes(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        s1,
        s3,
    )
    d11, d31, d13, d33, d33_spitzer = coefficients_from_modes(
        geom, f1_modes, f3_modes, ctx.nu_hat
    )
    residual = _residual_norm(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        s1,
        f1_modes,
    )
    return (
        d11,
        d31,
        d13,
        d33,
        d33_spitzer,
        f1_modes,
        f3_modes,
        residual,
        onsager_error(d31, d13),
        _stack_internal_systems(f1_modes, f3_modes),
        _stack_internal_systems(s1[:3], s3[:3]),
    )


def _stack_internal_systems(primary: Array, parallel: Array) -> Array:
    return jnp.stack((primary, primary, parallel))


def _monoenergetic_matrix(d11: Array, d31: Array, d13: Array, d33: Array) -> Array:
    return jnp.asarray([[d11, d11, d13], [d11, d11, d13], [d31, d31, d33]])


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
