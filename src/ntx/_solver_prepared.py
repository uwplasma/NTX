"""Prepared monoenergetic solve path and custom-VJP wrappers."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsp_linalg
from jax import Array

from ._solver_adjoint import (
    _coefficient_mode_pullback,
    _parameter_gradient_from_adjoint,
    _prepared_implicit_vjp_primal,
)
from ._solver_context import _operator_context
from ._solver_factorization import (
    _residual_norm,
    _solve_factorized_adjoint,
    _solve_factorized_modes,
    _solve_modes,
)
from ._solver_types import (
    CompiledPreparedSolver,
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    TransportResult,
    transport_result_from_arrays,
)
from .operators import (
    apply_nullspace_condition,
    operator_blocks,
    parameter_derivative_blocks,
    source_modes,
)
from .transport import coefficients_from_modes, onsager_error


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


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def solve_prepared_coefficient_vector_jvp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> Array:
    """Coefficient-vector solve with an explicit custom-JVP contract point."""

    return solve_prepared_coefficient_vector(prepared, case)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def solve_prepared_coefficient_vector_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> Array:
    """Coefficient-vector solve with an explicit custom-VJP contract point."""

    return solve_prepared_coefficient_vector(prepared, case)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def solve_prepared_coefficient_vector_iterative_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> Array:
    """Coefficient-vector solve with matrix-free primal and transpose solves.

    This reverse-oriented path avoids exposing dense LU factorization buffers to
    the enclosing compiler graph. It is intentionally opt-in because it trades
    execution speed for lower compile/runtime memory pressure.
    """

    return _solve_prepared_coefficient_vector_iterative_raw(
        prepared,
        case.nu_hat,
        case.resolved_epsi_hat(prepared.geometry.transport_psi_scale),
    )


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


def _zero_first_row(block: Array) -> Array:
    return block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))


def _prepared_coefficient_vector_jvp_from_values(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
    nu_hat_dot,
    epsi_hat_dot,
) -> tuple[Array, Array]:
    (
        coefficients,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
    ) = _prepared_implicit_vjp_primal(
        prepared,
        nu_hat,
        epsi_hat,
    )
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        nu_hat,
        epsi_hat,
    )

    source1_dot = []
    source3_dot = []
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = _zero_first_row(diagonal_nu)
            diagonal_epsi = _zero_first_row(diagonal_epsi)
        diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
        source1_dot.append(-(diagonal_dot @ f1_full[k]))
        source3_dot.append(-(diagonal_dot @ f3_full[k]))

    f1_dot = _solve_factorized_modes(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(source1_dot),
    )
    f3_dot = _solve_factorized_modes(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(source3_dot),
    )

    def coefficient_fn(modes1, modes3, nu_value):
        return jnp.stack(coefficients_from_modes(prepared.geometry, modes1, modes3, nu_value))

    _, coefficients_dot = jax.jvp(
        coefficient_fn,
        (f1_full[:3], f3_full[:3], ctx.nu_hat),
        (f1_dot[:3], f3_dot[:3], nu_hat_dot),
    )
    return coefficients, coefficients_dot


def _solve_prepared_coefficient_vector_jvp_rule(
    prepared: PreparedMonoenergeticSystem,
    primals: tuple[MonoenergeticCase],
    tangents: tuple[MonoenergeticCase],
) -> tuple[Array, Array]:
    (case,) = primals
    (case_dot,) = tangents
    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    nu_hat_dot = jnp.asarray(case_dot.nu_hat)
    if case.epsi_hat is not None:
        epsi_hat_dot = jnp.asarray(case_dot.epsi_hat)
    elif case.er_hat is not None:
        assert transport_scale is not None
        epsi_hat_dot = jnp.asarray(case_dot.er_hat) / jnp.asarray(transport_scale)
    else:
        epsi_hat_dot = jnp.zeros_like(resolved_epsi_hat)
    return _prepared_coefficient_vector_jvp_from_values(
        prepared,
        case.nu_hat,
        resolved_epsi_hat,
        nu_hat_dot,
        epsi_hat_dot,
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


def _apply_prepared_block_operator(
    prepared: PreparedMonoenergeticSystem,
    ctx,
    modes: Array,
) -> Array:
    values = []
    for k in range(prepared.grid.n_xi + 1):
        lower, diagonal, upper = operator_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal, upper = apply_nullspace_condition(diagonal, upper)
            assert upper is not None
        value = diagonal @ modes[k]
        if k > 0:
            value = value + lower @ modes[k - 1]
        if k < prepared.grid.n_xi:
            value = value + upper @ modes[k + 1]
        values.append(value)
    return jnp.stack(values)


def _apply_prepared_block_operator_transpose(
    prepared: PreparedMonoenergeticSystem,
    ctx,
    modes: Array,
) -> Array:
    values = []
    for k in range(prepared.grid.n_xi + 1):
        lower, diagonal, upper = operator_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal, upper = apply_nullspace_condition(diagonal, upper)
            assert upper is not None
        value = diagonal.T @ modes[k]
        if k > 0:
            _lower_prev, diagonal_prev, upper_prev = operator_blocks(
                ctx,
                k - 1,
                prepared.d_theta,
                prepared.d_zeta,
            )
            if k - 1 == 0:
                diagonal_prev, upper_prev = apply_nullspace_condition(diagonal_prev, upper_prev)
                assert upper_prev is not None
            value = value + upper_prev.T @ modes[k - 1]
        if k < prepared.grid.n_xi:
            lower_next, _diagonal_next, _upper_next = operator_blocks(
                ctx,
                k + 1,
                prepared.d_theta,
                prepared.d_zeta,
            )
            value = value + lower_next.T @ modes[k + 1]
        values.append(value)
    return jnp.stack(values)


def _solve_prepared_modes_bicgstab(
    prepared: PreparedMonoenergeticSystem,
    ctx,
    source: Array,
    *,
    transpose: bool = False,
) -> Array:
    source_shape = source.shape

    def matvec(flat_modes):
        modes = flat_modes.reshape(source_shape)
        if transpose:
            applied = _apply_prepared_block_operator_transpose(prepared, ctx, modes)
        else:
            applied = _apply_prepared_block_operator(prepared, ctx, modes)
        return applied.reshape((-1,))

    solution, _info = jsp_linalg.bicgstab(
        matvec,
        source.reshape((-1,)),
        tol=1.0e-10,
        atol=0.0,
        maxiter=max(40, 2 * int(prepared.grid.n_xi + 1)),
    )
    return solution.reshape(source_shape)


def _prepared_iterative_vjp_primal(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> tuple[Array, Array, Array]:
    geom = prepared.geometry
    grid = prepared.grid
    ctx = _operator_context(prepared.surface, geom, grid, nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    f1_full = _solve_prepared_modes_bicgstab(prepared, ctx, s1)
    f3_full = _solve_prepared_modes_bicgstab(prepared, ctx, s3)

    def coefficient_fn(modes1, modes3, nu_value):
        return jnp.stack(coefficients_from_modes(geom, modes1, modes3, nu_value))

    coefficients = coefficient_fn(f1_full[:3], f3_full[:3], ctx.nu_hat)
    return coefficients, f1_full, f3_full


def _solve_prepared_coefficient_vector_iterative_raw(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> Array:
    coefficients, _f1_full, _f3_full = _prepared_iterative_vjp_primal(
        prepared,
        nu_hat,
        epsi_hat,
    )
    return coefficients


def _solve_prepared_coefficient_vector_iterative_vjp_fwd(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, tuple[Array, Array, Array | None, bool, bool, Array, Array]]:
    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    coefficients, f1_full, f3_full = _prepared_iterative_vjp_primal(
        prepared,
        case.nu_hat,
        resolved_epsi_hat,
    )
    return coefficients, (
        jnp.asarray(case.nu_hat),
        resolved_epsi_hat,
        None if transport_scale is None else jnp.asarray(transport_scale),
        case.epsi_hat is not None,
        case.er_hat is not None,
        f1_full,
        f3_full,
    )


def _solve_prepared_coefficient_vector_iterative_vjp_bwd(
    prepared: PreparedMonoenergeticSystem,
    residuals: tuple[Array, Array, Array | None, bool, bool, Array, Array],
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
    lambda1 = _solve_prepared_modes_bicgstab(prepared, ctx, g1, transpose=True)
    lambda3 = _solve_prepared_modes_bicgstab(prepared, ctx, g3, transpose=True)
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


def _parameter_gradient_directional_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx,
    f1_full: Array,
    f3_full: Array,
    f1_dot: Array,
    f3_dot: Array,
    lambda1: Array,
    lambda3: Array,
    lambda1_dot: Array,
    lambda3_dot: Array,
) -> tuple[Array, Array]:
    nu_bar_dot = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    epsi_bar_dot = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = _zero_first_row(diagonal_nu)
            diagonal_epsi = _zero_first_row(diagonal_epsi)
        nu_bar_dot = nu_bar_dot - (
            jnp.vdot(lambda1_dot[k], diagonal_nu @ f1_full[k])
            + jnp.vdot(lambda1[k], diagonal_nu @ f1_dot[k])
            + jnp.vdot(lambda3_dot[k], diagonal_nu @ f3_full[k])
            + jnp.vdot(lambda3[k], diagonal_nu @ f3_dot[k])
        )
        epsi_bar_dot = epsi_bar_dot - (
            jnp.vdot(lambda1_dot[k], diagonal_epsi @ f1_full[k])
            + jnp.vdot(lambda1[k], diagonal_epsi @ f1_dot[k])
            + jnp.vdot(lambda3_dot[k], diagonal_epsi @ f3_full[k])
            + jnp.vdot(lambda3[k], diagonal_epsi @ f3_dot[k])
        )
    return nu_bar_dot, epsi_bar_dot


def solve_prepared_coefficient_vector_derivative_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    case_dot: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, MonoenergeticCase]:
    """Return a compact VJP and directional derivative of that VJP.

    This is the reverse-mode companion needed by callers that differentiate
    coefficient-solve derivative fields. It avoids tracing a JVP through the LU
    factorization by differentiating the implicit primal/adjoint systems
    directly for one monoenergetic case.
    """

    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    nu_hat_dot = jnp.asarray(case_dot.nu_hat)
    if case.epsi_hat is not None:
        epsi_hat_dot = jnp.asarray(case_dot.epsi_hat)
    elif case.er_hat is not None:
        assert transport_scale is not None
        epsi_hat_dot = jnp.asarray(case_dot.er_hat) / jnp.asarray(transport_scale)
    else:
        epsi_hat_dot = jnp.zeros_like(resolved_epsi_hat)

    (
        _coefficients,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
    ) = _prepared_implicit_vjp_primal(
        prepared,
        case.nu_hat,
        resolved_epsi_hat,
    )
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        resolved_epsi_hat,
    )

    source1_dot = []
    source3_dot = []
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = _zero_first_row(diagonal_nu)
            diagonal_epsi = _zero_first_row(diagonal_epsi)
        diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
        source1_dot.append(-(diagonal_dot @ f1_full[k]))
        source3_dot.append(-(diagonal_dot @ f3_full[k]))

    f1_dot = _solve_factorized_modes(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(source1_dot),
    )
    f3_dot = _solve_factorized_modes(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(source3_dot),
    )

    def _coefficient_pullback(modes1, modes3, nu_value):
        return _coefficient_mode_pullback(
            prepared.geometry,
            modes1,
            modes3,
            nu_value,
            coefficient_bar,
        )

    (
        f1_bar_low,
        f3_bar_low,
        nu_bar_direct,
    ), (
        f1_bar_low_dot,
        f3_bar_low_dot,
        nu_bar_direct_dot,
    ) = jax.jvp(
        _coefficient_pullback,
        (f1_full[:3], f3_full[:3], ctx.nu_hat),
        (f1_dot[:3], f3_dot[:3], nu_hat_dot),
    )

    g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
    g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
    g1_dot = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low_dot)
    g3_dot = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low_dot)

    lambda1 = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        g1,
    )
    lambda3 = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        g3,
    )

    adjoint_rhs1_dot = []
    adjoint_rhs3_dot = []
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = _zero_first_row(diagonal_nu)
            diagonal_epsi = _zero_first_row(diagonal_epsi)
        diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
        adjoint_rhs1_dot.append(g1_dot[k] - diagonal_dot.T @ lambda1[k])
        adjoint_rhs3_dot.append(g3_dot[k] - diagonal_dot.T @ lambda3[k])

    lambda1_dot = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(adjoint_rhs1_dot),
    )
    lambda3_dot = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(adjoint_rhs3_dot),
    )

    nu_bar_implicit, epsi_bar = _parameter_gradient_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
    )
    nu_bar_implicit_dot, epsi_bar_dot = _parameter_gradient_directional_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        f1_dot,
        f3_dot,
        lambda1,
        lambda3,
        lambda1_dot,
        lambda3_dot,
    )

    nu_bar = nu_bar_direct + nu_bar_implicit
    nu_bar_dot = nu_bar_direct_dot + nu_bar_implicit_dot
    if case.epsi_hat is not None:
        return (
            MonoenergeticCase(nu_hat=nu_bar, epsi_hat=epsi_bar, er_hat=None),
            MonoenergeticCase(nu_hat=nu_bar_dot, epsi_hat=epsi_bar_dot, er_hat=None),
        )
    if case.er_hat is not None:
        assert transport_scale is not None
        return (
            MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=epsi_bar / transport_scale),
            MonoenergeticCase(nu_hat=nu_bar_dot, epsi_hat=None, er_hat=epsi_bar_dot / transport_scale),
        )
    return (
        MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=None),
        MonoenergeticCase(nu_hat=nu_bar_dot, epsi_hat=None, er_hat=None),
    )


solve_prepared_coefficient_vector_vjp.defvjp(
    _solve_prepared_coefficient_vector_vjp_fwd,
    _solve_prepared_coefficient_vector_vjp_bwd,
)
solve_prepared_coefficient_vector_iterative_vjp.defvjp(
    _solve_prepared_coefficient_vector_iterative_vjp_fwd,
    _solve_prepared_coefficient_vector_iterative_vjp_bwd,
)

solve_prepared_coefficient_vector_jvp.defjvp(
    _solve_prepared_coefficient_vector_jvp_rule,
)


def _solve_prepared_coefficient_vector_raw(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> Array:
    values = _solve_prepared_arrays_from_values(prepared, nu_hat, epsi_hat)
    return jnp.stack(values[:5])


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


__all__ = [
    "compile_prepared_solver",
    "solve_prepared",
    "solve_prepared_coefficient_vector",
    "solve_prepared_coefficient_vector_derivative_vjp",
    "solve_prepared_coefficient_vector_iterative_vjp",
    "solve_prepared_coefficient_vector_jvp",
    "solve_prepared_coefficient_vector_vjp",
    "solve_prepared_internal",
]
