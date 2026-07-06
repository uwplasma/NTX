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


@jax.custom_vjp
def solve_prepared_coefficient_vector_recompute_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> Array:
    """Exact coefficient-vector VJP that recomputes factorization in backward.

    The standard exact custom VJP saves the dense LU/factorization state in the
    forward residual. This opt-in variant keeps only scalar inputs in the
    residual and rebuilds the exact factorization in the transpose rule.
    """

    return solve_prepared_coefficient_vector(prepared, case)


@jax.custom_vjp
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


@jax.custom_jvp
def solve_prepared_coefficient_vector_iterative_jvp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> Array:
    """Coefficient-vector solve with matrix-free primal and tangent solves."""

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


def _solve_prepared_coefficient_vector_recompute_vjp_fwd(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[
    Array,
    tuple[PreparedMonoenergeticSystem, Array, Array, Array | None, bool, bool],
]:
    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    coefficients, *_ = _prepared_implicit_vjp_primal(
        prepared,
        case.nu_hat,
        resolved_epsi_hat,
    )
    return coefficients, (
        prepared,
        jnp.asarray(case.nu_hat),
        resolved_epsi_hat,
        None if transport_scale is None else jnp.asarray(transport_scale),
        case.epsi_hat is not None,
        case.er_hat is not None,
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


def _solve_prepared_coefficient_vector_recompute_vjp_bwd(
    residuals: tuple[PreparedMonoenergeticSystem, Array, Array, Array | None, bool, bool],
    coefficient_bar: Array,
) -> tuple[None, MonoenergeticCase]:
    (
        prepared,
        nu_hat,
        resolved_epsi_hat,
        transport_scale,
        uses_epsi_hat,
        uses_er_hat,
    ) = residuals
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
        nu_hat,
        resolved_epsi_hat,
    )
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
        return (None, MonoenergeticCase(nu_hat=nu_bar, epsi_hat=epsi_bar, er_hat=None))
    if uses_er_hat:
        assert transport_scale is not None
        er_bar = epsi_bar / transport_scale
        return (None, MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=er_bar))
    return (None, MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=None))


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


def _prepared_coefficient_vector_iterative_jvp_from_values(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
    nu_hat_dot,
    epsi_hat_dot,
) -> tuple[Array, Array]:
    coefficients, f1_full, f3_full = _prepared_iterative_vjp_primal(
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

    f1_dot = _solve_prepared_modes_bicgstab(prepared, ctx, jnp.stack(source1_dot))
    f3_dot = _solve_prepared_modes_bicgstab(prepared, ctx, jnp.stack(source3_dot))

    def coefficient_fn(modes1, modes3, nu_value):
        return jnp.stack(coefficients_from_modes(prepared.geometry, modes1, modes3, nu_value))

    _, coefficients_dot = jax.jvp(
        coefficient_fn,
        (f1_full[:3], f3_full[:3], ctx.nu_hat),
        (f1_dot[:3], f3_dot[:3], nu_hat_dot),
    )
    return coefficients, coefficients_dot


def _solve_prepared_coefficient_vector_iterative_jvp_rule(
    primals: tuple[PreparedMonoenergeticSystem, MonoenergeticCase],
    tangents: tuple[PreparedMonoenergeticSystem, MonoenergeticCase],
) -> tuple[Array, Array]:
    prepared, case = primals
    _prepared_dot, case_dot = tangents
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
    return _prepared_coefficient_vector_iterative_jvp_from_values(
        prepared,
        case.nu_hat,
        resolved_epsi_hat,
        nu_hat_dot,
        epsi_hat_dot,
    )


def _solve_prepared_coefficient_vector_iterative_vjp_fwd(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[
    Array,
    tuple[
        PreparedMonoenergeticSystem,
        Array,
        Array,
        Array | None,
        bool,
        bool,
        Array,
        Array,
    ],
]:
    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    coefficients, f1_full, f3_full = _prepared_iterative_vjp_primal(
        prepared,
        case.nu_hat,
        resolved_epsi_hat,
    )
    return coefficients, (
        prepared,
        jnp.asarray(case.nu_hat),
        resolved_epsi_hat,
        None if transport_scale is None else jnp.asarray(transport_scale),
        case.epsi_hat is not None,
        case.er_hat is not None,
        f1_full,
        f3_full,
    )


def _solve_prepared_coefficient_vector_iterative_vjp_bwd(
    residuals: tuple[
        PreparedMonoenergeticSystem,
        Array,
        Array,
        Array | None,
        bool,
        bool,
        Array,
        Array,
    ],
    coefficient_bar: Array,
) -> tuple[PreparedMonoenergeticSystem, MonoenergeticCase]:
    (
        prepared,
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
        case_bar = MonoenergeticCase(nu_hat=nu_bar, epsi_hat=epsi_bar, er_hat=None)
        return (None, case_bar)
    if uses_er_hat:
        assert transport_scale is not None
        er_bar = epsi_bar / transport_scale
        case_bar = MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=er_bar)
        return (None, case_bar)
    case_bar = MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=None)
    return (None, case_bar)


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


def solve_prepared_coefficient_vector_lowdot_two_pullbacks(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_fn,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Fused exact pullback for two coefficient-derivative contractions.

    Higher-level transport models own the mapping from coefficient vectors to
    objective-specific coefficient cotangents, so this helper receives that
    mapping as a Python callback. NTX still owns the expensive implicit-solve
    pullback algebra and reuses one primal factorization for the base VJP and
    the two low-mode directional pullbacks.
    """

    from jax.scipy.linalg import lu_solve

    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    first_nu_dot = jnp.asarray(first_case_dot.nu_hat)
    second_nu_dot = jnp.asarray(second_case_dot.nu_hat)
    if first_case_dot.epsi_hat is not None:
        first_epsi_dot = jnp.asarray(first_case_dot.epsi_hat)
    elif first_case_dot.er_hat is not None:
        assert transport_scale is not None
        first_epsi_dot = jnp.asarray(first_case_dot.er_hat) / jnp.asarray(transport_scale)
    else:
        first_epsi_dot = jnp.zeros_like(resolved_epsi_hat)
    if second_case_dot.epsi_hat is not None:
        second_epsi_dot = jnp.asarray(second_case_dot.epsi_hat)
    elif second_case_dot.er_hat is not None:
        assert transport_scale is not None
        second_epsi_dot = jnp.asarray(second_case_dot.er_hat) / jnp.asarray(transport_scale)
    else:
        second_epsi_dot = jnp.zeros_like(resolved_epsi_hat)

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
        case.nu_hat,
        resolved_epsi_hat,
    )
    base_coefficient_bar, first_coefficient_bar, second_coefficient_bar = coefficient_bar_fn(
        coefficients
    )
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        resolved_epsi_hat,
    )
    mode_indices = jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32)

    def _zero_first_row_if_needed(block, k):
        zeroed = block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))
        return jnp.where(jnp.asarray(k) == 0, zeroed, block)

    def _take_mode(values, k):
        return jax.lax.dynamic_index_in_dim(values, k, axis=0, keepdims=False)

    def _solve_factorized_low_modes_scan(source_for_mode):
        mode_count = saved_lu.shape[0]
        last_index = mode_count - 1
        source0 = source_for_mode(jnp.asarray(0, dtype=jnp.int32))
        zero = jnp.zeros_like(source0)
        y_last = lu_solve(
            (_take_mode(saved_lu, last_index), _take_mode(saved_piv, last_index)),
            source_for_mode(jnp.asarray(last_index, dtype=jnp.int32)),
        )
        y0 = jnp.where(last_index == 0, y_last, zero)
        y1 = jnp.where(last_index == 1, y_last, zero)
        y2 = jnp.where(last_index == 2, y_last, zero)

        def _backward_y(carry, k):
            y_next, y0_value, y1_value, y2_value = carry
            rhs = source_for_mode(k) - _take_mode(saved_upper, k) @ y_next
            y_k = lu_solve(
                (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                rhs,
            )
            y0_value = jnp.where(k == 0, y_k, y0_value)
            y1_value = jnp.where(k == 1, y_k, y1_value)
            y2_value = jnp.where(k == 2, y_k, y2_value)
            return (y_k, y0_value, y1_value, y2_value), None

        (_, y0, y1, y2), _ = jax.lax.scan(
            _backward_y,
            (y_last, y0, y1, y2),
            jnp.arange(last_index, dtype=jnp.int32),
            reverse=True,
        )
        mode0 = y0
        mode1 = y1 - lu_solve(
            (_take_mode(saved_lu, 1), _take_mode(saved_piv, 1)),
            _take_mode(saved_lower, 1) @ mode0,
        )
        mode2 = y2 - lu_solve(
            (_take_mode(saved_lu, 2), _take_mode(saved_piv, 2)),
            _take_mode(saved_lower, 2) @ mode1,
        )
        return jnp.stack([mode0, mode1, mode2], axis=0)

    def _solve_factorized_adjoint_scan(source_bar):
        mode_count = source_bar.shape[0]
        last_index = mode_count - 1
        mu_last = _take_mode(source_bar, last_index)

        def _backward_mu(mu_next, k):
            propagated = lu_solve(
                (_take_mode(saved_lu, k + 1), _take_mode(saved_piv, k + 1)),
                mu_next,
                trans=1,
            )
            mu_k = _take_mode(source_bar, k) - _take_mode(saved_lower, k + 1).T @ propagated
            return mu_k, mu_k

        _, mu_tail = jax.lax.scan(
            _backward_mu,
            mu_last,
            jnp.arange(last_index, dtype=jnp.int32),
            reverse=True,
        )
        mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
        adjoint0 = lu_solve(
            (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
            _take_mode(mu, 0),
            trans=1,
        )

        def _forward_adjoint(adjoint_prev, k):
            rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
            adjoint_k = lu_solve(
                (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                rhs,
                trans=1,
            )
            return adjoint_k, adjoint_k

        _, adjoint_tail = jax.lax.scan(
            _forward_adjoint,
            adjoint0,
            jnp.arange(1, mode_count, dtype=jnp.int32),
        )
        return jnp.concatenate([adjoint0[None, ...], adjoint_tail], axis=0)

    def _parameter_source_matrix_for_mode(k):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
        diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
        f1_k = _take_mode(f1_full, k)
        f3_k = _take_mode(f3_full, k)
        return jnp.stack(
            [
                jnp.stack([diagonal_nu @ f1_k, diagonal_nu @ f3_k], axis=-1),
                jnp.stack([diagonal_epsi @ f1_k, diagonal_epsi @ f3_k], axis=-1),
            ],
            axis=-1,
        )

    def _contract_factorized_parameter_sources_scan(source_bar_matrix_for_mode):
        mode_count = saved_lu.shape[0]
        last_index = mode_count - 1
        mu_last = source_bar_matrix_for_mode(jnp.asarray(last_index, dtype=jnp.int32))

        def _backward_mu(mu_next, k):
            propagated = lu_solve(
                (_take_mode(saved_lu, k + 1), _take_mode(saved_piv, k + 1)),
                mu_next,
                trans=1,
            )
            mu_k = source_bar_matrix_for_mode(k) - _take_mode(saved_lower, k + 1).T @ propagated
            return mu_k, mu_k

        _, mu_tail = jax.lax.scan(
            _backward_mu,
            mu_last,
            jnp.arange(last_index, dtype=jnp.int32),
            reverse=True,
        )
        mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
        adjoint0 = lu_solve(
            (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
            _take_mode(mu, 0),
            trans=1,
        )
        source0 = _parameter_source_matrix_for_mode(jnp.asarray(0, dtype=jnp.int32))
        contract0 = jnp.sum(adjoint0[..., None] * source0, axis=(0, 1))

        def _forward_adjoint(carry, k):
            adjoint_prev, contract = carry
            rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
            adjoint_k = lu_solve(
                (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                rhs,
                trans=1,
            )
            source_k = _parameter_source_matrix_for_mode(k)
            contract = contract + jnp.sum(adjoint_k[..., None] * source_k, axis=(0, 1))
            return (adjoint_k, contract), None

        (_, contracted), _ = jax.lax.scan(
            _forward_adjoint,
            (adjoint0, contract0),
            jnp.arange(1, mode_count, dtype=jnp.int32),
        )
        return contracted

    def _base_pullback(coefficient_bar):
        f1_bar_low, f3_bar_low, nu_bar_direct = _coefficient_mode_pullback(
            prepared.geometry,
            f1_full[:3],
            f3_full[:3],
            ctx.nu_hat,
            coefficient_bar,
        )
        g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
        g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)

        def _source_bar_matrix_for_mode(k):
            return jnp.stack([_take_mode(g1, k), _take_mode(g3, k)], axis=-1)

        nu_bar_implicit, epsi_bar = -_contract_factorized_parameter_sources_scan(
            _source_bar_matrix_for_mode
        )
        return nu_bar_direct + nu_bar_implicit, epsi_bar

    def _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
        diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
        diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
        return (
            -(diagonal_dot @ _take_mode(f1_full, k)),
            -(diagonal_dot @ _take_mode(f3_full, k)),
        )

    def _packed_source_dot_matrix_for_mode(k):
        first_source1, first_source3 = _source_dot_pair_for_direction(
            k,
            first_nu_dot,
            first_epsi_dot,
        )
        second_source1, second_source3 = _source_dot_pair_for_direction(
            k,
            second_nu_dot,
            second_epsi_dot,
        )
        return jnp.stack(
            [first_source1, first_source3, second_source1, second_source3],
            axis=-1,
        )

    packed_f_dot_low_matrix = _solve_factorized_low_modes_scan(
        _packed_source_dot_matrix_for_mode
    )

    def _two_direction_pullbacks(
        coefficient_bar_pair,
        nu_hat_dot_pair,
        epsi_hat_dot_pair,
        f1_dot_low_pair,
        f3_dot_low_pair,
    ):
        def _coefficient_pullback_for_direction(
            coefficient_bar,
            f1_dot_low,
            f3_dot_low,
            nu_hat_dot,
        ):
            def _coefficient_pullback(modes1, modes3, nu_value):
                return _coefficient_mode_pullback(
                    prepared.geometry,
                    modes1,
                    modes3,
                    nu_value,
                    coefficient_bar,
                )

            return jax.jvp(
                _coefficient_pullback,
                (f1_full[:3], f3_full[:3], ctx.nu_hat),
                (f1_dot_low, f3_dot_low, nu_hat_dot),
            )

        (
            f1_bar_low_pair,
            f3_bar_low_pair,
            nu_bar_direct_pair,
        ), (
            f1_bar_low_dot_pair,
            f3_bar_low_dot_pair,
            nu_bar_direct_dot_pair,
        ) = jax.vmap(_coefficient_pullback_for_direction)(
            coefficient_bar_pair,
            f1_dot_low_pair,
            f3_dot_low_pair,
            nu_hat_dot_pair,
        )

        zeros_pair = jnp.zeros(
            (*f1_full.shape, coefficient_bar_pair.shape[0]),
            dtype=f1_full.dtype,
        )
        g1_pair = zeros_pair.at[:3, :, :].set(jnp.moveaxis(f1_bar_low_pair, 0, -1))
        g3_pair = zeros_pair.at[:3, :, :].set(jnp.moveaxis(f3_bar_low_pair, 0, -1))
        g1_dot_pair = zeros_pair.at[:3, :, :].set(
            jnp.moveaxis(f1_bar_low_dot_pair, 0, -1)
        )
        g3_dot_pair = zeros_pair.at[:3, :, :].set(
            jnp.moveaxis(f3_bar_low_dot_pair, 0, -1)
        )

        def _interleave_field_columns(field1_pair, field3_pair):
            return jnp.reshape(
                jnp.stack([field1_pair, field3_pair], axis=-1),
                (field1_pair.shape[0], field1_pair.shape[1], -1),
            )

        lambda_matrix = _solve_factorized_adjoint_scan(
            _interleave_field_columns(g1_pair, g3_pair)
        )
        lambda_pair_matrix = jnp.reshape(
            lambda_matrix,
            (lambda_matrix.shape[0], lambda_matrix.shape[1], coefficient_bar_pair.shape[0], 2),
        )
        lambda1_pair = jnp.moveaxis(lambda_pair_matrix[..., 0], -1, 1)
        lambda3_pair = jnp.moveaxis(lambda_pair_matrix[..., 1], -1, 1)

        def _diagonal_dot_pair(k):
            diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                ctx,
                k,
                prepared.d_theta,
                prepared.d_zeta,
            )
            diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
            diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
            return (
                nu_hat_dot_pair[:, None, None] * diagonal_nu[None, :, :]
                + epsi_hat_dot_pair[:, None, None] * diagonal_epsi[None, :, :]
            )

        def _adjoint_rhs_dot_matrix_for_mode(k):
            diagonal_dot_pair = _diagonal_dot_pair(k)
            lambda1_k = _take_mode(lambda1_pair, k)
            lambda3_k = _take_mode(lambda3_pair, k)
            rhs1_pair = (
                jnp.moveaxis(_take_mode(g1_dot_pair, k), -1, 0)
                - jnp.einsum("dji,dj->di", diagonal_dot_pair, lambda1_k)
            )
            rhs3_pair = (
                jnp.moveaxis(_take_mode(g3_dot_pair, k), -1, 0)
                - jnp.einsum("dji,dj->di", diagonal_dot_pair, lambda3_k)
            )
            return jnp.reshape(
                jnp.moveaxis(jnp.stack([rhs1_pair, rhs3_pair], axis=-1), 0, 1),
                (rhs1_pair.shape[1], -1),
            )

        def _accumulate_base_bars(carry, k):
            nu_bar, epsi_bar = carry
            diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                ctx,
                k,
                prepared.d_theta,
                prepared.d_zeta,
            )
            diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
            diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
            f1_k = _take_mode(f1_full, k)
            f3_k = _take_mode(f3_full, k)
            lambda1_k = _take_mode(lambda1_pair, k)
            lambda3_k = _take_mode(lambda3_pair, k)
            nu_bar = nu_bar - (
                jnp.einsum("dn,n->d", lambda1_k, diagonal_nu @ f1_k)
                + jnp.einsum("dn,n->d", lambda3_k, diagonal_nu @ f3_k)
            )
            epsi_bar = epsi_bar - (
                jnp.einsum("dn,n->d", lambda1_k, diagonal_epsi @ f1_k)
                + jnp.einsum("dn,n->d", lambda3_k, diagonal_epsi @ f3_k)
            )
            return (nu_bar, epsi_bar), None

        (
            nu_bar_implicit_pair,
            epsi_bar_pair,
        ), _ = jax.lax.scan(
            _accumulate_base_bars,
            (
                jnp.zeros((coefficient_bar_pair.shape[0],), dtype=prepared.grid.jax_dtype),
                jnp.zeros((coefficient_bar_pair.shape[0],), dtype=prepared.grid.jax_dtype),
            ),
            mode_indices,
        )

        def _parameter_source_matrix_for_mode(k):
            diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                ctx,
                k,
                prepared.d_theta,
                prepared.d_zeta,
            )
            diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
            diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
            f1_k = _take_mode(f1_full, k)
            f3_k = _take_mode(f3_full, k)
            return jnp.stack(
                [
                    jnp.stack([diagonal_nu @ f1_k, diagonal_nu @ f3_k], axis=-1),
                    jnp.stack([diagonal_epsi @ f1_k, diagonal_epsi @ f3_k], axis=-1),
                ],
                axis=-1,
            )

        def _contract_direction_parameter_sources_scan(source_bar_matrix_for_mode):
            direction_count = coefficient_bar_pair.shape[0]
            mode_count = saved_lu.shape[0]
            last_index = mode_count - 1
            mu_last = source_bar_matrix_for_mode(jnp.asarray(last_index, dtype=jnp.int32))

            def _backward_mu(mu_next, k):
                propagated = lu_solve(
                    (_take_mode(saved_lu, k + 1), _take_mode(saved_piv, k + 1)),
                    mu_next,
                    trans=1,
                )
                mu_k = source_bar_matrix_for_mode(k) - _take_mode(
                    saved_lower,
                    k + 1,
                ).T @ propagated
                return mu_k, mu_k

            _, mu_tail = jax.lax.scan(
                _backward_mu,
                mu_last,
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
            adjoint0 = lu_solve(
                (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                _take_mode(mu, 0),
                trans=1,
            )
            source0 = _parameter_source_matrix_for_mode(jnp.asarray(0, dtype=jnp.int32))
            adjoint0_pair = jnp.reshape(
                adjoint0,
                (adjoint0.shape[0], direction_count, 2),
            )
            contract0 = jnp.sum(
                adjoint0_pair[..., None] * source0[:, None, :, :],
                axis=(0, 2),
            )

            def _forward_adjoint(carry, k):
                adjoint_prev, contract = carry
                rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                adjoint_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                    trans=1,
                )
                source_k = _parameter_source_matrix_for_mode(k)
                adjoint_pair = jnp.reshape(
                    adjoint_k,
                    (adjoint_k.shape[0], direction_count, 2),
                )
                contract = contract + jnp.sum(
                    adjoint_pair[..., None] * source_k[:, None, :, :],
                    axis=(0, 2),
                )
                return (adjoint_k, contract), None

            (_, contracted), _ = jax.lax.scan(
                _forward_adjoint,
                (adjoint0, contract0),
                jnp.arange(1, mode_count, dtype=jnp.int32),
            )
            return contracted

        nu_bar_implicit_dot_pair, epsi_bar_dot_pair = (
            -_contract_direction_parameter_sources_scan(_adjoint_rhs_dot_matrix_for_mode)
        )

        def _source_dot_pair_for_mode(k):
            diagonal_dot_pair = _diagonal_dot_pair(k)
            return (
                -jnp.einsum("dij,j->di", diagonal_dot_pair, _take_mode(f1_full, k)),
                -jnp.einsum("dij,j->di", diagonal_dot_pair, _take_mode(f3_full, k)),
            )

        def _source_bar_pair_for_mode(lambdas, k):
            diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                ctx,
                k,
                prepared.d_theta,
                prepared.d_zeta,
            )
            diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
            diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
            lambda_k = _take_mode(lambdas, k)
            values = jnp.stack(
                [
                    jnp.einsum("ji,dj->di", diagonal_nu, lambda_k),
                    jnp.einsum("ji,dj->di", diagonal_epsi, lambda_k),
                ],
                axis=-1,
            )
            return jnp.reshape(jnp.moveaxis(values, 0, 1), (values.shape[1], -1))

        def _contract_direction_source_bar_pair_scan(source_dot_pair_for_mode, source_bar_pair_for_mode):
            direction_count = coefficient_bar_pair.shape[0]
            mode_count = saved_lu.shape[0]
            last_index = mode_count - 1
            mu_last = source_bar_pair_for_mode(jnp.asarray(last_index, dtype=jnp.int32))

            def _backward_mu(mu_next, k):
                propagated = lu_solve(
                    (_take_mode(saved_lu, k + 1), _take_mode(saved_piv, k + 1)),
                    mu_next,
                    trans=1,
                )
                mu_k = source_bar_pair_for_mode(k) - _take_mode(
                    saved_lower,
                    k + 1,
                ).T @ propagated
                return mu_k, mu_k

            _, mu_tail = jax.lax.scan(
                _backward_mu,
                mu_last,
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
            adjoint0 = lu_solve(
                (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                _take_mode(mu, 0),
                trans=1,
            )
            source0 = source_dot_pair_for_mode(jnp.asarray(0, dtype=jnp.int32))
            adjoint0_pair = jnp.reshape(
                adjoint0,
                (adjoint0.shape[0], direction_count, 2),
            )
            contract0 = jnp.sum(
                adjoint0_pair * source0.T[:, :, None],
                axis=0,
            )

            def _forward_adjoint(carry, k):
                adjoint_prev, contract = carry
                rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                adjoint_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                    trans=1,
                )
                source_k = source_dot_pair_for_mode(k)
                adjoint_pair = jnp.reshape(
                    adjoint_k,
                    (adjoint_k.shape[0], direction_count, 2),
                )
                contract = contract + jnp.sum(
                    adjoint_pair * source_k.T[:, :, None],
                    axis=0,
                )
                return (adjoint_k, contract), None

            (_, contracted), _ = jax.lax.scan(
                _forward_adjoint,
                (adjoint0, contract0),
                jnp.arange(1, mode_count, dtype=jnp.int32),
            )
            return contracted

        source1_dot_pair_for_mode = lambda k: _source_dot_pair_for_mode(k)[0]
        source3_dot_pair_for_mode = lambda k: _source_dot_pair_for_mode(k)[1]
        f1_field_dot_pair = _contract_direction_source_bar_pair_scan(
            source1_dot_pair_for_mode,
            lambda k: _source_bar_pair_for_mode(lambda1_pair, k),
        )
        f3_field_dot_pair = _contract_direction_source_bar_pair_scan(
            source3_dot_pair_for_mode,
            lambda k: _source_bar_pair_for_mode(lambda3_pair, k),
        )
        nu_bar_implicit_dot_pair = (
            nu_bar_implicit_dot_pair - f1_field_dot_pair[:, 0] - f3_field_dot_pair[:, 0]
        )
        epsi_bar_dot_pair = (
            epsi_bar_dot_pair - f1_field_dot_pair[:, 1] - f3_field_dot_pair[:, 1]
        )
        nu_bar_pair = nu_bar_direct_pair + nu_bar_implicit_pair
        nu_bar_dot_pair = nu_bar_direct_dot_pair + nu_bar_implicit_dot_pair
        return (
            nu_bar_pair,
            epsi_bar_pair,
            nu_bar_dot_pair,
            epsi_bar_dot_pair,
        )

    def _scan_direction_pullbacks(
        coefficient_bar_pair,
        nu_hat_dot_pair,
        epsi_hat_dot_pair,
        f1_dot_low_pair,
        f3_dot_low_pair,
    ):
        def _contract_factorized_source_bar_pair_scan(source_dot_for_mode, source_bar_for_mode):
            mode_count = saved_lu.shape[0]
            last_index = mode_count - 1
            mu_last = source_bar_for_mode(jnp.asarray(last_index, dtype=jnp.int32))

            def _backward_mu(mu_next, k):
                propagated = lu_solve(
                    (_take_mode(saved_lu, k + 1), _take_mode(saved_piv, k + 1)),
                    mu_next,
                    trans=1,
                )
                mu_k = source_bar_for_mode(k) - _take_mode(saved_lower, k + 1).T @ propagated
                return mu_k, mu_k

            _, mu_tail = jax.lax.scan(
                _backward_mu,
                mu_last,
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
            adjoint0 = lu_solve(
                (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                _take_mode(mu, 0),
                trans=1,
            )
            source0 = source_dot_for_mode(jnp.asarray(0, dtype=jnp.int32))
            contract0 = jnp.sum(adjoint0 * source0[:, None], axis=0)

            def _forward_adjoint(carry, k):
                adjoint_prev, contract = carry
                rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                adjoint_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                    trans=1,
                )
                source_k = source_dot_for_mode(k)
                contract = contract + jnp.sum(adjoint_k * source_k[:, None], axis=0)
                return (adjoint_k, contract), None

            (_, contracted), _ = jax.lax.scan(
                _forward_adjoint,
                (adjoint0, contract0),
                jnp.arange(1, mode_count, dtype=jnp.int32),
            )
            return contracted

        def _one_direction_pullback(xs):
            coefficient_bar, nu_hat_dot, epsi_hat_dot, f1_dot_low, f3_dot_low = xs

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
                (f1_dot_low, f3_dot_low, nu_hat_dot),
            )

            g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
            g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
            g1_dot = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low_dot)
            g3_dot = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low_dot)

            lambda1 = _solve_factorized_adjoint_scan(lambda k: _take_mode(g1, k))
            lambda3 = _solve_factorized_adjoint_scan(lambda k: _take_mode(g3, k))

            def _diagonal_dot(k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                return nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi

            def _adjoint_rhs_dot_for_mode(lambda_field, g_dot, k):
                return _take_mode(g_dot, k) - _diagonal_dot(k).T @ _take_mode(lambda_field, k)

            def _accumulate_base_bars(carry, k):
                nu_bar, epsi_bar = carry
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                f1_k = _take_mode(f1_full, k)
                f3_k = _take_mode(f3_full, k)
                lambda1_k = _take_mode(lambda1, k)
                lambda3_k = _take_mode(lambda3, k)
                nu_bar = nu_bar - lambda1_k @ (diagonal_nu @ f1_k) - lambda3_k @ (diagonal_nu @ f3_k)
                epsi_bar = epsi_bar - lambda1_k @ (diagonal_epsi @ f1_k) - lambda3_k @ (
                    diagonal_epsi @ f3_k
                )
                return (nu_bar, epsi_bar), None

            (
                nu_bar_implicit,
                epsi_bar,
            ), _ = jax.lax.scan(
                _accumulate_base_bars,
                (
                    jnp.asarray(0, dtype=prepared.grid.jax_dtype),
                    jnp.asarray(0, dtype=prepared.grid.jax_dtype),
                ),
                mode_indices,
            )

            def _adjoint_rhs_dot_matrix_for_mode(k):
                return jnp.stack(
                    [
                        _adjoint_rhs_dot_for_mode(lambda1, g1_dot, k),
                        _adjoint_rhs_dot_for_mode(lambda3, g3_dot, k),
                    ],
                    axis=-1,
                )

            nu_bar_implicit_dot, epsi_bar_dot = -_contract_factorized_parameter_sources_scan(
                _adjoint_rhs_dot_matrix_for_mode
            )

            def _source_bar_pair_for_mode(lambdas, k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                lambda_k = _take_mode(lambdas, k)
                return jnp.stack([diagonal_nu.T @ lambda_k, diagonal_epsi.T @ lambda_k], axis=-1)

            source1_dot_for_mode = lambda k: _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot)[0]
            source3_dot_for_mode = lambda k: _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot)[1]
            f1_field_dot = _contract_factorized_source_bar_pair_scan(
                source1_dot_for_mode,
                lambda k: _source_bar_pair_for_mode(lambda1, k),
            )
            f3_field_dot = _contract_factorized_source_bar_pair_scan(
                source3_dot_for_mode,
                lambda k: _source_bar_pair_for_mode(lambda3, k),
            )
            nu_bar_implicit_dot = nu_bar_implicit_dot - f1_field_dot[0] - f3_field_dot[0]
            epsi_bar_dot = epsi_bar_dot - f1_field_dot[1] - f3_field_dot[1]
            return (
                nu_bar_direct + nu_bar_implicit,
                epsi_bar,
                nu_bar_direct_dot + nu_bar_implicit_dot,
                epsi_bar_dot,
            )

        _, outputs = jax.lax.scan(
            lambda carry, xs: (carry, _one_direction_pullback(xs)),
            jnp.asarray(0, dtype=jnp.int32),
            (
                coefficient_bar_pair,
                nu_hat_dot_pair,
                epsi_hat_dot_pair,
                f1_dot_low_pair,
                f3_dot_low_pair,
            ),
        )
        return outputs

    base = _base_pullback(base_coefficient_bar)
    direction_nu_bar, direction_epsi_bar, direction_nu_bar_dot, direction_epsi_bar_dot = (
        _scan_direction_pullbacks(
            jnp.stack([first_coefficient_bar, second_coefficient_bar], axis=0),
            jnp.stack([first_nu_dot, second_nu_dot], axis=0),
            jnp.stack([first_epsi_dot, second_epsi_dot], axis=0),
            jnp.stack([packed_f_dot_low_matrix[..., 0], packed_f_dot_low_matrix[..., 2]], axis=0),
            jnp.stack([packed_f_dot_low_matrix[..., 1], packed_f_dot_low_matrix[..., 3]], axis=0),
        )
    )
    first = (
        direction_nu_bar[0],
        direction_epsi_bar[0],
        direction_nu_bar_dot[0],
        direction_epsi_bar_dot[0],
    )
    second = (
        direction_nu_bar[1],
        direction_epsi_bar[1],
        direction_nu_bar_dot[1],
        direction_epsi_bar_dot[1],
    )
    return (*base, *first, *second)


solve_prepared_coefficient_vector_vjp.defvjp(
    _solve_prepared_coefficient_vector_vjp_fwd,
    _solve_prepared_coefficient_vector_vjp_bwd,
)
solve_prepared_coefficient_vector_recompute_vjp.defvjp(
    _solve_prepared_coefficient_vector_recompute_vjp_fwd,
    _solve_prepared_coefficient_vector_recompute_vjp_bwd,
)
solve_prepared_coefficient_vector_iterative_vjp.defvjp(
    _solve_prepared_coefficient_vector_iterative_vjp_fwd,
    _solve_prepared_coefficient_vector_iterative_vjp_bwd,
)

solve_prepared_coefficient_vector_jvp.defjvp(
    _solve_prepared_coefficient_vector_jvp_rule,
)
solve_prepared_coefficient_vector_iterative_jvp.defjvp(
    _solve_prepared_coefficient_vector_iterative_jvp_rule,
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
    "solve_prepared_coefficient_vector_iterative_jvp",
    "solve_prepared_coefficient_vector_iterative_vjp",
    "solve_prepared_coefficient_vector_jvp",
    "solve_prepared_coefficient_vector_lowdot_two_pullbacks",
    "solve_prepared_coefficient_vector_recompute_vjp",
    "solve_prepared_coefficient_vector_vjp",
    "solve_prepared_internal",
]
