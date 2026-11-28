"""Reusable adjoint contractions and case-coordinate chain rules on prepared systems."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array

from .._geometry import GeometryOnGrid
from .._solver import (
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    _coefficient_mode_pullback,
    _operator_context,
    _parameter_gradient_from_adjoint,
    _prepared_implicit_vjp_primal,
    _solve_factorized_adjoint,
    _solve_factorized_modes,
)
from ..operators import (
    OperatorContext,
    apply_nullspace_condition,
    block_parameters,
    operator_blocks_from_parameters,
    parameter_derivative_blocks,
    source_modes,
)
from ..transport import coefficients_from_modes


def _geometry_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
):
    """Transpose the fixed residual and direct coefficients to geometry only.

    Reuse the solved modes and adjoints: ``g_bar = direct_bar - lambda.T F_g``.
    Resolved epsilon is held fixed; callers own the outer support chain.
    """

    def _residual_and_direct_coefficients(geometry):
        local_ctx = OperatorContext(
            surface=prepared.surface, geometry=geometry, nu_hat=ctx.nu_hat, epsi_hat=ctx.epsi_hat
        )
        inputs = (geometry, prepared.d_theta, prepared.d_zeta, block_parameters(local_ctx))
        return _compact_residual_and_direct_coefficients(
            inputs,
            surface=prepared.surface,
            n_xi=int(prepared.grid.n_xi),
            f1_full=f1_full,
            f3_full=f3_full,
        )

    _, pullback = jax.vjp(
        _residual_and_direct_coefficients,
        prepared.geometry,
    )
    (geometry_bar,) = pullback((coefficient_bar, -lambda1, -lambda3))
    return geometry_bar


def _prepared_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
):
    """Transpose the same fixed residual to all differentiable prepared leaves.

    This VJP includes surface, geometry and derivative operators without
    differentiating a factorization or solving another primal/adjoint system.
    """

    def _residual_and_direct_coefficients(prepared_value):
        return _compact_residual_and_direct_coefficients(
            _compact_prepared_residual_inputs(prepared_value, ctx.nu_hat, ctx.epsi_hat),
            surface=prepared_value.surface,
            n_xi=int(prepared.grid.n_xi),
            f1_full=f1_full,
            f3_full=f3_full,
        )

    _, pullback = jax.vjp(_residual_and_direct_coefficients, prepared)
    (prepared_bar,) = pullback((coefficient_bar, -lambda1, -lambda3))
    return prepared_bar


def _compact_prepared_residual_inputs(
    prepared: PreparedMonoenergeticSystem,
    nu_hat: Array,
    epsi_hat: Array,
):
    """Return the dynamic inputs to the fixed prepared residual.

    The dense residual depends on the complete prepared pytree only through
    the geometry used by the coefficient contraction, the two derivative
    matrices, and the compact block-parameter arrays.  Keeping this tuple
    explicit lets the expensive residual transpose stop there; a separate,
    small VJP subsequently chains its cotangent back to ``prepared``.
    """
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        nu_hat,
        epsi_hat,
    )
    return (
        prepared.geometry,
        prepared.d_theta,
        prepared.d_zeta,
        block_parameters(ctx),
    )


def _compact_residual_and_direct_coefficients(
    compact_inputs,
    *,
    surface,
    n_xi: int,
    f1_full: Array,
    f3_full: Array,
):
    """Fixed-primal residual expressed in compact prepared inputs only."""
    geometry, d_theta, d_zeta, params = compact_inputs
    local_ctx = OperatorContext(
        surface=surface,
        geometry=geometry,
        nu_hat=params["nu_hat"],
        epsi_hat=params["epsi_hat"],
    )
    source1, source3 = source_modes(local_ctx, n_xi)

    def _residual(modes, source):
        rows = []
        for mode_index in range(n_xi + 1):
            lower, diagonal, upper = operator_blocks_from_parameters(
                params,
                mode_index,
                d_theta,
                d_zeta,
            )
            if mode_index == 0:
                diagonal, upper = apply_nullspace_condition(diagonal, upper)
                assert upper is not None
            row = diagonal @ modes[mode_index] - source[mode_index]
            if mode_index > 0:
                row = row + lower @ modes[mode_index - 1]
            if mode_index < n_xi:
                row = row + upper @ modes[mode_index + 1]
            rows.append(row)
        return jnp.stack(rows)

    direct_coefficients = jnp.stack(
        coefficients_from_modes(
            geometry,
            f1_full[:3],
            f3_full[:3],
            params["nu_hat"],
        )
    )
    return (
        direct_coefficients,
        _residual(f1_full, source1),
        _residual(f3_full, source3),
    )


def _compact_prepared_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
):
    """Transpose the fixed residual to the compact input tuple.

    Call ``_compact_prepared_bar_to_prepared`` once after combining bars to
    finish the prepared-input chain without another dense residual transpose.
    """
    compact_inputs = _compact_prepared_residual_inputs(
        prepared,
        ctx.nu_hat,
        ctx.epsi_hat,
    )
    _, pullback = jax.vjp(
        lambda inputs: _compact_residual_and_direct_coefficients(
            inputs,
            surface=prepared.surface,
            n_xi=int(prepared.grid.n_xi),
            f1_full=f1_full,
            f3_full=f3_full,
        ),
        compact_inputs,
    )
    (compact_bar,) = pullback((coefficient_bar, -lambda1, -lambda3))
    return compact_bar


def _compact_prepared_bar_to_prepared(
    prepared: PreparedMonoenergeticSystem,
    *,
    nu_hat: Array,
    epsi_hat: Array,
    compact_bar,
):
    """Chain compact-input bars to prepared leaves, summing shared geometry paths."""
    _, pullback = jax.vjp(
        lambda prepared_value: _compact_prepared_residual_inputs(
            prepared_value,
            nu_hat,
            epsi_hat,
        ),
        prepared,
    )
    (prepared_bar,) = pullback(compact_bar)
    return prepared_bar


def _directional_compact_prepared_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    *,
    nu_hat: Array,
    epsi_hat: Array,
    nu_hat_dot: Array,
    epsi_hat_dot: Array,
    f1_full: Array,
    f3_full: Array,
    f1_dot: Array,
    f3_dot: Array,
    lambda1: Array,
    lambda3: Array,
    lambda1_dot: Array,
    lambda3_dot: Array,
    coefficient_bar: Array,
    coefficient_bar_dot: Array | None = None,
):
    """Directional compact-residual cotangent without a prepared VJP.

    The two outputs are the base and case-directional compact cotangents.
    The caller should retain only the directional cotangent for the primal
    prepared bar, combine it with the base compact bar, then execute one
    compact-to-prepared pullback.
    """

    def _compact_bar_from_dynamic_terms(
        nu_hat_value,
        epsi_hat_value,
        f1_value,
        f3_value,
        lambda1_value,
        lambda3_value,
        coefficient_bar_value,
    ):
        local_ctx = _operator_context(
            prepared.surface,
            prepared.geometry,
            prepared.grid,
            nu_hat_value,
            epsi_hat_value,
        )
        return _compact_prepared_gradient_from_adjoint(
            prepared,
            local_ctx,
            f1_value,
            f3_value,
            lambda1_value,
            lambda3_value,
            coefficient_bar_value,
        )

    if coefficient_bar_dot is None:
        coefficient_bar_dot = jnp.zeros_like(coefficient_bar)

    return jax.jvp(
        _compact_bar_from_dynamic_terms,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3, coefficient_bar),
        (
            nu_hat_dot,
            epsi_hat_dot,
            f1_dot,
            f3_dot,
            lambda1_dot,
            lambda3_dot,
            coefficient_bar_dot,
        ),
    )


def _directional_geometry_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    *,
    nu_hat: Array,
    epsi_hat: Array,
    nu_hat_dot: Array,
    epsi_hat_dot: Array,
    f1_full: Array,
    f3_full: Array,
    f1_dot: Array,
    f3_dot: Array,
    lambda1: Array,
    lambda3: Array,
    lambda1_dot: Array,
    lambda3_dot: Array,
    coefficient_bar: Array,
    coefficient_bar_dot: Array | None = None,
):
    """Return base and directional exact geometry bars without re-solving.

    The directional result is the derivative of the fixed-residual geometry
    pullback along the supplied case direction.  Primal and adjoint tangents
    are inputs, so the nested JVP never differentiates through an LU
    factorization or invokes another implicit solve.
    """

    def _geometry_bar_from_dynamic_terms(
        nu_hat_value,
        epsi_hat_value,
        f1_value,
        f3_value,
        lambda1_value,
        lambda3_value,
        coefficient_bar_value,
    ):
        local_ctx = OperatorContext(
            surface=prepared.surface,
            geometry=prepared.geometry,
            nu_hat=nu_hat_value,
            epsi_hat=epsi_hat_value,
        )
        return _geometry_gradient_from_adjoint(
            prepared,
            local_ctx,
            f1_value,
            f3_value,
            lambda1_value,
            lambda3_value,
            coefficient_bar_value,
        )

    if coefficient_bar_dot is None:
        coefficient_bar_dot = jnp.zeros_like(coefficient_bar)

    return jax.jvp(
        _geometry_bar_from_dynamic_terms,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3, coefficient_bar),
        (
            nu_hat_dot,
            epsi_hat_dot,
            f1_dot,
            f3_dot,
            lambda1_dot,
            lambda3_dot,
            coefficient_bar_dot,
        ),
    )


def _directional_prepared_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    *,
    nu_hat: Array,
    epsi_hat: Array,
    nu_hat_dot: Array,
    epsi_hat_dot: Array,
    f1_full: Array,
    f3_full: Array,
    f1_dot: Array,
    f3_dot: Array,
    lambda1: Array,
    lambda3: Array,
    lambda1_dot: Array,
    lambda3_dot: Array,
    coefficient_bar: Array,
    coefficient_bar_dot: Array | None = None,
):
    """Directional counterpart of :func:`_prepared_gradient_from_adjoint`."""

    def _prepared_bar_from_dynamic_terms(
        nu_hat_value,
        epsi_hat_value,
        f1_value,
        f3_value,
        lambda1_value,
        lambda3_value,
        coefficient_bar_value,
    ):
        local_ctx = _operator_context(
            prepared.surface, prepared.geometry, prepared.grid, nu_hat_value, epsi_hat_value
        )
        return _prepared_gradient_from_adjoint(
            prepared,
            local_ctx,
            f1_value,
            f3_value,
            lambda1_value,
            lambda3_value,
            coefficient_bar_value,
        )

    if coefficient_bar_dot is None:
        coefficient_bar_dot = jnp.zeros_like(coefficient_bar)

    return jax.jvp(
        _prepared_bar_from_dynamic_terms,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3, coefficient_bar),
        (
            nu_hat_dot,
            epsi_hat_dot,
            f1_dot,
            f3_dot,
            lambda1_dot,
            lambda3_dot,
            coefficient_bar_dot,
        ),
    )


def _solve_factorized_adjoint_field_pair(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    first_field_rhs: Array,
    second_field_rhs: Array,
) -> tuple[Array, Array]:
    """Solve two compatible adjoint field RHS through one matrix-RHS call.

    The leading block and unknown axes are shared and a trailing RHS-column
    axis is required. It is deliberately independent of the scalar low-dot
    helper; scalar vector RHS must use the established scalar solve.
    """

    if first_field_rhs.ndim != 3 or second_field_rhs.shape != first_field_rhs.shape:
        raise ValueError("Paired adjoint RHS must both have shape (mode, unknown, n_rhs).")
    packed = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.concatenate([first_field_rhs, second_field_rhs], axis=-1),
    )
    first_solution, second_solution = jnp.split(packed, 2, axis=-1)
    return first_solution, second_solution


def _solve_factorized_directional_adjoint_field_pair(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    first_field_adjoint: Array,
    second_field_adjoint: Array,
    first_field_rhs_dot: Array,
    second_field_rhs_dot: Array,
    diagonal_dot: Array,
) -> tuple[Array, Array]:
    """Solve paired directional adjoints for a trailing direction/RHS axis.

    ``diagonal_dot`` has layout ``(mode, direction, row, column)`` and the
    four field arrays have layout ``(mode, unknown, direction)``.  The RHS is
    the exact scalar formula ``g_dot - diagonal_dot.T @ lambda`` evaluated
    independently for each direction, then packed only for the factorized
    transpose solve.
    """

    expected_shape = first_field_adjoint.shape
    if (
        second_field_adjoint.shape != expected_shape
        or first_field_rhs_dot.shape != expected_shape
        or second_field_rhs_dot.shape != expected_shape
        or diagonal_dot.shape
        != (expected_shape[0], expected_shape[-1], expected_shape[1], expected_shape[1])
    ):
        raise ValueError("Directional paired-adjoint inputs have incompatible shapes.")
    first_rhs = first_field_rhs_dot - jnp.einsum("mdji,mjd->mid", diagonal_dot, first_field_adjoint)
    second_rhs = second_field_rhs_dot - jnp.einsum(
        "mdji,mjd->mid", diagonal_dot, second_field_adjoint
    )
    return _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        first_rhs,
        second_rhs,
    )


def _solve_factorized_multi_rhs_directional_adjoint_field_pair(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    first_field_adjoint: Array,
    second_field_adjoint: Array,
    first_field_rhs_dot: Array,
    second_field_rhs_dot: Array,
    diagonal_dot: Array,
) -> tuple[Array, Array]:
    """Solve directional adjoints for objective RHS columns in one call.

    This is the matrix-RHS counterpart of
    :func:`_solve_factorized_directional_adjoint_field_pair`.  The trailing
    axes are ``(rhs, direction)`` rather than only ``(direction,)``:

    - fields: ``(mode, unknown, rhs, direction)``;
    - diagonal derivatives: ``(mode, direction, row, column)``.

    The physical directions are shared by every objective RHS.  They are
    packed with the objective columns only at the block-tridiagonal transpose
    solve, then restored immediately.  In particular, this is not a ``vmap``
    around scalar implicit solves and it keeps no factorisation outside the
    caller's local anchor calculation.
    """

    expected_shape = first_field_adjoint.shape
    if (
        first_field_adjoint.ndim != 4
        or second_field_adjoint.shape != expected_shape
        or first_field_rhs_dot.shape != expected_shape
        or second_field_rhs_dot.shape != expected_shape
        or diagonal_dot.shape
        != (
            expected_shape[0],
            expected_shape[-1],
            expected_shape[1],
            expected_shape[1],
        )
    ):
        raise ValueError("Multi-RHS directional paired-adjoint inputs have incompatible shapes.")

    # ``diagonal_dot`` has no RHS axis: each physical case direction acts on
    # every objective adjoint column.  Keep that broadcasting explicit so a
    # later caller cannot accidentally introduce an objective-dependent
    # physical direction.
    first_rhs = first_field_rhs_dot - jnp.einsum(
        "mdji,mjrd->mird", diagonal_dot, first_field_adjoint
    )
    second_rhs = second_field_rhs_dot - jnp.einsum(
        "mdji,mjrd->mird", diagonal_dot, second_field_adjoint
    )
    mode_count, unknown_count, rhs_count, direction_count = first_rhs.shape
    first_packed = jnp.reshape(first_rhs, (mode_count, unknown_count, rhs_count * direction_count))
    second_packed = jnp.reshape(
        second_rhs, (mode_count, unknown_count, rhs_count * direction_count)
    )
    first_solution, second_solution = _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        first_packed,
        second_packed,
    )
    return (
        jnp.reshape(first_solution, first_rhs.shape),
        jnp.reshape(second_solution, second_rhs.shape),
    )


def _prepared_primal_context(prepared, case):
    """Share local primal setup without adding solves or JIT boundaries."""
    transport_scale = prepared.geometry.transport_psi_scale
    epsi = case.resolved_epsi_hat(transport_scale)
    primal = _prepared_implicit_vjp_primal(prepared, case.nu_hat, epsi)
    ctx = _operator_context(prepared.surface, prepared.geometry, prepared.grid, case.nu_hat, epsi)
    return transport_scale, ctx, primal


def pullback_prepared_coefficient_vector_case_and_prepared(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, PreparedMonoenergeticSystem]:
    """Exact grouped implicit pullback for case and complete prepared support.

    The returned prepared cotangent includes every differentiable leaf of the
    prepared NTX system.  It reuses the primal factorization and transpose
    solutions exactly as the geometry-only API does.
    """
    transport_scale, ctx, primal = _prepared_primal_context(prepared, case)
    (_coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper) = primal
    f1_bar_low, f3_bar_low, nu_hat_direct_bar = _coefficient_mode_pullback(
        prepared.geometry, f1_full[:3], f3_full[:3], ctx.nu_hat, coefficient_bar
    )
    g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
    g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
    lambda1 = _solve_factorized_adjoint(saved_lu, saved_piv, saved_lower, saved_upper, g1)
    lambda3 = _solve_factorized_adjoint(saved_lu, saved_piv, saved_lower, saved_upper, g3)
    nu_hat_implicit_bar, epsi_hat_bar = _parameter_gradient_from_adjoint(
        prepared, ctx, f1_full, f3_full, lambda1, lambda3
    )
    prepared_bar = _prepared_gradient_from_adjoint(
        prepared, ctx, f1_full, f3_full, lambda1, lambda3, coefficient_bar
    )
    prepared_bar = _add_er_coordinate_scale_bar(prepared_bar, case, epsi_hat_bar, transport_scale)
    case_bar = _case_cotangent_from_resolved_epsi_bar(
        case,
        nu_hat_bar=nu_hat_direct_bar + nu_hat_implicit_bar,
        epsi_hat_bar=epsi_hat_bar,
        transport_scale=transport_scale,
    )
    return case_bar, prepared_bar


def pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_bars: Array,
) -> tuple[MonoenergeticCase, dict[str, Array]]:
    """Database-only base-coefficient transpose to GeometryOnGrid bars.

    This is intentionally separate from the Lij low-dot helpers.  It shares
    one prepared primal and factorization across a leading table-bar RHS axis.
    Returning geometry bars lets a scan caller add every ``(nu, Er)`` case
    before one exact geometry-to-surface VJP, without materialising an
    RHS-batched prepared-system cotangent.
    """
    coefficient_bars = jnp.asarray(coefficient_bars)
    if coefficient_bars.ndim != 2:
        raise ValueError("coefficient_bars must have shape (n_rhs, n_coefficients).")
    if case.epsi_hat is None:
        raise ValueError("The database VMEC multi-RHS transpose requires explicit epsi_hat.")
    _, ctx, primal = _prepared_primal_context(prepared, case)
    (_coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper) = primal

    def _low_modes(one_bar):
        return _coefficient_mode_pullback(
            prepared.geometry, f1_full[:3], f3_full[:3], ctx.nu_hat, one_bar
        )

    f1_bar_low, f3_bar_low, nu_direct = jax.vmap(_low_modes)(coefficient_bars)
    rhs_count = coefficient_bars.shape[0]
    g1 = (
        jnp.zeros((*f1_full.shape, rhs_count), dtype=f1_full.dtype)
        .at[:3]
        .set(jnp.moveaxis(f1_bar_low, 0, -1))
    )
    g3 = (
        jnp.zeros((*f3_full.shape, rhs_count), dtype=f3_full.dtype)
        .at[:3]
        .set(jnp.moveaxis(f3_bar_low, 0, -1))
    )
    lambda1, lambda3 = _solve_factorized_adjoint_field_pair(
        saved_lu, saved_piv, saved_lower, saved_upper, g1, g3
    )

    def _parameter_bars(carry, mode_index):
        nu_bar, epsi_bar = carry
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx, mode_index, prepared.d_theta, prepared.d_zeta
        )
        diagonal_nu = jax.lax.cond(
            mode_index == 0, _zero_first_row, lambda value: value, diagonal_nu
        )
        diagonal_epsi = jax.lax.cond(
            mode_index == 0, _zero_first_row, lambda value: value, diagonal_epsi
        )
        f1_mode = jax.lax.dynamic_index_in_dim(f1_full, mode_index, axis=0, keepdims=False)
        f3_mode = jax.lax.dynamic_index_in_dim(f3_full, mode_index, axis=0, keepdims=False)
        lambda1_mode = jax.lax.dynamic_index_in_dim(lambda1, mode_index, axis=0, keepdims=False)
        lambda3_mode = jax.lax.dynamic_index_in_dim(lambda3, mode_index, axis=0, keepdims=False)
        return (
            nu_bar
            - jnp.einsum("nr,n->r", lambda1_mode, diagonal_nu @ f1_mode)
            - jnp.einsum("nr,n->r", lambda3_mode, diagonal_nu @ f3_mode),
            epsi_bar
            - jnp.einsum("nr,n->r", lambda1_mode, diagonal_epsi @ f1_mode)
            - jnp.einsum("nr,n->r", lambda3_mode, diagonal_epsi @ f3_mode),
        ), None

    (nu_implicit, epsi_bars), _ = jax.lax.scan(
        _parameter_bars,
        (
            jnp.zeros((rhs_count,), dtype=prepared.grid.jax_dtype),
            jnp.zeros((rhs_count,), dtype=prepared.grid.jax_dtype),
        ),
        jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32),
    )
    # Keep the fixed primal/factorisation and matrix-RHS adjoints above, but
    # use the exact geometry-only residual VJP here.  The earlier analytic
    # primitive contraction is fast for one case yet accumulates a measurable
    # Jacobian error across the different epsi values of a database scan.
    # This VJP is over ``GeometryOnGrid`` only: it does not materialise a
    # prepared-system cotangent or re-solve NTX.
    geometry_bars = jax.vmap(
        lambda lambda1_value, lambda3_value, coefficient_bar_value: _geometry_gradient_from_adjoint(
            prepared,
            ctx,
            f1_full,
            f3_full,
            lambda1_value,
            lambda3_value,
            coefficient_bar_value,
        ),
        in_axes=(2, 2, 0),
    )(lambda1, lambda3, coefficient_bars)
    return (
        MonoenergeticCase(
            nu_hat=nu_direct + nu_implicit,
            epsi_hat=epsi_bars,
            er_hat=None,
        ),
        geometry_bars,
    )


def _zero_first_row(block: Array) -> Array:
    """Keep the fixed nullspace row independent of operator parameters."""
    return block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))


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
    """Differentiate parameter contractions while preserving the fixed nullspace row."""
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


def _pullback_prepared_coefficient_vector_case_and_geometry_derivative_core(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    case_dot: MonoenergeticCase,
    coefficient_bar: Array,
    *,
    include_geometry: bool,
) -> tuple[
    MonoenergeticCase,
    MonoenergeticCase,
    GeometryOnGrid | None,
    GeometryOnGrid | None,
]:
    """Return exact case/profile and geometry bars for a coefficient direction.

    This is the reverse-mode companion needed by callers that differentiate
    coefficient-solve derivative fields. It avoids tracing a JVP through the LU
    factorization by differentiating the implicit primal/adjoint systems
    directly for one monoenergetic case. The geometry bars reuse those same
    primal and adjoint quantities.
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
        (
            f1_bar_low,
            f3_bar_low,
            nu_bar_direct,
        ),
        (
            f1_bar_low_dot,
            f3_bar_low_dot,
            nu_bar_direct_dot,
        ),
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
    if include_geometry:
        base_geometry_bar, directional_geometry_bar = _directional_geometry_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_hat_dot,
            epsi_hat_dot=epsi_hat_dot,
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=f1_dot,
            f3_dot=f3_dot,
            lambda1=lambda1,
            lambda3=lambda3,
            lambda1_dot=lambda1_dot,
            lambda3_dot=lambda3_dot,
            coefficient_bar=coefficient_bar,
        )
    else:
        base_geometry_bar = None
        directional_geometry_bar = None
    case_bars = (
        _case_cotangent_from_resolved_epsi_bar(
            case, nu_hat_bar=nu_bar, epsi_hat_bar=epsi_bar, transport_scale=transport_scale
        ),
        _case_cotangent_from_resolved_epsi_bar(
            case, nu_hat_bar=nu_bar_dot, epsi_hat_bar=epsi_bar_dot, transport_scale=transport_scale
        ),
    )
    return (
        case_bars[0],
        case_bars[1],
        base_geometry_bar,
        directional_geometry_bar,
    )


def solve_prepared_coefficient_vector_derivative_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    case_dot: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, MonoenergeticCase]:
    """Return the coefficient pullback and its directional derivative.

    With a fixed ``coefficient_bar`` the result is ``(J.T @ bar,
    D(J.T @ bar)[case_dot])``. This preserves the established NEOPAX order;
    it is the reverse of the two input bars of ``vjp(jvp(f, case_dot))``.
    """

    base_case_bar, tangent_case_bar, _base_geometry_bar, _tangent_geometry_bar = (
        _pullback_prepared_coefficient_vector_case_and_geometry_derivative_core(
            prepared,
            case,
            case_dot,
            coefficient_bar,
            include_geometry=False,
        )
    )
    return base_case_bar, tangent_case_bar


def _unpack_coefficient_bars(result):
    """Read the common callback contract, preserving caller-owned auxiliary data."""
    if len(result) not in {4, 5}:
        raise ValueError(
            "coefficient_bars_and_aux_fn must return "
            "(base_bars, first_bars, second_bars, auxiliary) optionally followed "
            "by (first_bars_dot, second_bars_dot)."
        )
    first_dot, second_dot = (
        result[4] if len(result) == 5 else (jnp.zeros_like(result[1]), jnp.zeros_like(result[2]))
    )
    return *result[:4], first_dot, second_dot


def _case_cotangent_from_resolved_epsi_bar(
    case: MonoenergeticCase,
    *,
    nu_hat_bar: Array,
    epsi_hat_bar: Array,
    transport_scale: float | Array | None,
) -> MonoenergeticCase:
    """Rebuild a case cotangent in the input's public coordinate system."""

    if case.epsi_hat is not None:
        return MonoenergeticCase(nu_hat=nu_hat_bar, epsi_hat=epsi_hat_bar, er_hat=None)
    if case.er_hat is not None:
        assert transport_scale is not None
        return MonoenergeticCase(
            nu_hat=nu_hat_bar,
            epsi_hat=None,
            er_hat=epsi_hat_bar / transport_scale,
        )
    return MonoenergeticCase(nu_hat=nu_hat_bar, epsi_hat=None, er_hat=None)


def _add_er_coordinate_scale_bar(
    prepared_bar: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    epsi_hat_bar: Array,
    transport_scale: float | Array | None,
) -> PreparedMonoenergeticSystem:
    """Apply ``epsi_hat = er_hat / transport_scale`` to a prepared bar."""

    if case.er_hat is None:
        return prepared_bar
    assert transport_scale is not None
    return dataclasses.replace(
        prepared_bar,
        geometry=dataclasses.replace(
            prepared_bar.geometry,
            transport_psi_scale=(
                prepared_bar.geometry.transport_psi_scale
                - epsi_hat_bar * jnp.asarray(case.er_hat) / jnp.asarray(transport_scale) ** 2
            ),
        ),
    )


def _sum_prepared_cotangent_trees(*trees: PreparedMonoenergeticSystem):
    """Sum prepared cotangents while preserving JAX's static ``float0`` leaves."""

    def _sum_leaves(*values):
        if jnp.asarray(values[0]).dtype == jax.dtypes.float0:
            return values[0]
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result

    return jax.tree_util.tree_map(_sum_leaves, *trees)
