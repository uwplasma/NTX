"""Prepared monoenergetic solve path and custom-VJP wrappers."""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .geometry import GeometryOnGrid
from ._solver_adjoint import (
    _case_and_geometry_gradient_from_adjoint,
    _coefficient_mode_pullback,
    _directional_geometry_gradient_from_adjoint,
    _directional_prepared_gradient_from_adjoint,
    _geometry_gradient_from_adjoint,
    _prepared_gradient_from_adjoint,
    _parameter_gradient_from_adjoint,
    _parameter_gradient_from_adjoint_multi_rhs,
    _prepared_implicit_vjp_primal,
)
from ._solver_context import _operator_context
from ._solver_factorization import (
    _factorize_prepared_modes,
    _full_mode_residual_norm,
    _solve_factorized_adjoint,
    _solve_factorized_modes,
    _solve_modes_with_tail_residual,
)
from ._solver_types import (
    CompiledPreparedSolver,
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    ResidualAuditResult,
    TransportResult,
    transport_result_from_arrays,
)
from .operators import parameter_derivative_blocks, source_modes
from .transport import coefficients_from_modes, onsager_error


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
        raise ValueError(
            "Paired adjoint RHS must both have shape (mode, unknown, n_rhs)."
        )
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
        or diagonal_dot.shape != (
            expected_shape[0], expected_shape[-1], expected_shape[1], expected_shape[1]
        )
    ):
        raise ValueError("Directional paired-adjoint inputs have incompatible shapes.")
    first_rhs = first_field_rhs_dot - jnp.einsum(
        "mdji,mjd->mid", diagonal_dot, first_field_adjoint
    )
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
        raise ValueError(
            "Multi-RHS directional paired-adjoint inputs have incompatible shapes."
        )

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
    first_packed = jnp.reshape(
        first_rhs, (mode_count, unknown_count, rhs_count * direction_count)
    )
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


def solve_prepared(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    *,
    adjoint_window: int | None = None,
) -> TransportResult:
    """Solve one monoenergetic case using precomputed geometry and derivatives.

    ``adjoint_window`` bounds the reverse pass; see
    :func:`ntx.advise_adjoint_window`. It has no effect on a forward solve.
    """

    return transport_result_from_arrays(
        _solve_prepared_arrays(prepared, case, adjoint_window=adjoint_window)
    )


def solve_prepared_internal(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Solve one prepared monoenergetic case and return `(Dij, f, s)` low-order arrays."""

    values = _solve_prepared_arrays(prepared, case)
    result = transport_result_from_arrays(values)
    dij = _monoenergetic_matrix(result.D11, result.D31, result.D13, result.D33)
    return dij, values[9], values[10]


def audit_prepared_residuals(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> ResidualAuditResult:
    """Reconstruct all Legendre modes and independently audit every block row.

    This diagnostic stores full block factors and modes. Use it for validation,
    not in memory-constrained production scans.
    """
    grid = prepared.grid
    epsi_hat = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
    ctx = _operator_context(
        prepared.surface, prepared.geometry, grid, case.nu_hat, epsi_hat
    )
    source, parallel_source = source_modes(ctx, grid.n_xi)
    retained, _, tail_residual = _solve_modes_with_tail_residual(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        source,
        parallel_source,
    )
    factors = _factorize_prepared_modes(
        ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta
    )
    full_modes = _solve_factorized_modes(*factors, source)
    full_residual = _full_mode_residual_norm(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        source,
        full_modes,
    )
    return ResidualAuditResult(
        tail_eliminated_l2=tail_residual,
        full_system_l2=full_residual,
        retained_mode_max_abs_error=jnp.max(jnp.abs(retained - full_modes[:3])),
        n_modes=grid.n_xi + 1,
    )


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


def pullback_prepared_coefficient_vector_case_and_geometry(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, GeometryOnGrid]:
    """Exact grouped implicit pullback for case/profile and NTX geometry.

    Unlike :func:`solve_prepared_coefficient_vector_vjp`, this is an explicit
    pullback API. It retains the existing custom-VJP contract (``prepared`` is
    non-differentiable there), while exposing the exact prepared-geometry
    cotangent required by callers that already manage a higher-level support
    payload. No primal or adjoint system is solved twice.
    """

    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
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
    f1_bar_low, f3_bar_low, nu_hat_direct_bar = _coefficient_mode_pullback(
        prepared.geometry,
        f1_full[:3],
        f3_full[:3],
        ctx.nu_hat,
        coefficient_bar,
    )
    g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
    g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
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
    nu_hat_bar, epsi_hat_bar, geometry_bar = _case_and_geometry_gradient_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
        coefficient_bar,
        nu_hat_direct_bar,
    )
    if case.epsi_hat is not None:
        case_bar = MonoenergeticCase(
            nu_hat=nu_hat_bar,
            epsi_hat=epsi_hat_bar,
            er_hat=None,
        )
    elif case.er_hat is not None:
        assert transport_scale is not None
        case_bar = MonoenergeticCase(
            nu_hat=nu_hat_bar,
            epsi_hat=None,
            er_hat=epsi_hat_bar / transport_scale,
        )
    else:
        case_bar = MonoenergeticCase(
            nu_hat=nu_hat_bar,
            epsi_hat=None,
            er_hat=None,
        )
    return case_bar, geometry_bar


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
    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    (
        _coefficients,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
    ) = _prepared_implicit_vjp_primal(prepared, case.nu_hat, resolved_epsi_hat)
    ctx = _operator_context(
        prepared.surface, prepared.geometry, prepared.grid, case.nu_hat, resolved_epsi_hat
    )
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
    if case.epsi_hat is not None:
        case_bar = MonoenergeticCase(nu_hat=nu_hat_direct_bar + nu_hat_implicit_bar, epsi_hat=epsi_hat_bar, er_hat=None)
    elif case.er_hat is not None:
        assert transport_scale is not None
        # The fixed-residual prepared pullback holds the resolved epsilon
        # coordinate fixed.  The public ``er_hat`` representation additionally
        # depends on ``geometry.transport_psi_scale`` through
        # epsilon = er_hat / transport_psi_scale.
        prepared_bar = dataclasses.replace(
            prepared_bar,
            geometry=dataclasses.replace(
                prepared_bar.geometry,
                transport_psi_scale=(
                    prepared_bar.geometry.transport_psi_scale
                    - epsi_hat_bar
                    * jnp.asarray(case.er_hat)
                    / jnp.asarray(transport_scale) ** 2
                ),
            ),
        )
        case_bar = MonoenergeticCase(
            nu_hat=nu_hat_direct_bar + nu_hat_implicit_bar,
            epsi_hat=None,
            er_hat=epsi_hat_bar / transport_scale,
        )
    else:
        case_bar = MonoenergeticCase(
            nu_hat=nu_hat_direct_bar + nu_hat_implicit_bar, epsi_hat=None, er_hat=None
        )
    return case_bar, prepared_bar


def pullback_prepared_coefficient_vector_case_and_prepared_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_bars: Array,
) -> tuple[MonoenergeticCase, PreparedMonoenergeticSystem]:
    """Exact prepared-system pullback for a leading coefficient-RHS axis.

    This is an opt-in API, separate from the scalar custom-VJP and scalar
    explicit pullback APIs above.  It evaluates the implicit primal once,
    retains its unbatched block factors, and packs the two kinetic fields of
    every requested coefficient cotangent into one matrix-RHS transpose solve.
    The returned case and prepared cotangents carry the same leading RHS axis
    as ``coefficient_bars``.

    It intentionally covers only the base coefficient pullback.  The
    low-dot/two-direction transport helper has additional tangent terms and
    will use a dedicated native multi-RHS implementation rather than silently
    changing this scalar-compatible contract.
    """

    coefficient_bars = jnp.asarray(coefficient_bars)
    if coefficient_bars.ndim != 2:
        raise ValueError(
            "coefficient_bars must have shape (n_rhs, n_coefficients)."
        )

    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    (
        _coefficients,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
    ) = _prepared_implicit_vjp_primal(prepared, case.nu_hat, resolved_epsi_hat)
    ctx = _operator_context(
        prepared.surface, prepared.geometry, prepared.grid, case.nu_hat, resolved_epsi_hat
    )

    def _coefficient_bar_to_low_modes(one_coefficient_bar):
        return _coefficient_mode_pullback(
            prepared.geometry,
            f1_full[:3],
            f3_full[:3],
            ctx.nu_hat,
            one_coefficient_bar,
        )

    f1_bar_low, f3_bar_low, nu_hat_direct_bars = jax.vmap(
        _coefficient_bar_to_low_modes
    )(coefficient_bars)
    rhs_count = coefficient_bars.shape[0]
    g1 = jnp.zeros((*f1_full.shape, rhs_count), dtype=f1_full.dtype).at[:3].set(
        jnp.moveaxis(f1_bar_low, 0, -1)
    )
    g3 = jnp.zeros((*f3_full.shape, rhs_count), dtype=f3_full.dtype).at[:3].set(
        jnp.moveaxis(f3_bar_low, 0, -1)
    )
    lambda1, lambda3 = _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        g1,
        g3,
    )

    def _parameter_bars(carry, mode_index):
        nu_bar, epsi_bar = carry
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            mode_index,
            prepared.d_theta,
            prepared.d_zeta,
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
        nu_bar = nu_bar - (
            jnp.einsum("nr,n->r", lambda1_mode, diagonal_nu @ f1_mode)
            + jnp.einsum("nr,n->r", lambda3_mode, diagonal_nu @ f3_mode)
        )
        epsi_bar = epsi_bar - (
            jnp.einsum("nr,n->r", lambda1_mode, diagonal_epsi @ f1_mode)
            + jnp.einsum("nr,n->r", lambda3_mode, diagonal_epsi @ f3_mode)
        )
        return (nu_bar, epsi_bar), None

    (nu_hat_implicit_bars, epsi_hat_bars), _ = jax.lax.scan(
        _parameter_bars,
        (
            jnp.zeros((rhs_count,), dtype=prepared.grid.jax_dtype),
            jnp.zeros((rhs_count,), dtype=prepared.grid.jax_dtype),
        ),
        jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32),
    )
    # ``PreparedMonoenergeticSystem`` has static metadata leaves.  Batch only
    # its dynamic cotangent leaves, then rebuild the original pytree, rather
    # than asking ``vmap`` to construct a batched instance with static fields.
    prepared_bar_tree = jax.tree_util.tree_structure(prepared)

    def _prepared_bar_leaves(one_lambda1, one_lambda3, one_coefficient_bar):
        prepared_bar = _prepared_gradient_from_adjoint(
            prepared,
            ctx,
            f1_full,
            f3_full,
            one_lambda1,
            one_lambda3,
            one_coefficient_bar,
        )
        return tuple(jax.tree_util.tree_leaves(prepared_bar))

    batched_prepared_bar_leaves = jax.vmap(
        _prepared_bar_leaves,
        in_axes=(2, 2, 0),
    )(lambda1, lambda3, coefficient_bars)
    prepared_bars = jax.tree_util.tree_unflatten(
        prepared_bar_tree,
        batched_prepared_bar_leaves,
    )

    if case.epsi_hat is not None:
        case_bars = MonoenergeticCase(
            nu_hat=nu_hat_direct_bars + nu_hat_implicit_bars,
            epsi_hat=epsi_hat_bars,
            er_hat=None,
        )
    elif case.er_hat is not None:
        assert transport_scale is not None
        prepared_bars = dataclasses.replace(
            prepared_bars,
            geometry=dataclasses.replace(
                prepared_bars.geometry,
                transport_psi_scale=(
                    prepared_bars.geometry.transport_psi_scale
                    - epsi_hat_bars
                    * jnp.asarray(case.er_hat)
                    / jnp.asarray(transport_scale) ** 2
                ),
            ),
        )
        case_bars = MonoenergeticCase(
            nu_hat=nu_hat_direct_bars + nu_hat_implicit_bars,
            epsi_hat=None,
            er_hat=epsi_hat_bars / transport_scale,
        )
    else:
        case_bars = MonoenergeticCase(
            nu_hat=nu_hat_direct_bars + nu_hat_implicit_bars,
            epsi_hat=None,
            er_hat=None,
        )
    return case_bars, prepared_bars


def _zero_first_row(block: Array) -> Array:
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
    if case.epsi_hat is not None:
        case_bars = (
            MonoenergeticCase(nu_hat=nu_bar, epsi_hat=epsi_bar, er_hat=None),
            MonoenergeticCase(nu_hat=nu_bar_dot, epsi_hat=epsi_bar_dot, er_hat=None),
        )
    elif case.er_hat is not None:
        assert transport_scale is not None
        case_bars = (
            MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=epsi_bar / transport_scale),
            MonoenergeticCase(nu_hat=nu_bar_dot, epsi_hat=None, er_hat=epsi_bar_dot / transport_scale),
        )
    else:
        case_bars = (
            MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=None),
            MonoenergeticCase(nu_hat=nu_bar_dot, epsi_hat=None, er_hat=None),
        )
    return (
        case_bars[0],
        case_bars[1],
        base_geometry_bar,
        directional_geometry_bar,
    )


def pullback_prepared_coefficient_vector_case_and_geometry_derivative(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    case_dot: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, MonoenergeticCase, GeometryOnGrid, GeometryOnGrid]:
    """Exact grouped directional pullback for case/profile and geometry."""

    return _pullback_prepared_coefficient_vector_case_and_geometry_derivative_core(
        prepared,
        case,
        case_dot,
        coefficient_bar,
        include_geometry=True,
    )


def solve_prepared_coefficient_vector_derivative_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    case_dot: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, MonoenergeticCase]:
    """Backward-compatible case-only view of the grouped directional rule."""

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


def _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_fn,
    *,
    include_geometry: bool,
    include_prepared: bool = False,
    return_coefficient_aux: bool,
    packed_support_directional_adjoint: bool = False,
    support_only: bool = False,
    primal_residuals=None,
    return_primal_residuals: bool = False,
):
    """Fused exact pullback for two coefficient-derivative contractions.

    Higher-level transport models own the mapping from coefficient vectors to
    objective-specific coefficient cotangents, so this helper receives that
    mapping as a Python callback. NTX still owns the expensive implicit-solve
    pullback algebra and reuses one primal factorization for the base VJP and
    the two low-mode directional pullbacks. ``include_geometry`` is static:
    the legacy public API keeps its existing case-only output and does not
    build the prepared-geometry cotangents. When ``return_coefficient_aux``
    is true, ``coefficient_bar_fn`` must additionally return a pytree after
    the three coefficient cotangents; it is forwarded unchanged. This lets a
    transport caller carry exact direct prefactor bars (for example ``drds``)
    without another NTX solve.
    """

    if include_geometry and include_prepared:
        raise ValueError("Only one prepared-support cotangent representation may be requested.")
    include_support = include_geometry or include_prepared
    if support_only and not include_support:
        raise ValueError("support_only requires a prepared-support representation.")
    if packed_support_directional_adjoint and not include_support:
        raise ValueError("Packed support directional adjoint requires a support pullback mode.")

    from jax.scipy.linalg import lu_solve

    if primal_residuals is None:
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
        ctx = _operator_context(
            prepared.surface,
            prepared.geometry,
            prepared.grid,
            case.nu_hat,
            resolved_epsi_hat,
        )
    else:
        (
            transport_scale,
            resolved_epsi_hat,
            first_nu_dot,
            first_epsi_dot,
            second_nu_dot,
            second_epsi_dot,
            coefficients,
            f1_full,
            f3_full,
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            packed_f_dot_low_matrix,
            first_coefficient_dot,
            second_coefficient_dot,
        ) = primal_residuals
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

        if include_support:
            lambda1 = _solve_factorized_adjoint_scan(g1)
            lambda3 = _solve_factorized_adjoint_scan(g3)
            nu_bar_implicit, epsi_bar = _parameter_gradient_from_adjoint(
                prepared,
                ctx,
                f1_full,
                f3_full,
                lambda1,
                lambda3,
            )
            support_bar = (
                _geometry_gradient_from_adjoint(
                    prepared, ctx, f1_full, f3_full, lambda1, lambda3, coefficient_bar
                )
                if include_geometry
                else _prepared_gradient_from_adjoint(
                    prepared, ctx, f1_full, f3_full, lambda1, lambda3, coefficient_bar
                )
            )
            return nu_bar_direct + nu_bar_implicit, epsi_bar, support_bar

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

    if primal_residuals is None:
        packed_f_dot_low_matrix = _solve_factorized_low_modes_scan(
            _packed_source_dot_matrix_for_mode
        )

        def _coefficient_vector_from_low_modes(modes1, modes3, nu_value):
            return jnp.stack(
                coefficients_from_modes(prepared.geometry, modes1, modes3, nu_value)
            )

        _, first_coefficient_dot = jax.jvp(
            _coefficient_vector_from_low_modes,
            (f1_full[:3], f3_full[:3], ctx.nu_hat),
            (
                packed_f_dot_low_matrix[..., 0],
                packed_f_dot_low_matrix[..., 1],
                first_nu_dot,
            ),
        )
        _, second_coefficient_dot = jax.jvp(
            _coefficient_vector_from_low_modes,
            (f1_full[:3], f3_full[:3], ctx.nu_hat),
            (
                packed_f_dot_low_matrix[..., 2],
                packed_f_dot_low_matrix[..., 3],
                second_nu_dot,
            ),
        )
    local_primal_residuals = (
        transport_scale,
        resolved_epsi_hat,
        first_nu_dot,
        first_epsi_dot,
        second_nu_dot,
        second_epsi_dot,
        coefficients,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        packed_f_dot_low_matrix,
        first_coefficient_dot,
        second_coefficient_dot,
    )
    # This is also useful on its own for callers that need the primal
    # coefficient vector and two exact parameter directions.  Keep the
    # factorization-local calculation here: it is the same established
    # algebra used by the grouped pullback below, but stops before any
    # reverse contractions are formed.
    if coefficient_bar_fn is None:
        primal_outputs = (coefficients, first_coefficient_dot, second_coefficient_dot)
        return (
            (primal_outputs, local_primal_residuals)
            if return_primal_residuals
            else primal_outputs
        )
    coefficient_bar_result = (
        coefficient_bar_fn(
            coefficients,
            first_coefficient_dot,
            second_coefficient_dot,
        )
        if return_coefficient_aux
        else coefficient_bar_fn(coefficients)
    )
    base_coefficient_bar, first_coefficient_bar, second_coefficient_bar = (
        coefficient_bar_result[:3]
    )
    coefficient_aux = coefficient_bar_result[3] if return_coefficient_aux else None
    # A differentiated transport-moment pullback can make its coefficient
    # cotangent depend on the NTX coefficient primal.  Carry that mixed term
    # into the two low-dot reverse directions when supplied.  The optional
    # fifth result keeps every established caller on the previous zero-tangent
    # contract.
    if return_coefficient_aux and len(coefficient_bar_result) > 4:
        first_coefficient_bar_dot, second_coefficient_bar_dot = coefficient_bar_result[4]
    else:
        first_coefficient_bar_dot = jnp.zeros_like(first_coefficient_bar)
        second_coefficient_bar_dot = jnp.zeros_like(second_coefficient_bar)

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
        coefficient_bar_dot_pair,
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
            (
                coefficient_bar,
                coefficient_bar_dot,
                nu_hat_dot,
                epsi_hat_dot,
                f1_dot_low,
                f3_dot_low,
            ) = xs

            def _coefficient_pullback(modes1, modes3, nu_value, coefficient_bar_value):
                return _coefficient_mode_pullback(
                    prepared.geometry,
                    modes1,
                    modes3,
                    nu_value,
                    coefficient_bar_value,
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
                (f1_full[:3], f3_full[:3], ctx.nu_hat, coefficient_bar),
                (f1_dot_low, f3_dot_low, nu_hat_dot, coefficient_bar_dot),
            )

            g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
            g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
            g1_dot = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low_dot)
            g3_dot = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low_dot)

            lambda1 = _solve_factorized_adjoint_scan(g1)
            lambda3 = _solve_factorized_adjoint_scan(g3)

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

            if support_only:
                # The support-only caller holds the local case fixed.  Do not
                # form any case/profile contractions: only the forward
                # directional fields and prepared-support adjoints are needed.
                source1_dot, source3_dot = jax.lax.map(
                    lambda k: _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot),
                    mode_indices,
                )
                f1_dot = _solve_factorized_modes(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source1_dot,
                )
                f3_dot = _solve_factorized_modes(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source3_dot,
                )
                lambda1_dot = _solve_factorized_adjoint_scan(
                    jax.lax.map(
                        lambda k: _adjoint_rhs_dot_for_mode(lambda1, g1_dot, k),
                        mode_indices,
                    )
                )
                lambda3_dot = _solve_factorized_adjoint_scan(
                    jax.lax.map(
                        lambda k: _adjoint_rhs_dot_for_mode(lambda3, g3_dot, k),
                        mode_indices,
                    )
                )
                base_support_bar, directional_support_bar = (
                    _directional_geometry_gradient_from_adjoint(
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
                        coefficient_bar_dot=coefficient_bar_dot,
                    )
                    if include_geometry
                    else _directional_prepared_gradient_from_adjoint(
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
                        coefficient_bar_dot=coefficient_bar_dot,
                    )
                )
                return base_support_bar, directional_support_bar

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
            if include_support:
                source1_dot, source3_dot = jax.lax.map(
                    lambda k: _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot),
                    mode_indices,
                )
                f1_dot = _solve_factorized_modes(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source1_dot,
                )
                f3_dot = _solve_factorized_modes(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source3_dot,
                )
                if packed_support_directional_adjoint:
                    diagonal_dot = jax.lax.map(_diagonal_dot, mode_indices)[:, None, :, :]
                    lambda1_dot_pair, lambda3_dot_pair = (
                        _solve_factorized_directional_adjoint_field_pair(
                            saved_lu,
                            saved_piv,
                            saved_lower,
                            saved_upper,
                            lambda1[..., None],
                            lambda3[..., None],
                            g1_dot[..., None],
                            g3_dot[..., None],
                            diagonal_dot,
                        )
                    )
                    lambda1_dot = lambda1_dot_pair[..., 0]
                    lambda3_dot = lambda3_dot_pair[..., 0]
                else:
                    lambda1_dot = _solve_factorized_adjoint_scan(
                        jax.lax.map(
                            lambda k: _adjoint_rhs_dot_for_mode(lambda1, g1_dot, k),
                            mode_indices,
                        )
                    )
                    lambda3_dot = _solve_factorized_adjoint_scan(
                        jax.lax.map(
                            lambda k: _adjoint_rhs_dot_for_mode(lambda3, g3_dot, k),
                            mode_indices,
                        )
                )
                base_support_bar, directional_support_bar = (
                    _directional_geometry_gradient_from_adjoint(
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
                        coefficient_bar_dot=coefficient_bar_dot,
                    )
                    if include_geometry
                    else _directional_prepared_gradient_from_adjoint(
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
                        coefficient_bar_dot=coefficient_bar_dot,
                    )
                )
                return (
                    nu_bar_direct + nu_bar_implicit,
                    epsi_bar,
                    nu_bar_direct_dot + nu_bar_implicit_dot,
                    epsi_bar_dot,
                    base_support_bar,
                    directional_support_bar,
                )
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
                coefficient_bar_dot_pair,
                nu_hat_dot_pair,
                epsi_hat_dot_pair,
                f1_dot_low_pair,
                f3_dot_low_pair,
            ),
        )
        return outputs

    base = _base_pullback(base_coefficient_bar)
    if include_prepared and case.er_hat is not None:
        # ``_prepared_gradient_from_adjoint`` holds resolved epsilon fixed.
        # The public er_hat representation instead has
        # epsilon = er_hat / transport_psi_scale, so retain the same scale
        # chain rule as pullback_prepared_coefficient_vector_case_and_prepared.
        base_nu_bar, base_epsi_bar, base_prepared_bar = base
        assert transport_scale is not None
        base_prepared_bar = dataclasses.replace(
            base_prepared_bar,
            geometry=dataclasses.replace(
                base_prepared_bar.geometry,
                transport_psi_scale=(
                    base_prepared_bar.geometry.transport_psi_scale
                    - base_epsi_bar
                    * jnp.asarray(case.er_hat)
                    / jnp.asarray(transport_scale) ** 2
                ),
            ),
        )
        base = (base_nu_bar, base_epsi_bar, base_prepared_bar)
    direction = _scan_direction_pullbacks(
        jnp.stack([first_coefficient_bar, second_coefficient_bar], axis=0),
        jnp.stack([first_coefficient_bar_dot, second_coefficient_bar_dot], axis=0),
        jnp.stack([first_nu_dot, second_nu_dot], axis=0),
        jnp.stack([first_epsi_dot, second_epsi_dot], axis=0),
        jnp.stack([packed_f_dot_low_matrix[..., 0], packed_f_dot_low_matrix[..., 2]], axis=0),
        jnp.stack([packed_f_dot_low_matrix[..., 1], packed_f_dot_low_matrix[..., 3]], axis=0),
    )
    if support_only:
        direction_base_support_bar, direction_support_bar_dot = direction
    elif include_support:
        (
            direction_nu_bar,
            direction_epsi_bar,
            direction_nu_bar_dot,
            direction_epsi_bar_dot,
            direction_base_support_bar,
            direction_support_bar_dot,
        ) = direction
    else:
        (
            direction_nu_bar,
            direction_epsi_bar,
            direction_nu_bar_dot,
            direction_epsi_bar_dot,
        ) = direction
    if include_support:
        base_nu_bar, base_epsi_bar, base_support_bar = base

        def _take_direction_support(support_bars, index):
            return jax.tree_util.tree_map(
                lambda value: jax.lax.dynamic_index_in_dim(
                    value,
                    index,
                    axis=0,
                    keepdims=False,
                ),
                support_bars,
            )

        if support_only:
            return (
                base_support_bar,
                _take_direction_support(direction_base_support_bar, 0),
                _take_direction_support(direction_support_bar_dot, 0),
                _take_direction_support(direction_base_support_bar, 1),
                _take_direction_support(direction_support_bar_dot, 1),
                coefficient_aux,
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
        first = (
            *first,
            _take_direction_support(direction_base_support_bar, 0),
            _take_direction_support(direction_support_bar_dot, 0),
        )
        second = (
            *second,
            _take_direction_support(direction_base_support_bar, 1),
            _take_direction_support(direction_support_bar_dot, 1),
        )
        result = (base_nu_bar, base_epsi_bar, base_support_bar, *first, *second)
        return (*result, coefficient_aux) if return_coefficient_aux else result
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


def _lowdot_two_pullback_native_multi_rhs_adjoint_fields(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
):
    """Build native matrix-RHS fields for the exact low-dot support pullback.

    This private building block deliberately stops before constructing any
    prepared-system cotangent pytree.  It evaluates the primal and the two
    physical forward directions once, then forms all base and directional
    implicit adjoints with trailing objective-RHS columns.  The caller can
    subsequently apply the prepared-gradient contractions to those fields
    without entering the scalar low-dot implicit helper once per objective.

    Returned field layouts are ``(mode, unknown, rhs)`` for base fields and
    ``(mode, unknown, rhs, direction)`` for directional fields.  The two
    physical direction fields themselves have no RHS axis.  This distinction
    is essential: physical forward directions are objective-independent, while
    cotangents are objective-dependent.
    """

    primal_outputs, primal_residuals = (
        _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
            prepared,
            case,
            first_case_dot,
            second_case_dot,
            None,
            include_geometry=False,
            return_coefficient_aux=True,
            return_primal_residuals=True,
        )
    )
    (
        transport_scale,
        resolved_epsi_hat,
        first_nu_dot,
        first_epsi_dot,
        second_nu_dot,
        second_epsi_dot,
        _coefficients,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        packed_f_dot_low_matrix,
        _first_coefficient_dot,
        _second_coefficient_dot,
    ) = primal_residuals
    del transport_scale

    coefficient_bars_and_aux = coefficient_bars_and_aux_fn(*primal_outputs)
    if len(coefficient_bars_and_aux) not in {4, 5}:
        raise ValueError(
            "coefficient_bars_and_aux_fn must return "
            "(base_bars, first_bars, second_bars, auxiliary) optionally followed "
            "by (first_bars_dot, second_bars_dot)."
        )
    base_coefficient_bars, first_coefficient_bars, second_coefficient_bars, auxiliary = (
        coefficient_bars_and_aux[:4]
    )
    base_coefficient_bars = jnp.asarray(base_coefficient_bars)
    first_coefficient_bars = jnp.asarray(first_coefficient_bars)
    second_coefficient_bars = jnp.asarray(second_coefficient_bars)
    if len(coefficient_bars_and_aux) == 5:
        first_coefficient_bars_dot, second_coefficient_bars_dot = (
            jnp.asarray(value) for value in coefficient_bars_and_aux[4]
        )
    else:
        first_coefficient_bars_dot = jnp.zeros_like(first_coefficient_bars)
        second_coefficient_bars_dot = jnp.zeros_like(second_coefficient_bars)
    if (
        base_coefficient_bars.ndim != 2
        or first_coefficient_bars.shape != base_coefficient_bars.shape
        or second_coefficient_bars.shape != base_coefficient_bars.shape
        or first_coefficient_bars_dot.shape != base_coefficient_bars.shape
        or second_coefficient_bars_dot.shape != base_coefficient_bars.shape
    ):
        raise ValueError(
            "All low-dot coefficient cotangent batches must have shape "
            "(n_rhs, n_coefficients)."
        )

    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        resolved_epsi_hat,
    )
    rhs_count = base_coefficient_bars.shape[0]
    direction_count = 2
    mode_count, unknown_count = f1_full.shape
    mode_indices = jnp.arange(mode_count, dtype=jnp.int32)

    def _zero_first_row_if_needed(block, k):
        zeroed = block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))
        return jnp.where(jnp.asarray(k) == 0, zeroed, block)

    def _coefficient_pullback(one_coefficient_bar):
        return _coefficient_mode_pullback(
            prepared.geometry,
            f1_full[:3],
            f3_full[:3],
            ctx.nu_hat,
            one_coefficient_bar,
        )

    def _field_rhs_from_low(low_fields):
        # ``low_fields`` is (rhs, retained-mode, unknown).  The factorized
        # solver uses mode/unknown/RHS ordering.
        return jnp.zeros(
            (mode_count, unknown_count, rhs_count), dtype=f1_full.dtype
        ).at[:3].set(jnp.moveaxis(low_fields, 0, -1))

    base_f1_low, base_f3_low, base_nu_direct = jax.vmap(_coefficient_pullback)(
        base_coefficient_bars
    )
    base_lambda1, base_lambda3 = _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        _field_rhs_from_low(base_f1_low),
        _field_rhs_from_low(base_f3_low),
    )

    nu_dots = jnp.stack([first_nu_dot, second_nu_dot])
    epsi_dots = jnp.stack([first_epsi_dot, second_epsi_dot])
    f1_dot_low = jnp.stack(
        [packed_f_dot_low_matrix[..., 0], packed_f_dot_low_matrix[..., 2]], axis=-1
    )
    f3_dot_low = jnp.stack(
        [packed_f_dot_low_matrix[..., 1], packed_f_dot_low_matrix[..., 3]], axis=-1
    )

    def _directional_coefficient_pullback(
        coefficient_bars,
        coefficient_bars_dot,
        f1_dot_low_value,
        f3_dot_low_value,
        nu_dot_value,
    ):
        def _one_rhs(one_coefficient_bar, one_coefficient_bar_dot):
            def _pullback_from_primal(
                modes1, modes3, nu_value, coefficient_bar_value
            ):
                return _coefficient_mode_pullback(
                    prepared.geometry,
                    modes1,
                    modes3,
                    nu_value,
                    coefficient_bar_value,
                )

            return jax.jvp(
                _pullback_from_primal,
                (f1_full[:3], f3_full[:3], ctx.nu_hat, one_coefficient_bar),
                (
                    f1_dot_low_value,
                    f3_dot_low_value,
                    nu_dot_value,
                    one_coefficient_bar_dot,
                ),
            )

        return jax.vmap(_one_rhs)(coefficient_bars, coefficient_bars_dot)

    first_directional = _directional_coefficient_pullback(
        first_coefficient_bars,
        first_coefficient_bars_dot,
        f1_dot_low[..., 0],
        f3_dot_low[..., 0],
        first_nu_dot,
    )
    second_directional = _directional_coefficient_pullback(
        second_coefficient_bars,
        second_coefficient_bars_dot,
        f1_dot_low[..., 1],
        f3_dot_low[..., 1],
        second_nu_dot,
    )
    directional_primal_low = tuple(
        jnp.stack([first_value, second_value], axis=-1)
        for first_value, second_value in zip(first_directional[0], second_directional[0], strict=True)
    )
    directional_dot_low = tuple(
        jnp.stack([first_value, second_value], axis=-1)
        for first_value, second_value in zip(first_directional[1], second_directional[1], strict=True)
    )
    # (rhs, retained-mode, unknown, direction) -> (mode, unknown, rhs, direction)
    directional_f1_low, directional_f3_low, directional_nu_direct = directional_primal_low
    directional_f1_low_dot, directional_f3_low_dot, directional_nu_direct_dot = (
        directional_dot_low
    )
    directional_f1_low = jnp.transpose(directional_f1_low, (1, 2, 0, 3))
    directional_f3_low = jnp.transpose(directional_f3_low, (1, 2, 0, 3))
    directional_f1_low_dot = jnp.transpose(directional_f1_low_dot, (1, 2, 0, 3))
    directional_f3_low_dot = jnp.transpose(directional_f3_low_dot, (1, 2, 0, 3))
    directional_shape = (mode_count, unknown_count, rhs_count, direction_count)
    directional_g1 = jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(
        directional_f1_low
    )
    directional_g3 = jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(
        directional_f3_low
    )
    directional_g1_dot = jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(
        directional_f1_low_dot
    )
    directional_g3_dot = jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(
        directional_f3_low_dot
    )
    directional_lambda1, directional_lambda3 = _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.reshape(directional_g1, (mode_count, unknown_count, rhs_count * direction_count)),
        jnp.reshape(directional_g3, (mode_count, unknown_count, rhs_count * direction_count)),
    )
    directional_lambda1 = jnp.reshape(directional_lambda1, directional_shape)
    directional_lambda3 = jnp.reshape(directional_lambda3, directional_shape)

    def _directional_sources_for_mode(k):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx, k, prepared.d_theta, prepared.d_zeta
        )
        diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
        diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
        diagonal_dot = (
            nu_dots[:, None, None] * diagonal_nu[None, :, :]
            + epsi_dots[:, None, None] * diagonal_epsi[None, :, :]
        )
        f1_value = jax.lax.dynamic_index_in_dim(f1_full, k, axis=0, keepdims=False)
        f3_value = jax.lax.dynamic_index_in_dim(f3_full, k, axis=0, keepdims=False)
        return (
            -jnp.einsum("dij,j->id", diagonal_dot, f1_value),
            -jnp.einsum("dij,j->id", diagonal_dot, f3_value),
            diagonal_dot,
        )

    source1_dot, source3_dot, diagonal_dot = jax.vmap(
        _directional_sources_for_mode
    )(mode_indices)
    f1_dot_full = _solve_factorized_modes(
        saved_lu, saved_piv, saved_lower, saved_upper, source1_dot
    )
    f3_dot_full = _solve_factorized_modes(
        saved_lu, saved_piv, saved_lower, saved_upper, source3_dot
    )
    directional_lambda1_dot, directional_lambda3_dot = (
        _solve_factorized_multi_rhs_directional_adjoint_field_pair(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            directional_lambda1,
            directional_lambda3,
            directional_g1_dot,
            directional_g3_dot,
            diagonal_dot,
        )
    )
    return (
        primal_outputs,
        base_coefficient_bars,
        first_coefficient_bars,
        second_coefficient_bars,
        auxiliary,
        f1_full,
        f3_full,
        f1_dot_full,
        f3_dot_full,
        base_lambda1,
        base_lambda3,
        directional_lambda1,
        directional_lambda3,
        directional_lambda1_dot,
        directional_lambda3_dot,
        nu_dots,
        epsi_dots,
        base_nu_direct,
        directional_nu_direct,
        directional_nu_direct_dot,
    )


def solve_prepared_coefficient_vector_two_directional_factorized(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Return a prepared solve and two exact case directions from one factorization.

    This is an isolated forward helper.  It evaluates the base coefficient
    vector and the exact linear responses to ``first_case_dot`` and
    ``second_case_dot`` by reusing the one prepared-system factorization.
    It does not install a custom derivative rule and does not alter the
    existing solve or pullback entry points.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        None,
        include_geometry=False,
        return_coefficient_aux=False,
    )


def _case_cotangent_from_resolved_epsi_bar(
    case: MonoenergeticCase,
    *,
    nu_hat_bar: Array,
    epsi_hat_bar: Array,
    transport_scale: Array | None,
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
    transport_scale: Array | None,
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
                - epsi_hat_bar
                * jnp.asarray(case.er_hat)
                / jnp.asarray(transport_scale) ** 2
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


def solve_prepared_coefficient_vector_two_directional_prepared_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Prepared-pytree wrapper around a local exact prepared-support custom VJP.

    This entry point is intentionally separate from the existing raw solve and
    grouped pullback APIs.  Its forward rule saves one local prepared-system
    factorization and the two directional low modes; its backward rule reuses
    those residuals instead of constructing a second primal NTX solve.
    """

    prepared_tree = jax.tree_util.tree_structure(prepared)
    prepared_leaves = tuple(jax.tree_util.tree_leaves(prepared))
    return _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp(
        prepared_tree,
        prepared_leaves,
        case,
        first_case_dot,
        second_case_dot,
    )


def _solve_prepared_coefficient_vector_two_directional_prepared_primal(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
):
    return solve_prepared_coefficient_vector_two_directional_factorized(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
    )


def _solve_prepared_coefficient_vector_two_directional_prepared_vjp_fwd(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
):
    outputs, primal_residuals = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        None,
        include_geometry=False,
        return_coefficient_aux=False,
        return_primal_residuals=True,
    )
    return outputs, (prepared, case, first_case_dot, second_case_dot, primal_residuals)


def _solve_prepared_coefficient_vector_two_directional_prepared_vjp_bwd(
    residuals,
    output_bars: tuple[Array, Array, Array],
):
    prepared, case, first_case_dot, second_case_dot, primal_residuals = residuals
    base_coefficient_bar, first_coefficient_bar, second_coefficient_bar = output_bars

    def _coefficient_bars_and_aux(_coefficients, _first_coefficients, _second_coefficients):
        return (
            base_coefficient_bar,
            first_coefficient_bar,
            second_coefficient_bar,
            jnp.asarray(0.0, dtype=base_coefficient_bar.dtype),
        )

    result = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        _coefficient_bars_and_aux,
        include_geometry=False,
        include_prepared=True,
        return_coefficient_aux=True,
        primal_residuals=primal_residuals,
    )
    (
        base_nu_hat_bar,
        base_epsi_hat_bar,
        base_prepared_bar,
        first_base_nu_hat_bar,
        first_base_epsi_hat_bar,
        first_nu_hat_bar,
        first_epsi_hat_bar,
        _first_base_prepared_bar,
        first_prepared_bar,
        second_base_nu_hat_bar,
        second_base_epsi_hat_bar,
        second_nu_hat_bar,
        second_epsi_hat_bar,
        _second_base_prepared_bar,
        second_prepared_bar,
        _auxiliary_bar,
    ) = result
    transport_scale = primal_residuals[0]
    prepared_bar = _sum_prepared_cotangent_trees(
        base_prepared_bar,
        first_prepared_bar,
        second_prepared_bar,
    )
    # The grouped core has already applied this chain rule to the base output
    # bar.  Apply only the two directional-output contributions here.
    prepared_bar = _add_er_coordinate_scale_bar(
        prepared_bar,
        case,
        first_epsi_hat_bar + second_epsi_hat_bar,
        transport_scale,
    )
    prepared_bar = _add_er_coordinate_scale_bar(
        prepared_bar,
        first_case_dot,
        first_base_epsi_hat_bar,
        transport_scale,
    )
    prepared_bar = _add_er_coordinate_scale_bar(
        prepared_bar,
        second_case_dot,
        second_base_epsi_hat_bar,
        transport_scale,
    )
    case_bar = _case_cotangent_from_resolved_epsi_bar(
        case,
        nu_hat_bar=(
            base_nu_hat_bar + first_nu_hat_bar + second_nu_hat_bar
        ),
        epsi_hat_bar=(
            base_epsi_hat_bar + first_epsi_hat_bar + second_epsi_hat_bar
        ),
        transport_scale=transport_scale,
    )
    first_case_dot_bar = _case_cotangent_from_resolved_epsi_bar(
        first_case_dot,
        nu_hat_bar=first_base_nu_hat_bar,
        epsi_hat_bar=first_base_epsi_hat_bar,
        transport_scale=transport_scale,
    )
    second_case_dot_bar = _case_cotangent_from_resolved_epsi_bar(
        second_case_dot,
        nu_hat_bar=second_base_nu_hat_bar,
        epsi_hat_bar=second_base_epsi_hat_bar,
        transport_scale=transport_scale,
    )
    return prepared_bar, case_bar, first_case_dot_bar, second_case_dot_bar


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp(
    prepared_tree,
    prepared_leaves,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    prepared = jax.tree_util.tree_unflatten(prepared_tree, prepared_leaves)
    return _solve_prepared_coefficient_vector_two_directional_prepared_primal(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
    )


def _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp_fwd(
    prepared_tree,
    prepared_leaves,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
):
    prepared = jax.tree_util.tree_unflatten(prepared_tree, prepared_leaves)
    outputs, residuals = _solve_prepared_coefficient_vector_two_directional_prepared_vjp_fwd(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
    )
    return outputs, residuals


def _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp_bwd(
    prepared_tree,
    residuals,
    output_bars: tuple[Array, Array, Array],
):
    prepared_bar, case_bar, first_case_dot_bar, second_case_dot_bar = (
        _solve_prepared_coefficient_vector_two_directional_prepared_vjp_bwd(
            residuals,
            output_bars,
        )
    )
    return (
        tuple(jax.tree_util.tree_leaves(prepared_bar)),
        case_bar,
        first_case_dot_bar,
        second_case_dot_bar,
    )


_solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp.defvjp(
    _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp_fwd,
    _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp_bwd,
)


def solve_prepared_coefficient_vector_lowdot_two_pullbacks(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_fn,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Case-only public view of the fused low-order pullback kernel."""

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_fn,
        include_geometry=False,
        return_coefficient_aux=False,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_fn,
):
    """Fused exact pullback returning case and prepared-geometry cotangents.

    This opt-in API preserves the original 10 case cotangents and additionally
    returns one geometry cotangent for the base response plus base/directional
    geometry cotangents for each supplied case direction.  The primal
    factorization is shared across all three contractions.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_fn,
        include_geometry=True,
        return_coefficient_aux=False,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Geometry-returning fused pullback with caller-defined direct auxiliaries.

    ``coefficient_bar_and_aux_fn(coefficients, first_coeff_dot,
    second_coeff_dot)`` returns ``(base_bar, first_bar, second_bar,
    auxiliary)``. ``auxiliary`` is not interpreted by NTX; it is evaluated
    from the already available coefficient vector and its two exact tangent
    vectors, then returned with the implicit bars.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=True,
        return_coefficient_aux=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Return only geometry support bars of the fused low-dot pullback.

    This is the geometry-only counterpart of the prepared-support-only helper.
    It retains the exact base and two directional implicit adjoints, but omits
    unused case/profile contractions and does not construct complete prepared
    cotangent trees.  The result is ``(base_geometry,
    first_base_geometry, first_directional_geometry,
    second_base_geometry, second_directional_geometry, auxiliary)``.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=True,
        return_coefficient_aux=True,
        support_only=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Fused exact pullback returning a complete prepared-system cotangent.

    This is the support-payload-safe variant of the geometry-only helper. Its
    prepared bars include ``surface``, ``geometry``, and derivative-operator
    leaves while retaining the same one-factorization grouped adjoint.
    """
    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=False,
        include_prepared=True,
        return_coefficient_aux=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Return only the prepared-support bars of the fused low-dot pullback.

    This is an isolated API for callers differentiating a response with
    respect to a prepared NTX payload and an external direct auxiliary (for
    example ``drds``).  It intentionally discards the case/profile bars:
    callers in that lane hold the local case fixed.  The underlying grouped
    implicit algebra and primal factorization are exactly the established
    prepared-support implementation above, while the directional case/profile
    contractions are omitted rather than merely discarded.

    The result is ``(base_prepared, first_base_prepared,
    first_directional_prepared, second_base_prepared,
    second_directional_prepared, auxiliary)``.  This keeps the smaller
    support-only algebra isolated from every existing public helper.
    """

    result = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=False,
        include_prepared=True,
        return_coefficient_aux=True,
        support_only=True,
    )
    return result


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
):
    """Exact batched support-only low-dot pullback with one local primal.

    ``coefficient_bars_and_aux_fn`` receives the unbatched base coefficient
    vector and its two case-direction coefficient vectors.  It must return
    ``(base_bars, first_bars, second_bars, auxiliary)`` with a common leading
    RHS axis.  It may additionally return
    ``(first_bars_dot, second_bars_dot)`` as a fifth item when the directional
    coefficient cotangents themselves depend on the local NTX primal. The local
    NTX primal modes, block factorisation, and two forward
    directional mode solves are constructed *once*, then each RHS performs
    only its cotangent-dependent support adjoints.  The factorisation-local
    residual is scoped to this call; it is neither returned nor retained by a
    transport timestep/segment record.

    When ``return_primal_outputs`` is true, the unbatched
    ``(coefficients, first_coefficient_dot, second_coefficient_dot)`` tuple is
    returned beside the RHS-batched pullback result.  This lets an immediate
    caller construct its local primal response for interpolation without
    evaluating NTX again.  It does not retain the factorisation after return.

    This is deliberately separate from the scalar API.  It is the native
    multi-RHS building block for a future NEOPAX adapter and changes no
    existing solver or reverse-mode dispatch.
    """

    primal_outputs, primal_residuals = (
        _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
            prepared,
            case,
            first_case_dot,
            second_case_dot,
            None,
            include_geometry=False,
            include_prepared=True,
            return_coefficient_aux=True,
            support_only=True,
            return_primal_residuals=True,
        )
    )
    coefficient_bars_and_aux = coefficient_bars_and_aux_fn(*primal_outputs)
    if len(coefficient_bars_and_aux) not in {4, 5}:
        raise ValueError(
            "coefficient_bars_and_aux_fn must return "
            "(base_bars, first_bars, second_bars, auxiliary) optionally followed "
            "by (first_bars_dot, second_bars_dot)."
        )
    base_bars, first_bars, second_bars, auxiliary = coefficient_bars_and_aux[:4]
    if len(coefficient_bars_and_aux) == 5:
        first_bars_dot, second_bars_dot = coefficient_bars_and_aux[4]
    else:
        first_bars_dot = jnp.zeros_like(first_bars)
        second_bars_dot = jnp.zeros_like(second_bars)

    prepared_bar_tree = jax.tree_util.tree_structure(prepared)

    def _one_rhs(
        base_bar,
        first_bar,
        second_bar,
        auxiliary,
        first_bar_dot,
        second_bar_dot,
    ):
        result = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
            prepared,
            case,
            first_case_dot,
            second_case_dot,
            lambda _base, _first, _second: (
                base_bar,
                first_bar,
                second_bar,
                auxiliary,
                (first_bar_dot, second_bar_dot),
            ),
            include_geometry=False,
            include_prepared=True,
            return_coefficient_aux=True,
            support_only=True,
            primal_residuals=primal_residuals,
        )
        # Keep static prepared-system metadata outside ``vmap``.  Returning a
        # PreparedMonoenergeticSystem directly would ask JAX to batch its
        # validated/static fields; batch only the dynamic cotangent leaves and
        # reconstruct the five original pytrees afterwards.
        return (
            tuple(tuple(jax.tree_util.tree_leaves(value)) for value in result[:5]),
            result[5],
        )

    prepared_bar_leaf_groups, auxiliary = jax.vmap(_one_rhs)(
        base_bars,
        first_bars,
        second_bars,
        auxiliary,
        first_bars_dot,
        second_bars_dot,
    )
    result = (
        *(
            prepared_bar_tree.unflatten(leaves)
            for leaves in prepared_bar_leaf_groups
        ),
        auxiliary,
    )
    return (primal_outputs, result) if return_primal_outputs else result


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
    return_case_bars: bool = False,
):
    """Exact support-only low-dot pullback with native objective RHS solves.

    Unlike the older ``*_multi_rhs_and_aux`` experiment, this function does
    not ``vmap`` the scalar implicit pullback.  It builds all of the base and
    two directional adjoint fields with matrix-RHS factorized solves, then
    batches only the final prepared-gradient contractions.  Consequently the
    primal modes, physical forward directions, and block-tridiagonal adjoint
    factorization are each local to one call and are never recreated per
    objective RHS.

    It is intentionally opt-in and currently supports the explicit
    ``epsi_hat`` coordinate used by NEOPAX's local interpolated response. If
    ``return_case_bars`` is selected, it also returns the native matrix-RHS
    case contractions needed by NEOPAX's local support-coordinate chain.
    Those contractions reuse the adjoint fields formed here and do not create
    another NTX solve. The existing scalar APIs retain their broader public
    coordinate contracts.
    """

    if (
        case.epsi_hat is None
        or first_case_dot.epsi_hat is None
        or second_case_dot.epsi_hat is None
    ):
        raise ValueError(
            "The native multi-RHS prepared-support helper requires explicit epsi_hat cases."
        )
    (
        primal_outputs,
        base_coefficient_bars,
        first_coefficient_bars,
        second_coefficient_bars,
        auxiliary,
        f1_full,
        f3_full,
        f1_dot_full,
        f3_dot_full,
        base_lambda1,
        base_lambda3,
        directional_lambda1,
        directional_lambda3,
        directional_lambda1_dot,
        directional_lambda3_dot,
        nu_dots,
        epsi_dots,
        base_nu_direct,
        directional_nu_direct,
        directional_nu_direct_dot,
    ) = _lowdot_two_pullback_native_multi_rhs_adjoint_fields(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bars_and_aux_fn,
    )
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        case.epsi_hat,
    )
    prepared_tree = jax.tree_util.tree_structure(prepared)

    def _one_rhs(
        base_coefficient_bar,
        first_coefficient_bar,
        second_coefficient_bar,
        first_coefficient_bar_dot,
        second_coefficient_bar_dot,
        base_lambda1_value,
        base_lambda3_value,
        directional_lambda1_value,
        directional_lambda3_value,
        directional_lambda1_dot_value,
        directional_lambda3_dot_value,
    ):
        base_prepared = _prepared_gradient_from_adjoint(
            prepared,
            ctx,
            f1_full,
            f3_full,
            base_lambda1_value,
            base_lambda3_value,
            base_coefficient_bar,
        )
        first_base_prepared, first_directional_prepared = (
            _directional_prepared_gradient_from_adjoint(
                prepared,
                nu_hat=ctx.nu_hat,
                epsi_hat=ctx.epsi_hat,
                nu_hat_dot=nu_dots[0],
                epsi_hat_dot=epsi_dots[0],
                f1_full=f1_full,
                f3_full=f3_full,
                f1_dot=f1_dot_full[..., 0],
                f3_dot=f3_dot_full[..., 0],
                lambda1=directional_lambda1_value[..., 0],
                lambda3=directional_lambda3_value[..., 0],
                lambda1_dot=directional_lambda1_dot_value[..., 0],
                lambda3_dot=directional_lambda3_dot_value[..., 0],
                coefficient_bar=first_coefficient_bar,
                coefficient_bar_dot=first_coefficient_bar_dot,
            )
        )
        second_base_prepared, second_directional_prepared = (
            _directional_prepared_gradient_from_adjoint(
                prepared,
                nu_hat=ctx.nu_hat,
                epsi_hat=ctx.epsi_hat,
                nu_hat_dot=nu_dots[1],
                epsi_hat_dot=epsi_dots[1],
                f1_full=f1_full,
                f3_full=f3_full,
                f1_dot=f1_dot_full[..., 1],
                f3_dot=f3_dot_full[..., 1],
                lambda1=directional_lambda1_value[..., 1],
                lambda3=directional_lambda3_value[..., 1],
                lambda1_dot=directional_lambda1_dot_value[..., 1],
                lambda3_dot=directional_lambda3_dot_value[..., 1],
                coefficient_bar=second_coefficient_bar,
                coefficient_bar_dot=second_coefficient_bar_dot,
            )
        )

        return tuple(
            tuple(jax.tree_util.tree_leaves(value))
            for value in (
                base_prepared,
                first_base_prepared,
                first_directional_prepared,
                second_base_prepared,
                second_directional_prepared,
            )
        )

    prepared_bar_leaf_groups = jax.vmap(_one_rhs)(
        base_coefficient_bars,
        first_coefficient_bars,
        second_coefficient_bars,
        first_coefficient_bars_dot,
        second_coefficient_bars_dot,
        jnp.moveaxis(base_lambda1, 2, 0),
        jnp.moveaxis(base_lambda3, 2, 0),
        jnp.moveaxis(directional_lambda1, 2, 0),
        jnp.moveaxis(directional_lambda3, 2, 0),
        jnp.moveaxis(directional_lambda1_dot, 2, 0),
        jnp.moveaxis(directional_lambda3_dot, 2, 0),
    )
    result = (
        *(prepared_tree.unflatten(leaves) for leaves in prepared_bar_leaf_groups),
        auxiliary,
    )
    if return_case_bars:
        def _parameter_gradient_from_values(
            nu_hat_value,
            epsi_hat_value,
            f1_value,
            f3_value,
            lambda1_value,
            lambda3_value,
        ):
            return _parameter_gradient_from_adjoint_multi_rhs(
                prepared,
                _operator_context(
                    prepared.surface,
                    prepared.geometry,
                    prepared.grid,
                    nu_hat_value,
                    epsi_hat_value,
                ),
                f1_value,
                f3_value,
                lambda1_value,
                lambda3_value,
            )

        base_nu_implicit, base_epsi_bar = _parameter_gradient_from_values(
            ctx.nu_hat, ctx.epsi_hat, f1_full, f3_full, base_lambda1, base_lambda3
        )
        (
            (first_base_nu_implicit, first_base_epsi_bar),
            (first_nu_implicit_dot, first_epsi_bar_dot),
        ) = jax.jvp(
            _parameter_gradient_from_values,
            (
                ctx.nu_hat, ctx.epsi_hat, f1_full, f3_full,
                directional_lambda1[..., 0], directional_lambda3[..., 0],
            ),
            (
                nu_dots[0], epsi_dots[0], f1_dot_full[..., 0], f3_dot_full[..., 0],
                directional_lambda1_dot[..., 0], directional_lambda3_dot[..., 0],
            ),
        )
        (
            (second_base_nu_implicit, second_base_epsi_bar),
            (second_nu_implicit_dot, second_epsi_bar_dot),
        ) = jax.jvp(
            _parameter_gradient_from_values,
            (
                ctx.nu_hat, ctx.epsi_hat, f1_full, f3_full,
                directional_lambda1[..., 1], directional_lambda3[..., 1],
            ),
            (
                nu_dots[1], epsi_dots[1], f1_dot_full[..., 1], f3_dot_full[..., 1],
                directional_lambda1_dot[..., 1], directional_lambda3_dot[..., 1],
            ),
        )
        case_bar_components = (
            base_nu_direct + base_nu_implicit,
            base_epsi_bar,
            directional_nu_direct[..., 0] + first_base_nu_implicit,
            first_base_epsi_bar,
            directional_nu_direct_dot[..., 0] + first_nu_implicit_dot,
            first_epsi_bar_dot,
            directional_nu_direct[..., 1] + second_base_nu_implicit,
            second_base_epsi_bar,
            directional_nu_direct_dot[..., 1] + second_nu_implicit_dot,
            second_epsi_bar_dot,
        )
        return (
            (primal_outputs, result, case_bar_components)
            if return_primal_outputs
            else (result, case_bar_components)
        )
    return (primal_outputs, result) if return_primal_outputs else result


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux_packed_support_adjoint(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Prepared-support low-dot pullback with paired directional adjoint solves.

    Experimental and isolated: only the two ``lambda_dot`` field solves in
    the prepared-support branch are packed. All scalar APIs retain their
    established implementation.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=False,
        include_prepared=True,
        return_coefficient_aux=True,
        packed_support_directional_adjoint=True,
    )


solve_prepared_coefficient_vector_vjp.defvjp(
    _solve_prepared_coefficient_vector_vjp_fwd,
    _solve_prepared_coefficient_vector_vjp_bwd,
)


def _solve_prepared_coefficient_vector_raw(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
    *,
    adjoint_window: int | None = None,
) -> Array:
    values = _solve_prepared_arrays_from_values(
        prepared, nu_hat, epsi_hat, adjoint_window=adjoint_window
    )
    return jnp.stack(values[:5])


def _solve_prepared_arrays(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    *,
    adjoint_window: int | None = None,
) -> tuple[Array, ...]:
    return _solve_prepared_arrays_from_values(
        prepared,
        case.nu_hat,
        case.resolved_epsi_hat(prepared.geometry.transport_psi_scale),
        adjoint_window=adjoint_window,
    )


def _solve_prepared_arrays_from_values(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
    *,
    adjoint_window: int | None = None,
) -> tuple[Array, ...]:
    geom = prepared.geometry
    grid = prepared.grid
    ctx = _operator_context(prepared.surface, geom, grid, nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    f1_modes, f3_modes, residual = _solve_modes_with_tail_residual(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        s1,
        s3,
        adjoint_window,
    )
    d11, d31, d13, d33, d33_spitzer = coefficients_from_modes(
        geom, f1_modes, f3_modes, ctx.nu_hat
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
    "audit_prepared_residuals",
    "compile_prepared_solver",
    "solve_prepared",
    "solve_prepared_coefficient_vector",
    "solve_prepared_coefficient_vector_two_directional_factorized",
    "solve_prepared_coefficient_vector_two_directional_prepared_vjp",
    "pullback_prepared_coefficient_vector_case_and_prepared",
    "solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux",
    "solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux",
    "solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux",
    "solve_prepared_coefficient_vector_vjp",
    "solve_prepared_internal",
]
