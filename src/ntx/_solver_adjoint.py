"""Prepared-solver adjoint and custom-VJP helper algebra."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ._solver_context import _operator_context
from ._solver_factorization import (
    _factorize_prepared_modes,
    _solve_factorized_modes,
)
from ._solver_types import PreparedMonoenergeticSystem
from .operators import (
    OperatorContext,
    apply_nullspace_condition,
    block_parameters,
    operator_blocks,
    operator_blocks_from_parameters,
    parameter_derivative_blocks,
    source_modes,
)
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


def _parameter_gradient_from_adjoint_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
) -> tuple[Array, Array]:
    """Case bars for a trailing batch of already-solved adjoint fields.

    This is the matrix-RHS counterpart of
    :func:`_parameter_gradient_from_adjoint`.  It only contracts existing
    primal and adjoint fields; it neither factorizes nor solves.  Keeping the
    RHS column explicit avoids ``vmap`` tracing one complete parameter-JVP
    graph per objective in the experimental native support path.
    """
    if lambda1.ndim != 3 or lambda3.shape != lambda1.shape:
        raise ValueError(
            "lambda1 and lambda3 must have shape (mode, unknown, n_rhs)."
        )

    def zero_first_row(block: Array) -> Array:
        return block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))

    n_rhs = lambda1.shape[-1]
    nu_bar = jnp.zeros((n_rhs,), dtype=prepared.grid.jax_dtype)
    epsi_bar = jnp.zeros((n_rhs,), dtype=prepared.grid.jax_dtype)
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
        f1_nu = diagonal_nu @ f1_full[k]
        f3_nu = diagonal_nu @ f3_full[k]
        f1_epsi = diagonal_epsi @ f1_full[k]
        f3_epsi = diagonal_epsi @ f3_full[k]
        nu_bar = nu_bar - (
            jnp.sum(lambda1[k] * f1_nu[:, None], axis=0)
            + jnp.sum(lambda3[k] * f3_nu[:, None], axis=0)
        )
        epsi_bar = epsi_bar - (
            jnp.sum(lambda1[k] * f1_epsi[:, None], axis=0)
            + jnp.sum(lambda3[k] * f3_epsi[:, None], axis=0)
        )
    return nu_bar, epsi_bar


def _geometry_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
):
    """Return the exact geometry cotangent from prepared primal/adjoint modes.

    This is the geometry counterpart of :func:`_parameter_gradient_from_adjoint`.
    It deliberately differentiates the *fixed* block residual, not the factorized
    solve.  Consequently it reuses ``f1_full``, ``f3_full``, ``lambda1`` and
    ``lambda3`` already computed by the implicit case adjoint:

    ``dL/dg = dL_direct/dg - lambda.T d(A(g) f - s(g))/dg``.

    The result is a cotangent for ``prepared.geometry``.  The caller is
    responsible for chaining it to its higher-level support/geometry payload.
    """

    n_xi = int(prepared.grid.n_xi)

    def _conditioned_blocks(local_ctx: OperatorContext, mode_index: int):
        lower, diagonal, upper = operator_blocks(
            local_ctx,
            mode_index,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if mode_index == 0:
            diagonal, upper = apply_nullspace_condition(diagonal, upper)
            assert upper is not None
        return lower, diagonal, upper

    def _residual_and_direct_coefficients(geometry):
        local_ctx = OperatorContext(
            surface=prepared.surface,
            geometry=geometry,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
        )
        source1, source3 = source_modes(local_ctx, n_xi)

        def _residual(modes, source):
            rows = []
            for mode_index in range(n_xi + 1):
                lower, diagonal, upper = _conditioned_blocks(local_ctx, mode_index)
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
                local_ctx.nu_hat,
            )
        )
        return (
            direct_coefficients,
            _residual(f1_full, source1),
            _residual(f3_full, source3),
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
    """Exact fixed-primal cotangent for every differentiable prepared leaf.

    This is the full-prepared counterpart of
    :func:`_geometry_gradient_from_adjoint`.  Crucially, its VJP is only over
    the fixed residual and direct coefficient contraction: it does *not*
    differentiate a factorization or execute a second primal/adjoint solve.
    Thus the already available ``f`` and ``lambda`` modes are shared for the
    surface, geometry, derivative-operator, and grid-dependent support leaves.
    """
    n_xi = int(prepared.grid.n_xi)

    def _residual_and_direct_coefficients(prepared_value):
        local_ctx = _operator_context(
            prepared_value.surface,
            prepared_value.geometry,
            prepared_value.grid,
            ctx.nu_hat,
            ctx.epsi_hat,
        )
        source1, source3 = source_modes(local_ctx, n_xi)

        def _residual(modes, source):
            rows = []
            for mode_index in range(n_xi + 1):
                lower, diagonal, upper = operator_blocks(
                    local_ctx,
                    mode_index,
                    prepared_value.d_theta,
                    prepared_value.d_zeta,
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
                prepared_value.geometry,
                f1_full[:3],
                f3_full[:3],
                local_ctx.nu_hat,
            )
        )
        return (
            direct_coefficients,
            _residual(f1_full, source1),
            _residual(f3_full, source3),
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
    """Exact fixed-primal cotangent for compact prepared residual inputs.

    This has the same mathematical residual/direct-coefficient contraction as
    :func:`_prepared_gradient_from_adjoint`, but deliberately stops at the
    compact input tuple.  It neither factorizes nor solves.  The caller uses
    :func:`_compact_prepared_bar_to_prepared` to apply the cheap remaining
    prepared-input chain once to the final combined cotangent.
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
    """Chain a compact residual-input cotangent back to ``prepared``.

    This transpose intentionally contains no dense operator construction.  In
    particular, shared geometry leaves receive the sum of their direct
    geometry and block-parameter contributions here, exactly as in the old
    all-in-one prepared VJP.
    """
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


def _case_and_geometry_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
    nu_hat_direct_bar: Array,
):
    """Return exact case and prepared-geometry bars from one implicit adjoint.

    ``nu_hat_direct_bar`` is the direct coefficient contribution returned by
    :func:`_coefficient_mode_pullback`; the factorized adjoint solutions are
    shared by the case and geometry terms.
    """

    nu_hat_implicit_bar, epsi_hat_bar = _parameter_gradient_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
    )
    geometry_bar = _geometry_gradient_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
        coefficient_bar,
    )
    return (
        nu_hat_direct_bar + nu_hat_implicit_bar,
        epsi_hat_bar,
        geometry_bar,
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


def _combined_prepared_gradient_from_adjoint_multi_rhs_oracle(
    prepared: PreparedMonoenergeticSystem,
    *,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    first_f1_dot: Array,
    first_f3_dot: Array,
    second_f1_dot: Array,
    second_f3_dot: Array,
    nu_dots: Array,
    epsi_dots: Array,
    base_lambda1: Array,
    base_lambda3: Array,
    directional_lambda1: Array,
    directional_lambda3: Array,
    directional_lambda1_dot: Array,
    directional_lambda3_dot: Array,
    base_coefficient_bars: Array,
    first_coefficient_bars: Array,
    second_coefficient_bars: Array,
    first_coefficient_bars_dot: Array,
    second_coefficient_bars_dot: Array,
) -> tuple[Array, ...]:
    """Oracle contract for the native combined prepared-support transpose.

    The grouped native low-dot path already supplies one factorization and its
    base/directional adjoint fields with a trailing RHS axis.  This helper
    defines the *final combined prepared-bar* contract that a future explicit
    RHS-axis transpose must reproduce.  It deliberately uses the established
    scalar gradient routines under ``vmap`` for now; it is a numerical oracle,
    not the production optimisation.  Keeping it separate prevents any change
    to current public or reverse-mode behaviour while the native contraction
    is derived and tested.

    The return is a tuple of dynamic prepared leaves, each shaped
    ``(n_rhs, *prepared_leaf.shape)``.  Static prepared metadata is never
    reconstructed or batched here.
    """

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
        _, first_directional_prepared = _directional_prepared_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_dots[0],
            epsi_hat_dot=epsi_dots[0],
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=first_f1_dot,
            f3_dot=first_f3_dot,
            lambda1=directional_lambda1_value[..., 0],
            lambda3=directional_lambda3_value[..., 0],
            lambda1_dot=directional_lambda1_dot_value[..., 0],
            lambda3_dot=directional_lambda3_dot_value[..., 0],
            coefficient_bar=first_coefficient_bar,
            coefficient_bar_dot=first_coefficient_bar_dot,
        )
        _, second_directional_prepared = _directional_prepared_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_dots[1],
            epsi_hat_dot=epsi_dots[1],
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=second_f1_dot,
            f3_dot=second_f3_dot,
            lambda1=directional_lambda1_value[..., 1],
            lambda3=directional_lambda3_value[..., 1],
            lambda1_dot=directional_lambda1_dot_value[..., 1],
            lambda3_dot=directional_lambda3_dot_value[..., 1],
            coefficient_bar=second_coefficient_bar,
            coefficient_bar_dot=second_coefficient_bar_dot,
        )

        combined_leaves = []
        for primal_leaf, base_leaf, first_leaf, second_leaf in zip(
            jax.tree_util.tree_leaves(prepared),
            jax.tree_util.tree_leaves(base_prepared),
            jax.tree_util.tree_leaves(first_directional_prepared),
            jax.tree_util.tree_leaves(second_directional_prepared),
            strict=True,
        ):
            primal_value = jnp.asarray(primal_leaf)
            if not jnp.issubdtype(primal_value.dtype, jnp.inexact):
                combined_leaves.append(
                    jnp.zeros(primal_value.shape, dtype=jnp.float64)
                )
            elif (
                jnp.asarray(base_leaf).dtype == jax.dtypes.float0
                or jnp.asarray(first_leaf).dtype == jax.dtypes.float0
                or jnp.asarray(second_leaf).dtype == jax.dtypes.float0
            ):
                combined_leaves.append(jnp.zeros_like(primal_value))
            else:
                combined_leaves.append(base_leaf + first_leaf + second_leaf)
        return tuple(combined_leaves)

    return jax.vmap(_one_rhs)(
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
