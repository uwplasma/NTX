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
    operator_blocks,
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
            coefficient_bar,
        )

    return jax.jvp(
        _geometry_bar_from_dynamic_terms,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3),
        (nu_hat_dot, epsi_hat_dot, f1_dot, f3_dot, lambda1_dot, lambda3_dot),
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
):
    """Directional counterpart of :func:`_prepared_gradient_from_adjoint`."""
    def _prepared_bar_from_dynamic_terms(
        nu_hat_value, epsi_hat_value, f1_value, f3_value, lambda1_value, lambda3_value
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
            coefficient_bar,
        )

    return jax.jvp(
        _prepared_bar_from_dynamic_terms,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3),
        (nu_hat_dot, epsi_hat_dot, f1_dot, f3_dot, lambda1_dot, lambda3_dot),
    )
