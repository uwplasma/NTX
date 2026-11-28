"""Grouped primal and two-directional pullbacks sharing one local factorization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._solver import (
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    _coefficient_mode_pullback,
    _operator_context,
    _parameter_gradient_from_adjoint,
    _prepared_implicit_vjp_primal,
    _solve_factorized_modes,
)
from ..operators import parameter_derivative_blocks
from ..transport import coefficients_from_modes
from .adjoints import (
    _add_er_coordinate_scale_bar,
    _directional_geometry_gradient_from_adjoint,
    _directional_prepared_gradient_from_adjoint,
    _geometry_gradient_from_adjoint,
    _prepared_gradient_from_adjoint,
    _solve_factorized_directional_adjoint_field_pair,
)


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
            return jnp.stack(coefficients_from_modes(prepared.geometry, modes1, modes3, nu_value))

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
            (primal_outputs, local_primal_residuals) if return_primal_residuals else primal_outputs
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
    base_coefficient_bar, first_coefficient_bar, second_coefficient_bar = coefficient_bar_result[:3]
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
                nu_bar = (
                    nu_bar - lambda1_k @ (diagonal_nu @ f1_k) - lambda3_k @ (diagonal_nu @ f3_k)
                )
                epsi_bar = (
                    epsi_bar
                    - lambda1_k @ (diagonal_epsi @ f1_k)
                    - lambda3_k @ (diagonal_epsi @ f3_k)
                )
                return (nu_bar, epsi_bar), None

            (
                (
                    nu_bar_implicit,
                    epsi_bar,
                ),
                _,
            ) = jax.lax.scan(
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

            def source1_dot_for_mode(k):
                return _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot)[0]

            def source3_dot_for_mode(k):
                return _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot)[1]

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
        base = (
            base[0],
            base[1],
            _add_er_coordinate_scale_bar(base[2], case, base[1], transport_scale),
        )
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

        first: tuple = (
            direction_nu_bar[0],
            direction_epsi_bar[0],
            direction_nu_bar_dot[0],
            direction_epsi_bar_dot[0],
        )
        second: tuple = (
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
