"""Matrix-RHS adjoints and compact VMEC support pullbacks for several objectives."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from .._geometry import VmecSurface
from .._solver import (
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    _coefficient_mode_pullback,
    _operator_context,
    _solve_factorized_modes,
)
from ..operators import parameter_derivative_blocks
from .adjoints import (
    _compact_prepared_bar_to_prepared,
    _compact_prepared_gradient_from_adjoint,
    _compact_prepared_residual_inputs,
    _directional_compact_prepared_gradient_from_adjoint,
    _directional_prepared_gradient_from_adjoint,
    _prepared_gradient_from_adjoint,
    _solve_factorized_adjoint_field_pair,
    _solve_factorized_multi_rhs_directional_adjoint_field_pair,
    _unpack_coefficient_bars,
)
from .geometry import (
    _coefficient_mode_pullback_multi_rhs_direct,
    _directional_coefficient_mode_pullback_multi_rhs_direct,
    _directional_native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs_direct,
    _native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs,
    _parameter_gradient_from_adjoint_multi_rhs,
    vmec_geometry_bars_to_coefficients_multi_rhs,
)
from .responses import (
    _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core,
)


def _lowdot_two_pullback_native_multi_rhs_adjoint_fields(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    direct_coefficient_pullback: bool = False,
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

    primal_outputs, primal_residuals = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        None,
        include_geometry=False,
        return_coefficient_aux=True,
        return_primal_residuals=True,
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

    base, first, second, auxiliary, first_dot, second_dot = _unpack_coefficient_bars(
        coefficient_bars_and_aux_fn(*primal_outputs)
    )
    (
        base_coefficient_bars,
        first_coefficient_bars,
        second_coefficient_bars,
        first_coefficient_bars_dot,
        second_coefficient_bars_dot,
    ) = map(jnp.asarray, (base, first, second, first_dot, second_dot))
    if (
        base_coefficient_bars.ndim != 2
        or first_coefficient_bars.shape != base_coefficient_bars.shape
        or second_coefficient_bars.shape != base_coefficient_bars.shape
        or first_coefficient_bars_dot.shape != base_coefficient_bars.shape
        or second_coefficient_bars_dot.shape != base_coefficient_bars.shape
    ):
        raise ValueError(
            "All low-dot coefficient cotangent batches must have shape (n_rhs, n_coefficients)."
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
        return (
            jnp.zeros((mode_count, unknown_count, rhs_count), dtype=f1_full.dtype)
            .at[:3]
            .set(jnp.moveaxis(low_fields, 0, -1))
        )

    if direct_coefficient_pullback:
        base_f1_low, base_f3_low, base_nu_direct = _coefficient_mode_pullback_multi_rhs_direct(
            prepared.geometry, ctx.nu_hat, base_coefficient_bars
        )
    else:
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
        if direct_coefficient_pullback:
            # ``_coefficient_mode_pullback`` is independent of the retained
            # primal modes.  The direct helper therefore needs only the two
            # coefficient-bar batches and the explicit Spitzer ``nu``
            # direction; ``f*_dot_low_value`` are intentionally absent.
            return _directional_coefficient_mode_pullback_multi_rhs_direct(
                prepared.geometry,
                ctx.nu_hat,
                nu_dot_value,
                coefficient_bars,
                coefficient_bars_dot,
            )

        def _one_rhs(one_coefficient_bar, one_coefficient_bar_dot):
            def _pullback_from_primal(modes1, modes3, nu_value, coefficient_bar_value):
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
        for first_value, second_value in zip(
            first_directional[0], second_directional[0], strict=True
        )
    )
    directional_dot_low = tuple(
        jnp.stack([first_value, second_value], axis=-1)
        for first_value, second_value in zip(
            first_directional[1], second_directional[1], strict=True
        )
    )
    # (rhs, retained-mode, unknown, direction) -> (mode, unknown, rhs, direction)
    directional_f1_low, directional_f3_low, directional_nu_direct = directional_primal_low
    directional_f1_low_dot, directional_f3_low_dot, directional_nu_direct_dot = directional_dot_low
    directional_f1_low = jnp.transpose(directional_f1_low, (1, 2, 0, 3))
    directional_f3_low = jnp.transpose(directional_f3_low, (1, 2, 0, 3))
    directional_f1_low_dot = jnp.transpose(directional_f1_low_dot, (1, 2, 0, 3))
    directional_f3_low_dot = jnp.transpose(directional_f3_low_dot, (1, 2, 0, 3))
    directional_shape = (mode_count, unknown_count, rhs_count, direction_count)
    directional_g1 = (
        jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(directional_f1_low)
    )
    directional_g3 = (
        jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(directional_f3_low)
    )
    directional_g1_dot = (
        jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(directional_f1_low_dot)
    )
    directional_g3_dot = (
        jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(directional_f3_low_dot)
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

    source1_dot, source3_dot, diagonal_dot = jax.vmap(_directional_sources_for_mode)(mode_indices)
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
        first_coefficient_bars_dot,
        second_coefficient_bars_dot,
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


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
    return_case_bars: bool = False,
    _compact_result: bool = False,
    _compact_residual_transpose: bool = False,
    return_vmec_coefficient_bars: bool = False,
    native_vmec_coefficient_bars_only: bool = False,
    native_vmec_direct_directional_product_rule: bool = False,
    native_direct_coefficient_pullback: bool = False,
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

    if case.epsi_hat is None or first_case_dot.epsi_hat is None or second_case_dot.epsi_hat is None:
        raise ValueError(
            "The native multi-RHS prepared-support helper requires explicit epsi_hat cases."
        )
    (
        primal_outputs,
        base_coefficient_bars,
        first_coefficient_bars,
        second_coefficient_bars,
        first_coefficient_bars_dot,
        second_coefficient_bars_dot,
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
        direct_coefficient_pullback=native_direct_coefficient_pullback,
    )
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        case.epsi_hat,
    )
    prepared_tree = jax.tree_util.tree_structure(prepared)

    if _compact_residual_transpose and not _compact_result:
        raise ValueError(
            "The compact residual transpose has only the compact prepared-bar contract."
        )

    if return_vmec_coefficient_bars and not _compact_result:
        raise ValueError(
            "Native VMEC coefficient bars require the combined compact prepared contract."
        )
    if native_vmec_coefficient_bars_only and not return_vmec_coefficient_bars:
        raise ValueError(
            "native_vmec_coefficient_bars_only requires return_vmec_coefficient_bars=True."
        )
    if native_vmec_coefficient_bars_only and _compact_residual_transpose:
        raise ValueError(
            "native_vmec_coefficient_bars_only does not support the compact residual contract."
        )
    if native_vmec_direct_directional_product_rule and not return_vmec_coefficient_bars:
        raise ValueError(
            "native_vmec_direct_directional_product_rule requires VMEC coefficient bars."
        )

    if _compact_residual_transpose:
        compact_tree = jax.tree_util.tree_structure(
            _compact_prepared_residual_inputs(
                prepared,
                ctx.nu_hat,
                ctx.epsi_hat,
            )
        )

        def _combine_compact_leaf(*leaves):
            if any(jnp.asarray(leaf).dtype == jax.dtypes.float0 for leaf in leaves):
                return leaves[0]
            return sum(leaves)

        def _one_rhs_compact_residual(
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
            base_compact = _compact_prepared_gradient_from_adjoint(
                prepared,
                ctx,
                f1_full,
                f3_full,
                base_lambda1_value,
                base_lambda3_value,
                base_coefficient_bar,
            )
            _, first_directional_compact = _directional_compact_prepared_gradient_from_adjoint(
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
            _, second_directional_compact = _directional_compact_prepared_gradient_from_adjoint(
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
            combined_compact = jax.tree_util.tree_map(
                _combine_compact_leaf,
                base_compact,
                first_directional_compact,
                second_directional_compact,
            )
            return tuple(jax.tree_util.tree_leaves(combined_compact))

        compact_bar_leaf_groups = jax.vmap(_one_rhs_compact_residual)(
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

        def _one_compact_bar_to_prepared(*compact_leaves):
            compact_bar = jax.tree_util.tree_unflatten(compact_tree, compact_leaves)
            prepared_bar = _compact_prepared_bar_to_prepared(
                prepared,
                nu_hat=ctx.nu_hat,
                epsi_hat=ctx.epsi_hat,
                compact_bar=compact_bar,
            )
            return tuple(jax.tree_util.tree_leaves(prepared_bar))

        prepared_bar_leaf_groups = jax.vmap(_one_compact_bar_to_prepared)(*compact_bar_leaf_groups)
        result = (jax.tree_util.tree_unflatten(prepared_tree, prepared_bar_leaf_groups), auxiliary)

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

        if _compact_result:
            # Only these three terms are cotangents of the primal prepared
            # system.  The two ``*_base`` trees belong to the physical case
            # directions and are consumed below for the compact case chain;
            # returning them as full prepared pytrees needlessly multiplies
            # the objective-batched temporary payload.
            def _combine_primal_prepared_leaf(primal_leaf, base_leaf, first_leaf, second_leaf):
                primal_arr = jnp.asarray(primal_leaf)
                if not jnp.issubdtype(primal_arr.dtype, jnp.inexact):
                    return jnp.zeros(primal_arr.shape, dtype=jnp.float64)
                return base_leaf + first_leaf + second_leaf

            combined_prepared = jax.tree_util.tree_map(
                _combine_primal_prepared_leaf,
                prepared,
                base_prepared,
                first_directional_prepared,
                second_directional_prepared,
            )
            return tuple(jax.tree_util.tree_leaves(combined_prepared))
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

    if native_vmec_coefficient_bars_only:
        # The caller routes the complete prepared-system contribution through
        # ``native_vmec_coefficient_bars``.  Do not stage the generic
        # prepared-gradient contraction merely to return a support payload
        # which NEOPAX will discard.  Keep a shape-compatible zero tree so
        # the established outer scan/carry ABI remains unchanged for now.
        rhs_count = int(jnp.asarray(base_coefficient_bars).shape[0])

        def _zero_prepared_leaf(primal_leaf):
            primal_arr = jnp.asarray(primal_leaf)
            return jnp.zeros((rhs_count,) + primal_arr.shape, dtype=jnp.float64)

        result = (
            jax.tree_util.tree_map(_zero_prepared_leaf, prepared),
            auxiliary,
        )
    elif not _compact_residual_transpose:
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
            (jax.tree_util.tree_unflatten(prepared_tree, prepared_bar_leaf_groups), auxiliary)
            if _compact_result
            else (
                *(
                    jax.tree_util.tree_unflatten(prepared_tree, leaves)
                    for leaves in prepared_bar_leaf_groups
                ),
                auxiliary,
            )
        )
    native_vmec_coefficient_bars = None
    if return_vmec_coefficient_bars:
        # The local support cotangent contains only the primal prepared-system
        # terms: base plus the two directional low-dot contributions.  The
        # two physical-direction *base* cotangents belong exclusively to the
        # case chain below and must not be added to this geometry result.
        #
        # Keep those three terms as primitive geometry bars until after their
        # sum.  Mapping each one separately through the sampled-field/Fourier
        # transpose changes floating-point reduction order relative to the
        # combined prepared VJP and needlessly stages that large transpose
        # three times inside the segment graph.
        native_vmec_base = _native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs(
            prepared,
            ctx,
            f1_full,
            f3_full,
            base_lambda1,
            base_lambda3,
            base_coefficient_bars,
        )

        def _directional_native_primitive_bars(
            direction_index,
            coefficient_bars,
            coefficient_bars_dot,
        ):
            def _primitive_from_dynamic_terms(
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
                return _native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs(
                    prepared,
                    local_ctx,
                    f1_value,
                    f3_value,
                    lambda1_value,
                    lambda3_value,
                    coefficient_bar_value,
                )

            _base, directional = jax.jvp(
                _primitive_from_dynamic_terms,
                (
                    ctx.nu_hat,
                    ctx.epsi_hat,
                    f1_full,
                    f3_full,
                    directional_lambda1[..., direction_index],
                    directional_lambda3[..., direction_index],
                    coefficient_bars,
                ),
                (
                    nu_dots[direction_index],
                    epsi_dots[direction_index],
                    f1_dot_full[..., direction_index],
                    f3_dot_full[..., direction_index],
                    directional_lambda1_dot[..., direction_index],
                    directional_lambda3_dot[..., direction_index],
                    coefficient_bars_dot,
                ),
            )
            return directional

        if native_vmec_direct_directional_product_rule:

            def _direct_directional_primitive_bars(
                direction_index,
                coefficient_bars,
                coefficient_bars_dot,
            ):
                return _directional_native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs_direct(
                    prepared,
                    nu_hat=ctx.nu_hat,
                    epsi_hat=ctx.epsi_hat,
                    nu_hat_dot=nu_dots[direction_index],
                    epsi_hat_dot=epsi_dots[direction_index],
                    f1_full=f1_full,
                    f3_full=f3_full,
                    f1_dot=f1_dot_full[..., direction_index],
                    f3_dot=f3_dot_full[..., direction_index],
                    lambda1=directional_lambda1[..., direction_index],
                    lambda3=directional_lambda3[..., direction_index],
                    lambda1_dot=directional_lambda1_dot[..., direction_index],
                    lambda3_dot=directional_lambda3_dot[..., direction_index],
                    coefficient_bars=coefficient_bars,
                    coefficient_bars_dot=coefficient_bars_dot,
                )

            native_vmec_first_directional = _direct_directional_primitive_bars(
                0, first_coefficient_bars, first_coefficient_bars_dot
            )
            native_vmec_second_directional = _direct_directional_primitive_bars(
                1, second_coefficient_bars, second_coefficient_bars_dot
            )
        else:
            native_vmec_first_directional = _directional_native_primitive_bars(
                0,
                first_coefficient_bars,
                first_coefficient_bars_dot,
            )
            native_vmec_second_directional = _directional_native_primitive_bars(
                1,
                second_coefficient_bars,
                second_coefficient_bars_dot,
            )
        native_vmec_primitive_bars = {
            name: (
                native_vmec_base[name]
                + native_vmec_first_directional[name]
                + native_vmec_second_directional[name]
            )
            for name in native_vmec_base
        }
        native_vmec_coefficient_bars = vmec_geometry_bars_to_coefficients_multi_rhs(
            cast(VmecSurface, prepared.surface),
            prepared.geometry,
            native_vmec_primitive_bars,
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
                ctx.nu_hat,
                ctx.epsi_hat,
                f1_full,
                f3_full,
                directional_lambda1[..., 0],
                directional_lambda3[..., 0],
            ),
            (
                nu_dots[0],
                epsi_dots[0],
                f1_dot_full[..., 0],
                f3_dot_full[..., 0],
                directional_lambda1_dot[..., 0],
                directional_lambda3_dot[..., 0],
            ),
        )
        (
            (second_base_nu_implicit, second_base_epsi_bar),
            (second_nu_implicit_dot, second_epsi_bar_dot),
        ) = jax.jvp(
            _parameter_gradient_from_values,
            (
                ctx.nu_hat,
                ctx.epsi_hat,
                f1_full,
                f3_full,
                directional_lambda1[..., 1],
                directional_lambda3[..., 1],
            ),
            (
                nu_dots[1],
                epsi_dots[1],
                f1_dot_full[..., 1],
                f3_dot_full[..., 1],
                directional_lambda1_dot[..., 1],
                directional_lambda3_dot[..., 1],
            ),
        )
        # Keep the established ten case components even in the compact view.
        # They are small compared with the prepared-system trees, and routing
        # them through NEOPAX's already-proven combination preserves exactly
        # the native mapped contract.  Only the five large prepared trees are
        # compacted to one.
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
        if return_vmec_coefficient_bars:
            return (
                (primal_outputs, result, case_bar_components, native_vmec_coefficient_bars)
                if return_primal_outputs
                else (result, case_bar_components, native_vmec_coefficient_bars)
            )
        return (
            (primal_outputs, result, case_bar_components)
            if return_primal_outputs
            else (result, case_bar_components)
        )
    if return_vmec_coefficient_bars:
        return (
            (primal_outputs, result, native_vmec_coefficient_bars)
            if return_primal_outputs
            else (result, native_vmec_coefficient_bars)
        )
    return (primal_outputs, result) if return_primal_outputs else result


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_and_aux(  # noqa: E501 - preserve the existing public API name
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
    return_case_bars: bool = False,
):
    """Combine the base and two directional prepared bars into one compact tree.

    The native primal and matrix-RHS adjoints are shared. Optional case bars
    retain the ten-component contract used by the outer coordinate chain.
    """

    return solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bars_and_aux_fn,
        return_primal_outputs=return_primal_outputs,
        return_case_bars=return_case_bars,
        _compact_result=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_residual_and_aux(  # noqa: E501 - preserve the existing public API name
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
    return_case_bars: bool = False,
):
    """Return compact bars with dense residual and prepared-input transposes split."""
    return solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bars_and_aux_fn,
        return_primal_outputs=return_primal_outputs,
        return_case_bars=return_case_bars,
        _compact_result=True,
        _compact_residual_transpose=True,
    )
