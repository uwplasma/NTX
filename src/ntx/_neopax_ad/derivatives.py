"""Factorized coefficient derivatives and their prepared-support custom VJP entry points."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .._solver import MonoenergeticCase, PreparedMonoenergeticSystem, _solve_factorized_modes
from ..operators import parameter_derivative_blocks
from ..transport import coefficients_from_modes
from .adjoints import (
    _add_er_coordinate_scale_bar,
    _case_cotangent_from_resolved_epsi_bar,
    _prepared_primal_context,
    _sum_prepared_cotangent_trees,
    _unpack_coefficient_bars,
)
from .responses import (
    _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core,
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


def solve_prepared_coefficient_vector_hessian_factorized(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return coefficient value, gradient, and Hessian from one factorization.

    The prepared NTX operator is affine in the resolved local coordinates
    ``(nu_hat, epsi_hat)`` and its source is parameter independent.  For
    ``A f = b`` this gives the exact directional systems

    ``A f_p = -A_p f`` and
    ``A f_pq = -(A_p f_q + A_q f_p)``.

    Consequently one base factorization can solve all base, first-, and
    second-direction fields.  This is intentionally an explicit implicit
    derivative primitive: it does not differentiate through the existing
    custom-VJP solver entry point.
    """

    _, ctx, primal = _prepared_primal_context(prepared, case)
    (coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper) = primal
    mode_indices = jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32)

    def _parameter_blocks(mode_index):
        dnu, depsi = parameter_derivative_blocks(
            ctx,
            mode_index,
            prepared.d_theta,
            prepared.d_zeta,
        )
        dnu = jax.lax.cond(
            mode_index == 0,
            lambda value: value.at[0, :].set(jnp.zeros_like(value[0, :])),
            lambda value: value,
            dnu,
        )
        depsi = jax.lax.cond(
            mode_index == 0,
            lambda value: value.at[0, :].set(jnp.zeros_like(value[0, :])),
            lambda value: value,
            depsi,
        )
        return dnu, depsi

    dnu_blocks, depsi_blocks = jax.vmap(_parameter_blocks)(mode_indices)

    def _solve_first_fields(field):
        rhs = jnp.stack(
            (
                -jnp.einsum("mij,mj->mi", dnu_blocks, field),
                -jnp.einsum("mij,mj->mi", depsi_blocks, field),
            ),
            axis=-1,
        )
        return _solve_factorized_modes(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            rhs,
        )

    f1_first = _solve_first_fields(f1_full)
    f3_first = _solve_first_fields(f3_full)

    def _solve_second_fields(first_fields):
        f_nu = first_fields[..., 0]
        f_epsi = first_fields[..., 1]
        rhs = jnp.stack(
            (
                -2.0 * jnp.einsum("mij,mj->mi", dnu_blocks, f_nu),
                -(
                    jnp.einsum("mij,mj->mi", dnu_blocks, f_epsi)
                    + jnp.einsum("mij,mj->mi", depsi_blocks, f_nu)
                ),
                -2.0 * jnp.einsum("mij,mj->mi", depsi_blocks, f_epsi),
            ),
            axis=-1,
        )
        return _solve_factorized_modes(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            rhs,
        )

    f1_second = _solve_second_fields(f1_first)
    f3_second = _solve_second_fields(f3_first)

    def _coefficient_first(f1_dot, f3_dot, nu_dot):
        output = jnp.stack(
            coefficients_from_modes(
                prepared.geometry,
                f1_dot[:3],
                f3_dot[:3],
                ctx.nu_hat,
            )
        )
        return output.at[4].set(-coefficients[4] * nu_dot / ctx.nu_hat)

    def _coefficient_second(f1_dotdot, f3_dotdot, nu_dot_a, nu_dot_b):
        output = jnp.stack(
            coefficients_from_modes(
                prepared.geometry,
                f1_dotdot[:3],
                f3_dotdot[:3],
                ctx.nu_hat,
            )
        )
        return output.at[4].set(2.0 * coefficients[4] * nu_dot_a * nu_dot_b / ctx.nu_hat**2)

    nu_one = jnp.asarray(1.0, dtype=ctx.nu_hat.dtype)
    zero = jnp.asarray(0.0, dtype=ctx.nu_hat.dtype)
    coefficient_nu = _coefficient_first(f1_first[..., 0], f3_first[..., 0], nu_one)
    coefficient_epsi = _coefficient_first(f1_first[..., 1], f3_first[..., 1], zero)
    coefficient_nunu = _coefficient_second(f1_second[..., 0], f3_second[..., 0], nu_one, nu_one)
    coefficient_nuepsi = _coefficient_second(f1_second[..., 1], f3_second[..., 1], nu_one, zero)
    coefficient_epsiepsi = _coefficient_second(f1_second[..., 2], f3_second[..., 2], zero, zero)
    return (
        coefficients,
        coefficient_nu,
        coefficient_epsi,
        coefficient_nunu,
        coefficient_nuepsi,
        coefficient_epsiepsi,
    )


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


def _solve_prepared_coefficient_vector_two_directional_prepared_vjp_fwd(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
):
    """Retain primal factors so the reverse rule does not solve the primal again."""
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
    """Combine output bars and restore the electric-field normalization chain rule."""
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
        nu_hat_bar=(base_nu_hat_bar + first_nu_hat_bar + second_nu_hat_bar),
        epsi_hat_bar=(base_epsi_hat_bar + first_epsi_hat_bar + second_epsi_hat_bar),
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
    """Keep static prepared metadata outside the differentiable argument leaves."""
    prepared = jax.tree_util.tree_unflatten(prepared_tree, prepared_leaves)
    return solve_prepared_coefficient_vector_two_directional_factorized(
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
    """Rebuild the prepared tree while retaining reusable primal residuals."""
    prepared = jax.tree_util.tree_unflatten(prepared_tree, prepared_leaves)
    return _solve_prepared_coefficient_vector_two_directional_prepared_vjp_fwd(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
    )


def _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp_bwd(
    prepared_tree,
    residuals,
    output_bars: tuple[Array, Array, Array],
):
    """Return numeric leaf bars without differentiating the prepared tree metadata."""
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

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
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


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
):
    """Return RHS-batched prepared-support bars with one local primal.

    The callback takes ``(coefficients, first_dot, second_dot)`` and returns
    ``(base_bars, first_bars, second_bars, auxiliary)`` with a shared RHS axis.
    An optional fifth item supplies ``(first_bars_dot, second_bars_dot)``.
    Return the unbatched primal outputs too when ``return_primal_outputs`` is
    true. Factors remain local to this call; they are not part of the result.
    """

    primal_outputs, primal_residuals = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
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
    base_bars, first_bars, second_bars, auxiliary, first_bars_dot, second_bars_dot = (
        _unpack_coefficient_bars(coefficient_bars_and_aux_fn(*primal_outputs))
    )

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
            jax.tree_util.tree_unflatten(prepared_bar_tree, leaves)
            for leaves in prepared_bar_leaf_groups
        ),
        auxiliary,
    )
    return (primal_outputs, result) if return_primal_outputs else result


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux_packed_support_adjoint(  # noqa: E501 - preserve the existing public API name
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Pair the two directional adjoint field solves using the shared factors."""

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
