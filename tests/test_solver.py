from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    compile_prepared_scan_solver,
    compile_prepared_solver,
    example_surface,
    prepare_monoenergetic_system,
    solve_monoenergetic,
    solve_monoenergetic_internal,
    solve_monoenergetic_scan,
    solve_prepared,
    solve_prepared_coefficient_vector,
    solve_prepared_coefficient_vector_two_directional_factorized,
    solve_prepared_coefficient_vector_two_directional_prepared_vjp,
    pullback_prepared_coefficient_vector_case_and_prepared,
    pullback_prepared_coefficient_vector_case_and_prepared_multi_rhs,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_and_aux,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux_packed_support_adjoint,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry_and_aux,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux,
    solve_prepared_coefficient_vector_vjp,
    solve_prepared_internal,
)
from ntx.geometry import BoozerSurface
from ntx._solver_adjoint import _prepared_implicit_vjp_primal
from ntx._solver_prepared import (
    _solve_factorized_adjoint_field_pair,
    _solve_factorized_multi_rhs_directional_adjoint_field_pair,
)


def test_uniform_field_has_zero_radial_transport():
    surface = BoozerSurface(
        m=jnp.asarray([0]),
        n=jnp.asarray([0]),
        b_cos=jnp.asarray([1.0]),
        nfp=1,
        iota=0.6,
        psi_p=1.0,
        b_theta=0.1,
        b_zeta=1.0,
    )
    result = solve_monoenergetic(surface, GridSpec(5, 5, 4), MonoenergeticCase(1e-1))
    assert abs(float(result.D11)) < 1e-10
    assert abs(float(result.D31)) < 1e-10


def test_example_surface_returns_finite_coefficients():
    result = solve_monoenergetic(example_surface(), GridSpec(5, 5, 6), MonoenergeticCase(1e-2))
    values = jnp.asarray([result.D11, result.D31, result.D13, result.D33, result.D33_spitzer])
    assert jnp.all(jnp.isfinite(values))
    assert result.D33_spitzer > 0.0
    assert result.D11 >= -1e-10


def test_example_surface_coefficients_converge_on_small_grid_ladder():
    surface = example_surface()
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    grids = (
        GridSpec(5, 5, 4),
        GridSpec(7, 7, 6),
        GridSpec(9, 9, 8),
    )
    vectors = [
        jnp.asarray(
            [
                result.D11,
                result.D31,
                result.D13,
                result.D33,
                result.D33_spitzer,
            ]
        )
        for result in (solve_monoenergetic(surface, grid, case) for grid in grids)
    ]
    reference = vectors[-1]
    coarse_error = jnp.abs(vectors[0] - reference) / jnp.maximum(1.0, jnp.abs(reference))
    medium_error = jnp.abs(vectors[1] - reference) / jnp.maximum(1.0, jnp.abs(reference))
    assert jnp.all(medium_error < coarse_error)
    assert float(jnp.max(medium_error)) < 0.15


def test_n_xi_two_boundary_case_runs():
    result = solve_monoenergetic(example_surface(), GridSpec(5, 5, 2), MonoenergeticCase(1e-2))
    assert jnp.isfinite(result.D33_spitzer)


def test_vmap_parameter_scan_matches_single_solve_shape():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    scan = solve_monoenergetic_scan(surface, grid, jnp.asarray([1e-2, 2e-2]))
    single = solve_monoenergetic(surface, grid, MonoenergeticCase(1e-2))
    assert scan["D11"].shape == (2,)
    assert jnp.allclose(scan["D11"][0], single.D11)


def test_batched_parameter_scan_matches_full_surface_batching():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    nu = jnp.asarray([[1e-2, 2e-2], [3e-2, 4e-2]])
    er = jnp.asarray([[0.0, 1e-3], [2e-3, 3e-3]])
    full = solve_monoenergetic_scan(surface, grid, nu, er_hat=er)
    batched = solve_monoenergetic_scan(
        surface,
        grid,
        nu,
        er_hat=er,
        scan_batch_size=3,
    )
    for key, value in full.items():
        assert batched[key].shape == value.shape
        assert jnp.allclose(batched[key], value, rtol=1e-12, atol=1e-12)


def test_batched_parameter_scan_rejects_invalid_batch_size():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    with pytest.raises(ValueError, match="positive"):
        solve_monoenergetic_scan(
            surface,
            grid,
            jnp.asarray([1e-2]),
            scan_batch_size=0,
        )


@pytest.mark.parametrize("execution_mode", ["sequential", "vectorized"])
def test_compiled_prepared_scan_matches_pointwise_and_preserves_shape(execution_mode):
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    prepared = prepare_monoenergetic_system(surface, grid)
    nu = jnp.asarray([[1e-2, 2e-2], [3e-2, 4e-2]])
    er = jnp.asarray([[0.0, 1e-3], [2e-3, 3e-3]])
    compiled = compile_prepared_scan_solver(
        prepared,
        batch_size=8,
        execution_mode=execution_mode,
    )

    result = compiled(nu, er_hat=er)
    for index in range(nu.size):
        point = solve_prepared(
            prepared,
            MonoenergeticCase(nu.ravel()[index], er_hat=er.ravel()[index]),
        )
        for key in ("D11", "D31", "D13", "D33", "D33_spitzer"):
            assert result[key].shape == nu.shape
            assert jnp.allclose(
                result[key].ravel()[index],
                getattr(point, key),
                rtol=1e-12,
                atol=1e-12,
            )


def test_compiled_prepared_scan_reuses_fixed_bucket_across_lengths():
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    compiled = compile_prepared_scan_solver(
        prepared,
        batch_size=8,
        execution_mode="sequential",
    )

    short = compiled(jnp.logspace(-3, -2, 3))
    long = compiled(jnp.logspace(-3, -2, 11))
    assert short["D11"].shape == (3,)
    assert long["D11"].shape == (11,)
    assert compiled.batch_size == 8


def test_compiled_prepared_scan_auto_defaults_to_scalar_parity_path():
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    compiled = compile_prepared_scan_solver(prepared)

    assert compiled.execution_mode == "sequential"
    assert compiled.batch_size == 8


def test_compiled_prepared_scan_is_differentiable():
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    compiled = compile_prepared_scan_solver(
        prepared,
        batch_size=8,
        execution_mode="sequential",
    )
    nu = jnp.asarray([1e-2, 2e-2, 3e-2])

    gradient = jax.grad(lambda er: jnp.sum(compiled(nu, er_hat=jnp.full_like(nu, er))["D11"]))(1e-3)
    assert jnp.isfinite(gradient)


def test_compiled_prepared_scan_warmup_reports_executable_costs():
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    compiled = compile_prepared_scan_solver(
        prepared,
        batch_size=1,
        execution_mode="sequential",
    )

    report = compiled.warmup()
    assert report.lowering_seconds >= 0.0
    assert report.compilation_seconds >= 0.0
    assert report.first_execution_seconds >= 0.0
    assert report.warm_execution_seconds >= 0.0
    assert report.temporary_size_bytes is None or report.temporary_size_bytes >= 0


def test_compiled_prepared_scan_rejects_nonstandard_bucket_and_mode():
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    with pytest.raises(ValueError, match="fixed buckets"):
        compile_prepared_scan_solver(prepared, batch_size=3)
    with pytest.raises(ValueError, match="execution_mode"):
        compile_prepared_scan_solver(prepared, execution_mode="threads")


def test_compiled_prepared_scan_adds_actionable_oom_guidance():
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    compiled = compile_prepared_scan_solver(
        prepared,
        batch_size=1,
        execution_mode="sequential",
    )

    def raise_oom(_nu, _epsi):
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")

    compiled._solve_batch = raise_oom
    with pytest.raises(RuntimeError, match="smaller fixed batch bucket"):
        compiled(jnp.asarray([1e-2]))


def test_spitzer_scales_inverse_with_collisionality():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    low = solve_monoenergetic(surface, grid, MonoenergeticCase(1e-2))
    high = solve_monoenergetic(surface, grid, MonoenergeticCase(2e-2))
    assert jnp.allclose(low.D33_spitzer / high.D33_spitzer, 2.0, rtol=1e-10)


def test_prepared_system_matches_direct_solve():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    prepared = prepare_monoenergetic_system(surface, grid)
    direct = solve_monoenergetic(surface, grid, case).as_dict()
    cached = solve_prepared(prepared, case).as_dict()
    for key, value in direct.items():
        assert jnp.allclose(value, cached[key], rtol=1e-12, atol=1e-12)


def test_compiled_prepared_solver_matches_eager_solve():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    prepared = prepare_monoenergetic_system(surface, grid)
    eager = solve_prepared(prepared, case).as_dict()
    compiled = compile_prepared_solver(prepared)(case).as_dict()
    for key, value in eager.items():
        assert jnp.allclose(value, compiled[key], rtol=1e-12, atol=1e-12)


def test_solve_internal_matches_named_coefficients():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    result = solve_monoenergetic(surface, grid, case)
    dij, f_internal, s_internal = solve_monoenergetic_internal(surface, grid, case)
    assert dij.shape == (3, 3)
    assert f_internal.shape == (3, 3, grid.n_fs)
    assert s_internal.shape == (3, 3, grid.n_fs)
    assert jnp.allclose(dij[0, 0], result.D11)
    assert jnp.allclose(dij[2, 0], result.D31)
    assert jnp.allclose(dij[0, 2], result.D13)
    assert jnp.allclose(dij[2, 2], result.D33)


def test_prepared_internal_matches_direct_internal_solve():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    prepared = prepare_monoenergetic_system(surface, grid)
    direct = solve_monoenergetic_internal(surface, grid, case)
    cached = solve_prepared_internal(prepared, case)
    for lhs, rhs in zip(direct, cached, strict=True):
        assert jnp.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_gradients_exist_through_er_hat_and_nu_hat():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)

    def d11_from_er(er_hat):
        return solve_monoenergetic(surface, grid, MonoenergeticCase(1e-2, er_hat=er_hat)).D11

    def d11_from_nu(nu_hat):
        return solve_monoenergetic(surface, grid, MonoenergeticCase(nu_hat, er_hat=1e-3)).D11

    assert jnp.isfinite(jax.grad(d11_from_er)(1e-3))
    assert jnp.isfinite(jax.grad(d11_from_nu)(1e-2))


def test_gradient_exists_through_surface_coefficients():
    grid = GridSpec(5, 5, 4)
    m = jnp.asarray([0, 1, 1, 2], dtype=jnp.int32)
    n = jnp.asarray([0, 0, 1, -1], dtype=jnp.int32)

    def d11_from_bcos(b_cos):
        surface = BoozerSurface(
            m=m,
            n=n,
            b_cos=b_cos,
            nfp=5,
            iota=0.85,
            psi_p=1.0,
            b_theta=0.05,
            b_zeta=1.0,
        )
        return solve_monoenergetic(surface, grid, MonoenergeticCase(1e-2)).D11

    grad = jax.grad(d11_from_bcos)(jnp.asarray([1.0, 0.06, 0.025, 0.01]))
    assert grad.shape == (4,)
    assert jnp.all(jnp.isfinite(grad))


def test_jit_accepts_surface_argument_in_core_path():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    solve_d11 = jax.jit(
        lambda surf, er_hat: (
            solve_monoenergetic(surf, grid, MonoenergeticCase(1e-2, er_hat=er_hat)).D11
        )
    )
    value = solve_d11(surface, 1e-3)
    assert jnp.isfinite(value)


def test_compiled_prepared_solver_is_differentiable_in_er_hat():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    prepared = prepare_monoenergetic_system(surface, grid)
    compiled = compile_prepared_solver(prepared)

    grad = jax.grad(lambda er_hat: compiled(MonoenergeticCase(1e-2, er_hat=er_hat)).D11)(1e-3)
    assert jnp.isfinite(grad)


def test_prepared_coefficient_vector_matches_transport_result():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    prepared = prepare_monoenergetic_system(surface, grid)
    vector = solve_prepared_coefficient_vector(prepared, case)
    result = solve_prepared(prepared, case)
    assert vector.shape == (5,)
    assert jnp.allclose(
        vector,
        jnp.asarray([result.D11, result.D31, result.D13, result.D33, result.D33_spitzer]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_custom_vjp_coefficient_vector_matches_direct_forward_and_gradient():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    prepared = prepare_monoenergetic_system(surface, grid)
    case = MonoenergeticCase(1e-2, er_hat=1e-3)

    direct = solve_prepared_coefficient_vector(prepared, case)
    wrapped = solve_prepared_coefficient_vector_vjp(prepared, case)
    assert jnp.allclose(direct, wrapped, rtol=1e-12, atol=1e-12)

    direct_grad = jax.grad(
        lambda er_hat: solve_prepared_coefficient_vector(
            prepared,
            MonoenergeticCase(1e-2, er_hat=er_hat),
        )[0]
    )(1e-3)
    wrapped_grad = jax.grad(
        lambda er_hat: solve_prepared_coefficient_vector_vjp(
            prepared,
            MonoenergeticCase(1e-2, er_hat=er_hat),
        )[0]
    )(1e-3)
    assert jnp.isfinite(wrapped_grad)
    assert jnp.allclose(direct_grad, wrapped_grad, rtol=1e-10, atol=1e-12)


def test_grouped_prepared_pullback_matches_full_prepared_vjp():
    """The grouped rule must retain every differentiable prepared leaf."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    coefficient_bar = jnp.asarray([0.7, -0.2, 0.1, 0.3, -0.4])
    _case_bar, grouped_prepared_bar = pullback_prepared_coefficient_vector_case_and_prepared(
        prepared,
        case,
        coefficient_bar,
    )
    _, full_pullback = jax.vjp(
        lambda prepared_value: solve_prepared_coefficient_vector(prepared_value, case),
        prepared,
    )
    (reference_prepared_bar,) = full_pullback(coefficient_bar)
    for grouped_leaf, reference_leaf in zip(
        jax.tree_util.tree_leaves(grouped_prepared_bar),
        jax.tree_util.tree_leaves(reference_prepared_bar),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(reference_leaf).dtype, jnp.inexact):
            assert jnp.allclose(grouped_leaf, reference_leaf, rtol=1e-9, atol=1e-11)


def test_grouped_geometry_two_direction_pullback_matches_geometry_only_vjp():
    """The geometry-only rule must equal a VJP with fixed support operators."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
    zero_direction = MonoenergeticCase(0.0, epsi_hat=0.0)
    coefficient_bar = jnp.asarray([0.7, -0.2, 0.1, 0.3, -0.4])

    grouped = solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry(
        prepared,
        case,
        zero_direction,
        zero_direction,
        lambda coefficients: (
            coefficient_bar,
            jnp.zeros_like(coefficients),
            jnp.zeros_like(coefficients),
        ),
    )
    grouped_geometry_bar = grouped[2]
    _, pullback = jax.vjp(
        lambda geometry: solve_prepared_coefficient_vector(
            dataclasses.replace(prepared, geometry=geometry),
            case,
        ),
        prepared.geometry,
    )
    (reference_geometry_bar,) = pullback(coefficient_bar)
    for grouped_leaf, reference_leaf in zip(
        jax.tree_util.tree_leaves(grouped_geometry_bar),
        jax.tree_util.tree_leaves(reference_geometry_bar),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(reference_leaf).dtype, jnp.inexact):
            assert jnp.allclose(grouped_leaf, reference_leaf, rtol=1e-9, atol=1e-11)


def test_geometry_only_prepared_input_matches_full_vjp_active_geometry_bar():
    """Fixing grid-derived derivative blocks preserves the geometry cotangent."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
    coefficient_bar = jnp.asarray([0.7, -0.2, 0.1, 0.3, -0.4])

    _, full_pullback = jax.vjp(
        lambda prepared_value: solve_prepared_coefficient_vector(prepared_value, case),
        prepared,
    )
    (full_prepared_bar,) = full_pullback(coefficient_bar)
    _, geometry_pullback = jax.vjp(
        lambda geometry: solve_prepared_coefficient_vector(
            dataclasses.replace(prepared, geometry=geometry),
            case,
        ),
        prepared.geometry,
    )
    (geometry_only_bar,) = geometry_pullback(coefficient_bar)

    for full_leaf, geometry_leaf in zip(
        jax.tree_util.tree_leaves(full_prepared_bar.geometry),
        jax.tree_util.tree_leaves(geometry_only_bar),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(geometry_leaf).dtype, jnp.inexact):
            assert jnp.allclose(full_leaf, geometry_leaf, rtol=1e-9, atol=1e-11)

    # These bars are nonzero in the broad VJP, yet their tangent is zero for
    # the fixed GridSpec runtime payload.  The restricted boundary removes
    # this dense operator-transpose work exactly for that payload.
    assert float(jnp.max(jnp.abs(full_prepared_bar.d_theta))) > 1e-12
    assert float(jnp.max(jnp.abs(full_prepared_bar.d_zeta))) > 1e-12


def test_geometry_aux_two_direction_pullback_matches_prepared_geometry_projection():
    """Geometry-only grouped bars equal the active projection of prepared bars."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
    first_direction = MonoenergeticCase(0.0, epsi_hat=0.13)
    second_direction = MonoenergeticCase(0.17, epsi_hat=0.0)
    coefficient_bar = jnp.asarray([0.7, -0.2, 0.1, 0.3, -0.4])

    def _bars_and_aux(coefficients, first_coeff_dot, second_coeff_dot):
        return (
            coefficient_bar,
            -0.4 * coefficient_bar,
            0.6 * coefficient_bar,
            (coefficients, first_coeff_dot, second_coeff_dot),
        )

    geometry_result = solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry_and_aux(
        prepared, case, first_direction, second_direction, _bars_and_aux
    )
    prepared_result = solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux(
        prepared, case, first_direction, second_direction, _bars_and_aux
    )
    for geometry_index in (2, 7, 8, 13, 14):
        for geometry_leaf, prepared_leaf in zip(
            jax.tree_util.tree_leaves(geometry_result[geometry_index]),
            jax.tree_util.tree_leaves(prepared_result[geometry_index].geometry),
            strict=True,
        ):
            if jnp.issubdtype(jnp.asarray(geometry_leaf).dtype, jnp.inexact):
                assert jnp.allclose(geometry_leaf, prepared_leaf, rtol=1e-9, atol=1e-11)
    for geometry_aux, prepared_aux in zip(geometry_result[-1], prepared_result[-1], strict=True):
        assert jnp.allclose(geometry_aux, prepared_aux, rtol=1e-12, atol=1e-12)


def test_geometry_support_only_aux_matches_geometry_full_result():
    """Geometry support-only output omits case bars without changing support."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
    first_direction = MonoenergeticCase(0.0, epsi_hat=0.13)
    second_direction = MonoenergeticCase(0.17, epsi_hat=0.0)
    coefficient_bar = jnp.asarray([0.7, -0.2, 0.1, 0.3, -0.4])

    def _bars_and_aux(coefficients, first_coeff_dot, second_coeff_dot):
        return (
            coefficient_bar,
            -0.4 * coefficient_bar,
            0.6 * coefficient_bar,
            (coefficients, first_coeff_dot, second_coeff_dot),
        )

    support_result = solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux(
        prepared,
        case,
        first_direction,
        second_direction,
        _bars_and_aux,
    )
    full_result = solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry_and_aux(
        prepared,
        case,
        first_direction,
        second_direction,
        _bars_and_aux,
    )
    for support_index, full_index in zip((0, 1, 2, 3, 4), (2, 7, 8, 13, 14), strict=True):
        for support_leaf, full_leaf in zip(
            jax.tree_util.tree_leaves(support_result[support_index]),
            jax.tree_util.tree_leaves(full_result[full_index]),
            strict=True,
        ):
            if (
                jnp.asarray(support_leaf).dtype != jax.dtypes.float0
                and jnp.asarray(full_leaf).dtype != jax.dtypes.float0
                and jnp.issubdtype(jnp.asarray(full_leaf).dtype, jnp.inexact)
            ):
                assert jnp.allclose(support_leaf, full_leaf, rtol=1e-9, atol=1e-11)
    for support_aux, full_aux in zip(support_result[-1], full_result[-1], strict=True):
        assert jnp.allclose(support_aux, full_aux, rtol=1e-12, atol=1e-12)


def test_multi_rhs_prepared_pullback_matches_scalar_prepared_pullbacks():
    """The native matrix-RHS base rule must match every scalar RHS exactly."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    coefficient_bars = jnp.asarray(
        [[0.7, -0.2, 0.1, 0.3, -0.4], [-0.3, 0.8, -0.5, 0.2, 0.6]]
    )
    multi_case_bar, multi_prepared_bar = (
        pullback_prepared_coefficient_vector_case_and_prepared_multi_rhs(
            prepared,
            case,
            coefficient_bars,
        )
    )
    for rhs_index in range(coefficient_bars.shape[0]):
        scalar_case_bar, scalar_prepared_bar = (
            pullback_prepared_coefficient_vector_case_and_prepared(
                prepared,
                case,
                coefficient_bars[rhs_index],
            )
        )
        for multi_leaf, scalar_leaf in zip(
            jax.tree_util.tree_leaves(multi_case_bar),
            jax.tree_util.tree_leaves(scalar_case_bar),
            strict=True,
        ):
            if jnp.issubdtype(jnp.asarray(scalar_leaf).dtype, jnp.inexact):
                assert jnp.allclose(
                    multi_leaf[rhs_index], scalar_leaf, rtol=1e-9, atol=1e-11
                )
        for multi_leaf, scalar_leaf in zip(
            jax.tree_util.tree_leaves(multi_prepared_bar),
            jax.tree_util.tree_leaves(scalar_prepared_bar),
            strict=True,
        ):
            if jnp.issubdtype(jnp.asarray(scalar_leaf).dtype, jnp.inexact):
                assert jnp.allclose(
                    multi_leaf[rhs_index], scalar_leaf, rtol=1e-9, atol=1e-11
                )


@pytest.mark.parametrize("rhs_count", (1, 2, 4))
def test_lowdot_support_only_multi_rhs_matches_scalar_support_only(rhs_count):
    """The local multi-RHS helper must preserve every exact support bar."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    first_direction = MonoenergeticCase(0.0, er_hat=0.23)
    second_direction = MonoenergeticCase(-0.17, er_hat=0.0)
    base_bars = jnp.reshape(
        jnp.arange(rhs_count * 5, dtype=jnp.float64), (rhs_count, 5)
    ) / 13.0 - 0.4
    first_bars = jnp.flip(base_bars, axis=1) * 0.37
    second_bars = base_bars * -0.21
    auxiliary = jnp.linspace(-0.3, 0.4, rhs_count, dtype=jnp.float64)

    multi = (
        solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux(
            prepared,
            case,
            first_direction,
            second_direction,
            lambda _base, _first, _second: (
                base_bars,
                first_bars,
                second_bars,
                auxiliary,
            ),
        )
    )
    primal_outputs, multi_with_primal = (
        solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux(
            prepared,
            case,
            first_direction,
            second_direction,
            lambda _base, _first, _second: (
                base_bars,
                first_bars,
                second_bars,
                auxiliary,
            ),
            return_primal_outputs=True,
        )
    )
    assert all(value.shape == (5,) for value in primal_outputs)
    for with_primal_tree, multi_tree in zip(
        multi_with_primal, multi, strict=True
    ):
        for with_primal_leaf, multi_leaf in zip(
            jax.tree_util.tree_leaves(with_primal_tree),
            jax.tree_util.tree_leaves(multi_tree),
            strict=True,
        ):
            if jnp.issubdtype(jnp.asarray(multi_leaf).dtype, jnp.inexact):
                assert jnp.allclose(with_primal_leaf, multi_leaf, rtol=0.0, atol=0.0)
    assert multi[-1].shape == (rhs_count,)
    for rhs_index in range(rhs_count):
        scalar = solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux(
            prepared,
            case,
            first_direction,
            second_direction,
            lambda _base, _first, _second, index=rhs_index: (
                base_bars[index],
                first_bars[index],
                second_bars[index],
                auxiliary[index],
            ),
        )
        for multi_tree, scalar_tree in zip(multi[:-1], scalar[:-1], strict=True):
            for multi_leaf, scalar_leaf in zip(
                jax.tree_util.tree_leaves(multi_tree),
                jax.tree_util.tree_leaves(scalar_tree),
                strict=True,
            ):
                if jnp.issubdtype(jnp.asarray(scalar_leaf).dtype, jnp.inexact):
                    assert jnp.allclose(
                        multi_leaf[rhs_index], scalar_leaf, rtol=1e-9, atol=1e-11
                    )
        assert jnp.allclose(multi[-1][rhs_index], scalar[-1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("rhs_count", (1, 2, 4))
def test_native_lowdot_support_multi_rhs_matches_scalar_support_only(rhs_count):
    """Native RHS-column adjoints reproduce the scalar prepared-support rule."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
    first_direction = MonoenergeticCase(0.0, epsi_hat=0.23)
    second_direction = MonoenergeticCase(-0.17, epsi_hat=0.0)
    base_bars = jnp.reshape(
        jnp.arange(rhs_count * 5, dtype=jnp.float64), (rhs_count, 5)
    ) / 13.0 - 0.4
    first_bars = jnp.flip(base_bars, axis=1) * 0.37
    second_bars = base_bars * -0.21
    auxiliary = jnp.linspace(-0.3, 0.4, rhs_count, dtype=jnp.float64)

    native = (
        solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(
            prepared,
            case,
            first_direction,
            second_direction,
            lambda _base, _first, _second: (
                base_bars,
                first_bars,
                second_bars,
                auxiliary,
            ),
        )
    )
    for rhs_index in range(rhs_count):
        scalar = solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux(
            prepared,
            case,
            first_direction,
            second_direction,
            lambda _base, _first, _second, index=rhs_index: (
                base_bars[index],
                first_bars[index],
                second_bars[index],
                auxiliary[index],
            ),
        )
        for native_tree, scalar_tree in zip(native[:-1], scalar[:-1], strict=True):
            for native_leaf, scalar_leaf in zip(
                jax.tree_util.tree_leaves(native_tree),
                jax.tree_util.tree_leaves(scalar_tree),
                strict=True,
            ):
                if jnp.issubdtype(jnp.asarray(scalar_leaf).dtype, jnp.inexact):
                    assert jnp.allclose(
                        native_leaf[rhs_index], scalar_leaf, rtol=1e-9, atol=1e-11
                    )
        assert jnp.allclose(native[-1][rhs_index], scalar[-1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_native_lowdot_support_multi_rhs_compact_matches_full_contract(rhs_count):
    """The compact native view is exactly the required reduction of full output."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
    first_direction = MonoenergeticCase(0.0, epsi_hat=0.23)
    second_direction = MonoenergeticCase(-0.17, epsi_hat=0.0)
    base_bars = jnp.reshape(
        jnp.arange(rhs_count * 5, dtype=jnp.float64), (rhs_count, 5)
    ) / 13.0 - 0.4
    first_bars = jnp.flip(base_bars, axis=1) * 0.37
    second_bars = base_bars * -0.21
    auxiliary = jnp.linspace(-0.3, 0.4, rhs_count, dtype=jnp.float64)
    bar_fn = lambda _base, _first, _second: (
        base_bars,
        first_bars,
        second_bars,
        auxiliary,
    )

    full_primal, full_result, full_case = (
        solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(
            prepared,
            case,
            first_direction,
            second_direction,
            bar_fn,
            return_primal_outputs=True,
            return_case_bars=True,
        )
    )
    compact_primal, compact_result, compact_case = (
        solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_and_aux(
            prepared,
            case,
            first_direction,
            second_direction,
            bar_fn,
            return_primal_outputs=True,
            return_case_bars=True,
        )
    )
    full_base, _first_base, full_first_directional, _second_base, full_second_directional, full_auxiliary = full_result
    compact_prepared, compact_auxiliary = compact_result
    def _expected_prepared_leaf(primal_leaf, base_leaf, first_leaf, second_leaf):
        primal_arr = jnp.asarray(primal_leaf)
        if not jnp.issubdtype(primal_arr.dtype, jnp.inexact):
            return jnp.zeros(primal_arr.shape, dtype=jnp.float64)
        if (
            jnp.asarray(base_leaf).dtype == jax.dtypes.float0
            or jnp.asarray(first_leaf).dtype == jax.dtypes.float0
            or jnp.asarray(second_leaf).dtype == jax.dtypes.float0
        ):
            return jnp.zeros_like(primal_arr)
        return base_leaf + first_leaf + second_leaf

    expected_prepared_leaves = tuple(
        _expected_prepared_leaf(primal_leaf, base_leaf, first_leaf, second_leaf)
        for primal_leaf, base_leaf, first_leaf, second_leaf in zip(
            jax.tree_util.tree_leaves(prepared),
            jax.tree_util.tree_leaves(full_base),
            jax.tree_util.tree_leaves(full_first_directional),
            jax.tree_util.tree_leaves(full_second_directional),
            strict=True,
        )
    )
    for actual, expected in zip(compact_primal, full_primal, strict=True):
        assert jnp.allclose(actual, expected, rtol=0.0, atol=0.0)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(compact_prepared),
        expected_prepared_leaves,
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(expected).dtype, jnp.inexact):
            assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(compact_auxiliary, full_auxiliary, rtol=0.0, atol=0.0)
    for actual, expected in zip(compact_case, full_case, strict=True):
        assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_fused_prepared_two_direction_pullback_runs_and_matches_base_vjp():
    """The fused prepared-support branch must receive mode arrays, not callbacks."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    coefficient_bar = jnp.asarray([0.7, -0.2, 0.1, 0.3, -0.4])
    zero_direction = MonoenergeticCase(0.0, er_hat=0.0)

    result = solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux(
        prepared,
        case,
        zero_direction,
        zero_direction,
        lambda coefficients, first_coeff_dot, second_coeff_dot: (
            coefficient_bar,
            jnp.zeros_like(first_coeff_dot),
            jnp.zeros_like(second_coeff_dot),
            jnp.asarray(0.0, dtype=coefficients.dtype),
        ),
    )
    fused_prepared_bar = result[2]
    _, reference_prepared_bar = pullback_prepared_coefficient_vector_case_and_prepared(
        prepared,
        case,
        coefficient_bar,
    )
    for fused_leaf, reference_leaf in zip(
        jax.tree_util.tree_leaves(fused_prepared_bar),
        jax.tree_util.tree_leaves(reference_prepared_bar),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(reference_leaf).dtype, jnp.inexact):
            assert jnp.allclose(fused_leaf, reference_leaf, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("case_representation", ("er", "epsi"))
def test_factorized_two_directional_primal_matches_raw_jvps(case_representation):
    """One-factorization primal directions match the established raw JVPs."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    if case_representation == "er":
        case = MonoenergeticCase(1e-2, er_hat=1e-3)
        first_direction = MonoenergeticCase(0.13, er_hat=-0.27)
        second_direction = MonoenergeticCase(-0.19, er_hat=0.41)

        def raw_solution(nu_hat, field_value):
            return solve_prepared_coefficient_vector(
                prepared,
                MonoenergeticCase(nu_hat, er_hat=field_value),
            )

        primal = (case.nu_hat, case.er_hat)
        first_tangent = (first_direction.nu_hat, first_direction.er_hat)
        second_tangent = (second_direction.nu_hat, second_direction.er_hat)
    else:
        case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
        first_direction = MonoenergeticCase(0.13, epsi_hat=-0.27)
        second_direction = MonoenergeticCase(-0.19, epsi_hat=0.41)

        def raw_solution(nu_hat, field_value):
            return solve_prepared_coefficient_vector(
                prepared,
                MonoenergeticCase(nu_hat, epsi_hat=field_value),
            )

        primal = (case.nu_hat, case.epsi_hat)
        first_tangent = (first_direction.nu_hat, first_direction.epsi_hat)
        second_tangent = (second_direction.nu_hat, second_direction.epsi_hat)

    reference_base, reference_first = jax.jvp(raw_solution, primal, first_tangent)
    _, reference_second = jax.jvp(raw_solution, primal, second_tangent)
    actual_base, actual_first, actual_second = jax.jit(
        lambda: solve_prepared_coefficient_vector_two_directional_factorized(
            prepared,
            case,
            first_direction,
            second_direction,
        )
    )()
    assert jnp.allclose(actual_base, reference_base, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(actual_first, reference_first, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(actual_second, reference_second, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("case_representation", ("er", "epsi"))
def test_factorized_two_directional_prepared_vjp_matches_raw_vjp(case_representation):
    """The isolated prepared custom VJP matches the raw triple-output VJP."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    if case_representation == "er":
        case = MonoenergeticCase(1e-2, er_hat=1e-3)
        first_direction = MonoenergeticCase(0.13, er_hat=-0.27)
        second_direction = MonoenergeticCase(-0.19, er_hat=0.41)
    else:
        case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
        first_direction = MonoenergeticCase(0.13, epsi_hat=-0.27)
        second_direction = MonoenergeticCase(-0.19, epsi_hat=0.41)
    output_bars = (
        jnp.asarray([0.7, -0.2, 0.1, 0.3, -0.4]),
        jnp.asarray([-0.1, 0.3, 0.2, -0.4, 0.5]),
        jnp.asarray([0.4, 0.1, -0.6, 0.3, -0.2]),
    )

    def raw_two_directional(prepared_value, case_value, first_dot, second_dot):
        base = solve_prepared_coefficient_vector(prepared_value, case_value)
        _, first = jax.jvp(
            lambda local_case: solve_prepared_coefficient_vector(prepared_value, local_case),
            (case_value,),
            (first_dot,),
        )
        _, second = jax.jvp(
            lambda local_case: solve_prepared_coefficient_vector(prepared_value, local_case),
            (case_value,),
            (second_dot,),
        )
        return base, first, second

    _, raw_pullback = jax.vjp(
        raw_two_directional,
        prepared,
        case,
        first_direction,
        second_direction,
    )
    raw_bars = raw_pullback(output_bars)
    custom_bars = jax.jit(
        lambda prepared_value, case_value, first_dot, second_dot: jax.vjp(
            solve_prepared_coefficient_vector_two_directional_prepared_vjp,
            prepared_value,
            case_value,
            first_dot,
            second_dot,
        )[1](output_bars)
    )(
        prepared,
        case,
        first_direction,
        second_direction,
    )
    for custom_bar, raw_bar in zip(custom_bars, raw_bars, strict=True):
        for custom_leaf, raw_leaf in zip(
            jax.tree_util.tree_leaves(custom_bar),
            jax.tree_util.tree_leaves(raw_bar),
            strict=True,
        ):
            if jnp.issubdtype(jnp.asarray(raw_leaf).dtype, jnp.inexact):
                assert jnp.allclose(custom_leaf, raw_leaf, rtol=1e-9, atol=1e-11)


def test_packed_support_directional_adjoint_matches_scalar_prepared_helper():
    """Packing only lambda-dot field solves must preserve every support bar."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    first_direction = MonoenergeticCase(0.0, er_hat=0.2)
    second_direction = MonoenergeticCase(0.1, er_hat=0.0)

    def coefficient_bar_and_aux(coefficients, first_coefficient_dot, second_coefficient_dot):
        return (
            jnp.asarray([0.7, -0.2, 0.1, 0.3, -0.4]),
            jnp.asarray([-0.1, 0.3, 0.2, -0.4, 0.5]),
            jnp.asarray([0.4, 0.1, -0.6, 0.3, -0.2]),
            jnp.asarray(0.0, dtype=coefficients.dtype),
        )

    reference = solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux(
        prepared,
        case,
        first_direction,
        second_direction,
        coefficient_bar_and_aux,
    )
    packed = (
        solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux_packed_support_adjoint(
            prepared,
            case,
            first_direction,
            second_direction,
            coefficient_bar_and_aux,
        )
    )
    for packed_leaf, reference_leaf in zip(
        jax.tree_util.tree_leaves(packed),
        jax.tree_util.tree_leaves(reference),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(reference_leaf).dtype, jnp.inexact):
            assert jnp.allclose(packed_leaf, reference_leaf, rtol=1e-9, atol=1e-11)


def test_multi_rhs_directional_adjoint_packing_matches_explicit_columns():
    """Objective RHS columns use one packed transpose system, not scalar VJPs."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, er_hat=1e-3)
    _, f1_full, _, saved_lu, saved_piv, saved_lower, saved_upper = (
        _prepared_implicit_vjp_primal(
            prepared,
            case.nu_hat,
            case.resolved_epsi_hat(prepared.geometry.transport_psi_scale),
        )
    )
    mode_count, unknown_count = f1_full.shape
    rhs_count = 3
    direction_count = 2
    shape = (mode_count, unknown_count, rhs_count, direction_count)
    values = jnp.arange(
        mode_count * unknown_count * rhs_count * direction_count,
        dtype=f1_full.dtype,
    ).reshape(shape)
    first_adjoint = values / 13.0
    second_adjoint = (values + 1.0) / 17.0
    first_rhs_dot = (values - 2.0) / 19.0
    second_rhs_dot = (values + 3.0) / 23.0
    diagonal_dot = jnp.broadcast_to(
        jnp.stack(
            [
                jnp.eye(unknown_count, dtype=f1_full.dtype),
                -2.0 * jnp.eye(unknown_count, dtype=f1_full.dtype),
            ]
        ),
        (mode_count, direction_count, unknown_count, unknown_count),
    )

    first_solution, second_solution = (
        _solve_factorized_multi_rhs_directional_adjoint_field_pair(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            first_adjoint,
            second_adjoint,
            first_rhs_dot,
            second_rhs_dot,
            diagonal_dot,
        )
    )
    first_rhs = first_rhs_dot - jnp.einsum(
        "mdji,mjrd->mird", diagonal_dot, first_adjoint
    )
    second_rhs = second_rhs_dot - jnp.einsum(
        "mdji,mjrd->mird", diagonal_dot, second_adjoint
    )
    first_reference, second_reference = _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.reshape(first_rhs, (mode_count, unknown_count, rhs_count * direction_count)),
        jnp.reshape(second_rhs, (mode_count, unknown_count, rhs_count * direction_count)),
    )
    assert jnp.allclose(
        first_solution,
        jnp.reshape(first_reference, shape),
        rtol=1e-10,
        atol=1e-12,
    )
    assert jnp.allclose(
        second_solution,
        jnp.reshape(second_reference, shape),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("case_representation", ("er", "epsi"))
def test_support_only_lowdot_prepared_helper_matches_full_prepared_helper(case_representation):
    """Support-only prepared bars match the full helper in both case representations."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    if case_representation == "er":
        case = MonoenergeticCase(1e-2, er_hat=1e-3)
        first_direction = MonoenergeticCase(0.0, er_hat=0.2)
        second_direction = MonoenergeticCase(0.1, er_hat=0.0)
    else:
        case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
        first_direction = MonoenergeticCase(0.0, epsi_hat=0.2)
        second_direction = MonoenergeticCase(0.1, epsi_hat=0.0)

    def coefficient_bar_and_aux(coefficients, first_coefficient_dot, second_coefficient_dot):
        return (
            jnp.asarray([0.7, -0.2, 0.1, 0.3, -0.4]),
            jnp.asarray([-0.1, 0.3, 0.2, -0.4, 0.5]),
            jnp.asarray([0.4, 0.1, -0.6, 0.3, -0.2]),
            jnp.sum(coefficients)
            + jnp.sum(first_coefficient_dot)
            - jnp.sum(second_coefficient_dot),
        )

    full = solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux(
        prepared,
        case,
        first_direction,
        second_direction,
        coefficient_bar_and_aux,
    )
    support_only = jax.jit(
        lambda: solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux(
            prepared,
            case,
            first_direction,
            second_direction,
            coefficient_bar_and_aux,
        )
    )()
    expected = (full[2], full[7], full[8], full[13], full[14], full[15])
    for support_only_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(support_only),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
            assert jnp.allclose(support_only_leaf, expected_leaf, rtol=1e-9, atol=1e-11)
