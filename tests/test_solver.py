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
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_residual_and_aux,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux_packed_support_adjoint,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry_and_aux,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux,
    solve_prepared_coefficient_vector_vjp,
    solve_prepared_internal,
)
from ntx.geometry import BoozerSurface, VmecSurface
from ntx._geometry_eval import geometry_on_grid, vmec_sampled_field_bars_to_coefficients_multi_rhs, vmec_geometry_bars_to_coefficients_multi_rhs
from ntx._solver_adjoint import (
    _compact_prepared_bar_to_prepared,
    _compact_prepared_gradient_from_adjoint,
    _combined_prepared_gradient_from_adjoint_multi_rhs_oracle,
    _directional_compact_prepared_gradient_from_adjoint,
    _directional_prepared_gradient_from_adjoint,
    _block_parameters_bar_from_coefficient_bars_multi_rhs,
    _fixed_residual_block_coefficient_bars_multi_rhs,
    _fixed_residual_source_geometry_bars_multi_rhs,
    _direct_coefficient_geometry_bars_multi_rhs,
    _native_vmec_coefficient_bars_from_fixed_adjoint_multi_rhs,
    _directional_native_vmec_coefficient_bars_from_fixed_adjoint_multi_rhs,
    _prepared_gradient_from_adjoint,
    _prepared_implicit_vjp_primal,
)
from ntx._solver_context import _operator_context
from ntx.operators import (
    apply_nullspace_condition,
    block_parameters,
    build_block,
    coefficients_from_parameters,
)
from ntx.transport import coefficients_from_modes
from ntx._geometry_eval import (
    b2_mean_bars_multi_rhs, evaluate_fourier_series,
    fourier_series_coefficient_bars_multi_rhs, radial_drift_bars_multi_rhs,
    volume_prime_bars_multi_rhs,
)
from ntx._solver_prepared import (
    _lowdot_two_pullback_native_multi_rhs_adjoint_fields,
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


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_native_lowdot_combined_prepared_rhs_oracle_matches_compact_contract(rhs_count):
    """The future RHS-axis transpose has one exact, isolated output contract."""
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

    fields = _lowdot_two_pullback_native_multi_rhs_adjoint_fields(
        prepared,
        case,
        first_direction,
        second_direction,
        bar_fn,
    )
    (
        _primal_outputs,
        native_base_bars,
        native_first_bars,
        native_second_bars,
        native_first_bars_dot,
        native_second_bars_dot,
        _native_auxiliary,
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
        _base_nu_direct,
        _directional_nu_direct,
        _directional_nu_direct_dot,
    ) = fields
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        case.epsi_hat,
    )
    oracle_leaves = _combined_prepared_gradient_from_adjoint_multi_rhs_oracle(
        prepared,
        ctx=ctx,
        f1_full=f1_full,
        f3_full=f3_full,
        first_f1_dot=f1_dot_full[..., 0],
        first_f3_dot=f3_dot_full[..., 0],
        second_f1_dot=f1_dot_full[..., 1],
        second_f3_dot=f3_dot_full[..., 1],
        nu_dots=nu_dots,
        epsi_dots=epsi_dots,
        base_lambda1=base_lambda1,
        base_lambda3=base_lambda3,
        directional_lambda1=directional_lambda1,
        directional_lambda3=directional_lambda3,
        directional_lambda1_dot=directional_lambda1_dot,
        directional_lambda3_dot=directional_lambda3_dot,
        base_coefficient_bars=native_base_bars,
        first_coefficient_bars=native_first_bars,
        second_coefficient_bars=native_second_bars,
        first_coefficient_bars_dot=native_first_bars_dot,
        second_coefficient_bars_dot=native_second_bars_dot,
    )
    _, compact_result, _ = (
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
    compact_prepared, _compact_auxiliary = compact_result
    for actual, expected in zip(
        jax.tree_util.tree_leaves(compact_prepared), oracle_leaves, strict=True
    ):
        if jnp.issubdtype(jnp.asarray(expected).dtype, jnp.inexact):
            assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_compact_prepared_residual_transpose_matches_full_prepared_transpose(rhs_count):
    """Split residual/support transpose retains base and both low-dot terms."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
    first_direction = MonoenergeticCase(0.0, epsi_hat=0.23)
    second_direction = MonoenergeticCase(-0.17, epsi_hat=0.0)
    base_bars = jnp.reshape(
        jnp.arange(rhs_count * 5, dtype=jnp.float64), (rhs_count, 5)
    ) / 13.0 - 0.4
    first_bars = jnp.flip(base_bars, axis=1) * 0.37
    second_bars = base_bars * -0.21
    bar_fn = lambda _base, _first, _second: (
        base_bars,
        first_bars,
        second_bars,
        jnp.zeros((rhs_count,), dtype=jnp.float64),
    )
    (
        _primal_outputs,
        native_base_bars,
        native_first_bars,
        native_second_bars,
        native_first_bars_dot,
        native_second_bars_dot,
        _auxiliary,
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
        _base_nu_direct,
        _directional_nu_direct,
        _directional_nu_direct_dot,
    ) = _lowdot_two_pullback_native_multi_rhs_adjoint_fields(
        prepared,
        case,
        first_direction,
        second_direction,
        bar_fn,
    )
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        case.epsi_hat,
    )

    for rhs_index in range(rhs_count):
        base_reference = _prepared_gradient_from_adjoint(
            prepared,
            ctx,
            f1_full,
            f3_full,
            base_lambda1[..., rhs_index],
            base_lambda3[..., rhs_index],
            native_base_bars[rhs_index],
        )
        base_compact = _compact_prepared_gradient_from_adjoint(
            prepared,
            ctx,
            f1_full,
            f3_full,
            base_lambda1[..., rhs_index],
            base_lambda3[..., rhs_index],
            native_base_bars[rhs_index],
        )
        base_actual = _compact_prepared_bar_to_prepared(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            compact_bar=base_compact,
        )

        _, first_reference = _directional_prepared_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_dots[0],
            epsi_hat_dot=epsi_dots[0],
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=f1_dot_full[..., 0],
            f3_dot=f3_dot_full[..., 0],
            lambda1=directional_lambda1[..., rhs_index, 0],
            lambda3=directional_lambda3[..., rhs_index, 0],
            lambda1_dot=directional_lambda1_dot[..., rhs_index, 0],
            lambda3_dot=directional_lambda3_dot[..., rhs_index, 0],
            coefficient_bar=native_first_bars[rhs_index],
            coefficient_bar_dot=native_first_bars_dot[rhs_index],
        )
        _, first_compact = _directional_compact_prepared_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_dots[0],
            epsi_hat_dot=epsi_dots[0],
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=f1_dot_full[..., 0],
            f3_dot=f3_dot_full[..., 0],
            lambda1=directional_lambda1[..., rhs_index, 0],
            lambda3=directional_lambda3[..., rhs_index, 0],
            lambda1_dot=directional_lambda1_dot[..., rhs_index, 0],
            lambda3_dot=directional_lambda3_dot[..., rhs_index, 0],
            coefficient_bar=native_first_bars[rhs_index],
            coefficient_bar_dot=native_first_bars_dot[rhs_index],
        )
        _, second_compact = _directional_compact_prepared_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_dots[1],
            epsi_hat_dot=epsi_dots[1],
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=f1_dot_full[..., 1],
            f3_dot=f3_dot_full[..., 1],
            lambda1=directional_lambda1[..., rhs_index, 1],
            lambda3=directional_lambda3[..., rhs_index, 1],
            lambda1_dot=directional_lambda1_dot[..., rhs_index, 1],
            lambda3_dot=directional_lambda3_dot[..., rhs_index, 1],
            coefficient_bar=native_second_bars[rhs_index],
            coefficient_bar_dot=native_second_bars_dot[rhs_index],
        )
        _, second_reference = _directional_prepared_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_dots[1],
            epsi_hat_dot=epsi_dots[1],
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=f1_dot_full[..., 1],
            f3_dot=f3_dot_full[..., 1],
            lambda1=directional_lambda1[..., rhs_index, 1],
            lambda3=directional_lambda3[..., rhs_index, 1],
            lambda1_dot=directional_lambda1_dot[..., rhs_index, 1],
            lambda3_dot=directional_lambda3_dot[..., rhs_index, 1],
            coefficient_bar=native_second_bars[rhs_index],
            coefficient_bar_dot=native_second_bars_dot[rhs_index],
        )

        def _add_compact(*bars):
            def _add_leaves(*leaves):
                if any(
                    jnp.asarray(leaf).dtype == jax.dtypes.float0 for leaf in leaves
                ):
                    return leaves[0]
                return sum(leaves)

            return jax.tree_util.tree_map(
                _add_leaves,
                *bars,
            )

        combined_actual = _compact_prepared_bar_to_prepared(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            compact_bar=_add_compact(base_compact, first_compact, second_compact),
        )
        combined_reference = jax.tree_util.tree_map(
            lambda first, second, third: (
                first
                if (
                    jnp.asarray(first).dtype == jax.dtypes.float0
                    or jnp.asarray(second).dtype == jax.dtypes.float0
                    or jnp.asarray(third).dtype == jax.dtypes.float0
                )
                else first + second + third
            ),
            base_reference,
            first_reference,
            second_reference,
        )
        for actual, expected in zip(
            jax.tree_util.tree_leaves(combined_actual),
            jax.tree_util.tree_leaves(combined_reference),
            strict=True,
        ):
            if jnp.issubdtype(jnp.asarray(expected).dtype, jnp.inexact):
                assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_native_lowdot_compact_residual_contract_matches_compact_contract(rhs_count):
    """The new split transpose is exactly the existing compact native result."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    case = MonoenergeticCase(1e-2, epsi_hat=1e-3)
    first_direction = MonoenergeticCase(0.0, epsi_hat=0.23)
    second_direction = MonoenergeticCase(-0.17, epsi_hat=0.0)
    base_bars = jnp.reshape(
        jnp.arange(rhs_count * 5, dtype=jnp.float64), (rhs_count, 5)
    ) / 13.0 - 0.4
    bar_fn = lambda _base, _first, _second: (
        base_bars,
        jnp.flip(base_bars, axis=1) * 0.37,
        base_bars * -0.21,
        jnp.linspace(-0.3, 0.4, rhs_count, dtype=jnp.float64),
    )
    reference = (
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
    actual = (
        solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_residual_and_aux(
            prepared,
            case,
            first_direction,
            second_direction,
            bar_fn,
            return_primal_outputs=True,
            return_case_bars=True,
        )
    )
    for actual_tree, expected_tree in zip(actual, reference, strict=True):
        for actual_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(actual_tree),
            jax.tree_util.tree_leaves(expected_tree),
            strict=True,
        ):
            if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
                assert jnp.allclose(actual_leaf, expected_leaf, rtol=1e-9, atol=1e-11)


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


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_fixed_residual_block_coefficient_bars_multi_rhs_match_generic_vjp(rhs_count):
    """The first native RHS-axis residual transpose matches the block VJP.

    This deliberately isolates dense block coefficients from the later
    geometry/source and prepared-pytree chains.  The generic VJP is an oracle
    only; the implementation under test performs direct RHS contractions.
    """
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    n_modes = prepared.grid.n_xi + 1
    n_fs = prepared.geometry.b.size
    f1_full = jnp.reshape(
        jnp.linspace(-0.7, 0.9, n_modes * n_fs), (n_modes, n_fs)
    )
    f3_full = jnp.reshape(
        jnp.linspace(0.5, -0.8, n_modes * n_fs), (n_modes, n_fs)
    )
    lambda1 = jnp.reshape(
        jnp.linspace(-0.4, 0.6, n_modes * n_fs * rhs_count),
        (n_modes, n_fs, rhs_count),
    )
    lambda3 = jnp.flip(lambda1, axis=1) * -0.73

    actual = _fixed_residual_block_coefficient_bars_multi_rhs(
        prepared, f1_full, f3_full, lambda1, lambda3
    )

    def residual_from_blocks(lower_blocks, diagonal_blocks, upper_blocks):
        def residual(modes):
            rows = []
            for mode_index in range(n_modes):
                lower = lower_blocks[mode_index]
                diagonal = diagonal_blocks[mode_index]
                upper = upper_blocks[mode_index]
                if mode_index == 0:
                    diagonal, upper = apply_nullspace_condition(diagonal, upper)
                row = diagonal @ modes[mode_index]
                if mode_index > 0:
                    row = row + lower @ modes[mode_index - 1]
                if mode_index < n_modes - 1:
                    row = row + upper @ modes[mode_index + 1]
                rows.append(row)
            return jnp.stack(rows)

        return residual(f1_full), residual(f3_full)

    # Convert block cotangents back to the packed coefficient representation.
    # The linear build-block map is intentionally used only by the oracle.
    def one_packed_rhs(lambda1_value, lambda3_value):
        def residual_from_packed(lower_packed, diagonal_packed, upper_packed):
            lower = jax.vmap(
                lambda value: build_block(value, prepared.d_theta, prepared.d_zeta)
            )(lower_packed)
            diagonal = jax.vmap(
                lambda value: build_block(value, prepared.d_theta, prepared.d_zeta)
            )(diagonal_packed)
            upper = jax.vmap(
                lambda value: build_block(value, prepared.d_theta, prepared.d_zeta)
            )(upper_packed)
            return residual_from_blocks(lower, diagonal, upper)

        zeros = jnp.zeros((n_modes, 3, n_fs), dtype=f1_full.dtype)
        _, packed_pullback = jax.vjp(residual_from_packed, zeros, zeros, zeros)
        return packed_pullback((-lambda1_value, -lambda3_value))

    expected = jax.vmap(one_packed_rhs, in_axes=(2, 2), out_axes=1)(lambda1, lambda3)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        assert jnp.allclose(actual_leaf, expected_leaf, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_block_parameters_bar_multi_rhs_matches_coefficient_vjp(rhs_count):
    """Native coefficient-to-parameter transpose retains every RHS column."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    ctx = _operator_context(
        prepared.surface, prepared.geometry, prepared.grid, 1.1e-2, 1.7e-3
    )
    params = block_parameters(ctx)
    m, n = prepared.grid.n_xi + 1, prepared.grid.n_fs
    bars = tuple(
        jnp.reshape(jnp.linspace(-0.4 + offset, 0.6 + offset, m * rhs_count * 3 * n),
                    (m, rhs_count, 3, n))
        for offset in (0.0, 0.13, -0.21)
    )
    actual = _block_parameters_bar_from_coefficient_bars_multi_rhs(
        prepared, params, *bars
    )

    def all_coefficients(value):
        return tuple(
            jnp.stack(part) for part in zip(
                *(coefficients_from_parameters(value, k) for k in range(m)), strict=True
            )
        )

    _, pullback = jax.vjp(all_coefficients, params)
    expected = jax.vmap(
        lambda lower, diagonal, upper: pullback((lower, diagonal, upper))[0],
        in_axes=(1, 1, 1),
    )(*bars)
    for key, expected_leaf in expected.items():
        assert jnp.allclose(actual[key], expected_leaf, rtol=1e-10, atol=1e-12), key


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_fixed_residual_source_geometry_bars_multi_rhs_match_source_vjp(rhs_count):
    """Native source transpose preserves each RHS column and residual sign."""
    modes, n = 5, 6
    lambda1 = jnp.reshape(jnp.linspace(-0.4, 0.7, modes * n * rhs_count), (modes, n, rhs_count))
    lambda3 = jnp.flip(lambda1, axis=1) * -0.6
    actual_b, actual_drift = _fixed_residual_source_geometry_bars_multi_rhs(lambda1, lambda3)
    def sources(b, drift):
        s1 = jnp.zeros((modes, n), dtype=b.dtype).at[0].set(-(2.0 / 3.0) * drift).at[2].set(-(1.0 / 3.0) * drift)
        s3 = jnp.zeros((modes, n), dtype=b.dtype).at[1].set(b)
        return -s1, -s3  # residual contribution: ``A @ f - source``
    b = jnp.linspace(0.9, 1.2, n); drift = jnp.linspace(-0.3, 0.2, n)
    _, pullback = jax.vjp(sources, b, drift)
    expected_b, expected_drift = jax.vmap(lambda l1, l3: pullback((-l1, -l3)), in_axes=(2, 2))(lambda1, lambda3)
    assert jnp.allclose(actual_b, expected_b, rtol=1e-10, atol=1e-12)
    assert jnp.allclose(actual_drift, expected_drift, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_direct_coefficient_geometry_bars_multi_rhs_match_fixed_mode_vjp(rhs_count):
    """Native direct transport transpose matches the fixed-mode primitive VJP."""
    prepared = prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    g = prepared.geometry
    n = prepared.grid.n_fs
    f1 = jnp.reshape(jnp.linspace(-0.5, 0.8, 3 * n), (3, n))
    f3 = jnp.reshape(jnp.linspace(0.7, -0.4, 3 * n), (3, n))
    bars = jnp.reshape(jnp.linspace(-0.3, 0.6, rhs_count * 5), (rhs_count, 5))
    actual = _direct_coefficient_geometry_bars_multi_rhs(prepared, f1, f3, jnp.asarray(1.1e-2), bars)
    def direct(drift, jacobian, volume, psi, b, b0, nu):
        local = dataclasses.replace(g, radial_drift_spatial=drift, jacobian=jacobian,
            volume_prime=volume, coefficient_psi_scale=psi, b=b, b0=b0)
        return jnp.stack(coefficients_from_modes(local, f1, f3, nu))
    values = (g.radial_drift_spatial, g.jacobian, g.volume_prime,
              g.coefficient_psi_scale, g.b, g.b0, jnp.asarray(1.1e-2))
    _, pullback = jax.vjp(direct, *values)
    expected = jax.vmap(lambda bar: pullback(bar), in_axes=0)(bars)
    for key, expected_leaf in zip(actual, expected, strict=True):
        assert jnp.allclose(actual[key], expected_leaf, rtol=1e-10, atol=1e-12), key


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_fourier_series_coefficient_bars_multi_rhs_match_vjp(rhs_count):
    """Native Fourier transpose matches the value-and-derivative VJP."""
    m = jnp.asarray([0, 1, 2]); n = jnp.asarray([0, 1, -1])
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 4, endpoint=False)[:, None]
    zeta = jnp.linspace(0.0, 0.7, 5, endpoint=False)[None, :]
    bars = tuple(jnp.reshape(jnp.linspace(-0.3 + shift, 0.4 + shift, rhs_count * 20), (rhs_count, 4, 5)) for shift in (0.0, 0.1, -0.2))
    actual = fourier_series_coefficient_bars_multi_rhs(m, n, theta, zeta, nfp=5, value_bar=bars[0], d_theta_bar=bars[1], d_zeta_bar=bars[2])
    def forward(cosine, sine):
        return evaluate_fourier_series(m, n, cosine, theta, zeta, nfp=5, sin_coeffs=sine)
    _, pullback = jax.vjp(forward, jnp.asarray([1.0, -0.2, 0.3]), jnp.asarray([0.1, 0.4, -0.1]))
    expected = jax.vmap(lambda v, dt, dz: pullback((v, dt, dz)), in_axes=(0, 0, 0))(*bars)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        assert jnp.allclose(actual_leaf, expected_leaf, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_vmec_sampled_field_mapper_matches_field_vjps(rhs_count):
    """The VMEC wrapper preserves the Fourier transpose for every field."""
    base = dict(path=__import__("pathlib").Path("fixture.nc"), requested_psi_n=.2,
        psi_n=.2, nfp=2, ns=3, mpol=2, ntor=1, total_mode_count=2,
        loaded_mode_count=2, iota=.6, m=jnp.asarray([0,1]), n=jnp.asarray([0,1]),
        b0=1., psi_a_hat=1., phi_edge=1., r_n=.5, r_hat=.5,
        dpsi_hat_dr_hat=1., dr_hat_dpsi_hat=1., transport_psi_scale=1.)
    surface = VmecSurface(**base, b_cos=jnp.asarray([1.,.1]), jacobian_cos=jnp.asarray([1.,.02]),
        b_sub_theta_cos=jnp.asarray([.2,.01]), b_sub_zeta_cos=jnp.asarray([1.1,.03]),
        b_sup_theta_cos=jnp.asarray([.3,.04]), b_sup_zeta_cos=jnp.asarray([1.2,.05]))
    geometry = geometry_on_grid(surface, GridSpec(4, 5, 2))
    bars = {name: jnp.reshape(jnp.linspace(-.2 + i*.03, .3+i*.03, rhs_count*20), (rhs_count,4,5))
            for i, name in enumerate(("b", "d_b_dtheta", "d_b_dzeta", "jacobian", "b_sub_theta", "b_sub_zeta", "b_sup_theta", "b_sup_zeta"))}
    actual = vmec_sampled_field_bars_to_coefficients_multi_rhs(surface, geometry, bars)
    _, b_pb = jax.vjp(
        lambda c: evaluate_fourier_series(surface.m, surface.n, c, geometry.theta_2d,
                                            geometry.zeta_2d, nfp=surface.nfp),
        surface.b_cos,
    )
    b_expected = jax.vmap(
        lambda value, dtheta, dzeta: b_pb((value, dtheta, dzeta))[0]
    )(bars["b"], bars["d_b_dtheta"], bars["d_b_dzeta"])
    assert jnp.allclose(actual["b_cos"], b_expected, rtol=1e-10, atol=1e-12)
    for field, coeff in (("jacobian", "jacobian_cos"), ("b_sub_theta", "b_sub_theta_cos"), ("b_sub_zeta", "b_sub_zeta_cos"), ("b_sup_theta", "b_sup_theta_cos"), ("b_sup_zeta", "b_sup_zeta_cos")):
        _, pb = jax.vjp(lambda c: evaluate_fourier_series(surface.m, surface.n, c, geometry.theta_2d, geometry.zeta_2d, nfp=surface.nfp)[0], getattr(surface, coeff))
        expected = jax.vmap(lambda bar: pb(bar)[0])(bars[field])
        assert jnp.allclose(actual[coeff], expected, rtol=1e-10, atol=1e-12), coeff


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_native_vmec_fixed_adjoint_bars_match_generic_prepared_vjp(rhs_count):
    """Full native VMEC base pullback matches the established prepared VJP."""
    base = dict(path=__import__("pathlib").Path("fixture.nc"), requested_psi_n=.2, psi_n=.2, nfp=2, ns=3, mpol=2, ntor=1, total_mode_count=2, loaded_mode_count=2, iota=.6, m=jnp.asarray([0,1]), n=jnp.asarray([0,1]), b0=1., psi_a_hat=1., phi_edge=1., r_n=.5, r_hat=.5, dpsi_hat_dr_hat=1., dr_hat_dpsi_hat=1., transport_psi_scale=1.)
    surface = VmecSurface(**base, b_cos=jnp.asarray([1.,.1]), jacobian_cos=jnp.asarray([1.,.02]), b_sub_theta_cos=jnp.asarray([.2,.01]), b_sub_zeta_cos=jnp.asarray([1.1,.03]), b_sup_theta_cos=jnp.asarray([.3,.04]), b_sup_zeta_cos=jnp.asarray([1.2,.05]))
    prepared = prepare_monoenergetic_system(surface, GridSpec(4, 5, 2))
    ctx = _operator_context(surface, prepared.geometry, prepared.grid, .011, .002)
    modes, n = 3, prepared.grid.n_fs
    f1 = jnp.reshape(jnp.linspace(-.4,.6,modes*n),(modes,n)); f3 = jnp.reshape(jnp.linspace(.5,-.3,modes*n),(modes,n))
    l1 = jnp.reshape(jnp.linspace(-.2,.3,modes*n*rhs_count),(modes,n,rhs_count)); l3 = l1*.7
    bars = jnp.reshape(jnp.linspace(-.3,.4,rhs_count*5),(rhs_count,5))
    actual = _native_vmec_coefficient_bars_from_fixed_adjoint_multi_rhs(prepared,ctx,f1,f3,l1,l3,bars)
    def one(l1v,l3v,bar):
        compact = _compact_prepared_gradient_from_adjoint(prepared,ctx,f1,f3,l1v,l3v,bar)
        prepared_bar = _compact_prepared_bar_to_prepared(prepared,nu_hat=ctx.nu_hat,epsi_hat=ctx.epsi_hat,compact_bar=compact)
        _, surface_pullback = jax.vjp(lambda value: prepare_monoenergetic_system(value, prepared.grid), surface)
        return surface_pullback(prepared_bar)[0]
    expected = [one(l1[..., index], l3[..., index], bars[index]) for index in range(rhs_count)]
    for key in actual:
        expected_value = jnp.stack([getattr(value, key) for value in expected])
        assert jnp.allclose(actual[key], expected_value,rtol=1e-10,atol=1e-12), key


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_directional_native_vmec_fixed_adjoint_bars_match_generic_jvp(rhs_count):
    """Native lowdot VMEC transpose preserves the full prepared JVP chain.

    This is deliberately a tiny CPU gate: it compares the new explicit
    coefficient transpose, including its two directional fields, with the
    established differentiated prepared-support pullback.  It does not run a
    transport rollout or create any profile/dump output.
    """
    base = dict(path=__import__("pathlib").Path("fixture.nc"), requested_psi_n=.2, psi_n=.2, nfp=2, ns=3, mpol=2, ntor=1, total_mode_count=2, loaded_mode_count=2, iota=.6, m=jnp.asarray([0,1]), n=jnp.asarray([0,1]), b0=1., psi_a_hat=1., phi_edge=1., r_n=.5, r_hat=.5, dpsi_hat_dr_hat=1., dr_hat_dpsi_hat=1., transport_psi_scale=1.)
    surface = VmecSurface(**base, b_cos=jnp.asarray([1.,.1]), jacobian_cos=jnp.asarray([1.,.02]), b_sub_theta_cos=jnp.asarray([.2,.01]), b_sub_zeta_cos=jnp.asarray([1.1,.03]), b_sup_theta_cos=jnp.asarray([.3,.04]), b_sup_zeta_cos=jnp.asarray([1.2,.05]))
    prepared = prepare_monoenergetic_system(surface, GridSpec(4, 5, 2))
    modes, n = 3, prepared.grid.n_fs
    nu, epsi = jnp.asarray(.011), jnp.asarray(.002)
    f1 = jnp.reshape(jnp.linspace(-.4,.6,modes*n),(modes,n)); f3 = jnp.reshape(jnp.linspace(.5,-.3,modes*n),(modes,n))
    l1 = jnp.reshape(jnp.linspace(-.2,.3,modes*n*rhs_count),(modes,n,rhs_count)); l3 = l1*.7
    bars = jnp.reshape(jnp.linspace(-.3,.4,rhs_count*5),(rhs_count,5))
    nu_dot, epsi_dot = jnp.asarray(.003), jnp.asarray(-.004)
    f1_dot, f3_dot = f1 * -.11, f3 * .13
    l1_dot, l3_dot, bars_dot = l1 * .17, l3 * -.19, bars * .23
    actual, actual_dot = _directional_native_vmec_coefficient_bars_from_fixed_adjoint_multi_rhs(
        prepared, nu_hat=nu, epsi_hat=epsi, nu_hat_dot=nu_dot,
        epsi_hat_dot=epsi_dot, f1_full=f1, f3_full=f3, f1_dot=f1_dot,
        f3_dot=f3_dot, lambda1=l1, lambda3=l3, lambda1_dot=l1_dot,
        lambda3_dot=l3_dot, coefficient_bars=bars,
        coefficient_bars_dot=bars_dot,
    )
    coeff_names = ("b_cos", "jacobian_cos", "b_sub_theta_cos", "b_sub_zeta_cos", "b_sup_theta_cos", "b_sup_zeta_cos")
    def generic(nu_value, epsi_value, f1_value, f3_value, l1_value, l3_value, bar_value):
        ctx = _operator_context(surface, prepared.geometry, prepared.grid, nu_value, epsi_value)
        per_rhs = []
        for index in range(rhs_count):
            compact = _compact_prepared_gradient_from_adjoint(prepared, ctx, f1_value, f3_value, l1_value[..., index], l3_value[..., index], bar_value[index])
            prepared_bar = _compact_prepared_bar_to_prepared(prepared, nu_hat=nu_value, epsi_hat=epsi_value, compact_bar=compact)
            _, surface_pullback = jax.vjp(lambda value: prepare_monoenergetic_system(value, prepared.grid), surface)
            per_rhs.append(surface_pullback(prepared_bar)[0])
        return tuple(jnp.stack([getattr(value, name) for value in per_rhs]) for name in coeff_names)
    expected, expected_dot = jax.jvp(
        generic, (nu, epsi, f1, f3, l1, l3, bars),
        (nu_dot, epsi_dot, f1_dot, f3_dot, l1_dot, l3_dot, bars_dot),
    )
    for name, base_value, dot_value in zip(coeff_names, expected, expected_dot, strict=True):
        assert jnp.allclose(actual[name], base_value, rtol=1e-10, atol=1e-12), name
        assert jnp.allclose(actual_dot[name], dot_value, rtol=1e-10, atol=1e-12), name


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_native_lowdot_vmec_coefficient_return_matches_combined_prepared_pullback(rhs_count):
    """The opt-in lowdot return is the exact compact prepared surface chain."""
    base = dict(path=__import__("pathlib").Path("fixture.nc"), requested_psi_n=.2, psi_n=.2, nfp=2, ns=3, mpol=2, ntor=1, total_mode_count=2, loaded_mode_count=2, iota=.6, m=jnp.asarray([0,1]), n=jnp.asarray([0,1]), b0=1., psi_a_hat=1., phi_edge=1., r_n=.5, r_hat=.5, dpsi_hat_dr_hat=1., dr_hat_dpsi_hat=1., transport_psi_scale=1.)
    surface = VmecSurface(**base, b_cos=jnp.asarray([1.,.1]), jacobian_cos=jnp.asarray([1.,.02]), b_sub_theta_cos=jnp.asarray([.2,.01]), b_sub_zeta_cos=jnp.asarray([1.1,.03]), b_sup_theta_cos=jnp.asarray([.3,.04]), b_sup_zeta_cos=jnp.asarray([1.2,.05]))
    prepared = prepare_monoenergetic_system(surface, GridSpec(4, 5, 2))
    case = MonoenergeticCase(.011, epsi_hat=.002)
    first_direction = MonoenergeticCase(0., epsi_hat=.013)
    second_direction = MonoenergeticCase(.011, epsi_hat=0.)
    base_bars = jnp.reshape(jnp.linspace(-.3,.4,rhs_count * 5), (rhs_count, 5))
    first_bars, second_bars = base_bars * .21, base_bars * -.17
    base_dot, first_dot, second_dot = base_bars * .07, first_bars * -.11, second_bars * .19
    def bar_fn(_base, _first, _second):
        return base_bars, first_bars, second_bars, jnp.zeros((rhs_count,)), (first_dot, second_dot)
    _primal, (prepared_bar, _aux), native = (
        solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(
            prepared, case, first_direction, second_direction, bar_fn,
            return_primal_outputs=True, _compact_result=True,
            return_vmec_coefficient_bars=True,
        )
    )
    _, pullback = jax.vjp(
        lambda value: prepare_monoenergetic_system(value, prepared.grid), surface
    )
    def _one_rhs_prepared_bar(rhs_index):
        return jax.tree_util.tree_map(
            lambda leaf: (
                leaf[rhs_index]
                if jnp.asarray(leaf).ndim > 0
                and int(jnp.asarray(leaf).shape[0]) == rhs_count
                else leaf
            ),
            prepared_bar,
        )
    expected = tuple(
        pullback(_one_rhs_prepared_bar(rhs_index))[0]
        for rhs_index in range(rhs_count)
    )
    for name in ("b_cos", "jacobian_cos", "b_sub_theta_cos", "b_sub_zeta_cos", "b_sup_theta_cos", "b_sup_zeta_cos"):
        expected_value = jnp.stack([getattr(value, name) for value in expected])
        assert jnp.allclose(native[name], expected_value, rtol=1e-10, atol=1e-12), name


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_vmec_geometry_bar_chain_matches_geometry_vjp(rhs_count):
    """Full private VMEC primitive reverse matches ``geometry_on_grid`` VJP."""
    base = dict(path=__import__("pathlib").Path("fixture.nc"), requested_psi_n=.2, psi_n=.2, nfp=2, ns=3, mpol=2, ntor=1, total_mode_count=2, loaded_mode_count=2, iota=.6, m=jnp.asarray([0,1]), n=jnp.asarray([0,1]), b0=1., psi_a_hat=1., phi_edge=1., r_n=.5, r_hat=.5, dpsi_hat_dr_hat=1., dr_hat_dpsi_hat=1., transport_psi_scale=1.)
    surface = VmecSurface(**base, b_cos=jnp.asarray([1.,.1]), jacobian_cos=jnp.asarray([1.,.02]), b_sub_theta_cos=jnp.asarray([.2,.01]), b_sub_zeta_cos=jnp.asarray([1.1,.03]), b_sup_theta_cos=jnp.asarray([.3,.04]), b_sup_zeta_cos=jnp.asarray([1.2,.05]))
    grid = GridSpec(4, 5, 2); geometry = geometry_on_grid(surface, grid)
    names = ("b", "d_b_dtheta", "d_b_dzeta", "jacobian", "b_sub_theta", "b_sub_zeta", "b_sup_theta", "b_sup_zeta", "radial_drift_spatial")
    bars = {name: jnp.reshape(jnp.linspace(-.2+i*.02,.3+i*.02,rhs_count*20),(rhs_count,4,5)) for i,name in enumerate(names)}
    bars["volume_prime"] = jnp.linspace(-.1,.2,rhs_count); bars["b2_mean"] = jnp.linspace(.05,.15,rhs_count)
    actual = vmec_geometry_bars_to_coefficients_multi_rhs(surface, geometry, bars)
    def forward(*coeffs):
        local = dataclasses.replace(surface, b_cos=coeffs[0], jacobian_cos=coeffs[1], b_sub_theta_cos=coeffs[2], b_sub_zeta_cos=coeffs[3], b_sup_theta_cos=coeffs[4], b_sup_zeta_cos=coeffs[5])
        g = geometry_on_grid(local, grid)
        return tuple(getattr(g, name) for name in names) + (g.volume_prime, g.b2_mean)
    inputs = (surface.b_cos,surface.jacobian_cos,surface.b_sub_theta_cos,surface.b_sub_zeta_cos,surface.b_sup_theta_cos,surface.b_sup_zeta_cos)
    _, pb = jax.vjp(forward,*inputs)
    expected = jax.vmap(lambda *cot: pb(cot), in_axes=(0,)*11)(*(bars[n] for n in names),bars["volume_prime"],bars["b2_mean"])
    for key, value in zip(("b_cos","jacobian_cos","b_sub_theta_cos","b_sub_zeta_cos","b_sup_theta_cos","b_sup_zeta_cos"),expected,strict=True):
        assert jnp.allclose(actual[key], value, rtol=1e-10, atol=1e-12), key


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_radial_drift_bars_multi_rhs_match_vjp(rhs_count):
    """Native radial-drift transpose matches the pointwise VJP."""
    shape = (4, 5)
    b = jnp.ones(shape) * 1.2; dt = jnp.ones(shape) * 0.13; dz = jnp.ones(shape) * -0.07
    jac = jnp.ones(shape) * 0.9; bt = jnp.ones(shape) * 0.2; bz = jnp.ones(shape) * 0.8
    bars = jnp.reshape(jnp.linspace(-0.4, 0.5, rhs_count * 20), (rhs_count, *shape))
    actual = radial_drift_bars_multi_rhs(b, dt, dz, jac, bt, bz, bars)
    def drift(bv, dtv, dzv, jv, btv, bzv): return (btv*dzv-bzv*dtv)/(jv*bv**3)
    _, pb = jax.vjp(drift, b, dt, dz, jac, bt, bz)
    expected = jax.vmap(lambda bar: pb(bar))(bars)
    for key, value in zip(actual, expected, strict=True):
        assert jnp.allclose(actual[key], value, rtol=1e-10, atol=1e-12), key


@pytest.mark.parametrize("rhs_count", (1, 3))
def test_volume_and_b2_mean_bars_multi_rhs_match_vjp(rhs_count):
    """Native scalar geometry reductions match their batched VJPs."""
    grid = GridSpec(4, 5, 2)
    from ntx.grids import periodic_grid
    angular = periodic_grid(grid, 3)
    b = jnp.reshape(jnp.linspace(0.8, 1.3, 20), (4, 5))
    jac = jnp.reshape(jnp.linspace(0.7, 1.1, 20), (4, 5))
    volume_bars = jnp.linspace(-0.2, 0.3, rhs_count)
    b2_bars = jnp.linspace(0.1, 0.4, rhs_count)
    actual_volume = volume_prime_bars_multi_rhs(volume_bars, jac, angular)
    actual_b, actual_j = b2_mean_bars_multi_rhs(b, jac, b2_bars, angular)
    def reductions(bv, jv):
        volume = jnp.sum(jv) * angular.dtheta * angular.dzeta
        return volume, jnp.sum(bv**2*jv)*angular.dtheta*angular.dzeta/volume
    _, pb = jax.vjp(reductions, b, jac)
    expected_volume = jax.vmap(lambda bar: pb((bar, 0.0))[1])(volume_bars)
    expected_b, expected_j = jax.vmap(lambda bar: pb((0.0, bar)))(b2_bars)
    assert jnp.allclose(actual_volume, expected_volume, rtol=1e-10, atol=1e-12)
    assert jnp.allclose(actual_b, expected_b, rtol=1e-10, atol=1e-12)
    assert jnp.allclose(actual_j, expected_j, rtol=1e-10, atol=1e-12)
