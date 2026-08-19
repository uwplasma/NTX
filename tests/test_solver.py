from __future__ import annotations

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
    pullback_prepared_coefficient_vector_case_and_prepared,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux,
    solve_prepared_coefficient_vector_vjp,
    solve_prepared_internal,
)
from ntx.geometry import BoozerSurface


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
