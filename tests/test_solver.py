from __future__ import annotations

import jax
import jax.numpy as jnp

from ntx import (
    GridSpec,
    MonoenergeticCase,
    compile_prepared_solver,
    example_surface,
    prepare_monoenergetic_system,
    solve_monoenergetic,
    solve_monoenergetic_internal,
    solve_monoenergetic_scan,
    solve_prepared,
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
        lambda surf, er_hat: solve_monoenergetic(
            surf, grid, MonoenergeticCase(1e-2, er_hat=er_hat)
        ).D11
    )
    value = solve_d11(surface, 1e-3)
    assert jnp.isfinite(value)


def test_compiled_prepared_solver_is_differentiable_in_er_hat():
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    prepared = prepare_monoenergetic_system(surface, grid)
    compiled = compile_prepared_solver(prepared)

    grad = jax.grad(
        lambda er_hat: compiled(MonoenergeticCase(1e-2, er_hat=er_hat)).D11
    )(1e-3)
    assert jnp.isfinite(grad)
