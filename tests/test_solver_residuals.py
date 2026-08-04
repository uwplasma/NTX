"""Independent residual gates for retained and full Legendre systems."""

import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    audit_prepared_residuals,
    example_surface,
    prepare_monoenergetic_system,
    solve_prepared,
)
from ntx._solver import _operator_context
from ntx._solver import (
    _factorize_prepared_modes,
    _full_mode_residual_norm,
    _full_mode_transpose_relative_residual_norm,
    _solve_factorized_adjoint,
    _solve_factorized_modes,
)
from ntx.operators import apply_nullspace_condition, operator_blocks, source_modes


def _full_primary_solution(n_xi=6):
    surface = example_surface()
    grid = GridSpec(5, 7, n_xi)
    case = MonoenergeticCase(1.0e-2, er_hat=1.0e-3)
    prepared = prepare_monoenergetic_system(surface, grid)
    epsi_hat = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
    ctx = _operator_context(surface, prepared.geometry, grid, case.nu_hat, epsi_hat)
    source, _ = source_modes(ctx, grid.n_xi)
    factors = _factorize_prepared_modes(ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta)
    modes = _solve_factorized_modes(*factors, source)
    return prepared, case, ctx, source, modes


@pytest.mark.parametrize("n_xi", [2, 16, 32, 63])
def test_tail_eliminated_and_full_residuals_reach_roundoff(n_xi):
    prepared, case, ctx, source, modes = _full_primary_solution(n_xi)
    result = solve_prepared(prepared, case)
    audit = audit_prepared_residuals(prepared, case)
    assert float(result.residual_l2) < 1.0e-12
    assert result.schur_residual_l2 is result.residual_l2
    assert float(audit.tail_eliminated_l2) < 1.0e-12
    assert float(audit.full_system_l2) < 1.0e-12
    assert audit.schur_residual_l2 is audit.tail_eliminated_l2
    assert audit.full_system_residual_l2 is audit.full_system_l2
    assert float(audit.retained_mode_max_abs_error) < 1.0e-12
    assert audit.n_modes == prepared.grid.n_xi + 1


def test_factorized_solution_matches_materialized_dense_operator():
    prepared, _, ctx, source, modes = _full_primary_solution(4)
    block_size = prepared.grid.n_fs
    n_blocks = prepared.grid.n_xi + 1
    dense = jnp.zeros((n_blocks * block_size, n_blocks * block_size))

    for k in range(n_blocks):
        lower, diagonal, upper = operator_blocks(ctx, k, prepared.d_theta, prepared.d_zeta)
        if k == 0:
            diagonal, upper = apply_nullspace_condition(diagonal, upper)
        row = slice(k * block_size, (k + 1) * block_size)
        dense = dense.at[row, row].set(diagonal)
        if k > 0:
            previous = slice((k - 1) * block_size, k * block_size)
            dense = dense.at[row, previous].set(lower)
        if k + 1 < n_blocks:
            following = slice((k + 1) * block_size, (k + 2) * block_size)
            dense = dense.at[row, following].set(upper)

    dense_rhs = source.reshape(-1)
    dense_modes = jnp.linalg.solve(dense, dense_rhs).reshape(source.shape)
    dense_residual = jnp.linalg.norm(dense @ dense_modes.reshape(-1) - dense_rhs)
    dense_residual /= jnp.sqrt(dense_rhs.size)

    assert jnp.allclose(modes, dense_modes, rtol=1.0e-11, atol=1.0e-12)
    assert float(dense_residual) < 1.0e-12
    assert (
        float(
            _full_mode_residual_norm(
                ctx,
                prepared.grid.n_xi,
                prepared.d_theta,
                prepared.d_zeta,
                source,
                dense_modes,
            )
        )
        < 1.0e-12
    )


def test_full_residual_detects_a_perturbed_tail_mode():
    prepared, _, ctx, source, modes = _full_primary_solution()
    perturbed = modes.at[-1, 0].add(1.0e-4)
    residual = _full_mode_residual_norm(
        ctx,
        prepared.grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        source,
        perturbed,
    )
    assert float(residual) > 1.0e-7


def test_full_residual_requires_the_complete_legendre_tail():
    prepared, _, ctx, source, modes = _full_primary_solution()
    with pytest.raises(ValueError, match=r"n_xi \+ 1"):
        _full_mode_residual_norm(
            ctx,
            prepared.grid.n_xi,
            prepared.d_theta,
            prepared.d_zeta,
            source[:3],
            modes[:3],
        )


def test_transpose_residual_reaches_roundoff_and_detects_perturbation():
    prepared, _, ctx, _, _ = _full_primary_solution()
    factors = _factorize_prepared_modes(
        ctx,
        prepared.grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
    )
    source_bar = jnp.linspace(
        -1.0,
        1.0,
        (prepared.grid.n_xi + 1) * prepared.grid.n_fs,
    ).reshape((prepared.grid.n_xi + 1, prepared.grid.n_fs))
    adjoint = _solve_factorized_adjoint(*factors, source_bar)

    residual = _full_mode_transpose_relative_residual_norm(
        ctx,
        prepared.grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        source_bar,
        adjoint,
    )
    perturbed = _full_mode_transpose_relative_residual_norm(
        ctx,
        prepared.grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        source_bar,
        adjoint.at[-1, 0].add(1.0e-4),
    )
    assert float(residual) < 1.0e-11
    assert float(perturbed) > 1.0e-7
