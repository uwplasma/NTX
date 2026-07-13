"""Independent residual gates for retained and full Legendre systems."""

import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    audit_prepared_residuals,
    example_surface,
    prepare_monoenergetic_system,
    solve_prepared,
)
from ntx._solver_context import _operator_context
from ntx._solver_factorization import (
    _factorize_prepared_modes,
    _full_mode_residual_norm,
    _solve_factorized_modes,
)
from ntx.operators import source_modes


def _full_primary_solution():
    surface = example_surface()
    grid = GridSpec(5, 7, 6)
    case = MonoenergeticCase(1.0e-2, er_hat=1.0e-3)
    prepared = prepare_monoenergetic_system(surface, grid)
    epsi_hat = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
    ctx = _operator_context(surface, prepared.geometry, grid, case.nu_hat, epsi_hat)
    source, _ = source_modes(ctx, grid.n_xi)
    factors = _factorize_prepared_modes(
        ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta
    )
    modes = _solve_factorized_modes(*factors, source)
    return prepared, case, ctx, source, modes


def test_tail_eliminated_and_full_residuals_reach_roundoff():
    prepared, case, ctx, source, modes = _full_primary_solution()
    result = solve_prepared(prepared, case)
    audit = audit_prepared_residuals(prepared, case)
    assert float(result.residual_l2) < 1.0e-12
    assert float(audit.tail_eliminated_l2) < 1.0e-12
    assert float(audit.full_system_l2) < 1.0e-12
    assert float(audit.retained_mode_max_abs_error) < 1.0e-12
    assert audit.n_modes == prepared.grid.n_xi + 1


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
