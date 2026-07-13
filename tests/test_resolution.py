"""Fourier geometry sampling gates."""

import jax
import jax.numpy as jnp
import pytest

from ntx import (
    BoozerSurface,
    GridSpec,
    MonoenergeticCase,
    geometry_resolution_report,
    solve_monoenergetic,
)


def _surface_with_modes(m, n):
    count = len(m)
    return BoozerSurface(
        m=jnp.asarray(m),
        n=jnp.asarray(n),
        b_cos=jnp.linspace(1.0, 0.01, count),
        nfp=3,
        iota=0.7,
        psi_p=1.0,
        b_theta=0.1,
        b_zeta=1.0,
    )


def test_resolution_report_uses_one_field_period_mode_numbers():
    surface = _surface_with_modes([0, 3, -2], [0, -4, 2])
    report = geometry_resolution_report(surface, GridSpec(9, 11, 4))
    assert (report.m_min, report.m_max) == (-2, 3)
    assert (report.n_min, report.n_max) == (-4, 2)
    assert report.theta_nyquist_floor == 7
    assert report.zeta_nyquist_floor == 9
    assert report.resolved
    assert report.errors == ()


def test_resolution_report_warns_near_nyquist():
    report = geometry_resolution_report(
        _surface_with_modes([0, 2], [0, 2]), GridSpec(5, 5, 4)
    )
    assert report.resolved
    assert len(report.warnings) == 2


def test_undersampled_geometry_fails_before_solve():
    surface = _surface_with_modes([0, 3], [0, 4])
    grid = GridSpec(5, 7, 4)
    report = geometry_resolution_report(surface, grid)
    assert report.status == "undersampled"
    assert len(report.errors) == 2
    with pytest.raises(ValueError, match="Nyquist floor"):
        solve_monoenergetic(
            surface,
            grid,
            MonoenergeticCase(1.0e-2),
            require_resolved_geometry=True,
        )


def test_resolution_report_rejects_invalid_warning_floor():
    with pytest.raises(ValueError, match="at least 1"):
        geometry_resolution_report(
            _surface_with_modes([0], [0]),
            GridSpec(3, 3, 2),
            warning_oversampling=0.5,
        )


def test_jitted_surface_keeps_existing_transform_contract():
    surface = _surface_with_modes([0, 1], [0, 1])
    grid = GridSpec(5, 5, 4)
    solve = jax.jit(
        lambda traced_surface: solve_monoenergetic(
            traced_surface, grid, MonoenergeticCase(1.0e-2)
        ).D11
    )
    assert jnp.isfinite(solve(surface))
