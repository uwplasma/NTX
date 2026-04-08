from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic

VMEC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wout_w7x_standardConfig.nc"


def test_vmec_zero_field_has_small_onsager_residual():
    surface = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25)
    result = solve_monoenergetic(
        surface,
        GridSpec(9, 11, 6),
        MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0),
    )
    assert result.D11 >= -1e-10
    assert result.D33_spitzer > 0.0
    assert result.onsager_residual < 5e-3


def test_vmec_resolution_change_is_stable():
    surface = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25)
    coarse = solve_monoenergetic(
        surface,
        GridSpec(7, 9, 4),
        MonoenergeticCase(nu_hat=1e-3, epsi_hat=1e-3),
    )
    fine = solve_monoenergetic(
        surface,
        GridSpec(9, 11, 6),
        MonoenergeticCase(nu_hat=1e-3, epsi_hat=1e-3),
    )
    coarse_values = jnp.asarray([coarse.D11, coarse.D31, coarse.D13, coarse.D33])
    fine_values = jnp.asarray([fine.D11, fine.D31, fine.D13, fine.D33])
    assert jnp.all(jnp.isfinite(coarse_values))
    assert jnp.all(jnp.isfinite(fine_values))
    relative = jnp.abs((fine_values - coarse_values) / fine_values)
    assert jnp.all(relative < jnp.asarray([0.4, 0.4, 0.4, 0.4]))
