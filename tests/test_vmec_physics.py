from __future__ import annotations

import jax.numpy as jnp

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic

from .fixture_data import SAMPLE_WOUT


def test_vmec_surface_solves_finite_transport():
    surface = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25)
    result = solve_monoenergetic(
        surface,
        GridSpec(9, 11, 10),
        MonoenergeticCase(nu_hat=1.0e-3, epsi_hat=0.0),
    )
    values = jnp.asarray([result.D11, result.D31, result.D13, result.D33, result.D33_spitzer])
    assert jnp.all(jnp.isfinite(values))


def test_vmec_zero_and_small_field_runs_are_close():
    surface = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25)
    zero_field = solve_monoenergetic(
        surface,
        GridSpec(9, 11, 10),
        MonoenergeticCase(nu_hat=1.0e-3, epsi_hat=0.0),
    )
    small_field = solve_monoenergetic(
        surface,
        GridSpec(9, 11, 10),
        MonoenergeticCase(nu_hat=1.0e-3, epsi_hat=1.0e-4),
    )
    zero_values = jnp.asarray([zero_field.D11, zero_field.D31, zero_field.D13, zero_field.D33])
    small_values = jnp.asarray([small_field.D11, small_field.D31, small_field.D13, small_field.D33])
    relative = jnp.abs((small_values - zero_values) / jnp.maximum(jnp.abs(zero_values), 1.0))
    assert jnp.max(relative) < 0.5
