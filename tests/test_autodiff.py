from __future__ import annotations

import jax.numpy as jnp

from ntx import (
    BootstrapOptimizationResult,
    GridSpec,
    example_bootstrap_current_optimization,
    example_inverse_problem,
    example_neopax_profile_autodiff,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
)

from .fixture_data import SAMPLE_NEOPAX, SAMPLE_WOUT


def test_inverse_problem_recovers_scalar_amplitude():
    result = example_inverse_problem(
        grid=GridSpec(7, 9, 6),
        target_amplitude=0.08,
        initial_amplitude=0.035,
        steps=20,
        learning_rate=0.45,
    )
    assert float(result.loss_history[-1]) < float(result.loss_history[0])
    assert jnp.isclose(result.inferred_amplitude, result.target_amplitude, rtol=5e-2, atol=5e-3)
    assert jnp.all(jnp.isfinite(result.fitted_response))


def test_neopax_profile_autodiff_reduces_profile_misfit():
    scan = load_neopax_reference_scan(SAMPLE_NEOPAX)
    surfaces = tuple(
        surface_from_vmec_jax_vmec_wout_file(SAMPLE_WOUT, s=float(rho_value**2))
        for rho_value in scan.rho
    )
    result = example_neopax_profile_autodiff(
        surfaces,
        rho=scan.rho,
        nu_v=scan.nu_v,
        Es=scan.Es,
        Er=scan.Er,
        drds=scan.drds,
        grid=GridSpec(7, 9, 6),
        steps=18,
        learning_rate=0.2,
    )
    assert float(result.loss_history[-1]) < float(result.loss_history[0])
    assert result.sensitivity_matrix.shape[1] == 2
    assert jnp.all(jnp.isfinite(result.fitted_d33_profile))


def test_bootstrap_current_optimization_improves_weighted_objective():
    scan = load_neopax_reference_scan(SAMPLE_NEOPAX)
    surfaces = tuple(
        surface_from_vmec_jax_vmec_wout_file(SAMPLE_WOUT, s=float(rho_value**2))
        for rho_value in scan.rho
    )
    result = example_bootstrap_current_optimization(
        surfaces,
        rho=scan.rho,
        nu_v=scan.nu_v,
        Es=scan.Es,
        Er=scan.Er,
        drds=scan.drds,
        grid=GridSpec(7, 9, 6),
        steps=10,
        learning_rate=1.2,
        regularization=1.0,
    )
    assert isinstance(result, BootstrapOptimizationResult)
    assert float(result.objective_history[-1]) >= float(result.objective_history[0])
    assert not jnp.isclose(result.optimized_scale, result.baseline_scale)
    assert jnp.all(jnp.isfinite(result.optimized_current_profile))
