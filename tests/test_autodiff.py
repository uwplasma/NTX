from __future__ import annotations

import sys
from types import ModuleType

import jax.numpy as jnp

from ntx import (
    BootstrapOptimizationResult,
    DerivativeAuditResult,
    GridSpec,
    example_bootstrap_current_optimization,
    example_derivative_audit,
    example_inverse_problem,
    example_neopax_profile_autodiff,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
)
from ntx.autodiff import _maybe_import_neopax

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


def test_derivative_audit_matches_finite_difference():
    result = example_derivative_audit(grid=GridSpec(7, 9, 6))
    assert isinstance(result, DerivativeAuditResult)
    amplitude_d11_error = jnp.max(
        jnp.abs(result.autodiff_d11_da - result.finite_difference_d11_da)
        / jnp.maximum(jnp.abs(result.finite_difference_d11_da), 1e-30)
    )
    amplitude_d33_error = jnp.max(
        jnp.abs(result.autodiff_d33_da - result.finite_difference_d33_da)
        / jnp.maximum(jnp.abs(result.finite_difference_d33_da), 1e-30)
    )
    er_d11_error = jnp.max(
        jnp.abs(result.autodiff_d11_der - result.finite_difference_d11_der)
        / jnp.maximum(jnp.abs(result.finite_difference_d11_der), 1e-30)
    )
    er_d33_error = jnp.max(
        jnp.abs(result.autodiff_d33_der - result.finite_difference_d33_der)
        / jnp.maximum(jnp.abs(result.finite_difference_d33_der), 1e-30)
    )
    assert amplitude_d11_error < 5e-2
    assert amplitude_d33_error < 5e-2
    assert er_d11_error < 5e-2
    assert er_d33_error < 5e-2


def test_maybe_import_neopax_uses_sys_path_fallback(monkeypatch, tmp_path):
    root = tmp_path / "NEOPAX"
    root.mkdir()
    fake_module = ModuleType("NEOPAX")
    monkeypatch.setattr("ntx.autodiff.find_neopax_root", lambda: root)
    monkeypatch.delitem(sys.modules, "NEOPAX", raising=False)

    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "NEOPAX":
            if str(root) not in sys.path:
                raise ModuleNotFoundError("NEOPAX")
            return fake_module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    imported = _maybe_import_neopax()
    assert imported is fake_module
