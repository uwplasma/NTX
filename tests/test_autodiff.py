from __future__ import annotations

import sys
from types import ModuleType

import jax.numpy as jnp
import pytest

from ntx import (
    BootstrapOptimizationResult,
    DerivativeAuditResult,
    GridSpec,
    example_bootstrap_current_optimization,
    example_derivative_audit,
    example_inverse_problem,
    example_neopax_profile_autodiff,
    example_neopax_profile_uncertainty,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
)
from ntx._autodiff_workflows import (
    _evaluate_d11_profile,
    _evaluate_d13_profile,
    _evaluate_d33_profile,
)
from ntx._autodiff_workflows import (
    example_neopax_profile_autodiff as _example_neopax_profile_autodiff,
)
from ntx._neopax_types import NeopaxMonoenergeticArrays
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


def test_neopax_profile_uncertainty_matches_linearized_and_monte_carlo_scales():
    scan = load_neopax_reference_scan(SAMPLE_NEOPAX)
    surfaces = tuple(
        surface_from_vmec_jax_vmec_wout_file(SAMPLE_WOUT, s=float(rho_value**2))
        for rho_value in scan.rho
    )
    result = example_neopax_profile_uncertainty(
        surfaces,
        rho=scan.rho,
        nu_v=scan.nu_v,
        Es=scan.Es,
        Er=scan.Er,
        drds=scan.drds,
        grid=GridSpec(5, 5, 4),
        steps=12,
        learning_rate=0.2,
        monte_carlo_samples=32,
        random_seed=7,
    )
    assert result.parameter_covariance.shape == (2, 2)
    assert jnp.all(jnp.isfinite(result.monte_carlo_d33_std))
    assert jnp.all(result.monte_carlo_d33_std > 0.0)
    relative_std_mismatch = jnp.max(
        jnp.abs(result.linearized_d33_std - result.monte_carlo_d33_std)
        / jnp.maximum(result.monte_carlo_d33_std, 1e-30)
    )
    mean_shift = jnp.max(
        jnp.abs(result.monte_carlo_d33_mean - result.fitted_d33_profile)
        / jnp.maximum(jnp.abs(result.fitted_d33_profile), 1e-30)
    )
    assert relative_std_mismatch < 1.05
    assert mean_shift < 1e-10
    assert jnp.allclose(jnp.diag(result.parameter_correlation), 1.0)


def test_neopax_profile_autodiff_optional_import_paths():
    scan = load_neopax_reference_scan(SAMPLE_NEOPAX)
    surfaces = tuple(
        surface_from_vmec_jax_vmec_wout_file(SAMPLE_WOUT, s=float(rho_value**2))
        for rho_value in scan.rho
    )
    with pytest.raises(RuntimeError, match="requires a NEOPAX importer callback"):
        _example_neopax_profile_autodiff(
            surfaces,
            rho=scan.rho,
            nu_v=scan.nu_v,
            Es=scan.Es,
            Er=scan.Er,
            drds=scan.drds,
            grid=GridSpec(5, 5, 4),
            steps=2,
            use_neopax_package=True,
        )
    called = {"value": False}

    def fake_importer():
        called["value"] = True
        return object()

    result = _example_neopax_profile_autodiff(
        surfaces,
        rho=scan.rho,
        nu_v=scan.nu_v,
        Es=scan.Es,
        Er=scan.Er,
        drds=scan.drds,
        grid=GridSpec(5, 5, 4),
        steps=2,
        use_neopax_package=True,
        maybe_import_neopax=fake_importer,
    )
    assert called["value"] is True
    assert result.parameter_history.shape[0] == 2


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


def test_maybe_import_neopax_uses_direct_import_when_available(monkeypatch):
    fake_module = ModuleType("NEOPAX")
    monkeypatch.setitem(sys.modules, "NEOPAX", fake_module)
    imported = _maybe_import_neopax()
    assert imported is fake_module


def test_maybe_import_neopax_raises_when_no_root_is_found(monkeypatch):
    monkeypatch.delitem(sys.modules, "NEOPAX", raising=False)
    monkeypatch.setattr("ntx.autodiff.find_neopax_root", lambda: None)

    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "NEOPAX":
            raise ModuleNotFoundError("NEOPAX")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ModuleNotFoundError):
        _maybe_import_neopax()


def test_autodiff_profile_interpolants_return_finite_arrays():
    arrays = NeopaxMonoenergeticArrays(
        a_b=jnp.asarray(1.0),
        rho=jnp.asarray([0.25, 0.5, 0.75]),
        nu_log=jnp.asarray([-4.0, -3.0]),
        Er_list=jnp.asarray(
            [
                [-4.0, -3.0],
                [-4.0, -3.0],
                [-4.0, -3.0],
            ]
        ),
        D11_log=jnp.asarray(
            [
                [[-1.0, -0.8], [-0.6, -0.4]],
                [[-1.1, -0.9], [-0.7, -0.5]],
                [[-1.2, -1.0], [-0.8, -0.6]],
            ]
        ),
        D13=jnp.asarray(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.2, 0.3], [0.4, 0.5]],
                [[0.3, 0.4], [0.5, 0.6]],
            ]
        ),
        D33=jnp.asarray(
            [
                [[1.0, 1.2], [1.4, 1.6]],
                [[1.1, 1.3], [1.5, 1.7]],
                [[1.2, 1.4], [1.6, 1.8]],
            ]
        ),
    )
    rho = arrays.rho
    er_profile = jnp.asarray([1.0e-4, 2.0e-4, 3.0e-4])
    nu_value = jnp.asarray(5.0e-4)
    d11 = _evaluate_d11_profile(arrays, rho, nu_value, er_profile)
    d13 = _evaluate_d13_profile(arrays, rho, nu_value, er_profile)
    d33 = _evaluate_d33_profile(arrays, rho, nu_value, er_profile)
    assert d11.shape == rho.shape
    assert d13.shape == rho.shape
    assert d33.shape == rho.shape
    assert jnp.all(jnp.isfinite(d11))
    assert jnp.all(jnp.isfinite(d13))
    assert jnp.all(jnp.isfinite(d33))
