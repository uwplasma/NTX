from __future__ import annotations

import inspect

import ntx
import ntx.core as ntx_core
import ntx.validation as ntx_validation
import ntx.workflows as ntx_workflows


def test_top_level_public_exports_are_unique():
    assert len(ntx.__all__) == len(set(ntx.__all__))


def test_top_level_public_callables_are_documented():
    undocumented = [
        name
        for name in ntx.__all__
        if (inspect.isfunction(getattr(ntx, name)) or inspect.isclass(getattr(ntx, name)))
        and not inspect.getdoc(getattr(ntx, name))
    ]

    assert undocumented == []


def test_core_namespace_preserves_flat_solver_api():
    assert ntx_core.MonoenergeticCase is ntx.MonoenergeticCase
    assert ntx_core.solve_monoenergetic is ntx.solve_monoenergetic
    assert ntx_core.solve_monoenergetic_scan is ntx.solve_monoenergetic_scan
    assert ntx_core.onsager_error is ntx.onsager_error


def test_workflow_namespace_preserves_flat_workflow_api():
    assert ntx_workflows.NeopaxScan is ntx.NeopaxScan
    assert ntx_workflows.build_ntx_neopax_scan is ntx.build_ntx_neopax_scan
    assert ntx_workflows.get_differentiable_neopax_fluxes is ntx.get_differentiable_neopax_fluxes
    assert ntx_workflows.solve_ambipolar_er_profile is ntx.solve_ambipolar_er_profile
    assert ntx_workflows.example_derivative_audit is ntx.example_derivative_audit


def test_validation_namespace_preserves_flat_validation_api():
    assert ntx_validation.PhysicsGate is ntx.PhysicsGate
    assert ntx_validation.physics_gate_registry is ntx.physics_gate_registry
    assert ntx_validation.evaluate_artifact_gates is ntx.evaluate_artifact_gates
