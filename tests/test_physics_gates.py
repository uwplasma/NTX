from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import (
    BoozerSurface,
    GridSpec,
    MonoenergeticCase,
    example_surface,
    onsager_error,
    solve_monoenergetic,
)
from ntx.physics_gates import (
    PhysicsGate,
    PhysicsGateResult,
    evaluate_artifact_gates,
    physics_gate_registry,
)
from ntx.validation.physics_gates import _evaluate_scalar_gate, _gate_by_name

ROOT = Path(__file__).resolve().parents[1]


def _constant_field_surface() -> BoozerSurface:
    return BoozerSurface(
        m=jnp.asarray([0], dtype=jnp.int32),
        n=jnp.asarray([0], dtype=jnp.int32),
        b_cos=jnp.asarray([1.0]),
        nfp=5,
        iota=0.85,
        psi_p=1.0,
        chi_p=0.85,
        b_theta=0.05,
        b_zeta=1.0,
    )


def test_physics_gate_registry_contains_expected_gate_families():
    names = {gate.name for gate in physics_gate_registry()}
    assert "onsager_symmetry" in names
    assert "monoenergetic_validation_summary" in names
    assert "p2_projection_exact_recovery" in names
    assert "low_order_collision_block_recovery" in names
    assert "observable_map_fixed" in names
    assert "intrinsic_ambipolarity_symmetric_limit" in names
    assert "momentum_conservation_null_mode" in names
    assert "particle_conservation_invariant" in names
    assert "energy_conservation_invariant" in names
    assert "collision_operator_self_adjointness" in names
    assert "entropy_production_nonnegative" in names
    assert "spitzer_inverse_collisionality_limit" in names
    assert "operator_parameter_derivative_consistency" in names
    assert "imported_boozer_handedness" in names
    assert "prepared_derivative_path_consistency" in names
    assert "geometry_control_derivative_stress" in names
    assert "file_backed_geometry_control_derivative_stress" in names
    assert "boundary_forward_mode_current_derivative_stress" in names
    assert "explicit_relaxed_boundary_current_derivative_stress" in names
    assert "implicit_equilibrium_derivative_open_stress" in names
    assert "bootstrap_current_optimization_gain" in names
    assert "w7x_integrated_rebuild_raw" in names
    assert "precise_qs_redl_vs_sfincs" in names
    assert "precise_qs_ntx_neopax_closure_stress" in names
    assert "pmax_convergence_precise_qs" in names
    assert "w7x_pmax_transfer_regression" in names


def test_evaluate_artifact_gates_reports_pass_fail_and_monitor(tmp_path):
    static_root = tmp_path / "docs" / "_static"
    static_root.mkdir(parents=True)
    (static_root / "validation_summary.json").write_text(
        json.dumps(
            {
                "summary_metrics": {
                    "dkes_finest_plotted_error": 0.12,
                    "vmec_finest_plotted_error": 0.18,
                }
            }
        )
    )
    (static_root / "bootstrap_current_reference_audit_w7x.json").write_text(
        json.dumps(
            {
                "bootstrap_current_errors": [
                    {"grid": [13, 17, 17], "max_relative_error": 0.5},
                    {"grid": [25, 25, 64], "max_relative_error": 0.01},
                ]
            }
        )
    )
    (static_root / "derivative_path_benchmark.json").write_text(
        json.dumps(
            {
                "max_relative_mismatch": [2.0e-6, 3.0e-6],
                "speedup_prepared_vs_direct": [1.5, 2.0],
            }
        )
    )
    (static_root / "geometry_control_derivative_benchmark.json").write_text(
        json.dumps({"summary_metrics": {"max_relative_mismatch": 1.0e-4}})
    )
    (
        static_root / "file_backed_geometry_control_derivative_benchmark.json"
    ).write_text(json.dumps({"summary_metrics": {"max_relative_mismatch": 2.0e-4}}))
    (
        static_root / "boundary_forward_mode_current_derivative_benchmark.json"
    ).write_text(json.dumps({"summary_metrics": {"max_relative_mismatch": 5.0e-7}}))
    (
        static_root / "explicit_relaxed_boundary_current_derivative_benchmark.json"
    ).write_text(
        json.dumps(
            {
                "summary_metrics": {
                    "max_relative_mismatch": 2.0e-5,
                    "max_ordinary_explicit_volume_relative_difference": 0.0,
                }
            }
        )
    )
    (
        static_root / "implicit_equilibrium_forward_mode_derivative_benchmark.json"
    ).write_text(json.dumps({"summary_metrics": {"max_relative_mismatch": 6.0}}))
    (static_root / "bootstrap_current_optimization.json").write_text(
        json.dumps({"weighted_gain": 1.08})
    )
    (static_root / "bootstrap_current_fixed_field_validation.json").write_text(
        json.dumps(
            {
                "cases": {
                    "qa": {
                        "max_relative_error_vs_sfincs_interior": {
                            "Redl": 0.08,
                            "NTX+NEOPAX": 0.9,
                        }
                    },
                    "qh": {
                        "max_relative_error_vs_sfincs_interior": {
                            "Redl": 0.04,
                            "NTX+NEOPAX": 1.1,
                        }
                    },
                }
            }
        )
    )

    results = {result.gate.name: result for result in evaluate_artifact_gates(tmp_path)}

    assert results["monoenergetic_validation_summary"].status == "pass"
    assert results["monoenergetic_validation_summary"].value == pytest.approx(0.18)
    assert results["w7x_integrated_rebuild_raw"].status == "pass"
    assert results["w7x_integrated_rebuild_raw"].value == 0.01
    assert results["prepared_derivative_path_consistency"].status == "pass"
    assert results["prepared_derivative_path_consistency"].value == pytest.approx(3.0e-6)
    assert results["geometry_control_derivative_stress"].status == "pass"
    assert results["geometry_control_derivative_stress"].value == pytest.approx(1.0e-4)
    assert results["file_backed_geometry_control_derivative_stress"].status == "pass"
    assert results["file_backed_geometry_control_derivative_stress"].value == (
        pytest.approx(2.0e-4)
    )
    assert results["boundary_forward_mode_current_derivative_stress"].status == "pass"
    assert results["boundary_forward_mode_current_derivative_stress"].value == (
        pytest.approx(5.0e-7)
    )
    assert (
        results["explicit_relaxed_boundary_current_derivative_stress"].status == "pass"
    )
    assert results["explicit_relaxed_boundary_current_derivative_stress"].value == (
        pytest.approx(2.0e-5)
    )
    assert results["implicit_equilibrium_derivative_open_stress"].status == "monitor"
    assert results["implicit_equilibrium_derivative_open_stress"].value == (
        pytest.approx(6.0)
    )
    assert results["bootstrap_current_optimization_gain"].status == "pass"
    assert results["bootstrap_current_optimization_gain"].value == pytest.approx(1.08)
    assert results["precise_qs_redl_vs_sfincs"].status == "pass"
    assert results["precise_qs_redl_vs_sfincs"].value == 0.08
    assert results["precise_qs_ntx_neopax_closure_stress"].status == "monitor"
    assert results["precise_qs_ntx_neopax_closure_stress"].value == 1.1
    assert results["pmax_convergence_precise_qs"].status == "missing"
    assert results["w7x_pmax_transfer_regression"].status == "missing"


def test_gate_and_result_as_dict_include_optional_details():
    gate = PhysicsGate(
        name="demo_gate",
        category="stress",
        metric="demo",
        relation="monitor",
        threshold=None,
        source="unit-test",
        rationale="demo rationale",
    )
    result = PhysicsGateResult(gate=gate, value=1.25, status="monitor", details="tracked")

    assert gate.as_dict() == {
        "name": "demo_gate",
        "category": "stress",
        "metric": "demo",
        "relation": "monitor",
        "threshold": None,
        "source": "unit-test",
        "rationale": "demo rationale",
    }
    assert result.as_dict() == {
        "gate": gate.as_dict(),
        "value": 1.25,
        "status": "monitor",
        "details": "tracked",
    }


def test_evaluate_artifact_gates_reports_missing_and_convergence_monitor(tmp_path):
    static_root = tmp_path / "docs" / "_static"
    static_root.mkdir(parents=True)
    (static_root / "closure_pmax_convergence.json").write_text(
        json.dumps(
            {
                "precise_qs_max_successive_change": 0.125,
                "w7x_max_relative_error": 0.02,
            }
        )
    )

    results = {result.gate.name: result for result in evaluate_artifact_gates(tmp_path)}

    assert results["monoenergetic_validation_summary"].status == "missing"
    assert results["w7x_integrated_rebuild_raw"].status == "missing"
    assert results["prepared_derivative_path_consistency"].status == "missing"
    assert results["geometry_control_derivative_stress"].status == "missing"
    assert results["file_backed_geometry_control_derivative_stress"].status == "missing"
    assert results["boundary_forward_mode_current_derivative_stress"].status == "missing"
    assert (
        results["explicit_relaxed_boundary_current_derivative_stress"].status
        == "missing"
    )
    assert results["implicit_equilibrium_derivative_open_stress"].status == "missing"
    assert results["bootstrap_current_optimization_gain"].status == "missing"
    assert "bootstrap_current_reference_audit_w7x.json" in results[
        "w7x_integrated_rebuild_raw"
    ].details
    assert results["precise_qs_redl_vs_sfincs"].status == "missing"
    assert results["precise_qs_ntx_neopax_closure_stress"].status == "missing"
    assert results["pmax_convergence_precise_qs"].status == "monitor"
    assert results["pmax_convergence_precise_qs"].value == pytest.approx(0.125)
    assert results["w7x_pmax_transfer_regression"].status == "monitor"
    assert results["w7x_pmax_transfer_regression"].value == pytest.approx(0.02)


def test_repository_artifact_gates_match_current_claim_statuses():
    results = {result.gate.name: result for result in evaluate_artifact_gates(ROOT)}

    assert results["monoenergetic_validation_summary"].status == "pass"
    assert results["monoenergetic_validation_summary"].value <= 2.5e-1
    assert results["w7x_integrated_rebuild_raw"].status == "pass"
    assert results["w7x_integrated_rebuild_raw"].value <= 2.0e-2
    assert results["prepared_derivative_path_consistency"].status == "pass"
    assert results["prepared_derivative_path_consistency"].value <= 1.0e-4
    assert results["geometry_control_derivative_stress"].status == "pass"
    assert results["geometry_control_derivative_stress"].value <= 2.0e-4
    assert results["file_backed_geometry_control_derivative_stress"].status == "pass"
    assert results["file_backed_geometry_control_derivative_stress"].value <= 5.0e-4
    assert results["boundary_forward_mode_current_derivative_stress"].status == "pass"
    assert results["boundary_forward_mode_current_derivative_stress"].value <= 1.0e-5
    assert (
        results["explicit_relaxed_boundary_current_derivative_stress"].status == "pass"
    )
    assert (
        results["explicit_relaxed_boundary_current_derivative_stress"].value
        <= 1.0e-4
    )
    assert results["implicit_equilibrium_derivative_open_stress"].status == "monitor"
    assert results["bootstrap_current_optimization_gain"].status == "pass"
    assert results["bootstrap_current_optimization_gain"].value >= 1.0
    assert results["precise_qs_redl_vs_sfincs"].status == "pass"
    assert results["precise_qs_redl_vs_sfincs"].value <= 1.0e-1
    assert results["precise_qs_ntx_neopax_closure_stress"].status == "monitor"
    assert results["pmax_convergence_precise_qs"].status == "monitor"
    assert results["w7x_pmax_transfer_regression"].status == "monitor"


def test_owned_surface_coefficient_convergence_and_onsager_gate():
    surface = example_surface()
    case = MonoenergeticCase(nu_hat=1.0e-2, er_hat=1.0e-3)
    n_xi_values = (6, 8, 10)
    results = [
        solve_monoenergetic(surface, GridSpec(7, 7, n_xi), case)
        for n_xi in n_xi_values
    ]
    reference = results[-1]
    reference_vector = jnp.asarray([reference.D11, reference.D31, reference.D33])

    relative_errors = []
    for result in results[:-1]:
        vector = jnp.asarray([result.D11, result.D31, result.D33])
        relative_errors.append(
            jnp.abs(vector - reference_vector) / jnp.maximum(jnp.abs(reference_vector), 1.0e-30)
        )

    coarser_error, finer_error = relative_errors
    assert jnp.all(finer_error < coarser_error)
    assert float(jnp.max(finer_error)) < 2.5e-1

    onsager_relative = onsager_error(reference.D31, reference.D13) / jnp.maximum(
        jnp.maximum(jnp.abs(reference.D31), jnp.abs(reference.D13)),
        1.0e-30,
    )
    assert float(onsager_relative) < 1.0e-3

    coarse_spatial = solve_monoenergetic(surface, GridSpec(5, 5, 8), case)
    fine_spatial = solve_monoenergetic(surface, GridSpec(7, 7, 8), case)
    coarse_vector = jnp.asarray([coarse_spatial.D11, coarse_spatial.D31, coarse_spatial.D33])
    fine_vector = jnp.asarray([fine_spatial.D11, fine_spatial.D31, fine_spatial.D33])
    spatial_change = jnp.abs(coarse_vector - fine_vector) / jnp.maximum(
        jnp.abs(fine_vector),
        1.0e-30,
    )
    assert float(jnp.max(spatial_change)) < 1.0e-1


def test_constant_field_symmetric_limit_has_no_radial_transport():
    surface = _constant_field_surface()
    result = solve_monoenergetic(
        surface,
        GridSpec(5, 5, 6),
        MonoenergeticCase(nu_hat=1.0e-2, er_hat=1.0e-3),
    )

    assert result.D11 == pytest.approx(0.0, abs=1.0e-12)
    assert result.D31 == pytest.approx(0.0, abs=1.0e-12)
    assert result.D13 == pytest.approx(0.0, abs=1.0e-12)
    assert result.D33 > 0.0
    assert result.D33 == pytest.approx(result.D33_spitzer, rel=1.0e-10, abs=1.0e-12)


def test_constant_field_spitzer_branch_scales_with_inverse_collisionality():
    surface = _constant_field_surface()
    grid = GridSpec(5, 5, 6)
    low_nu = 5.0e-3
    high_nu = 2.0e-2
    low_collision = solve_monoenergetic(
        surface,
        grid,
        MonoenergeticCase(nu_hat=low_nu, er_hat=0.0),
    )
    high_collision = solve_monoenergetic(
        surface,
        grid,
        MonoenergeticCase(nu_hat=high_nu, er_hat=0.0),
    )

    assert low_collision.D33 == pytest.approx(
        low_collision.D33_spitzer,
        rel=1.0e-10,
        abs=1.0e-12,
    )
    assert high_collision.D33 == pytest.approx(
        high_collision.D33_spitzer,
        rel=1.0e-10,
        abs=1.0e-12,
    )
    assert low_collision.D33_spitzer / high_collision.D33_spitzer == pytest.approx(
        high_nu / low_nu,
        rel=1.0e-12,
    )


def test_scalar_gate_helpers_cover_fail_greater_equal_and_lookup_error():
    le_gate = _gate_by_name("w7x_integrated_rebuild_raw")
    ge_gate = PhysicsGate(
        name="ge_gate",
        category="analytical",
        metric="demo",
        relation=">=",
        threshold=2.0,
        source="unit-test",
        rationale="demo rationale",
    )
    test_gate = _gate_by_name("onsager_symmetry")

    assert _evaluate_scalar_gate(le_gate, 0.5).status == "fail"
    assert _evaluate_scalar_gate(ge_gate, 3.0).status == "pass"
    assert _evaluate_scalar_gate(ge_gate, 1.0).status == "fail"
    assert _evaluate_scalar_gate(test_gate, 0.0).status == "monitor"
    with pytest.raises(KeyError):
        _gate_by_name("missing_gate")
