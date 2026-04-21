from __future__ import annotations

import json

from ntx.physics_gates import evaluate_artifact_gates, physics_gate_registry


def test_physics_gate_registry_contains_expected_gate_families():
    names = {gate.name for gate in physics_gate_registry()}
    assert "onsager_symmetry" in names
    assert "p2_projection_exact_recovery" in names
    assert "low_order_collision_block_recovery" in names
    assert "observable_map_fixed" in names
    assert "intrinsic_ambipolarity_symmetric_limit" in names
    assert "momentum_conservation_null_mode" in names
    assert "entropy_production_nonnegative" in names
    assert "w7x_integrated_rebuild_raw" in names
    assert "precise_qs_redl_vs_sfincs" in names
    assert "precise_qs_ntx_neopax_closure_stress" in names
    assert "pmax_convergence_precise_qs" in names
    assert "w7x_pmax_transfer_regression" in names


def test_evaluate_artifact_gates_reports_pass_fail_and_monitor(tmp_path):
    static_root = tmp_path / "docs" / "_static"
    static_root.mkdir(parents=True)
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

    assert results["w7x_integrated_rebuild_raw"].status == "pass"
    assert results["w7x_integrated_rebuild_raw"].value == 0.01
    assert results["precise_qs_redl_vs_sfincs"].status == "pass"
    assert results["precise_qs_redl_vs_sfincs"].value == 0.08
    assert results["precise_qs_ntx_neopax_closure_stress"].status == "monitor"
    assert results["precise_qs_ntx_neopax_closure_stress"].value == 1.1
    assert results["pmax_convergence_precise_qs"].status == "missing"
    assert results["w7x_pmax_transfer_regression"].status == "missing"
