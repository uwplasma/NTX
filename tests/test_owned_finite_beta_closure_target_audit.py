from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import owned_finite_beta_closure_target_audit as audit
from ntx.validation._finite_beta_closure_target import (
    field_radius_matched_response_audit,
    profile_closure_target_diagnostics,
)


def _source_response_payload() -> dict:
    rows = []
    for index, rho in enumerate((0.2, 0.4, 0.6, 0.8)):
        epsilon = 0.02 + 0.05 * rho
        response = 0.75 + 4.0 * epsilon
        rows.append(
            {
                "rho": rho,
                "neopax_x": 18,
                "n_order": 18,
                "effective_temperature_response_multiplier_to_redl": response,
                "public_neopax_relative_error_vs_redl": abs(response - 1.0),
                "source_channel_superposition_relative_residual": 1.0e-14,
                "redl_profile_drivers": {
                    "epsilon": epsilon,
                    "trapped_fraction": 0.18 + 0.2 * rho,
                    "log10_nu_e_star": -1.0 + 0.1 * index,
                    "L32": -0.05 + 0.02 * index,
                },
            }
        )
    return {
        "summary_metrics": {
            "high_order_neopax_x": 18,
            "high_order_n_order": 18,
        },
        "rows": rows,
    }


def _matched_source_channel_payload() -> dict:
    rows = []
    for neopax_x, n_order, error, multiplier in (
        (10, 12, 0.21, 0.82),
        (10, 18, 0.08, 1.09),
        (18, 18, 0.31, 0.76),
    ):
        rows.append(
            {
                "rho": 0.125,
                "neopax_x": neopax_x,
                "n_order": n_order,
                "x_to_order_ratio": neopax_x / n_order,
                "public_neopax_relative_error_vs_redl": error,
                "effective_temperature_response_multiplier_to_redl": multiplier,
                "effective_temperature_channel_relative_error_vs_redl": error,
                "source_channel_superposition_relative_residual": 1.0e-14,
                "dominant_effective_channel": "effective_temperature_force",
                "species_cancellation_factor": 40.0,
            }
        )
    return {
        "rows": rows,
        "summary_metrics": {
            "best_public_neopax_x": 10,
            "best_public_n_order": 18,
            "best_public_relative_error_vs_redl": 0.08,
            "best_effective_temperature_response_multiplier_to_redl": 1.09,
            "high_stable_neopax_x": 18,
            "high_stable_n_order": 18,
            "high_stable_public_relative_error_vs_redl": 0.31,
            "high_stable_effective_temperature_response_multiplier_to_redl": 0.76,
            "source_channel_superposition_gate_pass": True,
            "max_source_channel_superposition_relative_residual": 1.0e-14,
            "high_stable_dominant_effective_channel": "effective_temperature_force",
            "high_stable_effective_temperature_fraction_of_total": 1.0,
        },
    }


def _matched_quadrature_payload() -> dict:
    rows = []
    for neopax_x, n_order, error in (
        (10, 12, 0.21),
        (10, 18, 0.08),
        (18, 18, 0.31),
    ):
        rows.append(
            {
                "neopax_x": neopax_x,
                "n_order": n_order,
                "x_to_order_ratio": neopax_x / n_order,
                "stress_relative_error_total_vs_redl": error,
            }
        )
    return {
        "rows": rows,
        "summary_metrics": {
            "stress_rho": 0.125,
            "quadrature_stable_gate_pass_count": 0,
            "quadrature_stable_current_gate_pass": False,
            "underintegrated_gate_pass_count": 1,
            "best_stress_pass_rejected_as_underintegrated": True,
            "quadrature_aliasing_detected": True,
        },
    }


def test_build_payload_tracks_physical_driver_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "source_response.json"
    source.write_text(json.dumps(_source_response_payload()))

    direct = profile_closure_target_diagnostics(_source_response_payload())
    assert direct["summary_metrics"]["radius_count"] == 4
    assert direct["summary_metrics"]["epsilon_abs_pearson"] == pytest.approx(1.0)

    payload = audit.build_payload(
        source_response_json=source,
        matched_source_channel_json=None,
        matched_quadrature_json=None,
    )
    metrics = payload["summary_metrics"]

    assert metrics["radius_count"] == 4
    assert metrics["runtime_correction_applied"] is False
    assert metrics["best_single_physics_driver"] in {
        "epsilon",
        "trapped_fraction",
        "log10_nu_e_star",
        "redl_L32",
    }
    assert metrics["epsilon_abs_pearson"] == pytest.approx(1.0)
    assert metrics["best_leave_one_out_rmse"] is not None
    assert (
        metrics["best_leave_one_out_improvement_over_constant"]
        > 1.0
    )
    assert payload["closure_requirements"]
    assert payload["field_radius_matched_response_audit"] is None


def test_build_payload_cross_links_field_radius_matched_audits(tmp_path: Path) -> None:
    source = tmp_path / "source_response.json"
    matched_source = tmp_path / "matched_source.json"
    matched_quadrature = tmp_path / "matched_quadrature.json"
    source.write_text(json.dumps(_source_response_payload()))
    matched_source.write_text(json.dumps(_matched_source_channel_payload()))
    matched_quadrature.write_text(json.dumps(_matched_quadrature_payload()))

    direct = field_radius_matched_response_audit(
        matched_source_channel_json=matched_source,
        matched_quadrature_json=matched_quadrature,
        root=tmp_path,
    )
    assert direct is not None

    payload = audit.build_payload(
        source_response_json=source,
        matched_source_channel_json=matched_source,
        matched_quadrature_json=matched_quadrature,
    )
    metrics = payload["summary_metrics"]
    matched = payload["field_radius_matched_response_audit"]

    assert direct["source_artifact"] == "matched_source.json"
    assert direct["quadrature_artifact"] == "matched_quadrature.json"
    assert matched["same_stress_radius_between_artifacts"] is True
    assert matched["matched_setting_count"] == 3
    assert matched["best_public_setting"] == {"neopax_x": 10, "n_order": 18}
    assert matched["high_stable_setting"] == {"neopax_x": 18, "n_order": 18}
    assert matched["quadrature_stable_current_gate_pass"] is False
    assert matched["best_stress_pass_rejected_as_underintegrated"] is True
    assert matched["quadrature_aliasing_detected"] is True
    assert matched["runtime_correction_applied"] is False
    assert metrics["field_radius_matched_best_public_relative_error_vs_redl"] == 0.08
    assert (
        metrics[
            "field_radius_matched_high_stable_effective_temperature_response_multiplier_to_redl"
        ]
        == 0.76
    )


def test_closure_target_audit_writes_payload_and_figure(tmp_path: Path) -> None:
    source = tmp_path / "source_response.json"
    source.write_text(json.dumps(_source_response_payload()))
    output_prefix = tmp_path / "owned_finite_beta_closure_target_audit"

    payload = audit.build_payload(
        source_response_json=source,
        matched_source_channel_json=None,
        matched_quadrature_json=None,
    )
    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)

    written = json.loads(output_prefix.with_suffix(".json").read_text())
    assert written["summary_metrics"]["radius_count"] == 4
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
