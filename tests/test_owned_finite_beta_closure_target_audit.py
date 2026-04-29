from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import owned_finite_beta_closure_target_audit as audit


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


def test_build_payload_tracks_physical_driver_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "source_response.json"
    source.write_text(json.dumps(_source_response_payload()))

    payload = audit.build_payload(source_response_json=source)
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


def test_closure_target_audit_writes_payload_and_figure(tmp_path: Path) -> None:
    source = tmp_path / "source_response.json"
    source.write_text(json.dumps(_source_response_payload()))
    output_prefix = tmp_path / "owned_finite_beta_closure_target_audit"

    payload = audit.build_payload(source_response_json=source)
    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)

    written = json.loads(output_prefix.with_suffix(".json").read_text())
    assert written["summary_metrics"]["radius_count"] == 4
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
