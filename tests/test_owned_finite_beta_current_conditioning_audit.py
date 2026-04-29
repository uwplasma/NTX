from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import owned_finite_beta_current_conditioning_audit as example


def test_current_conditioning_audit_quantifies_required_coefficient_precision(
    tmp_path: Path,
):
    closure_json = tmp_path / "closure.json"
    observable_json = tmp_path / "observable.json"
    closure_json.write_text(
        json.dumps(
            {
                "coefficient_by_rho": [
                    {"rho": 0.25, "max_raw_relative_difference": 0.02},
                    {"rho": 0.50, "max_raw_relative_difference": 0.01},
                ]
            }
        )
        + "\n"
    )
    observable_json.write_text(
        json.dumps(
            {
                "summary_metrics": {"stress_rho": 0.25},
                "rows": [
                    {
                        "rho": 0.25,
                        "redl_current_over_root_fsab2": -10.0,
                        "relative_error_total_vs_redl": 0.30,
                        "species_momentum_correction_l1_over_root_fsab2": 200.0,
                        "applied_momentum_correction_over_root_fsab2": 20.0,
                        "residual_after_correction_over_species_correction_l1": 0.015,
                    },
                    {
                        "rho": 0.50,
                        "redl_current_over_root_fsab2": -100.0,
                        "relative_error_total_vs_redl": 0.08,
                        "species_momentum_correction_l1_over_root_fsab2": 50.0,
                        "applied_momentum_correction_over_root_fsab2": 25.0,
                        "residual_after_correction_over_species_correction_l1": 0.01,
                    },
                ],
            }
        )
        + "\n"
    )

    payload = example.build_payload(
        closure_json=closure_json,
        observable_json=observable_json,
    )
    metrics = payload["summary_metrics"]

    assert metrics["stress_current_condition_number"] == 20.0
    assert (
        metrics["stress_required_coefficient_relative_difference_for_current_gate"]
        == pytest.approx(0.005)
    )
    assert metrics["stress_coefficient_precision_gap_to_current_gate"] == 4.0
    assert (
        metrics["stress_coefficient_limited_current_relative_error_bound"]
        == pytest.approx(0.4)
    )
    assert (
        metrics["stress_observed_current_error_over_coefficient_bound"]
        == pytest.approx(0.75)
    )
    assert (
        payload["stress_radius"][
            "observed_residual_species_l1_over_coefficient_error"
        ]
        == pytest.approx(0.75)
    )

    output_prefix = tmp_path / "current_conditioning"
    example.write_payload(payload, output_prefix)
    example.build_figure(payload, output_prefix)
    assert json.loads(output_prefix.with_suffix(".json").read_text())["rows"]
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
