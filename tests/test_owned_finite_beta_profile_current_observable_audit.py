from __future__ import annotations

import json
from pathlib import Path

from examples import owned_finite_beta_profile_current_observable_audit as example


def test_profile_current_observable_audit_identifies_correction_amplitude(tmp_path: Path):
    bootstrap_json = tmp_path / "bootstrap.json"
    closure_json = tmp_path / "closure.json"
    bootstrap_json.write_text(
        json.dumps(
            {
                "redl": {
                    "rho": [0.25, 0.5],
                    "epsilon": [0.02, 0.05],
                    "trapped_fraction": [0.2, 0.3],
                    "L31": [0.1, 0.2],
                    "L32": [-0.01, -0.02],
                    "alpha": [-0.7, -0.6],
                    "nu_e_star": [0.4, 0.2],
                    "nu_i_star": [0.3, 0.1],
                },
                "ntx_neopax": {
                    "rho": [0.0, 0.25, 0.5, 1.0],
                    "density": [4.0, 3.5, 2.0, 1.0],
                    "temperature": [10.0, 8.0, 4.0, 1.0],
                    "A1_electron": [0.0, 1.0, 2.0, 3.0],
                    "A2_electron": [0.0, -1.0, -2.0, -3.0],
                    "L31_electron": [-4.0, -3.0, -2.0, -1.0],
                    "L32_electron": [-8.0, -6.0, -4.0, -2.0],
                    "root_fsab2": [1.0, 1.0, 1.0, 1.0],
                    "current_nomom_species": [
                        [-25.0, -20.0, -20.0, -10.0],
                        [-5.0, -10.0, -15.0, -25.0],
                    ],
                    "current_total_species": [
                        [-20.0, -5.0, -10.0, -5.0],
                        [-5.0, -10.0, -15.0, -20.0],
                    ],
                },
                "comparison": {
                    "rho": [0.25, 0.5],
                    "redl_current_over_root_fsab2": [-10.0, -20.0],
                    "ntx_neopax_nomom_over_root_fsab2": [-30.0, -35.0],
                    "ntx_neopax_total_over_root_fsab2": [-15.0, -25.0],
                    "momentum_order_scan": {
                        "2": {
                            "n_order": 2,
                            "relative_error_total_vs_redl": [1.0, 0.5],
                            "ntx_neopax_total_over_root_fsab2": [-20.0, -30.0],
                            "max_relative_error_total_vs_redl": 1.0,
                            "rms_relative_error_total_vs_redl": 0.8,
                        },
                        "4": {
                            "n_order": 4,
                            "relative_error_total_vs_redl": [0.5, 0.25],
                            "ntx_neopax_total_over_root_fsab2": [-15.0, -25.0],
                            "max_relative_error_total_vs_redl": 0.5,
                            "rms_relative_error_total_vs_redl": 0.4,
                        },
                    },
                },
            }
        )
        + "\n"
    )
    closure_json.write_text(
        json.dumps(
            {
                "summary_metrics": {
                    "inner_gap_coefficient_relative_difference": 0.02,
                    "inner_gap_bootstrap_relative_difference": 0.5,
                }
            }
        )
        + "\n"
    )

    payload = example.build_payload(
        bootstrap_json=bootstrap_json,
        closure_json=closure_json,
    )
    metrics = payload["summary_metrics"]

    assert payload["benchmark"] == "owned_finite_beta_profile_current_observable_audit"
    assert metrics["stress_rho"] == 0.25
    assert metrics["stress_relative_error_total_vs_redl"] == 0.5
    assert metrics["stress_relative_error_nomom_vs_redl"] == 2.0
    assert metrics["stress_applied_over_needed_correction"] == 0.75
    assert metrics["stress_residual_after_correction_over_needed"] == 0.25
    assert metrics["stress_species_correction_l1_over_root_fsab2"] == 15.0
    assert metrics["stress_species_correction_cancellation_amplification"] == 1.0
    assert (
        metrics["stress_residual_after_correction_over_species_correction_l1"]
        == 5.0 / 15.0
    )
    assert metrics["correction_sign_agreement_fraction"] == 1.0
    assert metrics["pmax_stress_error_monotone_nonincreasing"] is True
    assert metrics["profile_current_gate_pass"] is False
    assert payload["stress_radius"]["profile_drivers"]["trapped_fraction"] == 0.2
    assert payload["stress_radius"]["species_momentum_correction_over_root_fsab2"] == [
        15.0,
        0.0,
    ]

    output_prefix = tmp_path / "profile_current_observable_audit"
    example.write_payload(payload, output_prefix)
    example.build_figure(payload, output_prefix)
    assert json.loads(output_prefix.with_suffix(".json").read_text())["rows"]
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
