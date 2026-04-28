from __future__ import annotations

import json
from pathlib import Path

from examples import owned_finite_beta_closure_localization as example


def test_closure_localization_excludes_sidecar_and_matches_inner_gap(tmp_path: Path):
    sfincs_json = tmp_path / "sfincs.json"
    bootstrap_json = tmp_path / "bootstrap.json"

    sfincs_json.write_text(
        json.dumps(
            {
                "decks": [
                    {
                        "case_id": "finite_beta_fake",
                        "case_label": "Finite-beta fake",
                        "rho": 0.25,
                        "nu_prime": 1.0e-3,
                        "e_star": 0.0,
                        "transport_summary": {
                            "status": "complete",
                            "ntx_same_grid": {
                                "status": "complete",
                                "relative_difference": {
                                    "L13_bridge_vs_sfincs": 0.01,
                                    "L31_bridge_vs_sfincs": 0.02,
                                    "L33_bridge_vs_sfincs": 0.03,
                                    "L33_spitzer_bridge_vs_sfincs": 0.50,
                                },
                            },
                        },
                    },
                    {
                        "case_id": "finite_beta_fake",
                        "case_label": "Finite-beta fake",
                        "rho": 0.50,
                        "nu_prime": 1.0e-2,
                        "e_star": 0.0,
                        "transport_summary": {
                            "status": "complete",
                            "ntx_same_grid": {
                                "status": "complete",
                                "relative_difference": {
                                    "L13_bridge_vs_sfincs": 0.02,
                                    "L31_bridge_vs_sfincs": 0.04,
                                    "L33_bridge_vs_sfincs": 0.01,
                                    "L33_spitzer_bridge_vs_sfincs": 0.40,
                                },
                            },
                        },
                    },
                ]
            }
        )
        + "\n"
    )
    bootstrap_json.write_text(
        json.dumps(
            {
                "comparison": {
                    "rho": [0.25, 0.50, 0.75],
                    "relative_error_total_vs_redl": [0.30, 0.08, 0.06],
                    "relative_error_nomom_vs_redl": [1.2, 0.9, 0.7],
                    "redl_current_over_root_fsab2": [-1.0, -2.0, -3.0],
                    "ntx_neopax_total_over_root_fsab2": [-1.3, -2.16, -3.18],
                    "momentum_order_scan": {
                        "2": {
                            "n_order": 2,
                            "max_relative_error_total_vs_redl": 0.5,
                            "rms_relative_error_total_vs_redl": 0.3,
                        },
                        "4": {
                            "n_order": 4,
                            "max_relative_error_total_vs_redl": 0.3,
                            "rms_relative_error_total_vs_redl": 0.2,
                        },
                    },
                }
            }
        )
        + "\n"
    )

    payload = example.build_payload(
        sfincs_json=sfincs_json,
        bootstrap_json=bootstrap_json,
    )

    metrics = payload["summary_metrics"]
    assert metrics["max_same_grid_coefficient_relative_difference"] == 0.04
    assert metrics["max_bootstrap_total_relative_difference"] == 0.30
    assert metrics["inner_gap_rho"] == 0.25
    assert metrics["inner_gap_coefficient_relative_difference"] == 0.03
    assert metrics["inner_gap_current_to_coefficient_error_ratio"] == 10.0
    assert metrics["coefficient_gate_pass"] is True
    assert metrics["profile_current_gate_pass"] is False
    assert payload["coefficient_by_rho"][0]["max_sidecar_relative_difference"] == 0.50

    output_prefix = tmp_path / "closure_localization"
    example.write_payload(payload, output_prefix)
    example.build_figure(payload, output_prefix)
    assert json.loads(output_prefix.with_suffix(".json").read_text())["matched_radii"]
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
