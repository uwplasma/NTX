from __future__ import annotations

import json
from pathlib import Path

import pytest

import examples.owned_finite_beta_sfincs_jax_production_ladder_audit as audit


def _artifact(
    path: Path,
    *,
    grid: tuple[int, int, int],
    rel_scale: float,
    seconds: float | None,
) -> None:
    decks = []
    for rho in (1.0 / 7.0, 0.3):
        for nu_prime in (1.0e-3, 1.0e-2):
            decks.append(
                {
                    "case_id": "finite_beta_fake",
                    "rho": rho,
                    "nu_prime": nu_prime,
                    "e_star": 0.0,
                    "status": "complete",
                    "seconds": seconds,
                    "transport_summary": {
                        "status": "complete",
                        "ntx_same_grid": {
                            "status": "complete",
                            "relative_difference": {
                                "L13_bridge_vs_sfincs": rel_scale * (1.0 + rho),
                                "L31_bridge_vs_sfincs": rel_scale * (1.0 + nu_prime),
                                "L33_bridge_vs_sfincs": rel_scale,
                                "L33_spitzer_bridge_vs_sfincs": 2.0 * rel_scale,
                            },
                        },
                    },
                }
            )
    path.write_text(
        json.dumps(
            {
                "inputs": {
                    "grid": {
                        "Ntheta": grid[0],
                        "Nzeta": grid[1],
                        "Nxi": grid[2],
                        "Nx": 1,
                    },
                },
                "decks": decks,
            }
        )
        + "\n"
    )


def test_owned_finite_beta_sfincs_jax_production_ladder_audit_builds_payload_and_figure(
    tmp_path: Path,
):
    smoke = tmp_path / "smoke.json"
    production = tmp_path / "production.json"
    conditioning = tmp_path / "conditioning.json"
    observable = tmp_path / "observable.json"
    _artifact(smoke, grid=(25, 31, 32), rel_scale=1.0e-2, seconds=None)
    _artifact(production, grid=(35, 43, 48), rel_scale=1.2e-2, seconds=5.0)
    conditioning.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "rho": 1.0 / 7.0,
                        "required_coefficient_relative_difference_for_current_gate": 1.0e-3,
                    },
                    {
                        "rho": 0.3,
                        "required_coefficient_relative_difference_for_current_gate": 3.0e-3,
                    },
                ]
            }
        )
        + "\n"
    )
    observable.write_text(
        json.dumps(
            {
                "rows": [
                    {"rho": 1.0 / 7.0, "relative_error_total_vs_redl": 3.0e-1},
                    {"rho": 0.3, "relative_error_total_vs_redl": 8.0e-2},
                ]
            }
        )
        + "\n"
    )

    payload = audit.build_payload(
        smoke_json=smoke,
        production_json=production,
        conditioning_json=conditioning,
        observable_json=observable,
    )

    assert payload["benchmark"] == "owned_finite_beta_sfincs_jax_production_ladder_audit"
    assert payload["summary_metrics"]["completed_production_ladder_count"] == 4
    assert payload["summary_metrics"]["production_ladder_coefficient_gate_pass"] is True
    assert (
        payload["summary_metrics"]["production_ladder_current_conditioned_gate_pass"]
        is False
    )
    assert payload["summary_metrics"]["max_production_precision_gap_to_current_gate"] > 1.0
    assert payload["stress_row"]["rho"] == pytest.approx(1.0 / 7.0)

    output_prefix = tmp_path / "audit"
    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)
    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()


def test_owned_finite_beta_sfincs_jax_production_ladder_audit_requires_rows(
    tmp_path: Path,
):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"inputs": {"grid": {}}, "decks": []}) + "\n")
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps({"rows": []}) + "\n")

    with pytest.raises(ValueError, match="no completed same-grid rows"):
        audit.build_payload(
            smoke_json=empty,
            production_json=empty,
            conditioning_json=sidecar,
            observable_json=sidecar,
        )
