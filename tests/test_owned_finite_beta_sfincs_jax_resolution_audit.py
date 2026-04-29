from __future__ import annotations

import json
from pathlib import Path

import pytest

import examples.owned_finite_beta_sfincs_jax_resolution_audit as audit


def _write_probe(
    path: Path,
    *,
    ntheta: int,
    nzeta: int,
    nxi: int,
    min_bmn: float,
    rel: dict[str, float],
) -> None:
    path.write_text(
        json.dumps(
            {
                "inputs": {
                    "grid": {"Ntheta": ntheta, "Nzeta": nzeta, "Nxi": nxi, "Nx": 1},
                    "solverTolerance": 1.0e-7,
                    "min_Bmn_to_load": min_bmn,
                },
                "decks": [
                    {
                        "rho": 1.0 / 7.0,
                        "s": (1.0 / 7.0) ** 2,
                        "nu_prime": 1.0e-2,
                        "e_star": 0.0,
                        "status": "complete",
                        "seconds": 3.0,
                        "transport_summary": {
                            "status": "complete",
                            "ntx_same_grid": {
                                "status": "complete",
                                "relative_difference": rel,
                            },
                        },
                    }
                ],
            }
        )
        + "\n"
    )


def test_owned_finite_beta_sfincs_jax_resolution_audit_builds_payload_and_figure(
    tmp_path: Path,
):
    smoke = tmp_path / "smoke.json"
    production = tmp_path / "production.json"
    tight = tmp_path / "tight.json"
    conditioning = tmp_path / "conditioning.json"
    rel_smoke = {
        "L13_bridge_vs_sfincs": 2.0e-2,
        "L31_bridge_vs_sfincs": 2.1e-2,
        "L33_bridge_vs_sfincs": 1.2e-2,
        "L33_spitzer_bridge_vs_sfincs": 4.0e-3,
    }
    rel_production = {
        "L13_bridge_vs_sfincs": 2.05e-2,
        "L31_bridge_vs_sfincs": 2.08e-2,
        "L33_bridge_vs_sfincs": 1.25e-2,
        "L33_spitzer_bridge_vs_sfincs": 4.1e-3,
    }
    rel_tight = {
        "L13_bridge_vs_sfincs": 2.04e-2,
        "L31_bridge_vs_sfincs": 2.06e-2,
        "L33_bridge_vs_sfincs": 1.25e-2,
        "L33_spitzer_bridge_vs_sfincs": 4.1e-3,
    }
    _write_probe(smoke, ntheta=25, nzeta=31, nxi=32, min_bmn=1.0e-5, rel=rel_smoke)
    _write_probe(
        production,
        ntheta=35,
        nzeta=43,
        nxi=48,
        min_bmn=1.0e-5,
        rel=rel_production,
    )
    _write_probe(tight, ntheta=35, nzeta=43, nxi=48, min_bmn=1.0e-8, rel=rel_tight)
    conditioning.write_text(
        json.dumps(
            {
                "stress_radius": {
                    "required_coefficient_relative_difference_for_current_gate": 1.0e-3
                }
            }
        )
        + "\n"
    )

    payload = audit.build_payload(
        smoke_json=smoke,
        production_json=production,
        production_tight_harmonics_json=tight,
        conditioning_json=conditioning,
    )

    assert payload["benchmark"] == "owned_finite_beta_sfincs_jax_resolution_audit"
    metrics = payload["summary_metrics"]
    assert metrics["smoke_max_transport_relative_difference"] == pytest.approx(2.1e-2)
    assert metrics["production_max_transport_relative_difference"] == pytest.approx(2.08e-2)
    assert metrics["tight_harmonics_precision_gap_to_current_gate"] == pytest.approx(20.6)
    assert payload["rows"][2]["grid"]["Ntheta"] == 35
    assert payload["rows"][2]["min_Bmn_to_load"] == 1.0e-8

    output_prefix = tmp_path / "resolution_audit"
    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)
    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()


def test_owned_finite_beta_sfincs_jax_resolution_audit_requires_matching_deck(
    tmp_path: Path,
):
    probe = tmp_path / "probe.json"
    _write_probe(
        probe,
        ntheta=25,
        nzeta=31,
        nxi=32,
        min_bmn=1.0e-5,
        rel={
            "L13_bridge_vs_sfincs": 1.0e-2,
            "L31_bridge_vs_sfincs": 1.0e-2,
            "L33_bridge_vs_sfincs": 1.0e-2,
            "L33_spitzer_bridge_vs_sfincs": 1.0e-2,
        },
    )

    with pytest.raises(ValueError, match="no completed same-grid deck"):
        audit._extract_row("bad", probe, rho=0.5, nu_prime=1.0e-2, e_star=0.0)
