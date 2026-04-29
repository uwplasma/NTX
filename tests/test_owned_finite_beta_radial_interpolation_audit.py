from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import owned_finite_beta_radial_interpolation_audit as audit


def _bootstrap_payload(
    *,
    scan_rho: tuple[float, ...],
    total: tuple[float, ...],
) -> dict:
    rho = (0.2, 0.4, 0.6)
    redl = (-10.0, -20.0, -30.0)
    return {
        "case": {"id": "finite_beta_qa_pressure_current"},
        "profile_contract": {},
        "inputs": {
            "scan_rho": list(scan_rho),
            "Es": [0.0],
            "ntx_grid": {"n_theta": 5, "n_zeta": 7, "n_xi": 9},
            "field_radial_points": 5,
            "neopax_x": 4,
            "n_order": 6,
            "d33_mode": "spitzer",
            "momentum_orders": [2, 6],
            "mboz": 3,
            "nboz": 3,
            "redl_ntheta": 32,
            "helicity_n": 0,
            "min_bmn_to_load": 1.0e-5,
            "nu_v": [1.0e-4, 1.0e-3],
        },
        "comparison": {
            "rho": list(rho),
            "redl_current_over_root_fsab2": list(redl),
            "ntx_neopax_total_over_root_fsab2": list(total),
        },
    }


def test_build_payload_from_comparisons_tracks_interpolation_sensitivity() -> None:
    baseline = _bootstrap_payload(
        scan_rho=(0.1, 0.5, 0.9),
        total=(-13.0, -30.0, -27.0),
    )
    matched = _bootstrap_payload(
        scan_rho=(0.2, 0.4, 0.6),
        total=(-10.5, -19.0, -33.0),
    )

    payload = audit.build_payload_from_comparisons(
        baseline_payload=baseline,
        matched_payload=matched,
        baseline_path=Path("baseline.json"),
        matched_path=Path("matched.json"),
    )
    metrics = payload["summary_metrics"]

    assert metrics["baseline_scan_rho_count"] == 3
    assert metrics["field_radius_matched_scan_rho_count"] == 3
    assert metrics["runtime_interpolation_policy_changed"] is False
    assert metrics["runtime_correction_applied"] is False
    assert metrics["baseline_max_relative_error_total_vs_redl"] == pytest.approx(0.5)
    assert metrics["field_radius_matched_max_relative_error_total_vs_redl"] == pytest.approx(
        0.1
    )
    assert metrics["max_relative_error_improvement_factor"] == pytest.approx(5.0)
    assert metrics["field_radius_matched_current_gate_pass"] is True
    assert payload["open_work"]


def test_build_payload_from_comparisons_requires_same_rho() -> None:
    baseline = _bootstrap_payload(
        scan_rho=(0.1, 0.5, 0.9),
        total=(-13.0, -30.0, -27.0),
    )
    matched = _bootstrap_payload(
        scan_rho=(0.2, 0.4, 0.6),
        total=(-10.5, -19.0, -33.0),
    )
    matched["comparison"]["rho"] = [0.2, 0.45, 0.6]

    with pytest.raises(ValueError, match="share rho"):
        audit.build_payload_from_comparisons(
            baseline_payload=baseline,
            matched_payload=matched,
        )


def test_write_payload_and_figure(tmp_path: Path) -> None:
    baseline = _bootstrap_payload(
        scan_rho=(0.1, 0.5, 0.9),
        total=(-13.0, -30.0, -27.0),
    )
    matched = _bootstrap_payload(
        scan_rho=(0.2, 0.4, 0.6),
        total=(-10.5, -19.0, -33.0),
    )
    payload = audit.build_payload_from_comparisons(
        baseline_payload=baseline,
        matched_payload=matched,
    )
    output_prefix = tmp_path / "owned_finite_beta_radial_interpolation_audit"

    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)

    written = json.loads(output_prefix.with_suffix(".json").read_text())
    assert written["summary_metrics"]["radius_count"] == 3
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
