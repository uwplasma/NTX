from __future__ import annotations

import json
from pathlib import Path

from examples import owned_finite_beta_closure_quadrature_audit as audit


def _synthetic_bootstrap_payload() -> dict:
    return {
        "inputs": {
            "neopax_x": 10,
            "n_order": 12,
        },
        "comparison": {
            "rho": [0.25, 0.5],
            "relative_error_total_vs_redl": [0.31, 0.05],
        },
        "summary_metrics": {
            "max_relative_error_total_vs_redl_interior": 0.31,
        },
    }


def _synthetic_payload() -> dict:
    rows = [
        {
            "neopax_x": 10,
            "n_order": 12,
            "x_to_order_ratio": 10 / 12,
            "stress_relative_error_total_vs_redl": 0.31,
            "max_relative_error_total_vs_redl": 0.31,
            "closure_seconds": 1.0,
        },
        {
            "neopax_x": 10,
            "n_order": 14,
            "x_to_order_ratio": 10 / 14,
            "stress_relative_error_total_vs_redl": 0.04,
            "max_relative_error_total_vs_redl": 0.20,
            "closure_seconds": 1.1,
        },
        {
            "neopax_x": 18,
            "n_order": 12,
            "x_to_order_ratio": 18 / 12,
            "stress_relative_error_total_vs_redl": 0.40,
            "max_relative_error_total_vs_redl": 0.40,
            "closure_seconds": 1.2,
        },
        {
            "neopax_x": 18,
            "n_order": 14,
            "x_to_order_ratio": 18 / 14,
            "stress_relative_error_total_vs_redl": 0.39,
            "max_relative_error_total_vs_redl": 0.39,
            "closure_seconds": 1.3,
        },
    ]
    metrics = audit._summary_metrics(  # noqa: SLF001
        rows,
        bootstrap_payload=_synthetic_bootstrap_payload(),
    )
    return {
        "benchmark": "owned_finite_beta_closure_quadrature_audit",
        "rows": rows,
        "summary_metrics": metrics,
    }


def test_closure_quadrature_summary_flags_underintegrated_pass() -> None:
    payload = _synthetic_payload()
    metrics = payload["summary_metrics"]

    assert metrics["min_stress_gate_pass"] is True
    assert metrics["min_stress_neopax_x"] == 10
    assert metrics["min_stress_n_order"] == 14
    assert metrics["underintegrated_gate_pass_count"] == 1
    assert metrics["quadrature_aliasing_detected"] is True
    assert metrics["high_x_stress_error_monotone_nonincreasing_with_pmax"] is True
    assert metrics["reference_stress_relative_error"] == 0.31


def test_closure_quadrature_audit_writes_payload_and_figure(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    output_prefix = tmp_path / "closure_quadrature_audit"

    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)

    assert json.loads(output_prefix.with_suffix(".json").read_text())["rows"]
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
