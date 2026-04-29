from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from examples import owned_finite_beta_source_response_profile_audit as audit


def _synthetic_row(
    *,
    rho: float,
    multiplier: float,
    error: float,
    reconstruction: float = 1.0e-14,
) -> dict:
    target = -multiplier * 10.0
    candidate = -10.0
    return {
        "rho": rho,
        "neopax_x": 18,
        "n_order": 18,
        "x_to_order_ratio": 1.0,
        "redl_current_over_root_fsab2": target,
        "public_neopax_current_over_root_fsab2": candidate,
        "public_neopax_nomom_over_root_fsab2": -8.0,
        "public_neopax_relative_error_vs_redl": error,
        "full_vs_public_relative_difference": 1.0e-14,
        "source_channel_superposition_relative_residual": reconstruction,
        "effective_temperature_response_multiplier_to_redl": multiplier,
        "effective_temperature_channel_relative_error_vs_redl": abs(1.0 - multiplier),
        "effective_channel_response_multiplier_to_redl": {
            "density_electric_force": 0.5 * multiplier,
            "effective_temperature_force": multiplier,
            "parallel_electric_force": None,
        },
        "redl_profile_drivers": {
            "nu_e_star": 1.0e-3 * (1.0 + rho),
            "log10_nu_e_star": float(np.log10(1.0e-3 * (1.0 + rho))),
            "trapped_fraction": 0.2 + 0.1 * rho,
            "epsilon": 0.04 + 0.02 * rho,
            "L32": 0.1 + 0.03 * rho,
        },
        "source_decomposition": {
            "effective": {
                "current_by_channel_over_root_fsab2": {
                    "density_electric_force": -1.0,
                    "effective_temperature_force": candidate,
                    "parallel_electric_force": 0.0,
                }
            }
        },
        "redl_effective_channel_current_by_channel_over_root_fsab2": {
            "density_electric_force": -0.5 * multiplier,
            "effective_temperature_force": target,
            "parallel_electric_force": 0.0,
        },
    }


def test_summary_metrics_track_profile_response_range() -> None:
    rows = [
        _synthetic_row(rho=0.15, multiplier=0.70, error=0.40),
        _synthetic_row(rho=0.35, multiplier=0.85, error=0.20),
        _synthetic_row(rho=0.55, multiplier=0.95, error=0.05),
    ]

    metrics = audit._summary_metrics(rows)  # noqa: SLF001

    assert metrics["source_channel_superposition_gate_pass"] is True
    assert metrics["radius_count"] == 3
    assert metrics["high_order_neopax_x"] == 18
    assert metrics["high_order_n_order"] == 18
    assert metrics["high_order_temperature_response_multiplier_min"] == 0.70
    assert metrics["high_order_temperature_response_multiplier_max"] == 0.95
    assert metrics["high_order_temperature_response_multiplier_span"] == (
        0.95 - 0.70
    )
    assert metrics["high_order_stress_rho"] == 0.15
    assert metrics["temperature_response_correlation_with_log10_nu_e_star"] is not None


def test_profile_response_audit_writes_payload_and_figure(tmp_path: Path) -> None:
    output_prefix = tmp_path / "owned_finite_beta_source_response_profile_audit"
    rows = [
        _synthetic_row(rho=0.15, multiplier=0.70, error=0.40),
        _synthetic_row(rho=0.35, multiplier=0.85, error=0.20),
        _synthetic_row(rho=0.55, multiplier=0.95, error=0.05),
    ]
    payload = {
        "benchmark": "owned_finite_beta_source_response_profile_audit",
        "rows": rows,
        "summary_metrics": audit._summary_metrics(rows),  # noqa: SLF001
    }

    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)

    assert json.loads(output_prefix.with_suffix(".json").read_text())["rows"]
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
