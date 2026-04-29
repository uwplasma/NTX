from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples import owned_finite_beta_source_channel_audit as audit


def test_effective_source_decomposition_preserves_transport_rhs() -> None:
    projection = np.asarray(
        [
            [2.0, -3.0, 0.5],
            [1.0, 4.0, -0.25],
        ]
    )
    drives = np.asarray([7.0, -2.0, 0.0])

    effective_projection, effective_drives = audit._effective_projection_and_drives(  # noqa: SLF001
        projection,
        drives,
    )

    np.testing.assert_allclose(
        projection @ drives,
        effective_projection @ effective_drives,
    )
    assert effective_drives[0] == drives[0] + 1.5 * drives[1]
    assert effective_drives[1] == drives[1]
    assert effective_drives[2] == drives[2]


def test_source_channel_summary_tracks_high_stable_dominant_channel() -> None:
    rows = [
        {
            "neopax_x": 10,
            "n_order": 12,
            "x_to_order_ratio": 10 / 12,
            "source_channel_superposition_relative_residual": 1.0e-14,
            "full_vs_public_relative_difference": 1.0e-14,
            "public_neopax_relative_error_vs_redl": 0.31,
            "dominant_effective_channel": "effective_temperature_force",
            "effective_temperature_fraction_of_total": 0.9,
            "density_electric_fraction_of_total": 0.1,
            "parallel_electric_fraction_of_total": 0.0,
            "species_cancellation_factor": 10.0,
            "effective_temperature_response_multiplier_to_redl": 0.75,
            "effective_temperature_channel_relative_error_vs_redl": 0.33,
            "redl_effective_temperature_fraction_of_total": 1.1,
        },
        {
            "neopax_x": 18,
            "n_order": 18,
            "x_to_order_ratio": 1.0,
            "source_channel_superposition_relative_residual": 2.0e-14,
            "full_vs_public_relative_difference": 1.0e-14,
            "public_neopax_relative_error_vs_redl": 0.39,
            "dominant_effective_channel": "effective_temperature_force",
            "effective_temperature_fraction_of_total": 1.0,
            "density_electric_fraction_of_total": 1.0e-6,
            "parallel_electric_fraction_of_total": 0.0,
            "species_cancellation_factor": 82.0,
            "effective_temperature_response_multiplier_to_redl": 0.72,
            "effective_temperature_channel_relative_error_vs_redl": 0.39,
            "redl_effective_temperature_fraction_of_total": 1.0,
        },
    ]

    metrics = audit._summary_metrics(rows)  # noqa: SLF001

    assert metrics["source_channel_superposition_gate_pass"] is True
    assert metrics["best_public_neopax_x"] == 10
    assert metrics["best_public_current_gate_pass"] is False
    assert metrics["high_stable_neopax_x"] == 18
    assert (
        metrics["high_stable_dominant_effective_channel"]
        == "effective_temperature_force"
    )
    assert metrics["high_stable_species_cancellation_factor"] == 82.0
    assert metrics["high_stable_effective_temperature_response_multiplier_to_redl"] == 0.72
    assert metrics["best_effective_temperature_response_multiplier_to_redl"] == 0.75


def test_redl_effective_channel_targets_extract_density_and_temperature_terms() -> None:
    payload = {
        "redl": {
            "rho": [0.1, 0.2, 0.3],
            "density_gradient_term_over_root_fsab2": [1.0, 2.0, 3.0],
            "electron_temperature_gradient_term_over_root_fsab2": [4.0, 5.0, 6.0],
            "ion_temperature_gradient_term_over_root_fsab2": [7.0, 8.0, 9.0],
        }
    }

    targets = audit._redl_effective_channel_targets(payload, 0.2)  # noqa: SLF001

    assert targets["density_electric_force"] == 2.0
    assert targets["effective_temperature_force"] == 13.0
    assert targets["parallel_electric_force"] == 0.0


def test_load_or_build_scan_reuses_fallback_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "scan_cache"
    output_dir.mkdir()
    fallback = output_dir / "finite_beta_case_source_channel_scan.h5"
    fallback.write_bytes(b"cached")
    sentinel = object()

    monkeypatch.setattr(audit, "neopax_scan_requires_rebuild", lambda _path: False)
    monkeypatch.setattr(audit, "load_neopax_reference_scan", lambda _path: sentinel)

    def fail_build(*_args, **_kwargs):
        raise AssertionError("fallback cache should avoid rebuilding the NTX scan")

    monkeypatch.setattr(audit, "_build_scan_for_path", fail_build)

    scan, metadata = audit._load_or_build_scan(  # noqa: SLF001
        bootstrap_payload={
            "inputs": {
                "scan_rho": [0.2],
                "nu_v": [1.0e-4],
                "Es": [0.0],
            },
            "ntx_neopax": {"hdf5": {"path": str(tmp_path / "missing.h5")}},
        },
        case=SimpleNamespace(id="finite_beta_case"),
        field=SimpleNamespace(a_b=1.0),
        scan_grid=object(),
        output_dir=output_dir,
    )

    assert scan is sentinel
    assert metadata["scan_source"] == "cached_fallback_hdf5"
    assert metadata["scan_path"] == str(fallback)


def test_source_channel_audit_writes_payload_and_figure(tmp_path: Path) -> None:
    output_prefix = tmp_path / "owned_finite_beta_source_channel_audit"
    rows = []
    for neopax_x, n_order, error in ((10, 12, 0.31), (18, 18, 0.39)):
        rows.append(
            {
                "neopax_x": neopax_x,
                "n_order": n_order,
                "x_to_order_ratio": neopax_x / n_order,
                "redl_current_over_root_fsab2": -2.0e7,
                "public_neopax_current_over_root_fsab2": -2.8e7,
                "public_neopax_nomom_over_root_fsab2": 5.0e7,
                "public_neopax_relative_error_vs_redl": error,
                "source_channel_superposition_relative_residual": 1.0e-14,
                "full_vs_public_relative_difference": 1.0e-14,
                "matrix_condition_number": 1.0e3,
                "dominant_effective_channel": "effective_temperature_force",
                "effective_temperature_fraction_of_total": 1.0,
                "density_electric_fraction_of_total": 1.0e-6,
                "parallel_electric_fraction_of_total": 0.0,
                "species_cancellation_factor": 82.0,
                "source_decomposition": {
                    "effective": {
                        "current_by_channel_over_root_fsab2": {
                            "density_electric_force": -30.0,
                            "effective_temperature_force": -2.8e7,
                            "parallel_electric_force": 0.0,
                        }
                    }
                },
                "redl_effective_channel_current_by_channel_over_root_fsab2": {
                    "density_electric_force": -20.0,
                    "effective_temperature_force": -2.0e7,
                    "parallel_electric_force": 0.0,
                },
            }
        )
    payload = {
        "benchmark": "owned_finite_beta_source_channel_audit",
        "rows": rows,
        "summary_metrics": audit._summary_metrics(rows),  # noqa: SLF001
    }

    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)

    assert json.loads(output_prefix.with_suffix(".json").read_text())["rows"]
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
