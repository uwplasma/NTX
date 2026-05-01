from __future__ import annotations

import json
from pathlib import Path

import pytest

import examples.owned_finite_beta_sfincs_jax_profile_current_resolution_audit as audit


def _write_payload(root: Path, n_xi: int, current: float, *, converged: bool = True) -> None:
    path = root / f"Nxi_{n_xi}" / "payload.json"
    path.parent.mkdir(parents=True)
    payload = {
        "inputs": {
            "grid": {"n_theta": 17, "n_zeta": 21, "n_xi": n_xi, "nx": 5},
            "collision_operator": 1,
        },
        "decks": [
            {
                "rho": 1.0 / 7.0,
                "nu_n": 8.31565e-3,
                "status": "complete",
                "current_summary": {
                    "current_over_root_fsab2_am2": current,
                    "comparison": {
                        "redl_current_over_root_fsab2": -20.0,
                        "ntx_neopax_current_over_root_fsab2": -25.0,
                        "sfincs_jax_relative_error_vs_redl": abs(current + 20.0)
                        / 20.0,
                        "sfincs_jax_relative_error_vs_ntx_neopax": abs(current + 25.0)
                        / 25.0,
                        "ntx_neopax_relative_error_vs_redl": 0.25,
                    },
                    "solver": {
                        "linearSolverMethod": "sparse_pc_gmres",
                        "true_residual_gate_pass": converged,
                        "true_residual_over_target": 1.0e-6,
                    },
                },
            }
        ],
    }
    path.write_text(json.dumps(payload))


def test_profile_current_resolution_audit_summarizes_nxi_scan(tmp_path: Path):
    for n_xi, current in ((18, -20.0), (20, -22.0), (22, -24.0), (19, -31.0)):
        _write_payload(tmp_path, n_xi, current)

    payload = audit.build_payload(tmp_path)

    assert payload["benchmark"] == (
        "owned_finite_beta_sfincs_jax_profile_current_resolution_audit"
    )
    assert payload["summary_metrics"]["row_count"] == 4
    assert payload["summary_metrics"]["solver_converged_count"] == 4
    assert payload["summary_metrics"]["best_sfincs_jax_relative_error_vs_redl"] == 0.0
    assert payload["summary_metrics"]["ntx_neopax_gate_pass_count"] >= 1
    assert payload["summary_metrics"]["tail_even_odd_relative_gap"] == pytest.approx(
        abs(-31.0 - (-22.0)) / 31.0
    )


def test_profile_current_resolution_audit_writes_figure(tmp_path: Path):
    _write_payload(tmp_path, 18, -20.0)
    _write_payload(tmp_path, 19, -31.0, converged=False)
    payload = audit.build_payload(tmp_path)
    output_prefix = tmp_path / "resolution_audit"

    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)

    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    written = json.loads(output_prefix.with_suffix(".json").read_text())
    assert written["figure_png"] == str(output_prefix.with_suffix(".png"))
    assert written["figure_pdf"] == str(output_prefix.with_suffix(".pdf"))
