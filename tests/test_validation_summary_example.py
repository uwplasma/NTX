from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_validation_summary_example_writes_outputs(tmp_path):
    output_prefix = tmp_path / "validation_summary"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "validation_summary.py"),
            "--output-prefix",
            str(output_prefix),
            "--n-theta",
            "7",
            "--n-zeta",
            "9",
            "--n-xi",
            "6",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    assert output_prefix.with_suffix(".json").exists()


def test_validation_summary_example_writes_benchmark_metrics(tmp_path):
    output_prefix = tmp_path / "validation_summary"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "validation_summary.py"),
            "--output-prefix",
            str(output_prefix),
            "--n-theta",
            "7",
            "--n-zeta",
            "9",
            "--n-xi",
            "6",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["benchmark"] == "monoenergetic_validation_summary"
    assert payload["classification"] == "research-grade numerical validation"
    assert len(payload["literature_anchors"]) >= 2
    assert payload["grid"] == {"n_theta": 7, "n_zeta": 9, "n_xi": 6}
    assert len(payload["nu_hat"]) == 10
    assert len(payload["transport_curves"]["dkes_surface"]["D11"]) == 10
    assert len(payload["transport_curves"]["vmec_surface"]["onsager_relative"]) == 10
    assert len(payload["legendre_convergence"]["n_xi_values"]) == 5
    assert payload["summary_metrics"]["dkes_max_onsager_relative"] >= 0.0
