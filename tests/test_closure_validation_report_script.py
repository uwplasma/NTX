from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_closure_validation_report_writes_outputs(tmp_path):
    # Into a temporary prefix, not over the committed report: regenerating in
    # place left the working tree dirty and made consecutive suite runs
    # disagree with each other.
    prefix = tmp_path / "closure_validation_report"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_closure_validation_report.py"),
            "--output-prefix",
            str(prefix),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    markdown = prefix.with_suffix(".txt").read_text(encoding="utf-8")

    assert "precise_qs" in payload
    assert "claim_scope" in payload
    assert payload["closure_decision"]["status"] == "fixed-field-stress-gate-passed"
    assert "fixed_field_diagnostics" in payload
    assert "w7x_transfer" in payload
    assert "pmax_stress" in payload
    assert any(
        "fixed-field NTX+NEOPAX total-current stress" in item
        for item in payload["claim_scope"]["positive_gates"]
    )
    assert "avoid fitted bridge constants in the shipping runtime" in (
        payload["claim_scope"]["promotion_requirements"]
    )
    assert payload["precise_qs"]["qa"]["Redl"] <= 1.0e-1
    assert payload["fixed_field_diagnostics"]["qa"]["current_total"] <= 1.0e-1
    assert payload["w7x_transfer"]["raw_branch_error"] <= 2.0e-2
    assert "Closure Validation Report" in markdown
    assert "W7-X integrated transfer" in markdown
    assert "Fixed-field closure diagnostics" in markdown
    assert "not an independent species-current parity claim" in markdown
    assert "fixed-field-stress-gate-passed" in markdown
