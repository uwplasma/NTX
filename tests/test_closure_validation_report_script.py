from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_closure_validation_report_writes_outputs():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_closure_validation_report.py"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(
        (ROOT / "docs" / "_static" / "closure_validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    markdown = (
        ROOT / "docs" / "_static" / "closure_validation_report.txt"
    ).read_text(encoding="utf-8")

    assert "precise_qs" in payload
    assert "w7x_transfer" in payload
    assert "pmax_stress" in payload
    assert payload["precise_qs"]["qa"]["Redl"] <= 1.0e-1
    assert payload["w7x_transfer"]["raw_branch_error"] <= 2.0e-2
    assert "Closure Validation Report" in markdown
    assert "W7-X integrated transfer" in markdown
