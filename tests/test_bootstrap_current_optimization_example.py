from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_current_optimization_writes_outputs(tmp_path):
    output_prefix = tmp_path / "bootstrap_current_optimization"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "bootstrap_current_optimization.py"),
            "--steps",
            "12",
            "--output-prefix",
            str(output_prefix),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["weighted_gain"] > 0.0
    assert "serial_scan_seconds" in payload
