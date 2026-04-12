from __future__ import annotations

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
