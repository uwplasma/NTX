from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prepared_geometry_reuse_profile_writes_artifacts(tmp_path):
    output_prefix = tmp_path / "prepared_geometry_reuse_profile"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "prepared_geometry_reuse_profile.py"),
            "--preset",
            "smoke",
            "--case-counts",
            "2",
            "--output-prefix",
            str(output_prefix),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    json_path = output_prefix.with_suffix(".json")
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "prepared_geometry_reuse_profile"
    assert payload["summary_metrics"]["max_prepared_relative_mismatch"] == 0.0
    assert payload["summary_metrics"]["max_compiled_relative_mismatch"] < 1.0e-7
    assert payload["results"][0]["compiled_steady_speedup_vs_direct"] > 1.0
