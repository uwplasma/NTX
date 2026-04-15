from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_manuscript_artifacts_script_writes_outputs():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_manuscript_artifacts.py"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(
        (ROOT / "docs" / "_static" / "manuscript_artifacts.json").read_text(encoding="utf-8")
    )
    markdown = (ROOT / "docs" / "_static" / "manuscript_tables.md").read_text(encoding="utf-8")
    claims = (ROOT / "docs" / "_static" / "manuscript_claims.md").read_text(encoding="utf-8")

    assert "validation" in payload["tables"]
    assert "performance" in payload["tables"]
    assert "main_text" in payload["figure_sets"]
    assert "Bootstrap-Current Optimization" in markdown
    assert "| Commit |" in markdown
    assert "W7-X imported-workflow bootstrap-current convergence" in claims
