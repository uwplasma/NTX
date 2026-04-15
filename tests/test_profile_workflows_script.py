from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_profile_workflows_script_runs(tmp_path):
    output_json = tmp_path / "workflow-profile.json"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "JAX_ENABLE_X64": "1"}
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "profile_workflows.py"),
            "--surface",
            "vmec",
            "--output-json",
            str(output_json),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    payload = json.loads(proc.stdout)
    assert output_json.exists()
    assert payload["surface"] == "vmec"
    assert payload["prepare_monoenergetic_system_seconds"] > 0.0
    assert payload["scan_steady_seconds"] > 0.0
    assert payload["prepared_vector_steady_seconds"] > 0.0
    assert payload["native_bootstrap_seconds"] > 0.0
    assert payload["native_bootstrap_num_radii"] == 2
