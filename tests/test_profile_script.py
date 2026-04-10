from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_profile_runtime_script_runs(tmp_path):
    output_json = tmp_path / "runtime.json"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "JAX_ENABLE_X64": "1"}
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "profile_runtime.py"),
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
    assert {case["name"] for case in payload["cases"]} == {"dkes_w7x_scan", "vmec_w7x_scan"}
    assert all(case["scan_steady_seconds"] > 0.0 for case in payload["cases"])


def test_profile_runtime_script_backend_mismatch_fails(tmp_path):
    output_json = tmp_path / "runtime.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "JAX_ENABLE_X64": "1",
        "JAX_PLATFORM_NAME": "cpu",
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "profile_runtime.py"),
            "--backend",
            "gpu",
            "--output-json",
            str(output_json),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert proc.returncode != 0
    assert "requested --backend=gpu" in (proc.stdout + proc.stderr)
