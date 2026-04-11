from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_profile_multiprocess_runtime_script_runs(tmp_path):
    output_json = tmp_path / "multiprocess.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "JAX_ENABLE_X64": "1",
        "JAX_PLATFORM_NAME": "cpu",
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "profile_multiprocess_runtime.py"),
            "--backend",
            "cpu",
            "--workers",
            "2",
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
    assert payload["requested_backend"] == "cpu"
    assert payload["workers"] == 2
    assert {case["name"] for case in payload["cases"]} == {
        "dkes_sample_multiprocess",
        "vmec_sample_multiprocess",
    }
    assert all(case["max_abs_delta_d11"] < 1e-10 for case in payload["cases"])
