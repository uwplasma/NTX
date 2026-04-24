from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_profile_parallel_runtime_script_runs(tmp_path):
    output_json = tmp_path / "parallel.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "JAX_ENABLE_X64": "1",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "JAX_PLATFORM_NAME": "cpu",
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "profile_parallel_runtime.py"),
            "--output-json",
            str(output_json),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
        timeout=180,
    )
    payload = json.loads(proc.stdout)
    assert output_json.exists()
    assert payload["local_device_count"] >= 4
    assert {case["name"] for case in payload["cases"]} == {
        "dkes_sample_parallel",
        "vmec_sample_parallel",
    }
    assert all(case["max_abs_delta_d11"] < 1e-10 for case in payload["cases"])
