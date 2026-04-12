from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_scaling_script_runs(tmp_path):
    output_json = tmp_path / "scaling.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "JAX_ENABLE_X64": "1",
        "JAX_PLATFORM_NAME": "cpu",
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_scaling.py"),
            "--backend",
            "cpu",
            "--surface",
            "dkes",
            "--sizes",
            "4,8",
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
    assert payload["backend"] == "cpu"
    assert payload["surface"] == "dkes"
    assert payload["sizes"] == [4, 8]
    assert len(payload["results"]) == 2
    assert all(
        entry["max_abs_delta_serial_vs_multiprocess_d11"] < 1e-10
        for entry in payload["results"]
    )
