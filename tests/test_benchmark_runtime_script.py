from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_benchmark_reference_executable_script_cpu_skip_mode(tmp_path):
    output_path = tmp_path / "benchmark.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_reference_executable.py"),
            "--case",
            "w7x_eim_smoke",
            "--platform",
            "cpu",
            "--skip-reference",
            "--disable-preallocate",
            "--output-json",
            str(output_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_env(),
    )
    payload = json.loads(proc.stdout)
    assert payload["platform"] == "cpu"
    assert payload["mode"] == "eager"
    assert payload["xla_preallocate"] == "false"
    case = payload["cases"][0]
    assert case["case"]["name"] == "w7x_eim_smoke"
    assert case["ntx"]["max_gpu_memory_mib"] == 0
    assert case["ntx"]["first_run"]["coefficients"]["D11"] > 0.0
    assert output_path.exists()
