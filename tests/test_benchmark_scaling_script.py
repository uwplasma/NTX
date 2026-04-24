from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"


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
    assert payload["max_rss_mb"] > 0.0
    assert len(payload["results"]) == 2
    assert all(
        entry["max_abs_delta_serial_vs_multiprocess_d11"] < 1e-10
        for entry in payload["results"]
    )


def test_committed_performance_scaling_artifacts_have_memory_and_correctness_metadata():
    artifacts = (
        ("performance_scaling_cpu_smoke.json", "cpu"),
        ("performance_scaling_cpu_heavy.json", "cpu"),
        ("performance_scaling_gpu_smoke.json", "gpu"),
        ("performance_scaling_gpu_heavy.json", "gpu"),
    )
    required_result_keys = {
        "device_parallel_cases_per_second",
        "device_parallel_seconds",
        "device_parallel_speedup_vs_serial",
        "max_abs_delta_serial_vs_device_parallel_d11",
        "max_abs_delta_serial_vs_multiprocess_d11",
        "multiprocess_cases_per_second",
        "multiprocess_seconds",
        "multiprocess_speedup_vs_serial",
        "num_cases",
        "serial_cases_per_second",
        "serial_seconds",
    }

    for artifact_name, expected_backend in artifacts:
        path = STATIC / artifact_name
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["backend"] == expected_backend
        assert payload["surface"] == "dkes"
        assert payload["max_rss_mb"] > 0.0
        assert payload["local_device_count"] >= payload["healthy_parallel_device_count"] >= 1
        assert payload["sizes"] == [
            entry["num_cases"] for entry in payload["results"]
        ]

        for result in payload["results"]:
            assert required_result_keys <= result.keys()
            assert result["serial_seconds"] > 0.0
            assert result["multiprocess_seconds"] > 0.0
            assert result["device_parallel_seconds"] > 0.0
            assert result["serial_cases_per_second"] > 0.0
            assert result["multiprocess_cases_per_second"] > 0.0
            assert result["device_parallel_cases_per_second"] > 0.0
            assert result["multiprocess_speedup_vs_serial"] > 0.0
            assert result["device_parallel_speedup_vs_serial"] > 0.0
            assert result["max_abs_delta_serial_vs_multiprocess_d11"] < 5.0e-8
            assert result["max_abs_delta_serial_vs_device_parallel_d11"] < 5.0e-8
