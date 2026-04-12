from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_performance_scaling_example_writes_outputs(tmp_path):
    cpu_json = tmp_path / "cpu.json"
    gpu_json = tmp_path / "gpu.json"
    output_prefix = tmp_path / "performance_scaling"

    payload = {
        "results": [
            {
                "num_cases": 8,
                "serial_seconds": 1.0,
                "multiprocess_seconds": 2.0,
                "multiprocess_speedup_vs_serial": 0.5,
                "device_parallel_seconds": 1.2,
                "device_parallel_speedup_vs_serial": 0.8333333333,
            },
            {
                "num_cases": 64,
                "serial_seconds": 8.0,
                "multiprocess_seconds": 4.0,
                "multiprocess_speedup_vs_serial": 2.0,
                "device_parallel_seconds": 5.0,
                "device_parallel_speedup_vs_serial": 1.6,
            },
        ]
    }
    cpu_json.write_text(json.dumps(payload), encoding="utf-8")
    gpu_json.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "performance_scaling.py"),
            "--cpu-json",
            str(cpu_json),
            "--gpu-json",
            str(gpu_json),
            "--output-prefix",
            str(output_prefix),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
