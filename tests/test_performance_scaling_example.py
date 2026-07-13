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
    summary = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["artifact"] == "performance_scaling_summary"
    assert summary["cpu"]["multiprocess_crossover_cases"] == 64
    assert summary["gpu"]["device_parallel_crossover_cases"] == 64


def test_performance_strong_scaling_example_writes_outputs(tmp_path):
    cpu_json = tmp_path / "cpu_strong.json"
    gpu_json = tmp_path / "gpu_strong.json"
    output_prefix = tmp_path / "performance_strong_scaling"

    payload = {
        "artifact": "strong_scaling_benchmark",
        "backend": "cpu",
        "surface": "dkes",
        "grid": {"n_theta": 9, "n_zeta": 11, "n_xi": 6},
        "num_cases": 16,
        "local_device_count": 2,
        "healthy_parallel_device_count": 2,
        "max_rss_mb": 123.0,
        "serial": {"seconds": 4.0, "cases_per_second": 4.0},
        "device_parallel": [
            {
                "requested_device_count": 1,
                "effective_device_count": 1,
                "seconds": 4.0,
                "speedup_vs_serial": 1.0,
                "max_abs_delta_serial_d11": 0.0,
            },
            {
                "requested_device_count": 2,
                "effective_device_count": 2,
                "seconds": 2.5,
                "speedup_vs_serial": 1.6,
                "max_abs_delta_serial_d11": 0.0,
            },
        ],
        "multiprocess": [
            {
                "workers": 1,
                "seconds": 5.0,
                "speedup_vs_serial": 0.8,
                "max_abs_delta_serial_d11": 0.0,
            },
            {
                "workers": 2,
                "seconds": 3.0,
                "speedup_vs_serial": 1.3333333333,
                "max_abs_delta_serial_d11": 0.0,
            },
        ],
    }
    cpu_json.write_text(json.dumps(payload), encoding="utf-8")
    gpu_json.write_text(json.dumps({**payload, "backend": "gpu"}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "performance_strong_scaling.py"),
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
    summary = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["artifact"] == "strong_scaling_summary"
    assert summary["cpu"]["best_device_parallel_speedup_vs_serial"] == 1.6
    assert summary["gpu"]["best_multiprocess_speedup_vs_serial"] == 1.3333333333


def test_prepared_scan_benchmark_and_plot_write_synchronized_artifacts(tmp_path):
    cpu_json = tmp_path / "prepared_cpu.json"
    output_prefix = tmp_path / "prepared_scan_performance"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_prepared_scan.py"),
            "--backend",
            "cpu",
            "--sizes",
            "1,2",
            "--modes",
            "sequential:1,vectorized:1",
            "--repeats",
            "1",
            "--n-theta",
            "5",
            "--n-zeta",
            "5",
            "--n-xi",
            "4",
            "--output-json",
            str(cpu_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout
    payload = json.loads(cpu_json.read_text(encoding="utf-8"))
    assert payload["artifact"] == "prepared_scan_performance"
    assert payload["warmup"]["sequential-1"]["temporary_size_bytes"] >= 0
    assert payload["results"][1]["num_cases"] == 2

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "prepared_scan_performance.py"),
            "--cpu-json",
            str(cpu_json),
            "--gpu-json",
            str(cpu_json),
            "--output-prefix",
            str(output_prefix),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
