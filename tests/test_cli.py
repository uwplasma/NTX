from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_example_solve_runs():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ntx.cli",
            "solve",
            "--example",
            "--nu-hat",
            "1e-2",
            "--n-theta",
            "5",
            "--n-zeta",
            "5",
            "--n-xi",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    payload = json.loads(proc.stdout)
    assert "D11" in payload


def test_cli_dkes_solve_and_benchmark_runs():
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    dkes = root / "tests" / "fixtures" / "w7x_eim_sample.ddkes2.data"
    table = root / "tests" / "fixtures" / "reference_executable_reference_sample.dat"

    solve_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ntx.cli",
            "solve",
            "--dkes",
            str(dkes),
            "--nu-hat",
            "1e-5",
            "--n-theta",
            "5",
            "--n-zeta",
            "5",
            "--n-xi",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    solve_payload = json.loads(solve_proc.stdout)
    assert "D33" in solve_payload

    benchmark_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ntx.cli",
            "benchmark",
            "--dkes",
            str(dkes),
            str(table),
            "--nu-hat",
            "1e-5",
            "--er-hat",
            "1e-3",
            "--n-theta",
            "5",
            "--n-zeta",
            "5",
            "--n-xi",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    benchmark_payload = json.loads(benchmark_proc.stdout)
    assert "reference" in benchmark_payload
    assert "ntx_minus_reference" in benchmark_payload
