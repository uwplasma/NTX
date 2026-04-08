from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_input_toml(tmp_path: Path, *, verbose: bool) -> Path:
    root = Path(__file__).resolve().parents[1]
    dkes = root / "tests" / "fixtures" / "w7x_eim_sample.ddkes2.data"
    table = root / "tests" / "fixtures" / "reference_executable_reference_sample.dat"
    input_path = tmp_path / "run.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "dkes"',
                f'path = "{dkes}"',
                "",
                "[grid]",
                "n_theta = 5",
                "n_zeta = 5",
                "n_xi = 4",
                "",
                "[case]",
                "nu_hat = 1e-5",
                "er_hat = 1e-3",
                "",
                "[output]",
                'npz = "results.npz"',
                "include_modes = true",
                "",
                "[benchmark]",
                f'reference_table = "{table}"',
                "",
                "[logging]",
                f"verbose = {'true' if verbose else 'false'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return input_path


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


def test_cli_input_file_runs_and_writes_npz(tmp_path):
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    input_path = _write_input_toml(tmp_path, verbose=True)

    proc = subprocess.run(
        [sys.executable, "-m", "ntx.cli", str(input_path)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    output_npz = tmp_path / "results.npz"
    assert "NTX" in proc.stdout
    assert output_npz.exists()
    with np.load(output_npz) as data:
        assert "D11" in data
        assert "f1_modes" in data
        assert "reference_D11" in data
        assert "delta_D11" in data
