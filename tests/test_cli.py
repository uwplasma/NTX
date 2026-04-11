from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.fixture_data import SAMPLE_DKES, SAMPLE_WOUT


def _write_input_toml(tmp_path: Path, *, verbose: bool) -> Path:
    input_path = tmp_path / "run.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "dkes"',
                f'path = "{SAMPLE_DKES}"',
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
                "[logging]",
                f"verbose = {'true' if verbose else 'false'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return input_path


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return env


def test_cli_example_solve_runs():
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
        env=_env(),
    )
    payload = json.loads(proc.stdout)
    assert "D11" in payload


def test_cli_dkes_solve_runs():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ntx.cli",
            "solve",
            "--dkes",
            str(SAMPLE_DKES),
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
        env=_env(),
    )
    payload = json.loads(proc.stdout)
    assert "D33" in payload


def test_cli_vmec_solve_runs():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ntx.cli",
            "solve",
            "--vmec",
            str(SAMPLE_WOUT),
            "--psi-n",
            "0.25",
            "--nu-hat",
            "1e-3",
            "--epsi-hat",
            "1e-3",
            "--n-theta",
            "7",
            "--n-zeta",
            "9",
            "--n-xi",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_env(),
    )
    payload = json.loads(proc.stdout)
    assert "D33" in payload


def test_cli_vmec_er_hat_solve_runs():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ntx.cli",
            "solve",
            "--vmec",
            str(SAMPLE_WOUT),
            "--psi-n",
            "0.25",
            "--nu-hat",
            "1e-3",
            "--er-hat",
            "1e-3",
            "--n-theta",
            "7",
            "--n-zeta",
            "9",
            "--n-xi",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_env(),
    )
    payload = json.loads(proc.stdout)
    assert "D11" in payload
    assert payload["D33"] > 0.0


def test_cli_input_file_runs_and_writes_npz(tmp_path):
    input_path = _write_input_toml(tmp_path, verbose=True)
    proc = subprocess.run(
        [sys.executable, "-m", "ntx.cli", str(input_path)],
        check=True,
        text=True,
        capture_output=True,
        env=_env(),
    )

    output_npz = tmp_path / "results.npz"
    assert "NTX" in proc.stdout
    assert output_npz.exists()
    with np.load(output_npz) as data:
        assert "D11" in data
        assert "f1_modes" in data
        assert "b" in data
        assert "epsi_hat_resolved" in data
