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
