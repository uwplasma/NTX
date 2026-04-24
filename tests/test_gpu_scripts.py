from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return env


def test_gpu_regression_script_handles_missing_gpu():
    if any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("GPU is available; this coverage only exercises the no-GPU path")
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_gpu_regression.py")],
        text=True,
        capture_output=True,
        env=_env(),
        timeout=60,
    )
    assert proc.returncode == 1
    assert "No JAX GPU device is available" in proc.stdout
