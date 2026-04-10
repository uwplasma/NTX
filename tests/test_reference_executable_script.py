from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ntx._checkout_paths import find_reference_executable, repo_root

ROOT = repo_root()
BENCHMARK_EXE = find_reference_executable()
DKES_FIXTURE = ROOT / "tests" / "fixtures" / "w7x_eim_sample.ddkes2.data"
VMEC_FIXTURE = ROOT / "tests" / "fixtures" / "wout_w7x_standardConfig.nc"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    library_roots = [
        value.strip()
        for value in env.get("NTX_REFERENCE_LIBRARY_PATHS", "").split(os.pathsep)
        if value.strip()
    ]
    existing = [path for path in library_roots if Path(path).exists()]
    if existing:
        env["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(existing)
    return env


@pytest.mark.benchmark
@pytest.mark.skipif(
    BENCHMARK_EXE is None or not BENCHMARK_EXE.exists(),
    reason="local benchmark executable not available",
)
def test_compare_reference_executable_script_runs(tmp_path):
    input_path = tmp_path / "run.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "dkes"',
                f'path = "{DKES_FIXTURE}"',
                "",
                "[grid]",
                "n_theta = 5",
                "n_zeta = 5",
                "n_xi = 4",
                "",
                "[case]",
                "nu_hat = 1e-5",
                "er_hat = 0.0",
                "",
                "[output]",
                'npz = "results.npz"',
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_reference_executable.py"),
            str(input_path),
            "--reference-exe",
            str(BENCHMARK_EXE),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_env(),
    )
    payload = json.loads(proc.stdout)
    assert "ntx" in payload
    assert "reference" in payload
    assert "ntx_minus_reference" in payload


@pytest.mark.benchmark
@pytest.mark.skipif(
    BENCHMARK_EXE is None or not BENCHMARK_EXE.exists(),
    reason="local benchmark executable not available",
)
def test_compare_reference_executable_script_runs_for_vmec(tmp_path):
    input_path = tmp_path / "run_vmec.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "vmec"',
                f'path = "{VMEC_FIXTURE}"',
                "psi_n = 0.25",
                "",
                "[grid]",
                "n_theta = 5",
                "n_zeta = 5",
                "n_xi = 4",
                "",
                "[case]",
                "nu_hat = 1e-3",
                "epsi_hat = 0.0",
                "",
                "[output]",
                'npz = "results.npz"',
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_reference_executable.py"),
            str(input_path),
            "--reference-exe",
            str(BENCHMARK_EXE),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_env(),
    )
    payload = json.loads(proc.stdout)
    assert "ntx" in payload
    assert "reference" in payload
    assert "ntx_minus_reference" in payload
