from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .fixture_data import SAMPLE_BOOZMN, SAMPLE_NEOPAX, SAMPLE_WOUT

ROOT = Path(__file__).resolve().parents[1]


def _example_env() -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
    return env


def test_bootstrap_current_example_runs_with_vmec_input(tmp_path):
    output_prefix = tmp_path / "vmec_example"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "bootstrap_current_from_vmec_or_boozmn.py"),
            "--wout",
            str(SAMPLE_WOUT),
            "--skip-bootstrap",
            "--skip-sfincs",
            "--output-prefix",
            str(output_prefix),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_example_env(),
    )
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    assert output_prefix.with_suffix(".json").exists()


def test_bootstrap_current_example_runs_with_boozmn_input(tmp_path):
    output_prefix = tmp_path / "boozmn_example"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "bootstrap_current_from_vmec_or_boozmn.py"),
            "--wout",
            str(SAMPLE_WOUT),
            "--boozmn",
            str(SAMPLE_BOOZMN),
            "--reference-database",
            str(SAMPLE_NEOPAX),
            "--skip-bootstrap",
            "--skip-sfincs",
            "--output-prefix",
            str(output_prefix),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_example_env(),
    )
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    assert output_prefix.with_suffix(".json").exists()
