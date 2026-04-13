from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ntx._checkout_paths import find_neopax_root

ROOT = Path(__file__).resolve().parents[1]


def _example_env() -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
    return env


@pytest.mark.skipif(find_neopax_root() is None, reason="requires local W7-X reference inputs")
def test_bootstrap_current_reference_audit_runs(tmp_path):
    output_prefix = tmp_path / "w7x_audit"
    script = (ROOT / "examples" / "bootstrap_current_reference_audit_w7x.py").read_text()
    script = script.replace(
        'OUTPUT_PREFIX = ROOT / "docs" / "_static" / "bootstrap_current_reference_audit_w7x"',
        f'OUTPUT_PREFIX = Path(r"{output_prefix}")',
    )
    run_path = tmp_path / "bootstrap_current_reference_audit_w7x.py"
    run_path.write_text(script)
    subprocess.run(
        [sys.executable, str(run_path)],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_example_env(),
    )
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    assert output_prefix.with_suffix(".json").exists()
