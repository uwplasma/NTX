from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _example_env() -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
    return env


def test_bootstrap_current_example_runs(tmp_path):
    output_prefix = tmp_path / "bootstrap_response"
    script = (ROOT / "examples" / "bootstrap_current_from_vmec_or_boozmn.py").read_text()
    script = script.replace(
        'OUTPUT_PREFIX = ROOT / "docs" / "_static" / "bootstrap_current_from_vmec_or_boozmn"',
        f'OUTPUT_PREFIX = Path(r"{output_prefix}")',
    )
    run_path = tmp_path / "bootstrap_current_from_vmec_or_boozmn.py"
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
    payload = json.loads(output_prefix.with_suffix(".json").read_text())
    profile = payload["bootstrap_current_response"]
    assert len(profile) == 10
    assert max(abs(value) for value in profile) <= 1.0 + 1.0e-12
    assert max(abs(profile[index + 1] - profile[index]) for index in range(len(profile) - 1)) < 0.5
