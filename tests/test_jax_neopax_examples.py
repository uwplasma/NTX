import os
import subprocess
import sys
from pathlib import Path

import pytest

from ntx._checkout_paths import (
    find_booz_xform_jax_root,
    find_neopax_root,
    find_vmec_jax_example_input,
    find_vmec_jax_root,
    repo_root,
)

ROOT = repo_root()
VMEC_JAX_ROOT = find_vmec_jax_root()
BOOZ_XFORM_JAX_ROOT = find_booz_xform_jax_root()
NEOPAX_ROOT = find_neopax_root()
VMEC_EXAMPLE_INPUT = find_vmec_jax_example_input()

if not (
    ROOT.exists()
    and VMEC_JAX_ROOT is not None
    and BOOZ_XFORM_JAX_ROOT is not None
    and NEOPAX_ROOT is not None
    and VMEC_EXAMPLE_INPUT is not None
):
    pytest.skip("local JAX integration checkouts are not available", allow_module_level=True)

PYTHONPATH = os.pathsep.join(
    [
        str(ROOT / "src"),
        str(VMEC_JAX_ROOT),
        str(BOOZ_XFORM_JAX_ROOT),
        str(NEOPAX_ROOT),
    ]
)


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_neopax_with_ntx_example_runs():
    result = _run(ROOT / "examples" / "neopax_with_ntx.py")
    assert "D11_log shape:" in result.stdout
    assert "D33 shape:" in result.stdout


def test_vmec_jax_booz_xform_jax_ntx_example_runs():
    result = _run(
        ROOT / "examples" / "vmec_jax_booz_xform_jax_ntx.py",
        "--input",
        str(VMEC_EXAMPLE_INPUT),
        "--s",
        "0.25",
        "--mboz",
        "6",
        "--nboz",
        "0",
    )
    assert "D11:" in result.stdout
    assert "D33:" in result.stdout
