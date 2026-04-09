import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/rogeriojorge/local/.NTX")
PYTHONPATH = os.pathsep.join(
    [
        str(ROOT / "src"),
        "/Users/rogeriojorge/local/vmec_jax",
        "/Users/rogeriojorge/local/booz_xform_jax",
        "/Users/rogeriojorge/local/tests/NEOPAX",
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
        "/Users/rogeriojorge/local/vmec_jax/examples/data/input.circular_tokamak",
        "--s",
        "0.25",
        "--mboz",
        "6",
        "--nboz",
        "0",
    )
    assert "D11:" in result.stdout
    assert "D33:" in result.stdout
