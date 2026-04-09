from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/rogeriojorge/local/.NTX")


def test_qi_neopax_with_ntx_example_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "qi_neopax_with_ntx.py"),
            "--output",
            str(tmp_path / "qi_scan.h5"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "D11 shape:" in result.stdout
    assert "output:" in result.stdout
