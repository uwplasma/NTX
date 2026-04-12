from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_make_publication_figures_subset_writes_manifest(tmp_path):
    output_dir = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "make_publication_figures.py"),
            "--output-dir",
            str(output_dir),
            "--figures",
            "validation,performance_smoke",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest_path = output_dir / "publication_figure_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(payload) == {"validation", "performance_smoke"}
    assert output_dir.joinpath("validation_summary.png").exists()
    assert output_dir.joinpath("validation_summary.pdf").exists()
    assert output_dir.joinpath("performance_scaling_smoke.png").exists()
    assert output_dir.joinpath("performance_scaling_smoke.pdf").exists()
