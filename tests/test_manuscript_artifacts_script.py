from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_manuscript_artifacts_script_writes_outputs():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_manuscript_artifacts.py"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(
        (ROOT / "docs" / "_static" / "manuscript_artifacts.json").read_text(encoding="utf-8")
    )
    markdown = (ROOT / "docs" / "_static" / "manuscript_tables.md").read_text(encoding="utf-8")
    claims = (ROOT / "docs" / "_static" / "manuscript_claims.md").read_text(encoding="utf-8")

    assert "monoenergetic_validation" in payload["tables"]
    assert "fixed_field_validation" in payload["tables"]
    assert "validation" in payload["tables"]
    assert "geometry_control_derivatives" in payload["tables"]
    assert "file_backed_geometry_control_derivatives" in payload["tables"]
    assert "implicit_equilibrium_forward_mode_derivatives" in payload["tables"]
    assert "explicit_relaxed_boundary_current_derivatives" in payload["tables"]
    assert "performance" in payload["tables"]
    assert "benchmark_matrix" in payload
    assert payload["benchmark_matrix"]["summary"]["incomplete"] == 0
    assert "main_text" in payload["figure_sets"]
    figure_set_keys = set(payload["figure_sets"]["main_text"]) | set(
        payload["figure_sets"]["supplement"]
    )
    assert figure_set_keys == set(payload["figures"])
    assert "Benchmark Matrix" in markdown
    assert "Monoenergetic Validation Summary" in markdown
    assert "Fixed-Field Precise-QS Benchmark" in markdown
    assert "Geometry-Control Derivatives" in markdown
    assert "Boundary Forward-Mode Current Derivatives" in markdown
    assert "Implicit-Equilibrium Forward-Mode Derivatives" in markdown
    assert "Explicit-Relaxed Boundary Current Derivatives" in markdown
    assert "File-Backed Geometry-Control Derivatives" in markdown
    assert "Bootstrap-Current Optimization" in markdown
    assert "| Commit |" in markdown
    assert "monoenergetic validation-summary gate" in claims
    assert "fixed-field precise-QS benchmark" in claims
    assert "W7-X imported-workflow bootstrap-current convergence" in claims
    assert "three-harmonic geometry-control derivative stress benchmark" in claims
    assert "file-backed Boozer and VMEC geometry-control derivative stress benchmark" in claims
    assert "boundary-projected `vmec_jax -> booz_xform_jax -> NTX`" in claims
    assert "implicit fixed-boundary `vmec_jax -> booz_xform_jax -> NTX`" in claims
    assert "explicit-relaxed `vmec_jax -> booz_xform_jax -> NTX`" in claims
