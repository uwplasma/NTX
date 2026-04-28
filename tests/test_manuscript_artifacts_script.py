from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_SCRIPT = ROOT / "scripts" / "build_manuscript_artifacts.py"
FIGURE_SCRIPT = ROOT / "examples" / "make_publication_figures.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_manuscript_artifacts_script_writes_outputs():
    subprocess.run(
        [
            sys.executable,
            str(MANUSCRIPT_SCRIPT),
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
    assert "geometry_family_breadth" in payload["tables"]
    assert "geometry_family_transport" in payload["tables"]
    assert "owned_finite_beta_bootstrap_comparison" in payload["tables"]
    finite_beta_bootstrap = payload["tables"]["owned_finite_beta_bootstrap_comparison"]
    assert finite_beta_bootstrap["inputs"]["n_order"] >= 2
    assert finite_beta_bootstrap["inputs"]["d33_mode"]
    assert finite_beta_bootstrap["comparison"]["momentum_order_scan"]
    assert payload["claims"]["owned_finite_beta_bootstrap_nu_v_count"] >= 1
    assert payload["claims"]["owned_finite_beta_bootstrap_psi_p"] > 0.0
    assert "performance" in payload["tables"]
    assert "prepared_geometry_reuse" in payload["tables"]["performance"]
    assert "production" in payload["tables"]["performance"]
    assert "strong_scaling" in payload["tables"]["performance"]
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
    assert "Geometry-Family Breadth Summary" in markdown
    assert "Geometry-Family Transport Convergence" in markdown
    assert "Owned Finite-Beta Bootstrap-Current Stress" in markdown
    assert "Sonine-order max/RMS relative differences" in markdown
    assert "Boozer psi_p" in markdown
    assert "Bootstrap-Current Optimization" in markdown
    assert "Prepared-geometry reuse" in markdown
    assert "| Commit |" in markdown
    assert "monoenergetic validation-summary gate" in claims
    assert "fixed-field precise-QS benchmark" in claims
    assert "W7-X imported-workflow bootstrap-current convergence" in claims
    assert "three-harmonic geometry-control derivative stress benchmark" in claims
    assert "file-backed Boozer and VMEC geometry-control derivative stress benchmark" in claims
    assert "boundary-projected `vmec_jax -> booz_xform_jax -> NTX`" in claims
    assert "implicit fixed-boundary `vmec_jax -> booz_xform_jax -> NTX`" in claims
    assert "explicit-relaxed `vmec_jax -> booz_xform_jax -> NTX`" in claims
    assert "artifact-backed geometry-family breadth summary" in claims
    assert "geometry-family transport convergence stress diagnostic" in claims
    assert "owned finite-beta bootstrap-current stress audit" in claims
    assert "adaptive `nu/v` support points" in claims
    assert "production-grid CPU performance" in claims
    assert "fixed-workload CPU strong-scaling" in claims
    assert "prepared-geometry reuse profile" in claims


def test_manuscript_figure_sets_match_publication_presets():
    manuscript = _load_module(MANUSCRIPT_SCRIPT, "ntx_build_manuscript_artifacts")
    figures = _load_module(FIGURE_SCRIPT, "ntx_make_publication_figures_for_manuscript")

    payload = manuscript.build_payload()

    assert set(payload["figures"]) == figures.FIGURE_PRESETS["all"]
    assert set(payload["figure_sets"]["main_text"]) == figures.FIGURE_PRESETS["main_text"]
    assert set(payload["figure_sets"]["supplement"]) == figures.FIGURE_PRESETS["supplement"]
