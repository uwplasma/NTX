from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "geometry_family_breadth_summary.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ntx_geometry_family_breadth_summary",
        EXAMPLE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_geometry_family_breadth_summary_builds_from_committed_artifacts():
    module = _load_module()

    payload = module.build_payload()

    assert payload["benchmark"] == "geometry_family_breadth_summary"
    assert payload["classification"] == "artifact-backed geometry-breadth stress summary"
    assert payload["summary_metrics"]["active_case_count"] >= 7
    assert payload["summary_metrics"]["open_case_count"] == 2
    assert payload["summary_metrics"]["max_active_relative_mismatch"] < 5.0e-4
    assert payload["summary_metrics"]["max_open_relative_mismatch"] > 1.0e-1
    assert {case["status"] for case in payload["open_cases"]} == {"open"}


def test_geometry_family_breadth_summary_writes_outputs(tmp_path):
    module = _load_module()
    output_prefix = tmp_path / "geometry_family_breadth_summary"

    module.main(output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["benchmark"] == "geometry_family_breadth_summary"
    assert payload["summary_metrics"]["implicit_validated_objective_count"] == 1
    assert payload["summary_metrics"]["implicit_open_objective_count"] == 2
