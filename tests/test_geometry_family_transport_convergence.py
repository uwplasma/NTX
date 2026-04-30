from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "geometry_family_transport_convergence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ntx_geometry_family_transport_convergence",
        EXAMPLE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_case(module):
    return module.GeometryTransportCase(
        id="repo_vmec_fixture",
        label="Repo VMEC fixture",
        family="fixture",
        source="NTX tests",
        path=module.fixture_path("sample_wout.nc"),
        notes="unit-test fixture",
    )


def test_geometry_family_transport_convergence_builds_fixture_payload():
    module = _load_module()
    grids = (module.GridSpec(5, 5, 4), module.GridSpec(7, 7, 6))

    payload = module.build_payload(
        case_specs=(_fixture_case(module),),
        grids=grids,
        convergence_rtol=10.0,
    )

    assert payload["benchmark"] == "geometry_family_transport_convergence"
    assert payload["summary_metrics"]["case_count"] == 1
    assert payload["summary_metrics"]["successful_case_count"] == 1
    assert payload["summary_metrics"]["skipped_case_count"] == 0
    assert payload["summary_metrics"]["max_successful_residual_l2"] >= 0.0
    assert payload["summary_metrics"]["max_successful_relative_onsager_residual"] >= 0.0
    case = payload["cases"][0]
    assert case["id"] == "repo_vmec_fixture"
    assert case["surface"]["loaded_mode_count"] > 0
    assert len(case["grid_results"]) == 2
    assert set(case["last_step_relative_change"]) == {"D11", "D31", "D33"}
    assert set(case["last_step_absolute_change"]) == {"D11", "D31", "D33"}
    assert set(case["grid_results"][0]["coefficients"]) == {"D11", "D31", "D13", "D33"}
    assert case["quality_status"] in {"stress-pass", "monitor"}
    assert all(result["finite"] for result in case["grid_results"])


def test_geometry_family_transport_convergence_writes_outputs(tmp_path):
    module = _load_module()
    grids = (module.GridSpec(5, 5, 4), module.GridSpec(7, 7, 6))
    payload = module.build_payload(
        case_specs=(_fixture_case(module),),
        grids=grids,
        convergence_rtol=10.0,
    )
    output_prefix = tmp_path / "geometry_family_transport_convergence"

    module.write_payload(payload, output_prefix)
    module.build_figure(payload, output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    written = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert written["benchmark"] == "geometry_family_transport_convergence"
