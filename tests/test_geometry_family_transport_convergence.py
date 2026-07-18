from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

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


def test_upstream_layout_fallback_prefers_first_available_path(tmp_path):
    module = _load_module()
    legacy = tmp_path / "legacy" / "surface.nc"
    current = tmp_path / "current" / "surface.nc"
    legacy.parent.mkdir()
    current.parent.mkdir()
    legacy.touch()
    current.touch()
    cases = []

    module._add_first_case(
        cases,
        root=tmp_path,
        relative_paths=("current/surface.nc", "legacy/surface.nc"),
        case_id="qa",
        label="QA",
        family="QA",
        source="upstream examples",
        notes="layout compatibility test",
    )

    assert len(cases) == 1
    assert cases[0].path == current.resolve()


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
    assert payload["summary_metrics"]["promotion_case_count"] == 1
    assert payload["summary_metrics"]["diagnostic_only_case_count"] == 0
    assert payload["summary_metrics"]["skipped_case_count"] == 0
    summary = payload["summary_metrics"]
    assert summary["max_successful_schur_residual_l2"] >= 0.0
    assert summary["max_successful_residual_l2"] == summary["max_successful_schur_residual_l2"]
    for case in payload["cases"]:
        for result in case.get("grid_results", []):
            assert result["residual_l2"] == result["schur_residual_l2"]
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


def test_geometry_family_transport_convergence_enables_x64_before_loading():
    module = _load_module()
    payload = module.build_payload(
        case_specs=(_fixture_case(module),),
        grids=(module.GridSpec(5, 5, 4), module.GridSpec(7, 7, 6)),
        convergence_rtol=10.0,
    )

    assert payload["cases"][0]["surface"]["dtype"] == str(jnp.dtype(jnp.float64))


def test_geometry_family_transport_convergence_rejects_mixed_precision_ladder():
    module = _load_module()
    with pytest.raises(ValueError, match="same x64 policy"):
        module.build_payload(
            case_specs=(_fixture_case(module),),
            grids=(
                module.GridSpec(5, 5, 4, x64=True),
                module.GridSpec(7, 7, 6, dtype="float32", x64=False),
            ),
        )


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
