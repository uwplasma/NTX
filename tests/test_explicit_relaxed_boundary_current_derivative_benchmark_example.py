from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from ntx._checkout_paths import find_booz_xform_jax_root, find_neopax_root, find_vmex_root

ROOT = Path(__file__).resolve().parents[1]


def _has_boundary_stack() -> bool:
    vmec_root = find_vmex_root()
    if vmec_root is None:
        return False
    return (
        (vmec_root / "examples" / "data" / "input.LandremanPaul2021_QA_lowres").exists()
        and find_booz_xform_jax_root() is not None
        and find_neopax_root() is not None
    )


@pytest.mark.skipif(
    os.environ.get("NTX_RUN_HEAVY_BOUNDARY_EXAMPLES") != "1" or not _has_boundary_stack(),
    reason="requires NTX_RUN_HEAVY_BOUNDARY_EXAMPLES=1 and local vmex/booz_xform_jax/NEOPAX",
)
def test_explicit_relaxed_boundary_current_derivative_benchmark_writes_outputs(tmp_path):
    example_path = ROOT / "examples" / "explicit_relaxed_boundary_current_derivative_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "ntx_explicit_relaxed_boundary_current_derivative_benchmark",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_prefix = tmp_path / "explicit_relaxed_boundary_current_derivative_benchmark"
    module.main(output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["benchmark"] == "explicit_relaxed_boundary_current_derivative_benchmark"
    assert (
        payload["classification"]
        == "artifact-backed explicit-relaxed equilibrium forward-mode family stress benchmark"
    )
    assert payload["summary_metrics"]["max_relative_mismatch"] < 1.0e-3
    assert payload["summary_metrics"]["max_ordinary_explicit_volume_relative_difference"] < 1.0e-10
    case_ids = {case["id"] for case in payload["cases"]}
    assert case_ids == {"qa_lowres", "qh_warm_start"}
    for case in payload["cases"]:
        assert case["volume_metrics"]["ordinary_explicit_relative_difference"] < 1.0e-10
        objective_ids = {objective["id"] for objective in case["objectives"]}
        assert objective_ids == {
            "booz_xform_scalar",
            "ntx_transport_response",
            "ntx_neopax_integrated_current",
        }
