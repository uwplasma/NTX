from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from ntx._checkout_paths import (
    find_booz_xform_jax_root,
    find_neopax_root,
    find_vmec_jax_example_input,
)

ROOT = Path(__file__).resolve().parents[1]


def _has_boundary_stack() -> bool:
    return (
        find_vmec_jax_example_input() is not None
        and find_booz_xform_jax_root() is not None
        and find_neopax_root() is not None
    )


@pytest.mark.skipif(
    os.environ.get("NTX_RUN_HEAVY_BOUNDARY_EXAMPLES") != "1" or not _has_boundary_stack(),
    reason="requires NTX_RUN_HEAVY_BOUNDARY_EXAMPLES=1 and local vmec_jax/booz_xform_jax/NEOPAX",
)
def test_boundary_forward_mode_current_derivative_benchmark_writes_outputs(tmp_path):
    example_path = ROOT / "examples" / "boundary_forward_mode_current_derivative_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "ntx_boundary_forward_mode_current_derivative_benchmark",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_prefix = tmp_path / "boundary_forward_mode_current_derivative_benchmark"
    module.main(output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["benchmark"] == "boundary_forward_mode_current_derivative_benchmark"
    assert (
        payload["classification"]
        == "artifact-backed boundary-to-output forward-mode stress benchmark"
    )
    assert payload["summary_metrics"]["max_relative_mismatch"] < 1.0e-4
    objective_ids = {objective["id"] for objective in payload["objectives"]}
    assert objective_ids == {"ntx_transport_proxy", "ntx_neopax_integrated_current"}
