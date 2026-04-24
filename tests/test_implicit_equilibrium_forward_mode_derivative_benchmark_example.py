from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from ntx._checkout_paths import find_booz_xform_jax_root, find_vmec_jax_root

ROOT = Path(__file__).resolve().parents[1]


def _has_boundary_stack() -> bool:
    vmec_root = find_vmec_jax_root()
    if vmec_root is None:
        return False
    return (
        (vmec_root / "examples" / "data" / "input.LandremanPaul2021_QA_lowres").exists()
        and find_booz_xform_jax_root() is not None
    )


@pytest.mark.skipif(
    os.environ.get("NTX_RUN_HEAVY_BOUNDARY_EXAMPLES") != "1" or not _has_boundary_stack(),
    reason="requires NTX_RUN_HEAVY_BOUNDARY_EXAMPLES=1 and local vmec_jax/booz_xform_jax",
)
def test_implicit_equilibrium_forward_mode_derivative_benchmark_writes_outputs(tmp_path):
    example_path = ROOT / "examples" / "implicit_equilibrium_forward_mode_derivative_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "ntx_implicit_equilibrium_forward_mode_derivative_benchmark",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output_prefix = tmp_path / "implicit_equilibrium_forward_mode_derivative_benchmark"
    module.main(output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["benchmark"] == "implicit_equilibrium_forward_mode_derivative_benchmark"
    assert (
        payload["classification"]
        == "artifact-backed non-shipping implicit-equilibrium diagnostic"
    )
    assert payload["closure_decision"]["status"] == "closed-not-shipped"
    assert (
        payload["closure_decision"]["supported_equilibrium_derivative_path"]
        == "explicit_relaxed_fixed_boundary"
    )
    objective_ids = {objective["id"] for objective in payload["objectives"]}
    assert objective_ids == {
        "equilibrium_volume",
        "booz_xform_scalar",
        "ntx_transport_proxy",
    }
    objectives = {objective["id"]: objective for objective in payload["objectives"]}
    assert objectives["equilibrium_volume"]["status"] == "validated"
    assert objectives["equilibrium_volume"]["relative_mismatch"][0] < 1.0e-3
    assert objectives["booz_xform_scalar"]["status"] == "closed-not-shipped"
    assert objectives["booz_xform_scalar"]["relative_mismatch"][0] > 1.0e-1
    assert objectives["ntx_transport_proxy"]["status"] == "closed-not-shipped"
    assert objectives["ntx_transport_proxy"]["relative_mismatch"][0] > 1.0
    assert payload["summary_metrics"]["residual_contracts"] is False
    assert payload["residual_history"]
    assert payload["reverse_mode_diagnostic"]["objective_id"] == "booz_xform_scalar"
    assert payload["reverse_mode_diagnostic"]["status"] == "unsupported"
    assert payload["reverse_mode_diagnostic"]["error_type"] == "ValueError"
