from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_geometry_control_derivative_benchmark_writes_outputs(tmp_path):
    example_path = ROOT / "examples" / "geometry_control_derivative_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "ntx_geometry_control_derivative_benchmark",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_prefix = tmp_path / "geometry_control_derivative_benchmark"
    module.main(output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["benchmark"] == "geometry_control_derivative_benchmark"
    assert payload["classification"] == "artifact-backed autodiff stress benchmark"
    assert len(payload["control_modes"]) == 3
    assert payload["summary_metrics"]["max_relative_mismatch"] < 1.0e-2
