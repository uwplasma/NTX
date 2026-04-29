from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_current_robust_optimization_writes_outputs(tmp_path):
    example_path = ROOT / "examples" / "bootstrap_current_robust_optimization.py"
    spec = importlib.util.spec_from_file_location("ntx_bootstrap_current_robust", example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_prefix = tmp_path / "bootstrap_current_robust_optimization"
    module.main(
        output_prefix=output_prefix,
        steps=2,
        radial_points=3,
        grid=module.GridSpec(5, 5, 4),
        scale_grid_size=5,
        quadrature_order=3,
    )
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    for key in (
        "baseline_weighted_current_response",
        "optimized_weighted_current_response",
        "weighted_current_ratio",
        "weighted_current_relative_change",
        "robust_objective_initial",
        "robust_objective_final",
        "robust_objective_relative_change",
        "robust_gain",
    ):
        assert math.isfinite(payload[key])
    assert payload["radial_points"] == 3
    assert payload["scale_grid_size"] == 5
    assert payload["quadrature_order"] == 3
    assert payload["max_current_std"] >= 0.0
