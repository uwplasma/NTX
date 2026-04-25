from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_profile_basis_optimization_example_writes_outputs(tmp_path):
    example_path = ROOT / "examples" / "profile_basis_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "ntx_profile_basis_optimization_example",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_prefix = tmp_path / "profile_basis_optimization"
    module.main(output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["artifact"] == "profile_basis_optimization"
    assert payload["basis_count"] == 3
    assert payload["residual_l2_ratio"] >= 0.0
