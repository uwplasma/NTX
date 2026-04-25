from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_autodiff_profile_uncertainty_example_writes_outputs(tmp_path):
    example_path = ROOT / "examples" / "autodiff_profile_uncertainty.py"
    spec = importlib.util.spec_from_file_location("ntx_autodiff_profile_uncertainty", example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_prefix = tmp_path / "autodiff_profile_uncertainty"
    module.main(output_prefix=output_prefix, monte_carlo_samples=24, steps=16)
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    assert output_prefix.with_suffix(".json").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["basis_size"] == 3
    assert len(payload["parameter_std"]) == 3
    assert payload["hessian_probe_relative_error"] < 1.0e-7
    assert len(payload["fisher_eigenvalues"]) == 3
