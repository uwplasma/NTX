from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_derivative_path_benchmark_example_writes_outputs(tmp_path):
    example_path = ROOT / "examples" / "derivative_path_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "ntx_derivative_path_benchmark_example",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_prefix = tmp_path / "derivative_path_benchmark"
    module.main(output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    payload = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["grid"]["n_theta"] == 7
    assert payload["scan_sizes"][-1] == 32
    assert sorted(payload["gradient_channels"]) == ["dD11_dnu", "dD33_dEr"]
    assert max(payload["max_relative_mismatch"]) <= 1.0e-4
    assert min(payload["speedup_prepared_vs_direct"]) > 0.0
    assert payload["derivative_memory"]["scan_size"] == 32
    assert payload["derivative_memory"]["direct_reverse"]["temp_size_in_bytes"] >= 0
    assert (
        payload["derivative_memory"]["selective_recomputation"]["max_relative_mismatch_direct"]
        < 1.0e-4
    )
    audits = payload["validity_audit"]["entries"]
    assert [entry["valid"] for entry in audits] == [False, True, True]
    assert max(entry["prepared_adjoint_relative_error"] for entry in audits) < 1.0e-8
    assert max(entry["forward_relative_error"] for entry in audits) < 1.0e-8
    assert max(entry["finite_difference_relative_error"] for entry in audits) < 1.0e-4
    resolution = payload["resolution_audit"]
    assert resolution["two_successive_refinements_pass"] is True
    assert all(entry["valid"] for entry in resolution["entries"])
    assert all(
        entry["relative_change_previous"] <= resolution["relative_change_tolerance"]
        for entry in resolution["entries"][1:]
    )
