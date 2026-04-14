from __future__ import annotations

import importlib.util
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
