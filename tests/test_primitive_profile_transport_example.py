from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_primitive_profile_transport_example_writes_outputs(tmp_path):
    example_path = ROOT / "examples" / "primitive_profile_transport.py"
    spec = importlib.util.spec_from_file_location(
        "ntx_primitive_profile_transport_example",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_prefix = tmp_path / "primitive_profile_transport"
    module.main(output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
