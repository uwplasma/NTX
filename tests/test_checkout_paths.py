from __future__ import annotations

from pathlib import Path

from ntx import _checkout_paths as cp


def test_helper_discovery_from_env(monkeypatch, tmp_path):
    booz_root = tmp_path / "booz_xform_jax"
    neopax_root = tmp_path / "NEOPAX"
    vmec_root = tmp_path / "vmec_jax"
    example_input = vmec_root / "examples" / "data" / "input.circular_tokamak"
    example_input.parent.mkdir(parents=True)
    for path in (booz_root, neopax_root, vmec_root):
        path.mkdir(exist_ok=True)
    example_input.write_text("input", encoding="utf-8")

    monkeypatch.setenv("BOOZ_XFORM_JAX_ROOT", str(booz_root))
    monkeypatch.setenv("NEOPAX_ROOT", str(neopax_root))
    monkeypatch.setenv("VMEC_JAX_ROOT", str(vmec_root))

    assert cp.find_booz_xform_jax_root() == booz_root.resolve()
    assert cp.find_neopax_root() == neopax_root.resolve()
    assert cp.find_vmec_jax_root() == vmec_root.resolve()
    assert cp.find_vmec_jax_example_input() == example_input


def test_fixture_path_and_workspace_helpers():
    fixture = cp.fixture_path("sample_surface.ddkes2.data")
    assert fixture.name == "sample_surface.ddkes2.data"
    assert isinstance(cp.workspace_root(), Path)
    assert cp.find_vmec_jax_example_input("does-not-exist") is None
