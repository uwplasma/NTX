from __future__ import annotations

from pathlib import Path

from ntx import _checkout_paths as cp


def test_helper_discovery_from_env(monkeypatch, tmp_path):
    booz_root = tmp_path / "booz_xform_jax"
    neopax_root = tmp_path / "NEOPAX"
    sfincs_root = tmp_path / "sfincs_jax"
    sfincs_fortran_root = tmp_path / "sfincs"
    finite_beta_root = tmp_path / "single_stage_optimization_finite_beta"
    vmec_root = tmp_path / "vmec_jax"
    sfincs_executable = sfincs_fortran_root / "fortran" / "version3" / "sfincs"
    example_input = vmec_root / "examples" / "data" / "input.circular_tokamak"
    example_input.parent.mkdir(parents=True)
    for path in (booz_root, neopax_root, sfincs_root, finite_beta_root, vmec_root):
        path.mkdir(exist_ok=True)
    sfincs_executable.parent.mkdir(parents=True, exist_ok=True)
    sfincs_executable.write_text("", encoding="utf-8")
    example_input.write_text("input", encoding="utf-8")

    monkeypatch.setenv("BOOZ_XFORM_JAX_ROOT", str(booz_root))
    monkeypatch.setenv("NEOPAX_ROOT", str(neopax_root))
    monkeypatch.setenv("SFINCS_JAX_ROOT", str(sfincs_root))
    monkeypatch.setenv("SFINCS_ROOT", str(sfincs_fortran_root))
    monkeypatch.setenv("SINGLE_STAGE_FINITE_BETA_ROOT", str(finite_beta_root))
    monkeypatch.setenv("VMEC_JAX_ROOT", str(vmec_root))

    assert cp.find_booz_xform_jax_root() == booz_root.resolve()
    assert cp.find_neopax_root() == neopax_root.resolve()
    assert cp.find_sfincs_jax_root() == sfincs_root.resolve()
    assert cp.find_sfincs_executable() == sfincs_executable.resolve()
    assert cp.find_single_stage_finite_beta_root() == finite_beta_root.resolve()
    assert cp.find_vmec_jax_root() == vmec_root.resolve()
    assert cp.find_vmec_jax_example_input() == example_input


def test_fixture_path_and_workspace_helpers():
    fixture = cp.fixture_path("sample_surface.ddkes2.data")
    assert fixture.name == "sample_surface.ddkes2.data"
    assert isinstance(cp.workspace_root(), Path)
    assert cp.find_vmec_jax_example_input("does-not-exist") is None


def test_workspace_checkout_candidates_and_missing_optional_roots(monkeypatch, tmp_path):
    repo = tmp_path / "NTX"
    repo_src = repo / "src" / "ntx"
    repo_src.mkdir(parents=True)
    sibling = tmp_path / "alpha"
    sibling.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    nested = tests_dir / "beta"
    nested.mkdir()

    monkeypatch.setattr(cp, "repo_root", lambda: repo)
    monkeypatch.setattr(cp, "workspace_root", lambda: tmp_path)

    candidates = cp._workspace_checkout_candidates()
    resolved = {path.resolve() for path in candidates}
    assert sibling.resolve() in resolved
    assert nested.resolve() in resolved
    assert repo.resolve() not in resolved
    assert cp.find_simsopt_root() is None
    assert cp.find_sfincs_executable() is None
    assert cp.find_single_stage_finite_beta_root() is None
