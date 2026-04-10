from __future__ import annotations

from pathlib import Path

from ntx import _checkout_paths as cp


def test_reference_python_root_discovery_from_env(monkeypatch, tmp_path):
    root = tmp_path / "reference-python"
    root.mkdir()
    monkeypatch.setenv("NTX_REFERENCE_PYTHON_ROOT", str(root))
    assert cp.find_reference_python_root() == root.resolve()


def test_reference_python_root_discovery_from_workspace_signature(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    repo = workspace / "NTX"
    source_root = workspace / "candidate"
    (source_root / "Examples" / "DKES_like_database").mkdir(parents=True)
    signature = (
        source_root
        / "Examples"
        / "DKES_like_database"
        / "Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py"
    )
    signature.write_text(
        "print('x')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cp, "repo_root", lambda: repo)
    monkeypatch.delenv("NTX_REFERENCE_PYTHON_ROOT", raising=False)
    assert cp.find_reference_python_root() == source_root.resolve()


def test_reference_executable_discovery(monkeypatch, tmp_path):
    exe = tmp_path / "bin" / "main_reference.x"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("NTX_REFERENCE_EXECUTABLE", str(exe))
    assert cp.find_reference_executable() == exe.resolve()


def test_reference_executable_root_scan(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    repo = workspace / "NTX"
    root = workspace / "solver"
    exe = root / "bin" / "main_solver.x"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(cp, "repo_root", lambda: repo)
    monkeypatch.delenv("NTX_REFERENCE_EXECUTABLE", raising=False)
    monkeypatch.delenv("NTX_REFERENCE_EXECUTABLE_ROOT", raising=False)
    assert cp.find_reference_executable_root() == root.resolve()
    assert cp.find_reference_executable() == exe.resolve()


def test_fixture_path_and_optional_helpers():
    fixture = cp.fixture_path("w7x_eim_sample.ddkes2.data")
    assert fixture.name == "w7x_eim_sample.ddkes2.data"
    assert isinstance(cp.workspace_root(), Path)
    assert cp.find_vmec_jax_example_input("does-not-exist") is None
