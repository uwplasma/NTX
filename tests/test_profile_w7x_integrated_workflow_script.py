from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "profile_w7x_integrated_workflow.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "profile_w7x_integrated_workflow_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configure_xla_flags_sets_dump_path(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    dump_dir = tmp_path / "xla"
    flags = module._configure_xla_flags(dump_dir)
    assert f"--xla_dump_to={dump_dir}" in flags
    assert os.environ["XLA_FLAGS"] == flags


def test_configure_xla_flags_appends_existing_flags(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_enable_fast_math=false")
    dump_dir = tmp_path / "xla"
    flags = module._configure_xla_flags(dump_dir)
    assert flags.startswith("--xla_cpu_enable_fast_math=false")
    assert f"--xla_dump_to={dump_dir}" in flags


def test_configure_compilation_cache_sets_dir(tmp_path):
    module = _load_module()
    cache_dir = tmp_path / "jax-cache"
    configured = module._configure_compilation_cache(cache_dir)
    assert configured == str(cache_dir)
    assert cache_dir.exists()
