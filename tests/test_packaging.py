"""Shipping-oriented package surface tests."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import ntx

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_installed_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = project["project"]["version"]
    assert importlib.metadata.version("ntx") == ntx.__version__ == expected


def test_module_entrypoint_imports() -> None:
    assert ntx.__doc__ is not None


def test_runtime_dependencies_have_single_owners() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = tuple(project["project"]["dependencies"])
    normalized = {
        item.split(";", 1)[0].split("<", 1)[0].split(">", 1)[0].lower()
        for item in dependencies
    }

    assert "jax" in normalized
    assert "netcdf4" in normalized
    assert "jaxlib" not in normalized
    assert "scipy" not in normalized
    assert "typing-extensions" not in normalized
    assert "scipy" in {item.lower() for item in project["project"]["optional-dependencies"]["dev"]}
