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
