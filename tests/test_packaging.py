"""Shipping-oriented package surface tests."""

from __future__ import annotations

import importlib.metadata

import ntx


def test_package_version_matches_installed_metadata() -> None:
    assert importlib.metadata.version("ntx") == ntx.__version__ == "0.1.0"


def test_module_entrypoint_imports() -> None:
    assert ntx.__doc__ is not None
