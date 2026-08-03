"""Every module, and every public API symbol, has to say what it is for.

Two levels, deliberately different. A *module* docstring is what a reader hits
first when they open a file, and a package with twenty-four undocumented
modules is one where the reader has to reconstruct the layout from imports —
so every module needs one, private or not.

A *symbol* docstring is required only for the public API: names that do not
start with an underscore, in modules that do not start with an underscore.
Requiring one on every private helper would produce restatements of the
signature, which is worse than nothing because it looks like documentation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "ntx"


def _modules() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if p.name != "__init__.py")


def _public_api(tree: ast.Module) -> list[ast.AST]:
    """Top-level and class-level public definitions, not nested closures."""
    found: list[ast.AST] = []

    def scan(body: list[ast.stmt]) -> None:
        for node in body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if node.name.startswith("_"):
                continue
            found.append(node)
            if isinstance(node, ast.ClassDef):
                scan(node.body)

    scan(tree.body)
    return found


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_every_module_has_a_docstring(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), (
        f"{path.relative_to(PACKAGE)} has no module docstring. Say what the "
        "module owns and, if it was split out of another, why it is separate."
    )


def test_every_public_api_symbol_has_a_docstring() -> None:
    missing = []
    for path in _modules():
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _public_api(tree):
            if not ast.get_docstring(node):
                missing.append(f"{path.relative_to(PACKAGE)}::{node.name}")
    assert missing == [], (
        "public API symbols without a docstring: "
        + ", ".join(missing)
        + ". These are what a user reads; a private helper is exempt."
    )


def test_the_public_api_surface_is_actually_covered() -> None:
    """Guard the guard: if the scan finds nothing, it is not proving anything.

    A refactor that renamed every public module to a private one would make the
    check above pass vacuously. This pins that the surface being checked is
    still a real one.
    """
    total = sum(
        len(_public_api(ast.parse(p.read_text(encoding="utf-8"))))
        for p in _modules()
        if not p.name.startswith("_")
    )
    assert total >= 50, f"only {total} public API symbols scanned; the check has gone vacuous"
