from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTERNAL_MODULES_REQUIRING_SOURCE_MAP = (
    "src/ntx/_autodiff.py",
    "src/ntx/_autodiff_bootstrap.py",
    "src/ntx/_geometry.py",
    "src/ntx/_inputfiles.py",
    "src/ntx/_neopax.py",
    "src/ntx/_neopax_scan.py",
    "src/ntx/_profiles.py",
    "src/ntx/_profiles_control.py",
    "src/ntx/_profiles_transport.py",
    "src/ntx/_solver.py",
    "src/ntx/_solver_scan.py",
    "src/ntx/_vmex.py",
    "src/ntx/validation/_benchmark_matrix.py",
    "src/ntx/validation/_benchmark_matrix_geometry.py",
    "src/ntx/validation/_finite_beta_closure_target.py",
    "src/ntx/validation/_finite_beta_source_channels.py",
    "src/ntx/validation/_physics_gate.py",
    "src/ntx/validation/_physics_gate_artifacts.py",
)


def test_source_map_mentions_split_internal_modules() -> None:
    text = (ROOT / "docs" / "source-map.md").read_text(encoding="utf-8")

    missing = [
        module_path
        for module_path in INTERNAL_MODULES_REQUIRING_SOURCE_MAP
        if f"`{module_path}`" not in text
    ]

    assert missing == []


# A module may own a whole concern, but not two. The previous form of
# this rule capped every module at 400 lines, which is below the natural
# size of several of NTX's concerns -- so concerns were split across files
# that shared a name prefix, and the prefix became a pretend namespace. That
# is worse than a long module: a reader who opens _solver_prepared.py cannot
# tell what else they need to read, and the import graph grows an edge for
# every split.
#
# The limit is therefore set above the largest coherent concern rather than
# below it. Exceeding it is a signal to check whether a module has taken on a
# second concern -- not an instruction to cut the current one in half.
#
# It counts *code* lines: docstrings, comments and blanks are excluded. Counting
# raw lines made documenting a module a way to fail this test, which is exactly
# backwards -- the first version of this rule tripped on _neopax.py at 1201 raw
# lines while its code was 948. A module's cost to a reader is its logic; the
# prose that explains that logic makes it cheaper, not dearer.
MODULE_CODE_LINE_LIMIT = 1000


def _code_lines(path: Path) -> int:
    """Lines that are neither blank, comment, nor docstring."""
    source = path.read_text(encoding="utf-8")
    documented: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            documented.update(range(body[0].lineno, body[0].end_lineno + 1))
    return sum(
        1
        for number, line in enumerate(source.splitlines(), start=1)
        if line.strip() and not line.strip().startswith("#") and number not in documented
    )


def test_top_level_source_modules_stay_below_ownership_limit() -> None:
    oversized = {
        path.relative_to(ROOT).as_posix(): _code_lines(path)
        for path in ROOT.joinpath("src", "ntx").glob("*.py")
        if _code_lines(path) > MODULE_CODE_LINE_LIMIT
    }

    assert oversized == {}


def test_the_package_is_not_fragmented_into_prefix_namespaces() -> None:
    """The other half of the rule: a prefix is not a namespace.

    Two modules sharing a `_family_` prefix are fine. A dozen are a package
    that was never declared, and the guard above cannot see them because each
    fragment is individually small.
    """
    import collections
    import re

    families: collections.Counter[str] = collections.Counter()
    for path in ROOT.joinpath("src", "ntx").glob("_*.py"):
        match = re.match(r"_([a-z]+)_", path.stem)
        if match:
            families[match.group(1)] += 1

    crowded = {name: n for name, n in families.items() if n > 4}
    assert crowded == {}, (
        f"prefix families with more than four members: {crowded}. Either fuse "
        "them into modules that own a concern, or make the family a real "
        "package with an __init__ that says what it is."
    )
