from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
MAX_TRACKED_FILE_BYTES = 2 * MIB
MAX_TRACKED_TREE_BYTES = 25 * MIB
MAX_DOCS_STATIC_BYTES = 20 * MIB


def _tracked_files() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(
        ROOT / item.decode()
        for item in result.stdout.split(b"\0")
        if item
    )


def test_tracked_files_stay_below_two_mib():
    oversized = [
        (path.relative_to(ROOT), path.stat().st_size)
        for path in _tracked_files()
        if path.stat().st_size > MAX_TRACKED_FILE_BYTES
    ]

    assert oversized == []


def test_tracked_tree_and_artifact_directory_stay_small():
    tracked_files = _tracked_files()
    tracked_size = sum(path.stat().st_size for path in tracked_files)
    docs_static_size = sum(
        path.stat().st_size
        for path in tracked_files
        if path.relative_to(ROOT).parts[:2] == ("docs", "_static")
    )

    assert tracked_size <= MAX_TRACKED_TREE_BYTES
    assert docs_static_size <= MAX_DOCS_STATIC_BYTES


# Spelled so this file does not match itself.
MACHINE_PATH = re.compile(rb"/(Users|private/tmp)/|/home/[A-Za-z_]|[A-Z]:\\\\Users\\\\")


def test_tracked_text_files_hold_no_machine_paths():
    leaks = []
    for path in _tracked_files():
        data = path.read_bytes()
        if b"\0" not in data[:8192] and (match := MACHINE_PATH.search(data)):
            line = data.count(b"\n", 0, match.start()) + 1
            leaks.append(f"{path.relative_to(ROOT)}:{line}")
    assert leaks == []


def test_tracked_text_has_no_machine_paths_handles_or_attribution():
    """Tracked text must not contain personal handles or AI-attribution strings (paths: see above)."""
    import re
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    this = Path(__file__).relative_to(root).as_posix()
    patterns = {
        "personal handle": r"rogeriojorge|rjorge",
        "AI attribution": r"Co-Authored-By|Generated with|Claude|Anthropic|ChatGPT|Codex|Copilot",
    }
    handle_ok = {"CITATION.cff", "LICENSE", "pyproject.toml", ".zenodo.json"}
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.decode()
    bad = []
    for rel in (p for p in out.split("\0") if p and p != this):
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        text = re.sub(r"github\.com/rogeriojorge|github\.com/uwplasma", "", text)
        for name, pattern in patterns.items():
            if name == "personal handle" and rel.rsplit("/", 1)[-1] in handle_ok:
                continue
            bad += [f"{rel}: {name}: {m.group(0)}" for m in re.finditer(pattern, text)]
    assert not bad, "\n".join(bad[:20])
