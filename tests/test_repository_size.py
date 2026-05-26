from __future__ import annotations

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
