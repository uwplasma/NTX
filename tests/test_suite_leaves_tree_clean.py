"""The suite must not modify tracked files.

A test that regenerates a committed artifact in place makes the next run start
from a tree the previous run modified, so two runs of an unmodified checkout
disagree. That happened here: identical trees gave 55 failures on one run and
63 on the next, and attributing a failure to a change meant running both trees
from fresh clones.

The fix is that build scripts take an output location and tests pass a
temporary one. This pins that: any test that writes over a tracked file will
fail here rather than silently poisoning the run after it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Scripts that generate committed evidence. Each must accept an output
# location; running one without it, into the repository, is what caused the
# non-determinism.
GENERATORS = (
    ("scripts/build_manuscript_artifacts.py", "--output-dir"),
    ("scripts/build_closure_validation_report.py", "--output-prefix"),
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
@pytest.mark.parametrize("script,flag", GENERATORS)
def test_evidence_generators_accept_an_output_location(script: str, flag: str) -> None:
    text = (ROOT / script).read_text(encoding="utf-8")
    assert flag in text, (
        f"{script} must accept {flag} so tests can regenerate into a temporary "
        "directory instead of over the committed copy"
    )


def _tracked_status() -> set[str]:
    """Tracked paths git currently reports as modified."""
    status = _git("status", "--porcelain", "--", "docs", "src", "tests", "examples")
    return {
        line[3:] for line in status.splitlines() if line and not line.startswith("??")
    }


# Captured at collection, which happens before any test body runs. Comparing
# against this rather than against "clean" is what makes the check usable
# during development: a developer with uncommitted edits should not see a
# failure, only files the *suite itself* dirtied.
try:
    _BASELINE: set[str] | None = _tracked_status() if shutil.which("git") else None
except subprocess.CalledProcessError:  # pragma: no cover - not a checkout
    _BASELINE = None


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_the_suite_does_not_modify_tracked_files() -> None:
    """No test may dirty a tracked file that was clean when the run started.

    This compares against the collection-time snapshot, so it reports what the
    run changed rather than what the working tree happened to contain. It is
    ordering-sensitive by nature -- it can only see tests that ran before it --
    and the artifact generators it was written for sort earlier.
    """
    if _BASELINE is None:  # pragma: no cover - not a checkout
        pytest.skip("not a git checkout")

    newly_dirty = sorted(_tracked_status() - _BASELINE)
    assert newly_dirty == [], (
        "the suite modified tracked files that were clean when it started: "
        + ", ".join(newly_dirty)
        + ". Tests must write generated artifacts to tmp_path and compare "
        "against the committed copies, never regenerate them in place."
    )
