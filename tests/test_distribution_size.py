from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_distribution_size import check_distribution_sizes, distribution_kind

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("ntx.whl", "wheel"), ("ntx.tar.gz", "sdist"), ("ntx.zip", "sdist")],
)
def test_distribution_kind(name: str, expected: str):
    assert distribution_kind(Path(name)) == expected


def test_distribution_size_checks_missing_and_oversized_files(tmp_path):
    wheel = tmp_path / "ntx.whl"
    sdist = tmp_path / "ntx.tar.gz"
    wheel.write_bytes(b"x" * 11)
    sdist.write_bytes(b"x" * 9)

    violations = check_distribution_sizes(
        (wheel, sdist, tmp_path / "missing.whl"),
        max_wheel_bytes=10,
        max_sdist_bytes=10,
    )

    assert any("wheel limit" in violation for violation in violations)
    assert any("missing distribution" in violation for violation in violations)
    assert not any("sdist limit" in violation for violation in violations)


def test_distribution_size_script_accepts_bounded_files(tmp_path):
    wheel = tmp_path / "ntx.whl"
    sdist = tmp_path / "ntx.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_distribution_size.py"),
            str(wheel),
            str(sdist),
            "--max-wheel-bytes",
            "10",
            "--max-sdist-bytes",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wheel" in result.stdout
    assert "sdist" in result.stdout
