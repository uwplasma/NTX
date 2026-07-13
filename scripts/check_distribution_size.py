#!/usr/bin/env python3
"""Reject unexpectedly large NTX wheel and source distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

MIB = 1024 * 1024
DEFAULT_MAX_WHEEL_BYTES = 512 * 1024
DEFAULT_MAX_SDIST_BYTES = 2 * MIB


def distribution_kind(path: Path) -> str:
    """Return ``wheel`` or ``sdist`` for a supported distribution path."""

    if path.suffix == ".whl":
        return "wheel"
    if path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".zip")):
        return "sdist"
    msg = f"unsupported distribution filename: {path.name}"
    raise ValueError(msg)


def check_distribution_sizes(
    paths: tuple[Path, ...],
    *,
    max_wheel_bytes: int = DEFAULT_MAX_WHEEL_BYTES,
    max_sdist_bytes: int = DEFAULT_MAX_SDIST_BYTES,
) -> tuple[str, ...]:
    """Return human-readable size violations for existing distributions."""

    violations = []
    for path in paths:
        if not path.is_file():
            violations.append(f"missing distribution: {path}")
            continue
        kind = distribution_kind(path)
        limit = max_wheel_bytes if kind == "wheel" else max_sdist_bytes
        size = path.stat().st_size
        print(f"{path}: {size} bytes ({kind}, limit {limit} bytes)")
        if size > limit:
            violations.append(f"{path} is {size} bytes; {kind} limit is {limit} bytes")
    return tuple(violations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-wheel-bytes", type=int, default=DEFAULT_MAX_WHEEL_BYTES)
    parser.add_argument("--max-sdist-bytes", type=int, default=DEFAULT_MAX_SDIST_BYTES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    violations = check_distribution_sizes(
        tuple(args.paths),
        max_wheel_bytes=args.max_wheel_bytes,
        max_sdist_bytes=args.max_sdist_bytes,
    )
    if violations:
        raise SystemExit("\n".join(violations))


if __name__ == "__main__":
    main()
