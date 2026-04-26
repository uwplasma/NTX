#!/usr/bin/env python3
"""Open an NTX `.npz`, `.nc`, or `.h5` output file and make summary plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx.plotting import plot_run_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="NTX output `.nc`, `.npz`, or `.h5` file.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "output_file_summary",
        help="Prefix for PNG and PDF outputs.",
    )
    args = parser.parse_args()
    written = plot_run_output(args.output, output_prefix=args.output_prefix, formats=("png", "pdf"))
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
