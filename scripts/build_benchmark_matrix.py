#!/usr/bin/env python3
"""Write the maintained NTX benchmark matrix artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="NTX repository root")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "docs" / "_static" / "benchmark_matrix.json",
        help="Path for the machine-readable benchmark matrix.",
    )
    return parser.parse_args()


def main() -> None:
    from ntx.validation.benchmark_matrix import (
        evaluate_benchmark_matrix,
        write_benchmark_matrix_json,
    )

    args = parse_args()
    root = args.root.resolve()
    output_json = args.output_json.resolve()
    write_benchmark_matrix_json(root, output_json)

    evaluations = evaluate_benchmark_matrix(root)
    print("NTX benchmark matrix")
    for evaluation in evaluations:
        print(f"- {evaluation.entry.id} [{evaluation.entry.maturity}]: {evaluation.status}")
        for missing in evaluation.missing_required_paths:
            print(f"  missing: {missing}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
