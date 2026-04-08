"""Command-line interface for NTX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import coefficient_errors, nearest_reference_row, read_monoenergetic_table
from .config import enable_x64
from .geometry import example_surface
from .grids import GridSpec
from .solver import MonoenergeticCase, solve_monoenergetic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ntx")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="solve one monoenergetic case")
    solve.add_argument(
        "--example",
        action="store_true",
        help="use the built-in example Boozer surface",
    )
    solve.add_argument("--nu-hat", type=float, required=True)
    solve.add_argument("--epsi-hat", type=float, default=None)
    solve.add_argument("--er-hat", type=float, default=None)
    solve.add_argument("--n-theta", type=int, default=5)
    solve.add_argument("--n-zeta", type=int, default=5)
    solve.add_argument("--n-xi", type=int, default=6)

    sub.add_parser("inspect-surface", help="print the built-in example surface")
    benchmark = sub.add_parser("benchmark", help="compare an example solve to an external table")
    benchmark.add_argument("table", type=Path)
    benchmark.add_argument("--nu-hat", type=float, required=True)
    benchmark.add_argument("--er-hat", type=float, default=0.0)
    benchmark.add_argument("--n-theta", type=int, default=5)
    benchmark.add_argument("--n-zeta", type=int, default=5)
    benchmark.add_argument("--n-xi", type=int, default=6)

    args = parser.parse_args(argv)
    if args.command == "solve":
        if not args.example:
            parser.error("only --example is implemented in this v1 CLI")
        enable_x64(True)
        surface = example_surface()
        grid = GridSpec(args.n_theta, args.n_zeta, args.n_xi)
        case = MonoenergeticCase(args.nu_hat, epsi_hat=args.epsi_hat, er_hat=args.er_hat)
        print(
            json.dumps(
                solve_monoenergetic(surface, grid, case).as_dict(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "inspect-surface":
        print(example_surface())
        return 0
    if args.command == "benchmark":
        enable_x64(True)
        surface = example_surface()
        grid = GridSpec(args.n_theta, args.n_zeta, args.n_xi)
        case = MonoenergeticCase(args.nu_hat, er_hat=args.er_hat)
        result = solve_monoenergetic(surface, grid, case).as_dict()
        row = nearest_reference_row(read_monoenergetic_table(args.table), args.nu_hat, args.er_hat)
        reference_names = row.dtype.names
        if reference_names is None:
            parser.error("benchmark table must have named columns")
        print(
            json.dumps(
                {
                    "reference": {name: float(row[name]) for name in reference_names},
                    "ntx": result,
                    "ntx_minus_reference": coefficient_errors(result, row),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
