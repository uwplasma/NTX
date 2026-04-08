"""Command-line interface for NTX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmarks import coefficient_errors, nearest_reference_row, read_monoenergetic_table
from .config import enable_x64
from .geometry import example_surface
from .grids import GridSpec
from .inputfiles import run_from_input_file
from .io import load_dkes_surface
from .solver import MonoenergeticCase, solve_monoenergetic


def main(argv: list[str] | None = None) -> int:
    args_list = sys.argv[1:] if argv is None else argv
    if _looks_like_input_file(args_list):
        payload = run_from_input_file(args_list[0])
        print(json.dumps(payload["result"], indent=2, sort_keys=True))
        return 0

    parser = argparse.ArgumentParser(prog="ntx")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="solve one monoenergetic case")
    solve_surface = solve.add_mutually_exclusive_group(required=True)
    solve_surface.add_argument(
        "--example",
        action="store_true",
        help="use the built-in example Boozer surface",
    )
    solve_surface.add_argument("--dkes", type=Path, help="path to a DKES-format ddkes2.data file")
    solve.add_argument("--nu-hat", type=float, required=True)
    solve.add_argument("--epsi-hat", type=float, default=None)
    solve.add_argument("--er-hat", type=float, default=None)
    solve.add_argument("--n-theta", type=int, default=5)
    solve.add_argument("--n-zeta", type=int, default=5)
    solve.add_argument("--n-xi", type=int, default=6)

    inspect = sub.add_parser("inspect-surface", help="print a surface definition")
    inspect_surface = inspect.add_mutually_exclusive_group(required=True)
    inspect_surface.add_argument(
        "--example",
        action="store_true",
        help="print the built-in example Boozer surface",
    )
    inspect_surface.add_argument("--dkes", type=Path, help="path to a DKES-format ddkes2.data file")

    benchmark = sub.add_parser("benchmark", help="compare a solve to an external table")
    benchmark_surface = benchmark.add_mutually_exclusive_group(required=True)
    benchmark_surface.add_argument(
        "--example",
        action="store_true",
        help="use the built-in example Boozer surface",
    )
    benchmark_surface.add_argument(
        "--dkes",
        type=Path,
        help="path to a DKES-format ddkes2.data file",
    )
    benchmark.add_argument("table", type=Path)
    benchmark.add_argument("--nu-hat", type=float, required=True)
    benchmark.add_argument("--er-hat", type=float, default=0.0)
    benchmark.add_argument("--n-theta", type=int, default=5)
    benchmark.add_argument("--n-zeta", type=int, default=5)
    benchmark.add_argument("--n-xi", type=int, default=6)

    args = parser.parse_args(args_list)
    if args.command == "solve":
        enable_x64(True)
        surface = _load_surface(args)
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
        print(_load_surface(args))
        return 0
    if args.command == "benchmark":
        enable_x64(True)
        surface = _load_surface(args)
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


def _looks_like_input_file(argv: list[str]) -> bool:
    if len(argv) != 1:
        return False
    candidate = Path(argv[0]).expanduser()
    return candidate.suffix == ".toml" and candidate.exists()


def _load_surface(args):
    if getattr(args, "example", False):
        return example_surface()
    if getattr(args, "dkes", None) is not None:
        return load_dkes_surface(args.dkes)
    msg = "select either --example or --dkes"
    raise ValueError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
