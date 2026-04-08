"""Command-line interface for NTX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import enable_x64
from .geometry import example_surface
from .grids import GridSpec
from .inputfiles import run_from_input_file
from .io import load_dkes_surface, load_vmec_surface
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
    solve_surface.add_argument("--vmec", type=Path, help="path to a VMEC wout file")
    solve.add_argument("--nu-hat", type=float, required=True)
    solve.add_argument("--epsi-hat", type=float, default=None)
    solve.add_argument("--er-hat", type=float, default=None)
    solve.add_argument("--n-theta", type=int, default=5)
    solve.add_argument("--n-zeta", type=int, default=5)
    solve.add_argument("--n-xi", type=int, default=6)
    solve.add_argument("--psi-n", type=float, default=None)
    solve.add_argument("--vmec-radial-option", type=int, default=0)
    solve.add_argument("--vmec-nyquist-option", type=int, default=1)
    solve.add_argument("--vmec-mode-convention", default="reduced")
    solve.add_argument("--min-bmn-to-load", type=float, default=0.0)

    inspect = sub.add_parser("inspect-surface", help="print a surface definition")
    inspect_surface = inspect.add_mutually_exclusive_group(required=True)
    inspect_surface.add_argument(
        "--example",
        action="store_true",
        help="print the built-in example Boozer surface",
    )
    inspect_surface.add_argument("--dkes", type=Path, help="path to a DKES-format ddkes2.data file")
    inspect_surface.add_argument("--vmec", type=Path, help="path to a VMEC wout file")
    inspect.add_argument("--psi-n", type=float, default=None)
    inspect.add_argument("--vmec-radial-option", type=int, default=0)
    inspect.add_argument("--vmec-nyquist-option", type=int, default=1)
    inspect.add_argument("--vmec-mode-convention", default="reduced")
    inspect.add_argument("--min-bmn-to-load", type=float, default=0.0)

    args = parser.parse_args(args_list)
    if args.command == "solve":
        grid = GridSpec(args.n_theta, args.n_zeta, args.n_xi)
        enable_x64(grid.x64)
        surface = _load_surface(args)
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
    if getattr(args, "vmec", None) is not None:
        if args.psi_n is None:
            msg = "--psi-n is required with --vmec"
            raise ValueError(msg)
        return load_vmec_surface(
            args.vmec,
            psi_n=args.psi_n,
            vmec_radial_option=args.vmec_radial_option,
            vmec_nyquist_option=args.vmec_nyquist_option,
            vmec_mode_convention=args.vmec_mode_convention,
            min_bmn_to_load=args.min_bmn_to_load,
        )
    msg = "select one of --example, --dkes, or --vmec"
    raise ValueError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
