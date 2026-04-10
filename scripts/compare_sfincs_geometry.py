#!/usr/bin/env python3
# ruff: noqa: E402
"""Compare NTX VMEC geometry against a local sfincs_jax checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import GridSpec, compare_vmec_geometry_to_sfincs
from ntx._checkout_paths import find_sfincs_jax_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wout", type=Path, help="VMEC wout file")
    parser.add_argument("--psi-n", type=float, required=True, help="normalized toroidal-flux label")
    parser.add_argument("--n-theta", type=int, default=9)
    parser.add_argument("--n-zeta", type=int, default=11)
    parser.add_argument("--n-xi", type=int, default=6)
    parser.add_argument("--vmec-radial-option", type=int, default=0)
    parser.add_argument("--vmec-nyquist-option", type=int, default=1)
    parser.add_argument("--min-bmn-to-load", type=float, default=0.0)
    parser.add_argument(
        "--sfincs-repo",
        type=Path,
        default=find_sfincs_jax_root(),
        help="path to a local sfincs_jax checkout",
    )
    args = parser.parse_args(argv)
    if args.sfincs_repo is None:
        raise SystemExit("sfincs_jax checkout not found; pass --sfincs-repo explicitly")

    payload = compare_vmec_geometry_to_sfincs(
        wout_path=args.wout,
        psi_n=args.psi_n,
        grid=GridSpec(args.n_theta, args.n_zeta, args.n_xi),
        vmec_radial_option=args.vmec_radial_option,
        vmec_nyquist_option=args.vmec_nyquist_option,
        min_bmn_to_load=args.min_bmn_to_load,
        sfincs_repo=args.sfincs_repo,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
