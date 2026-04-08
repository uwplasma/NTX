#!/usr/bin/env python3
"""Compare NTX against archived DKES and SFINCS benchmark tables."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from ntx import GridSpec, MonoenergeticCase, load_dkes_surface, solve_monoenergetic
from ntx.benchmarks import (
    filter_reference_by_er_hat,
    nearest_reference_row,
    read_dkes_transport_scan,
    read_sfincs_transport_scan,
    relative_error,
)
from ntx.config import enable_x64

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional path for the JSON summary",
    )
    args = parser.parse_args(argv)
    enable_x64(True)

    payload = {
        "cases": [
            _compare_w7x_eim(),
            _compare_ciemat_qi(),
        ]
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _compare_w7x_eim() -> dict[str, object]:
    scale = 0.5237
    surface = load_dkes_surface(FIXTURES / "w7x_eim_full.ddkes2.data")
    grid = GridSpec(23, 55, 80)
    dkes = read_dkes_transport_scan(
        FIXTURES / "benchmarks" / "W7X-EIM" / "dkes_results.data",
        d11_scale=scale**-2,
        d31_scale=scale**-1,
    )
    sfincs_zero = read_sfincs_transport_scan(
        FIXTURES / "benchmarks" / "W7X-EIM" / "sfincs_er0.dat",
        er_hat=0.0,
        d11_scale=scale**-2,
        d31_scale=scale**-1,
    )
    sfincs_finite = read_sfincs_transport_scan(
        FIXTURES / "benchmarks" / "W7X-EIM" / "sfincs_er3e-4.dat",
        er_hat=3e-4,
        d11_scale=scale**-2,
        d31_scale=scale**-1,
    )
    comparisons = []
    for er_hat, sfincs in ((0.0, sfincs_zero), (3e-4, sfincs_finite)):
        nu_hat = 1e-5
        ntx = solve_monoenergetic(surface, grid, MonoenergeticCase(nu_hat=nu_hat, er_hat=er_hat))
        ntx_dict = ntx.as_dict()
        dkes_row = nearest_reference_row(filter_reference_by_er_hat(dkes, er_hat), nu_hat)
        sfincs_row = nearest_reference_row(sfincs, nu_hat)
        comparisons.append(
            {
                "er_hat": er_hat,
                "nu_hat": nu_hat,
                "ntx": ntx_dict,
                "dkes": _row_payload(dkes_row, ntx_dict),
                "sfincs": _row_payload(sfincs_row, ntx_dict),
            }
        )
    return {
        "name": "W7X-EIM",
        "surface_path": str(FIXTURES / "w7x_eim_full.ddkes2.data"),
        "grid": {"n_theta": 23, "n_zeta": 55, "n_xi": 80},
        "comparisons": comparisons,
    }


def _compare_ciemat_qi() -> dict[str, object]:
    scale = 0.4674
    surface = load_dkes_surface(FIXTURES / "ciemat_qi_s0250.ddkes2.data")
    grid = GridSpec(15, 119, 80)
    dkes = read_dkes_transport_scan(
        FIXTURES / "benchmarks" / "CIEMAT-QI" / "dkes_results.data",
        d11_scale=scale**-2,
        d31_scale=scale**-1,
    )
    sfincs_zero = read_sfincs_transport_scan(
        FIXTURES / "benchmarks" / "CIEMAT-QI" / "sfincs_er0.dat",
        er_hat=0.0,
        d11_scale=scale**-2,
        d31_scale=scale**-1,
    )
    nu_hat = 1e-4
    er_hat = 0.0
    ntx = solve_monoenergetic(surface, grid, MonoenergeticCase(nu_hat=nu_hat, er_hat=er_hat))
    ntx_dict = ntx.as_dict()
    dkes_row = nearest_reference_row(filter_reference_by_er_hat(dkes, er_hat), nu_hat)
    sfincs_row = nearest_reference_row(sfincs_zero, nu_hat)
    return {
        "name": "CIEMAT-QI",
        "surface_path": str(FIXTURES / "ciemat_qi_s0250.ddkes2.data"),
        "grid": {"n_theta": 15, "n_zeta": 119, "n_xi": 80},
        "comparisons": [
            {
                "er_hat": er_hat,
                "nu_hat": nu_hat,
                "ntx": ntx_dict,
                "dkes": _row_payload(dkes_row, ntx_dict),
                "sfincs": _row_payload(sfincs_row, ntx_dict),
            }
        ],
    }


def _row_payload(row, ntx: dict[str, float]) -> dict[str, float]:
    payload = {
        "D11": float(row["D11"]),
        "D31": float(row["D31"]),
    }
    if "D33" in row.dtype.names and row["D33"] == row["D33"]:
        payload["D33"] = float(row["D33"])
    payload["relative_error_D11"] = relative_error(ntx["D11"], payload["D11"])
    payload["relative_error_D31"] = relative_error(ntx["D31"], payload["D31"])
    if "D33" in payload:
        payload["relative_error_D33"] = relative_error(ntx["D33"], payload["D33"])
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
