#!/usr/bin/env python3
"""Compare NTX against archived thesis benchmark tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import (
    GridSpec,
    MonoenergeticCase,
    load_dkes_surface,
    load_magnetic_configuration_surface,
    solve_monoenergetic,
)
from ntx.benchmarks import (
    filter_reference_by_er_hat,
    nearest_reference_row,
    read_dkes_transport_scan,
    read_monoenergetic_table,
    read_sfincs_transport_scan,
    relative_error,
    select_monoenergetic_row,
)
from ntx.config import enable_x64
FIXTURES = ROOT / "tests" / "fixtures"
CASE_BUILDERS = {
    "W7X-EIM": lambda: _compare_w7x_eim(),
    "W7X-KJM": lambda: _compare_w7x_kjm(),
    "CIEMAT-QI": lambda: _compare_ciemat_qi(),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASE_BUILDERS),
        default=None,
        help="restrict the report to one or more named benchmark cases",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional path for the JSON summary",
    )
    args = parser.parse_args(argv)
    enable_x64(True)

    selected_cases = args.case or list(CASE_BUILDERS)
    payload = {"cases": [CASE_BUILDERS[name]() for name in selected_cases]}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _compare_w7x_eim() -> dict[str, object]:
    surface = load_dkes_surface(FIXTURES / "w7x_eim_full.ddkes2.data")
    grid = GridSpec(23, 55, 80)
    dkes = read_dkes_transport_scan(FIXTURES / "benchmarks" / "W7X-EIM" / "dkes_results.data")
    sfincs_zero = read_sfincs_transport_scan(
        FIXTURES / "benchmarks" / "W7X-EIM" / "sfincs_er0.dat",
        er_hat=0.0,
    )
    sfincs_finite = read_sfincs_transport_scan(
        FIXTURES / "benchmarks" / "W7X-EIM" / "sfincs_er3e-4.dat",
        er_hat=3e-4,
    )
    mono = read_monoenergetic_table(
        FIXTURES / "benchmarks" / "W7X-EIM" / "monoenergetic_reference.dat"
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
        if er_hat == 0.0:
            comparisons[-1]["monoenergetic"] = _mono_row_payload(
                select_monoenergetic_row(
                    mono,
                    nu_hat=nu_hat,
                    er_hat=er_hat,
                    n_theta=grid.n_theta,
                    n_zeta=grid.n_zeta,
                    n_xi=grid.n_xi,
                ),
                ntx_dict,
            )
    return {
        "name": "W7X-EIM",
        "surface_path": str(FIXTURES / "w7x_eim_full.ddkes2.data"),
        "grid": {"n_theta": 23, "n_zeta": 55, "n_xi": 80},
        "comparisons": comparisons,
    }


def _compare_w7x_kjm() -> dict[str, object]:
    surface = load_magnetic_configuration_surface(
        FIXTURES / "w7x_kjm_s0204.magnetic_configuration.dat"
    )
    grid = GridSpec(19, 79, 180)
    dkes = read_dkes_transport_scan(FIXTURES / "benchmarks" / "W7X-KJM" / "dkes_results.data")
    sfincs_zero = read_sfincs_transport_scan(
        FIXTURES / "benchmarks" / "W7X-KJM" / "sfincs_er0.dat",
        er_hat=0.0,
    )
    sfincs_finite = read_sfincs_transport_scan(
        FIXTURES / "benchmarks" / "W7X-KJM" / "sfincs_er3e-4.dat",
        er_hat=3e-4,
    )
    mono = read_monoenergetic_table(
        FIXTURES / "benchmarks" / "W7X-KJM" / "monoenergetic_reference.dat"
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
                "monoenergetic": _mono_row_payload(
                    select_monoenergetic_row(
                        mono,
                        nu_hat=nu_hat,
                        er_hat=er_hat,
                        n_theta=grid.n_theta,
                        n_zeta=grid.n_zeta,
                        n_xi=grid.n_xi,
                    ),
                    ntx_dict,
                ),
            }
        )
    return {
        "name": "W7X-KJM",
        "surface_path": str(FIXTURES / "w7x_kjm_s0204.magnetic_configuration.dat"),
        "grid": {"n_theta": 19, "n_zeta": 79, "n_xi": 180},
        "comparisons": comparisons,
    }


def _compare_ciemat_qi() -> dict[str, object]:
    surface = load_dkes_surface(FIXTURES / "ciemat_qi_s0250.ddkes2.data")
    grid = GridSpec(47, 215, 160)
    dkes = read_dkes_transport_scan(FIXTURES / "benchmarks" / "CIEMAT-QI" / "dkes_results.data")
    sfincs_zero = read_sfincs_transport_scan(
        FIXTURES / "benchmarks" / "CIEMAT-QI" / "sfincs_er0.dat",
        er_hat=0.0,
    )
    sfincs_finite = read_sfincs_transport_scan(
        FIXTURES / "benchmarks" / "CIEMAT-QI" / "sfincs_er1e-3.dat",
        er_hat=1e-3,
    )
    mono = read_monoenergetic_table(
        FIXTURES / "benchmarks" / "CIEMAT-QI" / "monoenergetic_reference.dat"
    )
    comparisons = []
    for er_hat, sfincs in ((0.0, sfincs_zero), (1e-3, sfincs_finite)):
        nu_hat = 1e-5
        ntx = solve_monoenergetic(surface, grid, MonoenergeticCase(nu_hat=nu_hat, er_hat=er_hat))
        ntx_dict = ntx.as_dict()
        dkes_row = nearest_reference_row(filter_reference_by_er_hat(dkes, er_hat), nu_hat)
        sfincs_row = nearest_reference_row(sfincs, nu_hat)
        comparison = {
            "er_hat": er_hat,
            "nu_hat": nu_hat,
            "ntx": ntx_dict,
            "dkes": _row_payload(dkes_row, ntx_dict),
            "sfincs": _row_payload(sfincs_row, ntx_dict),
        }
        if er_hat == 0.0:
            comparison["monoenergetic"] = _mono_row_payload(
                select_monoenergetic_row(
                    mono,
                    nu_hat=nu_hat,
                    er_hat=er_hat,
                    n_theta=grid.n_theta,
                    n_zeta=grid.n_zeta,
                    n_xi=grid.n_xi,
                ),
                ntx_dict,
            )
        comparisons.append(comparison)
    return {
        "name": "CIEMAT-QI",
        "surface_path": str(FIXTURES / "ciemat_qi_s0250.ddkes2.data"),
        "grid": {"n_theta": 47, "n_zeta": 215, "n_xi": 160},
        "comparisons": comparisons,
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


def _mono_row_payload(row, ntx: dict[str, float]) -> dict[str, float]:
    payload = {
        "D11": float(row["D11"]),
        "D31": float(row["D31"]),
        "D33": float(-row["D33"]),
    }
    payload["relative_error_D11"] = relative_error(ntx["D11"], payload["D11"])
    payload["relative_error_D31"] = relative_error(ntx["D31"], payload["D31"])
    payload["relative_error_D33"] = relative_error(ntx["D33"], payload["D33"])
    if "D33_spitzer" in (row.dtype.names or ()):
        payload["D33_spitzer"] = float(-row["D33_spitzer"])
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
