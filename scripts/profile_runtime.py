#!/usr/bin/env python3
"""Profile NTX scan throughput against a Python loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

from ntx import (
    GridSpec,
    MonoenergeticCase,
    load_dkes_surface,
    load_vmec_surface,
    solve_monoenergetic,
    solve_monoenergetic_scan,
)

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

    nu = jnp.logspace(-4, -2, 8)
    er = jnp.full((8,), 1e-3)
    payload = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "cases": [
            _profile_case(
                "dkes_w7x_scan",
                load_dkes_surface(FIXTURES / "w7x_eim_full.ddkes2.data"),
                GridSpec(9, 11, 6),
                nu,
                er,
            ),
            _profile_case(
                "vmec_w7x_scan",
                load_vmec_surface(FIXTURES / "wout_w7x_standardConfig.nc", psi_n=0.25),
                GridSpec(9, 11, 6),
                nu,
                er,
            ),
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _profile_case(name, surface, grid, nu, er) -> dict[str, object]:
    t0 = perf_counter()
    scan_first = solve_monoenergetic_scan(surface, grid, nu, er_hat=er)
    t1 = perf_counter()
    scan_second = solve_monoenergetic_scan(surface, grid, nu, er_hat=er)
    t2 = perf_counter()
    loop = [
        solve_monoenergetic(
            surface,
            grid,
            MonoenergeticCase(float(nu_hat), er_hat=float(er_hat)),
        ).as_dict()
        for nu_hat, er_hat in zip(nu.tolist(), er.tolist(), strict=True)
    ]
    t3 = perf_counter()
    return {
        "name": name,
        "grid": {"n_theta": grid.n_theta, "n_zeta": grid.n_zeta, "n_xi": grid.n_xi},
        "num_cases": int(nu.size),
        "scan_compile_and_run_seconds": t1 - t0,
        "scan_steady_seconds": t2 - t1,
        "loop_seconds": t3 - t2,
        "speedup_vs_loop": (t3 - t2) / max(t2 - t1, 1e-30),
        "scan_first_D11": float(scan_first["D11"][0]),
        "scan_second_D11": float(scan_second["D11"][0]),
        "loop_first_D11": float(loop[0]["D11"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
