#!/usr/bin/env python3
# ruff: noqa: E402
"""Profile serial versus device-parallel NTX scan throughput."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp

from ntx import (
    GridSpec,
    healthy_parallel_device_count,
    load_dkes_surface,
    load_vmec_surface,
    local_parallel_device_count,
    solve_monoenergetic_parallel_scan,
    solve_monoenergetic_scan,
)
from ntx.config import enable_x64

FIXTURES = ROOT / "tests" / "fixtures"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--num-cases",
        type=int,
        default=16,
        help="number of collisionality/electric-field scan points",
    )
    parser.add_argument(
        "--grid",
        type=_parse_grid,
        default=GridSpec(9, 11, 6),
        help="grid as Ntheta,Nzeta,Nxi",
    )
    args = parser.parse_args(argv)
    if args.num_cases < 1:
        raise ValueError("--num-cases must be positive")
    enable_x64(True)
    nu = jnp.logspace(-4, -2, args.num_cases)
    er = jnp.linspace(0.0, 2e-3, args.num_cases)
    payload = {
        "backend": jax.default_backend(),
        "healthy_parallel_device_count": healthy_parallel_device_count(),
        "local_device_count": local_parallel_device_count(),
        "devices": [str(device) for device in jax.local_devices()],
        "max_rss_mb": _max_rss_mb(),
        "cases": [
            _profile_case(
                "dkes_sample_parallel",
                load_dkes_surface(FIXTURES / "sample_surface.ddkes2.data"),
                args.grid,
                nu,
                er,
            ),
            _profile_case(
                "vmec_sample_parallel",
                load_vmec_surface(FIXTURES / "sample_wout.nc", psi_n=0.25),
                args.grid,
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


def _profile_case(name, surface, grid, nu, er):
    t0 = perf_counter()
    serial_first = solve_monoenergetic_scan(surface, grid, nu, er_hat=er)
    t1 = perf_counter()
    serial_second = solve_monoenergetic_scan(surface, grid, nu, er_hat=er)
    t2 = perf_counter()
    parallel_first = solve_monoenergetic_parallel_scan(surface, grid, nu, er_hat=er)
    t3 = perf_counter()
    parallel_second = solve_monoenergetic_parallel_scan(surface, grid, nu, er_hat=er)
    t4 = perf_counter()
    return {
        "name": name,
        "grid": {"n_theta": grid.n_theta, "n_zeta": grid.n_zeta, "n_xi": grid.n_xi},
        "num_cases": int(nu.size),
        "serial_compile_and_run_seconds": t1 - t0,
        "serial_steady_seconds": t2 - t1,
        "parallel_compile_and_run_seconds": t3 - t2,
        "parallel_steady_seconds": t4 - t3,
        "parallel_speedup_vs_serial": (t2 - t1) / max(t4 - t3, 1e-30),
        "max_abs_delta_d11": float(jnp.max(jnp.abs(serial_second["D11"] - parallel_second["D11"]))),
        "serial_first_D11": float(serial_first["D11"][0]),
        "parallel_first_D11": float(parallel_first["D11"][0]),
    }


def _parse_grid(value: str) -> GridSpec:
    try:
        pieces = tuple(int(piece) for piece in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid must have format Ntheta,Nzeta,Nxi") from exc
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("grid must have format Ntheta,Nzeta,Nxi")
    return GridSpec(*pieces)


def _max_rss_mb() -> float:
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


if __name__ == "__main__":
    raise SystemExit(main())
