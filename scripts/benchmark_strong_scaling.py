#!/usr/bin/env python3
# ruff: noqa: E402
"""Benchmark NTX strong scaling for one fixed monoenergetic scan workload."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = ROOT / "tests" / "fixtures"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--surface", choices=("dkes", "vmec"), default="dkes")
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--worker-counts", type=str, default="1,2,4")
    parser.add_argument("--device-counts", type=str, default="1,2,4")
    parser.add_argument("--n-theta", type=int, default=9)
    parser.add_argument("--n-zeta", type=int, default=11)
    parser.add_argument("--n-xi", type=int, default=6)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--skip-device-parallel",
        action="store_true",
        help="skip the single-process device-parallel strong-scaling path",
    )
    parser.add_argument(
        "--skip-multiprocess",
        action="store_true",
        help="skip the multiprocess strong-scaling path",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORM_NAME", args.backend)
    if args.backend == "gpu":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp

    from ntx import (
        GridSpec,
        healthy_parallel_device_count,
        load_dkes_surface,
        load_vmec_surface,
        local_parallel_device_count,
        solve_monoenergetic_multiprocess_scan,
        solve_monoenergetic_parallel_scan,
        solve_monoenergetic_scan,
    )
    from ntx.config import enable_x64

    enable_x64(True)

    if args.num_cases < 1:
        raise SystemExit("--num-cases must be positive")
    worker_counts = _parse_positive_ints(args.worker_counts, "--worker-counts")
    device_counts = _parse_positive_ints(args.device_counts, "--device-counts")

    if args.surface == "dkes":
        surface = load_dkes_surface(FIXTURES / "sample_surface.ddkes2.data")
    else:
        surface = load_vmec_surface(FIXTURES / "sample_wout.nc", psi_n=0.25)
    grid = GridSpec(args.n_theta, args.n_zeta, args.n_xi)

    nu = jnp.logspace(-4, -2, args.num_cases)
    er = jnp.linspace(0.0, 2e-3, args.num_cases)

    serial = _steady_timed(lambda: solve_monoenergetic_scan(surface, grid, nu, er_hat=er))
    local_devices = local_parallel_device_count()
    healthy_devices = healthy_parallel_device_count()

    device_parallel_results = []
    if not args.skip_device_parallel:
        for requested_count in device_counts:
            effective_count = min(requested_count, healthy_devices, args.num_cases)
            result = _steady_timed(
                lambda requested_count=requested_count: solve_monoenergetic_parallel_scan(
                    surface,
                    grid,
                    nu,
                    er_hat=er,
                    num_devices=requested_count,
                )
            )
            device_parallel_results.append(
                _scaling_entry(
                    result,
                    serial,
                    args.num_cases,
                    parallel_count=max(effective_count, 1),
                    count_key="requested_device_count",
                    count_value=requested_count,
                    extra={"effective_device_count": effective_count},
                )
            )

    multiprocess_results = []
    if not args.skip_multiprocess:
        for workers in worker_counts:
            result = _steady_timed(
                lambda workers=workers: solve_monoenergetic_multiprocess_scan(
                    surface,
                    grid,
                    nu,
                    er_hat=er,
                    backend=args.backend,
                    workers=workers,
                )
            )
            multiprocess_results.append(
                _scaling_entry(
                    result,
                    serial,
                    args.num_cases,
                    parallel_count=workers,
                    count_key="workers",
                    count_value=workers,
                )
            )

    payload = {
        "artifact": "strong_scaling_benchmark",
        "backend": jax.default_backend(),
        "surface": args.surface,
        "grid": {"n_theta": grid.n_theta, "n_zeta": grid.n_zeta, "n_xi": grid.n_xi},
        "num_cases": args.num_cases,
        "worker_counts": worker_counts,
        "device_counts": device_counts,
        "local_device_count": local_devices,
        "healthy_parallel_device_count": healthy_devices,
        "devices": [str(device) for device in jax.local_devices()],
        "max_rss_mb": _max_rss_mb(),
        "serial": {
            "seconds": serial["seconds"],
            "cases_per_second": float(args.num_cases) / max(serial["seconds"], 1e-30),
        },
        "device_parallel": device_parallel_results,
        "multiprocess": multiprocess_results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _parse_positive_ints(value: str, name: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise SystemExit(f"{name} must contain one or more positive integers")
    return values


def _scaling_entry(
    result: dict,
    serial: dict,
    num_cases: int,
    *,
    parallel_count: int,
    count_key: str,
    count_value: int,
    extra: dict[str, int] | None = None,
) -> dict[str, object]:
    seconds = float(result["seconds"])
    speedup = float(serial["seconds"]) / max(seconds, 1e-30)
    entry: dict[str, object] = {
        count_key: count_value,
        "seconds": seconds,
        "cases_per_second": float(num_cases) / max(seconds, 1e-30),
        "speedup_vs_serial": speedup,
        "parallel_efficiency_vs_serial": speedup / max(float(parallel_count), 1.0),
        "max_abs_delta_serial_d11": _max_abs_delta_d11(serial["coeffs"], result["coeffs"]),
    }
    if extra is not None:
        entry.update(extra)
    return entry


def _max_abs_delta_d11(reference: dict, candidate: dict) -> float:
    import jax.numpy as jnp

    return float(jnp.max(jnp.abs(reference["D11"] - candidate["D11"])))


def _timed(fn):
    t0 = perf_counter()
    coeffs = fn()
    t1 = perf_counter()
    return {"seconds": t1 - t0, "coeffs": coeffs}


def _steady_timed(fn):
    _timed(fn)
    return _timed(fn)


def _max_rss_mb() -> float:
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


if __name__ == "__main__":
    raise SystemExit(main())
