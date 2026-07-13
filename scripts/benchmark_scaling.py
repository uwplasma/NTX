#!/usr/bin/env python3
# ruff: noqa: E402
"""Benchmark NTX scan scaling across scan size for serial and parallel lanes."""

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
    parser.add_argument("--sizes", type=str, default="8,16,32,64")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--n-theta", type=int, default=9)
    parser.add_argument("--n-zeta", type=int, default=11)
    parser.add_argument("--n-xi", type=int, default=6)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--skip-device-parallel",
        action="store_true",
        help="skip the single-process device-parallel path",
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
    )
    from ntx.config import enable_x64

    enable_x64(True)

    sizes = tuple(int(item) for item in args.sizes.split(",") if item.strip())
    if not sizes:
        raise SystemExit("no scan sizes were provided")

    if args.surface == "dkes":
        surface = load_dkes_surface(FIXTURES / "sample_surface.ddkes2.data")
    else:
        surface = load_vmec_surface(FIXTURES / "sample_wout.nc", psi_n=0.25)
    grid = GridSpec(args.n_theta, args.n_zeta, args.n_xi)

    results = []
    for size in sizes:
        nu = jnp.logspace(-4, -2, size)
        er = jnp.linspace(0.0, 2e-3, size)
        results.append(
            _benchmark_size(
                surface,
                grid,
                nu,
                er,
                backend=args.backend,
                workers=args.workers,
                include_device_parallel=not args.skip_device_parallel,
            )
        )

    payload = {
        "backend": jax.default_backend(),
        "surface": args.surface,
        "grid": {"n_theta": grid.n_theta, "n_zeta": grid.n_zeta, "n_xi": grid.n_xi},
        "workers": args.workers,
        "local_device_count": local_parallel_device_count(),
        "healthy_parallel_device_count": healthy_parallel_device_count(),
        "devices": [str(device) for device in jax.local_devices()],
        "max_rss_mb": _max_rss_mb(),
        "sizes": list(sizes),
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _benchmark_size(
    surface,
    grid,
    nu,
    er,
    *,
    backend: str,
    workers: int,
    include_device_parallel: bool,
) -> dict[str, object]:
    import jax.numpy as jnp

    from ntx import (
        solve_monoenergetic_multiprocess_scan,
        solve_monoenergetic_parallel_scan,
        solve_monoenergetic_scan,
    )

    serial = _steady_timed(lambda: solve_monoenergetic_scan(surface, grid, nu, er_hat=er))
    device_parallel = None
    if include_device_parallel:
        device_parallel = _steady_timed(
            lambda: solve_monoenergetic_parallel_scan(surface, grid, nu, er_hat=er)
        )
    multiprocess = _steady_timed(
        lambda: solve_monoenergetic_multiprocess_scan(
            surface,
            grid,
            nu,
            er_hat=er,
            backend=backend,
            workers=workers,
        )
    )

    entry = {
        "num_cases": int(nu.size),
        "serial_seconds": serial["seconds"],
        "serial_cases_per_second": float(nu.size) / max(serial["seconds"], 1e-30),
        "multiprocess_seconds": multiprocess["seconds"],
        "multiprocess_cases_per_second": float(nu.size) / max(multiprocess["seconds"], 1e-30),
        "multiprocess_speedup_vs_serial": serial["seconds"] / max(multiprocess["seconds"], 1e-30),
        "max_abs_delta_serial_vs_multiprocess_d11": float(
            jnp.max(jnp.abs(serial["coeffs"]["D11"] - multiprocess["coeffs"]["D11"]))
        ),
    }
    if device_parallel is not None:
        entry["device_parallel_seconds"] = device_parallel["seconds"]
        entry["device_parallel_cases_per_second"] = float(nu.size) / max(
            device_parallel["seconds"],
            1e-30,
        )
        entry["device_parallel_speedup_vs_serial"] = serial["seconds"] / max(
            device_parallel["seconds"],
            1e-30,
        )
        entry["max_abs_delta_serial_vs_device_parallel_d11"] = float(
            jnp.max(jnp.abs(serial["coeffs"]["D11"] - device_parallel["coeffs"]["D11"]))
        )
    return entry


def _timed(fn):
    import jax

    t0 = perf_counter()
    coeffs = fn()
    jax.block_until_ready(coeffs)
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
