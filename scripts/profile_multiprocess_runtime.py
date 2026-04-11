#!/usr/bin/env python3
# ruff: noqa: E402
"""Profile serial versus multiprocess NTX scan throughput."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORM_NAME", args.backend)
    if args.backend == "gpu":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    from ntx import (
        GridSpec,
        load_dkes_surface,
        load_vmec_surface,
    )
    from ntx.config import enable_x64

    enable_x64(True)

    nu = jnp.logspace(-4, -2, 24)
    er = jnp.linspace(0.0, 2e-3, 24)
    payload = {
        "backend": jax.default_backend(),
        "requested_backend": args.backend,
        "workers": args.workers,
        "cases": [
            _profile_case(
                "dkes_sample_multiprocess",
                load_dkes_surface(FIXTURES / "sample_surface.ddkes2.data"),
                GridSpec(9, 11, 6),
                nu,
                er,
                args.backend,
                args.workers,
            ),
            _profile_case(
                "vmec_sample_multiprocess",
                load_vmec_surface(FIXTURES / "sample_wout.nc", psi_n=0.25),
                GridSpec(9, 11, 6),
                nu,
                er,
                args.backend,
                args.workers,
            ),
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _profile_case(name, surface, grid, nu, er, backend: str, workers: int):
    import jax.numpy as jnp

    from ntx import solve_monoenergetic_multiprocess_scan, solve_monoenergetic_scan

    t0 = perf_counter()
    serial = solve_monoenergetic_scan(surface, grid, nu, er_hat=er)
    t1 = perf_counter()
    parallel = solve_monoenergetic_multiprocess_scan(
        surface,
        grid,
        nu,
        er_hat=er,
        backend=backend,
        workers=workers,
    )
    t2 = perf_counter()
    return {
        "name": name,
        "serial_seconds": t1 - t0,
        "multiprocess_seconds": t2 - t1,
        "speedup_vs_serial": (t1 - t0) / max(t2 - t1, 1e-30),
        "max_abs_delta_d11": float(jnp.max(jnp.abs(serial["D11"] - parallel["D11"]))),
        "max_abs_delta_d33": float(jnp.max(jnp.abs(serial["D33"] - parallel["D33"]))),
    }


if __name__ == "__main__":
    raise SystemExit(main())
