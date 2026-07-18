#!/usr/bin/env python3
# ruff: noqa: E402
"""Benchmark reusable prepared NTX scans with synchronized JAX execution."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from dataclasses import asdict
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
    parser.add_argument("--sizes", default="1,8,32,128")
    parser.add_argument("--modes", default="sequential:8,vectorized:8,vectorized:32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-theta", type=int, default=17)
    parser.add_argument("--n-zeta", type=int, default=25)
    parser.add_argument("--n-xi", type=int, default=16)
    parser.add_argument("--compilation-cache-dir", type=Path, default=None)
    parser.add_argument("--explain-cache-misses", action="store_true")
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
        compile_prepared_scan_solver,
        configure_compilation_cache,
        load_dkes_surface,
        load_vmec_surface,
        prepare_monoenergetic_system,
    )

    jax.config.update("jax_enable_x64", True)
    if args.compilation_cache_dir is not None:
        configure_compilation_cache(
            args.compilation_cache_dir,
            explain_cache_misses=args.explain_cache_misses,
        )
    sizes = _positive_ints(args.sizes, "--sizes")
    modes = _parse_modes(args.modes)
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    if args.surface == "dkes":
        surface = load_dkes_surface(FIXTURES / "sample_surface.ddkes2.data")
    else:
        surface = load_vmec_surface(FIXTURES / "sample_wout.nc", psi_n=0.25)
    grid = GridSpec(args.n_theta, args.n_zeta, args.n_xi)

    started = perf_counter()
    prepared = prepare_monoenergetic_system(surface, grid)
    jax.block_until_ready(prepared.geometry.b)
    preparation_seconds = perf_counter() - started

    solvers = {}
    warmup = {}
    for label, mode, batch_size in modes:
        solver = compile_prepared_scan_solver(
            prepared,
            batch_size=batch_size,
            execution_mode=mode,
        )
        solvers[label] = solver
        warmup[label] = asdict(solver.warmup())

    results = []
    for size in sizes:
        nu = jnp.logspace(-5, -1, size)
        er = jnp.linspace(-1.0e-3, 1.0e-3, size)
        entries = {}
        reference = None
        for label, solver in solvers.items():
            samples = []
            values = None
            for _ in range(args.repeats):
                started = perf_counter()
                values = solver(nu, er_hat=er)
                jax.block_until_ready(values)
                samples.append(perf_counter() - started)
            assert values is not None
            vector = jnp.stack(
                [values[key] for key in ("D11", "D31", "D13", "D33", "D33_spitzer")],
                axis=-1,
            )
            entry = {
                "seconds_min": min(samples),
                "seconds_median": sorted(samples)[len(samples) // 2],
                "cases_per_second": size / max(min(samples), 1.0e-30),
            }
            if reference is None:
                reference = vector
                entry["max_abs_delta_reference"] = 0.0
                entry["max_relative_delta_reference"] = 0.0
            else:
                delta = jnp.abs(vector - reference)
                entry["max_abs_delta_reference"] = float(jnp.max(delta))
                entry["max_relative_delta_reference"] = float(
                    jnp.max(delta / jnp.maximum(jnp.abs(reference), 1.0e-30))
                )
            entries[label] = entry
        results.append({"num_cases": size, "modes": entries})

    payload = {
        "artifact": "prepared_scan_performance",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.local_devices()],
        "surface": args.surface,
        "grid": {"n_theta": grid.n_theta, "n_zeta": grid.n_zeta, "n_xi": grid.n_xi},
        "preparation_seconds": preparation_seconds,
        "max_rss_mb": _max_rss_mb(),
        "warmup": warmup,
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _positive_ints(value: str, name: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item < 1 for item in values):
        raise SystemExit(f"{name} must contain positive integers")
    return values


def _parse_modes(value: str) -> tuple[tuple[str, str, int], ...]:
    parsed = []
    for item in value.split(","):
        mode, separator, batch = item.strip().partition(":")
        if separator != ":" or mode not in ("sequential", "vectorized"):
            raise SystemExit("--modes entries must be sequential:N or vectorized:N")
        batch_size = int(batch)
        parsed.append((f"{mode}-{batch_size}", mode, batch_size))
    if not parsed:
        raise SystemExit("--modes must not be empty")
    return tuple(parsed)


def _max_rss_mb() -> float:
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_rss / (1024.0 * 1024.0) if sys.platform == "darwin" else max_rss / 1024.0


if __name__ == "__main__":
    raise SystemExit(main())
