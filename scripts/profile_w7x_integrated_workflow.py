#!/usr/bin/env python3
"""Profile the corrected integrated W7-X NTX+NEOPAX workflow."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import json
import os
import pstats
import resource
import sys
import time
from pathlib import Path
from typing import Any


def _configure_xla_flags(xla_dump_dir: Path | None) -> str | None:
    if xla_dump_dir is None:
        return None
    flags = (
        f"--xla_dump_to={xla_dump_dir} "
        "--xla_dump_hlo_as_text "
        "--xla_dump_hlo_as_proto "
        "--xla_gpu_enable_latency_hiding_scheduler=true"
    )
    existing = os.environ.get("XLA_FLAGS", "").strip()
    merged = f"{existing} {flags}".strip() if existing else flags
    os.environ["XLA_FLAGS"] = merged
    return merged


def _load_helper_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "bootstrap_current_w7x_rebuild_audit.py"
    spec = importlib.util.spec_from_file_location(
        "bootstrap_current_w7x_rebuild_audit_runtime",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _block_tree(tree: Any) -> Any:
    import jax

    jax.tree_util.tree_map(jax.block_until_ready, tree)
    return tree


def _max_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(raw) / (1024.0 * 1024.0)
    return float(raw) / 1024.0


def _write_cprofile(profile: cProfile.Profile, output_path: Path, sort_key: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.dump_stats(str(output_path))
    text_path = output_path.with_suffix(".txt")
    with text_path.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profile, stream=handle)
        stats.sort_stats(sort_key)
        stats.print_stats(80)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--xla-dump-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--cprofile-out", type=Path, default=None)
    parser.add_argument("--rebuild-scan", action="store_true")
    parser.add_argument(
        "--d33-mode",
        choices=("raw", "spitzer", "conductivity_difference"),
        default="raw",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    xla_flags = _configure_xla_flags(args.xla_dump_dir)

    module = _load_helper_module()
    import jax

    if args.backend is not None and jax.default_backend() != args.backend:
        raise SystemExit(
            f"requested --backend={args.backend} but JAX initialized {jax.default_backend()}"
        )

    trace_started = False
    if args.trace_dir is not None:
        jax.profiler.start_trace(str(args.trace_dir))
        trace_started = True

    profiler = cProfile.Profile()
    timings: dict[str, float] = {}
    points: dict[str, Any] = {}
    try:
        profiler.enable()

        start = time.perf_counter()
        reference_scan = module.load_neopax_reference_scan(module.REFERENCE_PATH)
        timings["reference_load_seconds"] = float(time.perf_counter() - start)

        if args.rebuild_scan and module.REBUILT_SCAN_PATH.exists():
            module.REBUILT_SCAN_PATH.unlink()

        start = time.perf_counter()
        rebuilt_path = module._rebuild_ntx_scan(reference_scan)
        timings["scan_prepare_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        rebuilt_scan = module.load_neopax_reference_scan(rebuilt_path)
        timings["rebuilt_scan_load_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        grid, field, species = module._build_species_and_field()
        _block_tree((field.rho_grid, species.temperature))
        timings["field_species_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        database = module.to_neopax_monoenergetic(
            rebuilt_scan,
            a_b=float(field.a_b),
            d33_mode=args.d33_mode,
        )
        _block_tree(database)
        timings["database_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        no_momentum = module.NEOPAX.get_Neoclassical_Fluxes(species, grid, field, database)
        _block_tree(no_momentum)
        timings["no_momentum_first_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        no_momentum_steady = module.NEOPAX.get_Neoclassical_Fluxes(species, grid, field, database)
        _block_tree(no_momentum_steady)
        timings["no_momentum_steady_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        with_momentum = module.NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction(
            species,
            grid,
            field,
            database,
        )
        _block_tree(with_momentum)
        timings["momentum_correction_first_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        with_momentum_steady = module.NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction(
            species,
            grid,
            field,
            database,
        )
        _block_tree(with_momentum_steady)
        timings["momentum_correction_steady_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        current = module._bootstrap_current_profile(database, grid, field, species)
        timings["current_reduction_seconds"] = float(time.perf_counter() - start)

        points["max_relative_error_vs_reference"] = float(
            module._max_relative_error(current, module.J_FINAL_REFERENCE)
        )
        midpoint = len(current) // 2
        points["midpoint_j_bootstrap"] = float(current[midpoint])
        points["reference_midpoint_j_bootstrap"] = float(module.J_FINAL_REFERENCE[midpoint])
    finally:
        profiler.disable()
        if trace_started:
            jax.profiler.stop_trace()

    if args.cprofile_out is not None:
        _write_cprofile(profiler, args.cprofile_out, sort_key="cumtime")

    payload = {
        "workflow": "w7x_integrated_ntx_neopax",
        "devices": [str(device) for device in jax.devices()],
        "default_backend": jax.default_backend(),
        "d33_mode": args.d33_mode,
        "trace_dir": str(args.trace_dir) if args.trace_dir is not None else None,
        "xla_dump_dir": str(args.xla_dump_dir) if args.xla_dump_dir is not None else None,
        "xla_flags": xla_flags,
        "rebuilt_scan": str(module.REBUILT_SCAN_PATH),
        "timings": timings,
        "max_rss_mb": _max_rss_mb(),
        "points": points,
        "notes": {
            "workflow": [
                "load W7-X reference scan",
                "prepare/rebuild NTX W7-X scan",
                "load rebuilt scan",
                "build NEOPAX field/species",
                "to_neopax_monoenergetic",
                "NEOPAX.get_Neoclassical_Fluxes",
                "NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction",
                "bootstrap-current reduction",
            ],
            "trace_format": (
                "TensorFlow/JAX profiler trace; open with TensorBoard profile "
                "plugin or Perfetto-compatible tooling"
            ),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
