#!/usr/bin/env python3
"""Profile the archive-backed fixed-field NTX+NEOPAX workflow."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
    module_path = root / "examples" / "fixed_field_parallel_flow_audit.py"
    spec = importlib.util.spec_from_file_location(
        "fixed_field_parallel_flow_audit_runtime",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("qa", "qh"), default="qa")
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--xla-dump-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    xla_flags = _configure_xla_flags(args.xla_dump_dir)

    module = _load_helper_module()
    import jax
    import jax.numpy as jnp
    import NEOPAX

    trace_started = False
    if args.trace_dir is not None:
        jax.profiler.start_trace(str(args.trace_dir))
        trace_started = True

    try:
        case = module._cases()[args.case]
        boozmn = module._ensure_boozmn(case)
        n_r = max(int(module.NTX_NEOPAX_RADIAL_POINTS), 9)
        timings: dict[str, float] = {}

        start = time.perf_counter()
        field = NEOPAX.Field.read_vmec_booz(n_r, str(case.wout_path), str(boozmn))
        timings["field_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        species = module._make_species(field, case)
        neopax_grid = NEOPAX.Grid.create_standard(n_r, 64, 2)
        nu_values = module._adaptive_nu_values(species, neopax_grid)
        timings["species_seconds"] = float(time.perf_counter() - start)

        profiles = module._archived_profiles(case)
        rho_field = module.np.asarray(field.rho_grid, dtype=float)
        rho_surface = module.np.clip(rho_field, 0.05, 0.95)
        drds = float(field.a_b) * 0.5 / module.np.clip(rho_surface, 0.05, None)
        archived_er = module._interp_profile(
            profiles.rho,
            profiles.electric_field_kv_per_m,
            rho_surface,
        )
        er_axis = float(module.np.median(archived_er)) * module.ER_AXIS_FACTORS
        er_values = module.np.repeat(er_axis[None, :], rho_surface.size, axis=0)

        start = time.perf_counter()
        surfaces = tuple(
            module.load_vmec_surface(
                case.wout_path,
                psi_n=float(rho_value**2),
                vmec_radial_option=0,
                vmec_nyquist_option=1,
                vmec_mode_convention="filtered_nyquist",
            )
            for rho_value in rho_surface
        )
        timings["surface_load_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        scan = module.build_ntx_neopax_scan_from_surfaces(
            surfaces,
            rho=jnp.asarray(rho_surface),
            nu_v=jnp.asarray(module.np.asarray(nu_values)),
            Er=jnp.asarray(module.np.asarray(er_values)),
            drds=jnp.asarray(module.np.asarray(drds)),
            grid=module.NTX_SURFACE_GRID,
            source_name=f"fixed_field_{case.name}",
        )
        _block_tree(scan)
        timings["ntx_scan_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        database = module.to_neopax_monoenergetic(scan, a_b=float(field.a_b))
        _block_tree(database)
        timings["database_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        no_momentum = module.NEOPAX.get_Neoclassical_Fluxes(species, neopax_grid, field, database)
        _block_tree(no_momentum)
        timings["no_momentum_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        with_momentum = module.NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction(
            species,
            neopax_grid,
            field,
            database,
        )
        _block_tree(with_momentum)
        timings["momentum_correction_seconds"] = float(time.perf_counter() - start)

    finally:
        if trace_started:
            jax.profiler.stop_trace()

    payload = {
        "case": args.case,
        "devices": [str(device) for device in jax.devices()],
        "default_backend": jax.default_backend(),
        "trace_dir": str(args.trace_dir) if args.trace_dir is not None else None,
        "xla_dump_dir": str(args.xla_dump_dir) if args.xla_dump_dir is not None else None,
        "xla_flags": xla_flags,
        "timings": timings,
        "max_rss_mb": _max_rss_mb(),
        "notes": {
            "workflow": [
                "field.read_vmec_booz",
                "species/profile reconstruction",
                "surface loading",
                "build_ntx_neopax_scan_from_surfaces",
                "to_neopax_monoenergetic",
                "NEOPAX.get_Neoclassical_Fluxes",
                "NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction",
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
