#!/usr/bin/env python3
# ruff: noqa: E402
"""Profile core NTX workflows to identify solver and bootstrap bottlenecks."""

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

import jax
import jax.numpy as jnp

from ntx import (
    GridSpec,
    MonoenergeticCase,
    PrimitiveSpeciesProfile,
    build_bootstrap_species_profile,
    build_ntx_neopax_scan_from_surfaces,
    evaluate_bootstrap_current,
    load_dkes_surface,
    load_vmec_surface,
    prepare_monoenergetic_system,
    solve_monoenergetic_scan,
    solve_prepared_coefficient_vector,
)
from ntx.config import enable_x64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default=None)
    parser.add_argument(
        "--surface",
        choices=("dkes", "vmec"),
        default="vmec",
        help="geometry family used for the workflow profile",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if args.backend is not None:
        if args.backend == "gpu" and not any(device.platform == "gpu" for device in jax.devices()):
            raise SystemExit("requested --backend=gpu but no JAX GPU device is available")
        if jax.default_backend() != args.backend:
            raise SystemExit(
                f"requested --backend={args.backend} but JAX initialized {jax.default_backend()}"
            )

    enable_x64(True)

    if args.surface == "dkes":
        surface = load_dkes_surface(FIXTURES / "sample_surface.ddkes2.data")
    else:
        surface = load_vmec_surface(FIXTURES / "sample_wout.nc", psi_n=0.25)

    payload = profile_workflows(surface, surface_name=args.surface)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def profile_workflows(surface, *, surface_name: str) -> dict[str, object]:
    grid = GridSpec(9, 11, 6)
    nu = jnp.logspace(-4, -2, 8)
    er = jnp.linspace(0.0, 2.0e-3, 8)

    prepared_setup = _timed(lambda: prepare_monoenergetic_system(surface, grid))
    prepared = prepared_setup["value"]
    prepared_vector = jax.jit(
        lambda nu_value, er_value: solve_prepared_coefficient_vector(
            prepared,
            MonoenergeticCase(nu_value, er_hat=er_value),
        )
    )

    scan_first = _timed(lambda: solve_monoenergetic_scan(surface, grid, nu, er_hat=er))
    scan_steady = _timed(lambda: solve_monoenergetic_scan(surface, grid, nu, er_hat=er))

    prepared_first = _timed(lambda: prepared_vector(nu[0], er[0]))
    prepared_steady = _timed(lambda: prepared_vector(nu[1], er[1]))

    bootstrap = _timed(lambda: _profile_native_bootstrap(surface, grid))
    bootstrap_result = bootstrap["value"]

    return {
        "backend": jax.default_backend(),
        "surface": surface_name,
        "devices": [str(device) for device in jax.devices()],
        "grid": {"n_theta": grid.n_theta, "n_zeta": grid.n_zeta, "n_xi": grid.n_xi},
        "prepare_monoenergetic_system_seconds": prepared_setup["seconds"],
        "scan_compile_and_run_seconds": scan_first["seconds"],
        "scan_steady_seconds": scan_steady["seconds"],
        "prepared_vector_compile_and_run_seconds": prepared_first["seconds"],
        "prepared_vector_steady_seconds": prepared_steady["seconds"],
        "native_bootstrap_seconds": bootstrap["seconds"],
        "native_bootstrap_num_radii": int(bootstrap_result.rho.size),
        "native_bootstrap_current_norm": float(jnp.linalg.norm(bootstrap_result.current_density)),
        "scan_first_D11": float(scan_first["value"]["D11"][0]),
        "scan_steady_D11": float(scan_steady["value"]["D11"][0]),
        "prepared_first_D11": float(prepared_first["value"][0]),
        "prepared_steady_D11": float(prepared_steady["value"][0]),
        "speedup_prepared_vs_scan_steady": scan_steady["seconds"]
        / max(prepared_steady["seconds"], 1.0e-30),
        "bootstrap_vs_scan_steady_ratio": bootstrap["seconds"]
        / max(scan_steady["seconds"], 1.0e-30),
    }


def _profile_native_bootstrap(surface, grid):
    import jax.numpy as jnp

    rho = jnp.asarray([0.25, 0.5])
    surfaces = (surface, surface)
    nu_v = jnp.asarray([1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2])
    es = jnp.asarray([[-1.0e-3, 0.0, 1.0e-3], [-2.0e-3, 0.0, 2.0e-3]])
    er_grid = es
    drds = jnp.asarray([1.0, 1.5])
    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er_grid,
        drds=drds,
        grid=grid,
        source_name="workflow_profile",
    )
    er_profile = jnp.asarray([0.0, 0.0])
    electron = build_bootstrap_species_profile(
        rho,
        PrimitiveSpeciesProfile(
            charge=-1.0,
            nu_v=jnp.full_like(rho, 1.0e-3),
            density=jnp.asarray([3.1e19, 2.9e19]),
            temperature=jnp.asarray([1400.0, 1100.0]),
            name="e",
        ),
        mass_mp=1.0 / 1836.15267343,
        er_profile=er_profile,
        a_b=1.0,
    )
    ion = build_bootstrap_species_profile(
        rho,
        PrimitiveSpeciesProfile(
            charge=1.0,
            nu_v=jnp.full_like(rho, 1.0e-3),
            density=jnp.asarray([3.1e19, 2.9e19]),
            temperature=jnp.asarray([1400.0, 1100.0]),
            name="i",
        ),
        mass_mp=1.0,
        er_profile=er_profile,
        a_b=1.0,
    )
    return evaluate_bootstrap_current(
        scan,
        (electron, ion),
        a_b=1.0,
        er_profile=er_profile,
        n_x=32,
    )


def _timed(fn):
    t0 = perf_counter()
    value = fn()
    t1 = perf_counter()
    return {"seconds": t1 - t0, "value": value}


if __name__ == "__main__":
    raise SystemExit(main())
