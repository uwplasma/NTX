#!/usr/bin/env python3
# ruff: noqa: E402
"""Run NTX GPU smoke/regression cases and write a JSON summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table

from ntx import (
    GridSpec,
    MonoenergeticCase,
    load_dkes_surface,
    load_vmec_surface,
    solve_monoenergetic,
)
from ntx.config import enable_x64


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    console = Console()
    enable_x64(True)
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if not gpu_devices:
        console.print("[bold red]No JAX GPU device is available.[/bold red]")
        return 1

    cases = [
        {
            "name": "dkes_sample_smoke",
            "surface": load_dkes_surface(
                ROOT / "tests" / "fixtures" / "sample_surface.ddkes2.data"
            ),
            "grid": GridSpec(5, 5, 4),
            "case": MonoenergeticCase(nu_hat=1e-5, er_hat=1e-3),
            "expected": {
                "D11": 0.009946201075081042,
                "D31": -0.1730016494448131,
                "D13": 0.17343732611105203,
                "D33": 301.4317825260738,
                "D33_spitzer": 66281.10706157789,
            },
        },
        {
            "name": "vmec_sample_smoke",
            "surface": load_vmec_surface(
                ROOT / "tests" / "fixtures" / "sample_wout.nc",
                psi_n=0.25,
            ),
            "grid": GridSpec(7, 9, 4),
            "case": MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3),
            "expected": {
                "D11": 0.004220692158278157,
                "D31": 0.02890447770891442,
                "D13": -0.02826287024440189,
                "D33": 318.55527391966154,
                "D33_spitzer": 665.6060710173264,
            },
        },
    ]

    summary: dict[str, Any] = {
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "gpu_devices": [str(device) for device in gpu_devices],
        "cases": [],
    }

    for entry in cases:
        surface = entry["surface"]
        grid = entry["grid"]
        case = entry["case"]
        solve = jax.jit(
            lambda surface=surface, grid=grid, case=case: _solve_coeff_vector(surface, grid, case)
        )

        t0 = time.perf_counter()
        warm = solve()
        warm.block_until_ready()
        compile_and_first_run_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        coeffs = solve()
        coeffs.block_until_ready()
        steady_run_s = time.perf_counter() - t1

        result_dict = {
            "D11": float(coeffs[0]),
            "D31": float(coeffs[1]),
            "D13": float(coeffs[2]),
            "D33": float(coeffs[3]),
            "D33_spitzer": float(coeffs[4]),
            "residual_l2": float(coeffs[5]),
            "onsager_residual": float(coeffs[6]),
        }
        deltas = {
            key: float(result_dict[key] - entry["expected"][key]) for key in entry["expected"]
        }
        max_rel_error = max(
            float(
                abs(result_dict[key] - entry["expected"][key])
                / max(abs(entry["expected"][key]), 1e-12)
            )
            for key in entry["expected"]
        )
        summary["cases"].append(
            {
                "name": entry["name"],
                "compile_and_first_run_s": compile_and_first_run_s,
                "steady_run_s": steady_run_s,
                "result": result_dict,
                "expected": entry["expected"],
                "delta": deltas,
                "max_relative_error": max_rel_error,
            }
        )

    table = Table(title="NTX GPU Regression")
    table.add_column("Case")
    table.add_column("Compile+Run [s]")
    table.add_column("Steady [s]")
    table.add_column("Max Rel Err")
    for entry in summary["cases"]:
        table.add_row(
            entry["name"],
            f"{entry['compile_and_first_run_s']:.6f}",
            f"{entry['steady_run_s']:.6f}",
            f"{entry['max_relative_error']:.3e}",
        )
    console.print(table)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"[bold green]Wrote[/bold green] [cyan]{args.output_json}[/cyan]")
    else:
        console.print_json(json.dumps(summary))

    failures = [
        entry["name"] for entry in summary["cases"] if entry["max_relative_error"] > 1e-6
    ]
    if failures:
        console.print(f"[bold red]Regression tolerance exceeded:[/bold red] {', '.join(failures)}")
        return 2
    return 0

def _solve_coeff_vector(surface, grid: GridSpec, case: MonoenergeticCase) -> jnp.ndarray:
    result = solve_monoenergetic(surface, grid, case)
    return jnp.asarray(
        [
            result.D11,
            result.D31,
            result.D13,
            result.D33,
            result.D33_spitzer,
            result.residual_l2,
            result.onsager_residual,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
