#!/usr/bin/env python3
"""Run NTX GPU smoke/regression cases and write a JSON summary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
from rich.console import Console
from rich.table import Table

from ntx import (
    GridSpec,
    MonoenergeticCase,
    load_dkes_surface,
    load_vmec_surface,
    solve_monoenergetic,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    console = Console()
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if not gpu_devices:
        console.print("[bold red]No JAX GPU device is available.[/bold red]")
        return 1

    cases = [
        {
            "name": "dkes_w7x_smoke",
            "surface": load_dkes_surface(
                ROOT / "tests" / "fixtures" / "w7x_eim_sample.ddkes2.data"
            ),
            "grid": GridSpec(5, 5, 4),
            "case": MonoenergeticCase(nu_hat=1e-5, er_hat=1e-3),
            "expected": {
                "D11": 0.0049033269042189735,
                "D31": 0.018601911381559297,
                "D13": -0.01605008327701478,
                "D33": 73.37743322156562,
                "D33_spitzer": 66287.9511900434,
            },
        },
        {
            "name": "vmec_w7x_smoke",
            "surface": load_vmec_surface(
                ROOT / "tests" / "fixtures" / "wout_w7x_standardConfig.nc",
                psi_n=0.25,
            ),
            "grid": GridSpec(7, 9, 4),
            "case": MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3),
            "expected": {
                "D11": 0.10146903492590549,
                "D31": 1.475206169374796,
                "D13": -1.4857960833097414,
                "D33": 244.9115457177769,
                "D33_spitzer": 668.9315902960439,
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
            lambda surface=surface, grid=grid, case=case: solve_monoenergetic(surface, grid, case)
        )

        t0 = time.perf_counter()
        warm = solve()
        warm.D33.block_until_ready()
        compile_and_first_run_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        result = solve()
        result.D33.block_until_ready()
        steady_run_s = time.perf_counter() - t1

        result_dict = result.as_dict()
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


if __name__ == "__main__":
    raise SystemExit(main())
