"""TOML-driven NTX run configuration and execution helpers."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .benchmarks import coefficient_errors, nearest_reference_row, read_monoenergetic_table
from .geometry import BoozerSurface, example_surface
from .grids import GridSpec
from .io import load_dkes_surface
from .solver import MonoenergeticCase, TransportResult, solve_monoenergetic


@dataclass(frozen=True)
class OutputSpec:
    npz: Path
    include_modes: bool = True


@dataclass(frozen=True)
class BenchmarkSpec:
    reference_table: Path | None = None


@dataclass(frozen=True)
class RunConfig:
    input_path: Path
    surface_type: str
    surface_path: Path | None
    grid: GridSpec
    case: MonoenergeticCase
    output: OutputSpec
    benchmark: BenchmarkSpec
    verbose: bool = True


def load_run_config(path: str | Path) -> RunConfig:
    """Load a TOML input file for `ntx input.toml` runs."""

    input_path = Path(path).expanduser().resolve()
    with input_path.open("rb") as stream:
        data = tomllib.load(stream)

    surface_data = _get_table(data, "surface")
    grid_data = _get_table(data, "grid")
    case_data = _get_table(data, "case")
    output_data = _get_optional_table(data, "output")
    benchmark_data = _get_optional_table(data, "benchmark")
    logging_data = _get_optional_table(data, "logging")

    surface_type = str(surface_data.get("type", "dkes"))
    surface_path_value = surface_data.get("path")
    surface_path = (
        None
        if surface_path_value is None
        else _resolve_relative_path(input_path, Path(str(surface_path_value)))
    )
    if surface_type == "dkes" and surface_path is None:
        msg = "surface.path is required when surface.type = 'dkes'"
        raise ValueError(msg)

    grid = GridSpec(
        n_theta=int(grid_data["n_theta"]),
        n_zeta=int(grid_data["n_zeta"]),
        n_xi=int(grid_data["n_xi"]),
        dtype=str(grid_data.get("dtype", "float64")),
        x64=bool(grid_data.get("x64", True)),
    )
    case = MonoenergeticCase(
        nu_hat=float(case_data["nu_hat"]),
        epsi_hat=_optional_float(case_data.get("epsi_hat")),
        er_hat=_optional_float(case_data.get("er_hat")),
    )
    output_npz_value = output_data.get("npz", input_path.with_suffix(".npz").name)
    output = OutputSpec(
        npz=_resolve_relative_path(input_path, Path(str(output_npz_value))),
        include_modes=bool(output_data.get("include_modes", True)),
    )
    benchmark_table = benchmark_data.get("reference_table")
    benchmark = BenchmarkSpec(
        reference_table=(
            None
            if benchmark_table is None
            else _resolve_relative_path(input_path, Path(str(benchmark_table)))
        )
    )
    return RunConfig(
        input_path=input_path,
        surface_type=surface_type,
        surface_path=surface_path,
        grid=grid,
        case=case,
        output=output,
        benchmark=benchmark,
        verbose=bool(logging_data.get("verbose", True)),
    )


def run_from_input_file(
    path: str | Path,
    *,
    console: Console | None = None,
) -> dict[str, Any]:
    """Execute an NTX run from a TOML input file and save a compressed `.npz`."""

    config = load_run_config(path)
    console = Console() if console is None else console

    if config.verbose:
        console.print(
            Panel.fit(
                f"[bold]NTX[/bold]\nInput file: [cyan]{config.input_path}[/cyan]",
                title="Run",
            )
        )

    surface = _load_surface(config)
    if config.verbose:
        console.print(_surface_table(surface, config))
        console.print(_case_table(config))
        console.print("[bold green]Solving monoenergetic system...[/bold green]")

    result = solve_monoenergetic(surface, config.grid, config.case)
    result_dict = result.as_dict()

    reference_row: np.void | None = None
    comparison: dict[str, float] | None = None
    if config.benchmark.reference_table is not None:
        if config.verbose:
            console.print(
                f"[bold blue]Loading reference table[/bold blue] "
                f"[cyan]{config.benchmark.reference_table}[/cyan]"
            )
        table = read_monoenergetic_table(config.benchmark.reference_table)
        reference_row = nearest_reference_row(
            table,
            config.case.nu_hat,
            0.0 if config.case.er_hat is None else config.case.er_hat,
        )
        comparison = coefficient_errors(result_dict, reference_row)

    save_run_npz(
        config.output.npz,
        config,
        surface,
        result,
        reference_row=reference_row,
        comparison=comparison,
    )

    if config.verbose:
        console.print(_result_table(result_dict))
        if reference_row is not None and comparison is not None:
            console.print(_comparison_table(reference_row, comparison))
        console.print(f"[bold green]Wrote[/bold green] [cyan]{config.output.npz}[/cyan]")

    return {
        "result": result_dict,
        "output_npz": str(config.output.npz),
        "comparison": comparison,
    }


def save_run_npz(
    path: str | Path,
    config: RunConfig,
    surface: BoozerSurface,
    result: TransportResult,
    *,
    reference_row: np.void | None,
    comparison: dict[str, float] | None,
) -> None:
    """Save run inputs, outputs, and optional benchmark data to `.npz`."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "input_path": np.asarray(str(config.input_path)),
        "surface_type": np.asarray(config.surface_type),
        "surface_path": np.asarray("" if config.surface_path is None else str(config.surface_path)),
        "n_theta": np.asarray(config.grid.n_theta),
        "n_zeta": np.asarray(config.grid.n_zeta),
        "n_xi": np.asarray(config.grid.n_xi),
        "dtype": np.asarray(config.grid.dtype),
        "x64": np.asarray(config.grid.x64),
        "nu_hat": np.asarray(config.case.nu_hat),
        "epsi_hat": np.asarray(
            np.nan if config.case.epsi_hat is None else float(config.case.epsi_hat)
        ),
        "er_hat": np.asarray(np.nan if config.case.er_hat is None else float(config.case.er_hat)),
        "surface_nfp": np.asarray(surface.nfp),
        "surface_iota": np.asarray(surface.iota),
        "surface_psi_p": np.asarray(surface.psi_p),
        "surface_b_theta": np.asarray(surface.b_theta),
        "surface_b_zeta": np.asarray(surface.b_zeta),
        "surface_b0": np.asarray(np.nan if surface.b0 is None else float(surface.b0)),
        "D11": np.asarray(float(result.D11)),
        "D31": np.asarray(float(result.D31)),
        "D13": np.asarray(float(result.D13)),
        "D33": np.asarray(float(result.D33)),
        "D33_spitzer": np.asarray(float(result.D33_spitzer)),
        "residual_l2": np.asarray(float(result.residual_l2)),
        "onsager_residual": np.asarray(float(result.onsager_residual)),
        "result_json": np.asarray(json.dumps(result.as_dict(), sort_keys=True)),
    }
    if config.output.include_modes:
        data["f1_modes"] = np.asarray(result.f1_modes)
        data["f3_modes"] = np.asarray(result.f3_modes)
    if reference_row is not None:
        reference_names = reference_row.dtype.names
        if reference_names is not None:
            for name in reference_names:
                data[f"reference_{name}"] = np.asarray(float(reference_row[name]))
    if comparison is not None:
        for key, value in comparison.items():
            data[f"delta_{key}"] = np.asarray(value)
    np.savez_compressed(output_path, **data)


def _load_surface(config: RunConfig) -> BoozerSurface:
    if config.surface_type == "example":
        return example_surface()
    if config.surface_type == "dkes" and config.surface_path is not None:
        return load_dkes_surface(config.surface_path)
    msg = f"unsupported surface.type {config.surface_type!r}"
    raise ValueError(msg)


def _get_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        msg = f"missing [{name}] table in input file"
        raise ValueError(msg)
    return section


def _get_optional_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name, {})
    if not isinstance(section, dict):
        msg = f"[{name}] must be a TOML table"
        raise ValueError(msg)
    return section


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _resolve_relative_path(input_path: Path, value: Path) -> Path:
    return value if value.is_absolute() else (input_path.parent / value).resolve()


def _surface_table(surface: BoozerSurface, config: RunConfig) -> Table:
    table = Table(title="Surface", show_header=True, header_style="bold magenta")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    table.add_row("type", config.surface_type)
    table.add_row("path", "-" if config.surface_path is None else str(config.surface_path))
    table.add_row("nfp", str(surface.nfp))
    table.add_row("iota", f"{surface.iota:.10g}")
    table.add_row("psi_p", f"{surface.psi_p:.10g}")
    table.add_row("B_theta", f"{surface.b_theta:.10g}")
    table.add_row("B_zeta", f"{surface.b_zeta:.10g}")
    table.add_row("modes", str(len(surface.m)))
    return table


def _case_table(config: RunConfig) -> Table:
    table = Table(title="Solve Parameters", show_header=True, header_style="bold magenta")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("n_theta", str(config.grid.n_theta))
    table.add_row("n_zeta", str(config.grid.n_zeta))
    table.add_row("n_xi", str(config.grid.n_xi))
    table.add_row("dtype", config.grid.dtype)
    table.add_row("x64", str(config.grid.x64))
    table.add_row("nu_hat", f"{config.case.nu_hat:.10g}")
    table.add_row(
        "er_hat",
        "-" if config.case.er_hat is None else f"{config.case.er_hat:.10g}",
    )
    table.add_row(
        "epsi_hat",
        "-" if config.case.epsi_hat is None else f"{config.case.epsi_hat:.10g}",
    )
    table.add_row("output_npz", str(config.output.npz))
    return table


def _result_table(result: dict[str, float]) -> Table:
    table = Table(title="Results", show_header=True, header_style="bold green")
    table.add_column("Coefficient")
    table.add_column("Value")
    for key in (
        "D11",
        "D31",
        "D13",
        "D33",
        "D33_spitzer",
        "residual_l2",
        "onsager_residual",
    ):
        table.add_row(key, f"{result[key]:.10g}")
    return table


def _comparison_table(reference_row: np.void, comparison: dict[str, float]) -> Table:
    table = Table(title="Reference Comparison", show_header=True, header_style="bold yellow")
    table.add_column("Coefficient")
    table.add_column("Reference")
    table.add_column("NTX - Reference")
    for key in ("D11", "D31", "D13", "D33"):
        table.add_row(key, f"{float(reference_row[key]):.10g}", f"{comparison[key]:.10g}")
    return table
