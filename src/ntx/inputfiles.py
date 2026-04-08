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

from .config import enable_x64
from .geometry import BoozerSurface, VmecSurface, example_surface, geometry_on_grid
from .grids import GridSpec
from .io import load_dkes_surface, load_vmec_surface
from .solver import MonoenergeticCase, TransportResult, solve_monoenergetic


@dataclass(frozen=True)
class SurfaceSpec:
    type: str
    path: Path | None = None
    psi_n: float | None = None
    vmec_radial_option: int = 0
    vmec_nyquist_option: int = 1
    min_bmn_to_load: float = 0.0


@dataclass(frozen=True)
class OutputSpec:
    npz: Path
    include_modes: bool = True


@dataclass(frozen=True)
class RunConfig:
    input_path: Path
    surface: SurfaceSpec
    grid: GridSpec
    case: MonoenergeticCase
    output: OutputSpec
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
    logging_data = _get_optional_table(data, "logging")

    surface_type = str(surface_data.get("type", "dkes"))
    surface_path_value = surface_data.get("path")
    surface_path = (
        None
        if surface_path_value is None
        else _resolve_relative_path(input_path, Path(str(surface_path_value)))
    )
    if surface_type in {"dkes", "vmec"} and surface_path is None:
        msg = f"surface.path is required when surface.type = {surface_type!r}"
        raise ValueError(msg)
    psi_n = _optional_float(surface_data.get("psi_n"))
    if surface_type == "vmec" and psi_n is None:
        msg = "surface.psi_n is required when surface.type = 'vmec'"
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
    return RunConfig(
        input_path=input_path,
        surface=SurfaceSpec(
            type=surface_type,
            path=surface_path,
            psi_n=psi_n,
            vmec_radial_option=int(surface_data.get("vmec_radial_option", 0)),
            vmec_nyquist_option=int(surface_data.get("vmec_nyquist_option", 1)),
            min_bmn_to_load=float(surface_data.get("min_bmn_to_load", 0.0)),
        ),
        grid=grid,
        case=case,
        output=output,
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

    enable_x64(config.grid.x64)
    surface = _load_surface(config.surface)
    if config.verbose:
        console.print(_surface_table(surface, config))
        console.print(_case_table(config, surface))
        console.print("[bold green]Solving monoenergetic system...[/bold green]")

    result = solve_monoenergetic(surface, config.grid, config.case)
    result_dict = result.as_dict()
    save_run_npz(config.output.npz, config, surface, result)

    if config.verbose:
        console.print(_result_table(result_dict))
        console.print(f"[bold green]Wrote[/bold green] [cyan]{config.output.npz}[/cyan]")

    return {
        "result": result_dict,
        "output_npz": str(config.output.npz),
    }


def save_run_npz(
    path: str | Path,
    config: RunConfig,
    surface: BoozerSurface | VmecSurface,
    result: TransportResult,
) -> None:
    """Save run inputs, outputs, and resolved geometry to `.npz`."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geom = geometry_on_grid(surface, config.grid)
    resolved_epsi_hat = config.case.resolved_epsi_hat(geom.psi_p)
    data: dict[str, Any] = {
        "input_path": np.asarray(str(config.input_path)),
        "surface_type": np.asarray(config.surface.type),
        "surface_path": np.asarray("" if config.surface.path is None else str(config.surface.path)),
        "surface_psi_n": np.asarray(
            np.nan if config.surface.psi_n is None else float(config.surface.psi_n)
        ),
        "surface_vmec_radial_option": np.asarray(config.surface.vmec_radial_option),
        "surface_vmec_nyquist_option": np.asarray(config.surface.vmec_nyquist_option),
        "surface_min_bmn_to_load": np.asarray(config.surface.min_bmn_to_load),
        "n_theta": np.asarray(config.grid.n_theta),
        "n_zeta": np.asarray(config.grid.n_zeta),
        "n_xi": np.asarray(config.grid.n_xi),
        "dtype": np.asarray(config.grid.dtype),
        "x64": np.asarray(config.grid.x64),
        "nu_hat": np.asarray(config.case.nu_hat),
        "epsi_hat_input": np.asarray(
            np.nan if config.case.epsi_hat is None else float(config.case.epsi_hat)
        ),
        "er_hat_input": np.asarray(
            np.nan if config.case.er_hat is None else float(config.case.er_hat)
        ),
        "epsi_hat_resolved": np.asarray(float(resolved_epsi_hat)),
        "surface_nfp": np.asarray(geom.nfp),
        "surface_iota": np.asarray(geom.iota),
        "surface_psi_p": np.asarray(np.nan if geom.psi_p is None else float(geom.psi_p)),
        "surface_b0": np.asarray(float(geom.b0)),
        "theta_grid": np.asarray(geom.grid.theta),
        "zeta_grid": np.asarray(geom.grid.zeta),
        "b": np.asarray(geom.b),
        "d_b_dtheta": np.asarray(geom.d_b_dtheta),
        "d_b_dzeta": np.asarray(geom.d_b_dzeta),
        "jacobian": np.asarray(geom.jacobian),
        "b_sub_theta": np.asarray(geom.b_sub_theta),
        "b_sub_zeta": np.asarray(geom.b_sub_zeta),
        "b_sup_theta": np.asarray(geom.b_sup_theta),
        "b_sup_zeta": np.asarray(geom.b_sup_zeta),
        "radial_drift_spatial": np.asarray(geom.radial_drift_spatial),
        "volume_prime": np.asarray(float(geom.volume_prime)),
        "b2_mean": np.asarray(float(geom.b2_mean)),
        "D11": np.asarray(float(result.D11)),
        "D31": np.asarray(float(result.D31)),
        "D13": np.asarray(float(result.D13)),
        "D33": np.asarray(float(result.D33)),
        "D33_spitzer": np.asarray(float(result.D33_spitzer)),
        "residual_l2": np.asarray(float(result.residual_l2)),
        "onsager_residual": np.asarray(float(result.onsager_residual)),
        "result_json": np.asarray(json.dumps(result.as_dict(), sort_keys=True)),
    }
    if isinstance(surface, BoozerSurface):
        data["surface_modes_m"] = np.asarray(surface.m)
        data["surface_modes_n"] = np.asarray(surface.n)
        data["surface_modes_b_cos"] = np.asarray(surface.b_cos)
        data["surface_b_theta"] = np.asarray(surface.b_theta)
        data["surface_b_zeta"] = np.asarray(surface.b_zeta)
    if isinstance(surface, VmecSurface):
        data["surface_modes_m"] = np.asarray(surface.m)
        data["surface_modes_n"] = np.asarray(surface.n)
        data["surface_modes_b_cos"] = np.asarray(surface.b_cos)
        data["surface_modes_jacobian_cos"] = np.asarray(surface.jacobian_cos)
        data["surface_modes_b_sub_theta_cos"] = np.asarray(surface.b_sub_theta_cos)
        data["surface_modes_b_sub_zeta_cos"] = np.asarray(surface.b_sub_zeta_cos)
        data["surface_modes_b_sup_theta_cos"] = np.asarray(surface.b_sup_theta_cos)
        data["surface_modes_b_sup_zeta_cos"] = np.asarray(surface.b_sup_zeta_cos)
    if config.output.include_modes:
        data["f1_modes"] = np.asarray(result.f1_modes)
        data["f3_modes"] = np.asarray(result.f3_modes)
    np.savez_compressed(output_path, **data)


def _load_surface(spec: SurfaceSpec) -> BoozerSurface | VmecSurface:
    if spec.type == "example":
        return example_surface()
    if spec.type == "dkes" and spec.path is not None:
        return load_dkes_surface(spec.path)
    if spec.type == "vmec" and spec.path is not None and spec.psi_n is not None:
        return load_vmec_surface(
            spec.path,
            psi_n=spec.psi_n,
            vmec_radial_option=spec.vmec_radial_option,
            vmec_nyquist_option=spec.vmec_nyquist_option,
            min_bmn_to_load=spec.min_bmn_to_load,
        )
    msg = f"unsupported surface.type {spec.type!r}"
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


def _surface_table(surface: BoozerSurface | VmecSurface, config: RunConfig) -> Table:
    table = Table(title="Surface", show_header=True, header_style="bold magenta")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    table.add_row("type", config.surface.type)
    table.add_row("path", "-" if config.surface.path is None else str(config.surface.path))
    if isinstance(surface, BoozerSurface):
        table.add_row("nfp", str(surface.nfp))
        table.add_row("iota", f"{surface.iota:.10g}")
        table.add_row("psi_p", f"{surface.psi_p:.10g}")
        table.add_row("B_theta", f"{surface.b_theta:.10g}")
        table.add_row("B_zeta", f"{surface.b_zeta:.10g}")
        table.add_row("modes", str(len(surface.m)))
    else:
        table.add_row("nfp", str(surface.nfp))
        table.add_row("psi_n", f"{surface.psi_n:.10g}")
        table.add_row("iota", f"{surface.iota:.10g}")
        table.add_row("B00", f"{surface.b0:.10g}")
        table.add_row("modes", str(len(surface.m)))
        table.add_row("vmec_radial_option", str(config.surface.vmec_radial_option))
        table.add_row("vmec_nyquist_option", str(config.surface.vmec_nyquist_option))
        table.add_row("min_bmn_to_load", f"{config.surface.min_bmn_to_load:.10g}")
    return table


def _case_table(config: RunConfig, surface: BoozerSurface | VmecSurface) -> Table:
    table = Table(title="Solve Parameters", show_header=True, header_style="bold magenta")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("n_theta", str(config.grid.n_theta))
    table.add_row("n_zeta", str(config.grid.n_zeta))
    table.add_row("n_xi", str(config.grid.n_xi))
    table.add_row("dtype", config.grid.dtype)
    table.add_row("x64", str(config.grid.x64))
    table.add_row("nu_hat", f"{config.case.nu_hat:.10g}")
    table.add_row("er_hat", "-" if config.case.er_hat is None else f"{config.case.er_hat:.10g}")
    table.add_row(
        "epsi_hat",
        "-" if config.case.epsi_hat is None else f"{config.case.epsi_hat:.10g}",
    )
    psi_p = surface.psi_p if isinstance(surface, VmecSurface) else surface.psi_p
    table.add_row("epsi_hat_resolved", f"{config.case.resolved_epsi_hat(psi_p):.10g}")
    table.add_row("output_npz", str(config.output.npz))
    table.add_row("include_modes", str(config.output.include_modes))
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
