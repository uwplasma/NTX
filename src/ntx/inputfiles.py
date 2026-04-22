"""TOML-driven NTX run configuration and execution helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel

from ._inputfiles_model import (
    OutputSpec,
    RunConfig,
    SurfaceSpec,
    _get_optional_table,
    _get_table,
    _load_surface,
    _optional_float,
    _resolve_relative_path,
    load_run_config,
)
from ._inputfiles_reporting import (
    _algorithm_metadata,
    _algorithm_table,
    _case_table,
    _geometry_metadata,
    _geometry_table,
    _mode_count,
    _output_table,
    _result_table,
    _source_sha256,
    _surface_metadata,
    _surface_metadata_table,
    _surface_source_path,
    _surface_source_text,
    _surface_table,
)
from .config import enable_x64
from .geometry import BoozerSurface, VmecSurface, geometry_on_grid
from .solver import TransportResult, solve_monoenergetic

__all__ = [
    "OutputSpec",
    "RunConfig",
    "SurfaceSpec",
    "load_run_config",
    "run_from_input_file",
    "save_run_npz",
    "_algorithm_metadata",
    "_algorithm_table",
    "_case_table",
    "_geometry_metadata",
    "_geometry_table",
    "_get_optional_table",
    "_get_table",
    "_load_surface",
    "_mode_count",
    "_optional_float",
    "_output_table",
    "_resolve_relative_path",
    "_source_sha256",
    "_surface_metadata",
    "_surface_metadata_table",
    "_surface_source_path",
    "_surface_source_text",
    "_surface_table",
]


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
    geom = geometry_on_grid(surface, config.grid)
    if config.verbose:
        console.print(_surface_table(surface, config))
        console.print(_surface_metadata_table(surface))
        console.print(_geometry_table(geom))
        console.print(_case_table(config, surface))
        console.print(_algorithm_table(config, geom))
        console.print("[bold green]Solving monoenergetic system...[/bold green]")

    result = solve_monoenergetic(surface, config.grid, config.case)
    result_dict = result.as_dict()
    save_run_npz(config.output.npz, config, surface, result, geometry=geom)

    if config.verbose:
        console.print(_result_table(result_dict))
        console.print(_output_table(config.output.npz, config))
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
    *,
    geometry=None,
) -> None:
    """Save run inputs, outputs, and resolved geometry to `.npz`."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geom = geometry if geometry is not None else geometry_on_grid(surface, config.grid)
    resolved_epsi_hat = config.case.resolved_epsi_hat(geom.transport_psi_scale)
    surface_meta = _surface_metadata(surface)
    geometry_meta = _geometry_metadata(geom)
    algorithm_meta = _algorithm_metadata(config, geom)
    source_path = _surface_source_path(surface)
    source_stat = None if source_path is None or not source_path.exists() else source_path.stat()
    source_sha256 = _source_sha256(source_path)
    source_text = _surface_source_text(surface, source_path)
    data: dict[str, Any] = {
        "input_path": np.asarray(str(config.input_path)),
        "input_toml_text": np.asarray(config.input_path.read_text(encoding="utf-8")),
        "surface_type": np.asarray(config.surface.type),
        "surface_path": np.asarray("" if config.surface.path is None else str(config.surface.path)),
        "surface_psi_n": np.asarray(
            np.nan if config.surface.psi_n is None else float(config.surface.psi_n)
        ),
        "surface_vmec_radial_option": np.asarray(config.surface.vmec_radial_option),
        "surface_vmec_nyquist_option": np.asarray(config.surface.vmec_nyquist_option),
        "surface_vmec_mode_convention": np.asarray(config.surface.vmec_mode_convention),
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
        "surface_transport_psi_scale": np.asarray(float(geom.transport_psi_scale)),
        "surface_coefficient_psi_scale": np.asarray(float(geom.coefficient_psi_scale)),
        "surface_b0": np.asarray(float(geom.b0)),
        "surface_mode_count": np.asarray(_mode_count(surface)),
        "surface_stellarator_symmetric": np.asarray(bool(surface.stellarator_symmetric)),
        "surface_source_name": np.asarray("" if source_path is None else source_path.name),
        "surface_source_size_bytes": np.asarray(
            np.nan if source_stat is None else float(source_stat.st_size)
        ),
        "surface_source_mtime": np.asarray(
            np.nan if source_stat is None else float(source_stat.st_mtime)
        ),
        "surface_source_sha256": np.asarray("" if source_sha256 is None else source_sha256),
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
        "surface_metadata_json": np.asarray(json.dumps(surface_meta, sort_keys=True)),
        "geometry_metadata_json": np.asarray(json.dumps(geometry_meta, sort_keys=True)),
        "algorithm_metadata_json": np.asarray(json.dumps(algorithm_meta, sort_keys=True)),
        "run_config_json": np.asarray(
            json.dumps(
                {
                    "surface": {
                        "type": config.surface.type,
                        "path": None if config.surface.path is None else str(config.surface.path),
                        "psi_n": config.surface.psi_n,
                        "vmec_radial_option": config.surface.vmec_radial_option,
                        "vmec_nyquist_option": config.surface.vmec_nyquist_option,
                        "vmec_mode_convention": config.surface.vmec_mode_convention,
                        "min_bmn_to_load": config.surface.min_bmn_to_load,
                    },
                    "grid": {
                        "n_theta": config.grid.n_theta,
                        "n_zeta": config.grid.n_zeta,
                        "n_xi": config.grid.n_xi,
                        "dtype": config.grid.dtype,
                        "x64": config.grid.x64,
                    },
                    "case": {
                        "nu_hat": config.case.nu_hat,
                        "epsi_hat": config.case.epsi_hat,
                        "er_hat": config.case.er_hat,
                    },
                    "output": {
                        "npz": str(config.output.npz),
                        "include_modes": config.output.include_modes,
                    },
                    "verbose": config.verbose,
                },
                sort_keys=True,
            )
        ),
        "result_json": np.asarray(json.dumps(result.as_dict(), sort_keys=True)),
    }
    if source_text is not None:
        data["surface_source_text"] = np.asarray(source_text)
    if isinstance(surface, BoozerSurface):
        data["surface_modes_m"] = np.asarray(surface.m)
        data["surface_modes_n"] = np.asarray(surface.n)
        data["surface_modes_b_cos"] = np.asarray(surface.b_cos)
        data["surface_b_theta"] = np.asarray(surface.b_theta)
        data["surface_b_zeta"] = np.asarray(surface.b_zeta)
        data["surface_chi_p"] = np.asarray(
            np.nan if surface.chi_p is None else float(surface.chi_p)
        )
    if isinstance(surface, VmecSurface):
        data["surface_modes_m"] = np.asarray(surface.m)
        data["surface_modes_n"] = np.asarray(surface.n)
        data["surface_modes_b_cos"] = np.asarray(surface.b_cos)
        data["surface_modes_jacobian_cos"] = np.asarray(surface.jacobian_cos)
        data["surface_modes_b_sub_theta_cos"] = np.asarray(surface.b_sub_theta_cos)
        data["surface_modes_b_sub_zeta_cos"] = np.asarray(surface.b_sub_zeta_cos)
        data["surface_modes_b_sup_theta_cos"] = np.asarray(surface.b_sup_theta_cos)
        data["surface_modes_b_sup_zeta_cos"] = np.asarray(surface.b_sup_zeta_cos)
        data["vmec_requested_psi_n"] = np.asarray(surface.requested_psi_n)
        data["vmec_selected_psi_n"] = np.asarray(surface.psi_n)
        data["vmec_ns"] = np.asarray(surface.ns)
        data["vmec_mpol"] = np.asarray(surface.mpol)
        data["vmec_ntor"] = np.asarray(surface.ntor)
        data["vmec_total_mode_count"] = np.asarray(surface.total_mode_count)
        data["vmec_loaded_mode_count"] = np.asarray(surface.loaded_mode_count)
        data["vmec_psi_a_hat"] = np.asarray(surface.psi_a_hat)
        data["vmec_phi_edge"] = np.asarray(surface.phi_edge)
        data["vmec_r_n"] = np.asarray(surface.r_n)
        data["vmec_r_hat"] = np.asarray(surface.r_hat)
        data["vmec_dpsi_hat_dr_hat"] = np.asarray(surface.dpsi_hat_dr_hat)
        data["vmec_dr_hat_dpsi_hat"] = np.asarray(surface.dr_hat_dpsi_hat)
        data["vmec_aminor_p"] = np.asarray(
            np.nan if surface.aminor_p is None else float(surface.aminor_p)
        )
    if config.output.include_modes:
        data["f1_modes"] = np.asarray(result.f1_modes)
        data["f3_modes"] = np.asarray(result.f3_modes)
    np.savez_compressed(output_path, **data)
