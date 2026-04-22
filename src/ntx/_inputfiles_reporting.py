"""Input-file reporting and metadata helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import jax
import numpy as np
from rich.table import Table

from ._inputfiles_model import RunConfig
from .geometry import BoozerSurface, VmecSurface


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
        table.add_row("chi_p", "-" if surface.chi_p is None else f"{surface.chi_p:.10g}")
        table.add_row("B_theta", f"{surface.b_theta:.10g}")
        table.add_row("B_zeta", f"{surface.b_zeta:.10g}")
        table.add_row("modes", str(len(surface.m)))
    else:
        table.add_row("nfp", str(surface.nfp))
        table.add_row("requested_psi_n", f"{surface.requested_psi_n:.10g}")
        table.add_row("selected_psi_n", f"{surface.psi_n:.10g}")
        table.add_row("iota", f"{surface.iota:.10g}")
        table.add_row("B00", f"{surface.b0:.10g}")
        table.add_row("r_n", f"{surface.r_n:.10g}")
        table.add_row("r_hat", f"{surface.r_hat:.10g}")
        table.add_row("loaded_modes", str(surface.loaded_mode_count))
        table.add_row("vmec_radial_option", str(config.surface.vmec_radial_option))
        table.add_row("vmec_nyquist_option", str(config.surface.vmec_nyquist_option))
        table.add_row("vmec_mode_convention", config.surface.vmec_mode_convention)
        table.add_row("min_bmn_to_load", f"{config.surface.min_bmn_to_load:.10g}")
    return table


def _surface_metadata_table(surface: BoozerSurface | VmecSurface) -> Table:
    table = Table(title="Surface Metadata", show_header=True, header_style="bold cyan")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    for key, value in _surface_metadata(surface).items():
        table.add_row(key, str(value))
    return table


def _geometry_table(geom) -> Table:
    table = Table(title="Geometry Statistics", show_header=True, header_style="bold blue")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in _geometry_metadata(geom).items():
        table.add_row(key, str(value))
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
    try:
        scale = surface.transport_psi_scale if isinstance(surface, VmecSurface) else surface.psi_p
        resolved_epsi_hat = f"{config.case.resolved_epsi_hat(scale):.10g}"
    except ValueError:
        resolved_epsi_hat = "requires transport normalization"
    table.add_row("epsi_hat_resolved", resolved_epsi_hat)
    if isinstance(surface, VmecSurface):
        table.add_row("dpsi_hat/dr_hat", f"{surface.dpsi_hat_dr_hat:.10g}")
        table.add_row("dr_hat/dpsi_hat", f"{surface.dr_hat_dpsi_hat:.10g}")
        table.add_row("coefficient_psi_scale", "1")
    table.add_row("output_npz", str(config.output.npz))
    table.add_row("include_modes", str(config.output.include_modes))
    return table


def _algorithm_table(config: RunConfig, geom) -> Table:
    table = Table(title="Algorithm", show_header=True, header_style="bold white")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    for key, value in _algorithm_metadata(config, geom).items():
        table.add_row(key, str(value))
    return table


def _result_table(result: dict[str, float]) -> Table:
    table = Table(title="Results", show_header=True, header_style="bold green")
    table.add_column("Coefficient")
    table.add_column("Value")
    for key in ("D11", "D31", "D13", "D33", "D33_spitzer", "residual_l2", "onsager_residual"):
        table.add_row(key, f"{result[key]:.10g}")
    return table


def _output_table(path: Path, config: RunConfig) -> Table:
    table = Table(title="Output Payload", show_header=True, header_style="bold yellow")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    table.add_row("npz_path", str(path))
    table.add_row(
        "stored_geometry",
        "B, derivatives, Jacobian, drifts, covariant and contravariant components",
    )
    table.add_row(
        "stored_metadata",
        "surface, geometry, algorithm, residuals, and transport coefficients",
    )
    table.add_row("stored_source_info", "input filename, file size, and modification time")
    table.add_row("stored_source_hash", "SHA-256 checksum of the source file when available")
    table.add_row(
        "stored_source_text",
        "raw text for DKES and other text surfaces when available",
    )
    table.add_row("stored_harmonics", "surface Fourier harmonics")
    table.add_row(
        "stored_modes",
        "f1_modes and f3_modes" if config.output.include_modes else "disabled",
    )
    return table


def _surface_metadata(surface: BoozerSurface | VmecSurface) -> dict[str, Any]:
    source_path = _surface_source_path(surface)
    source_stat = None if source_path is None or not source_path.exists() else source_path.stat()
    common: dict[str, Any] = {
        "mode_count": _mode_count(surface),
        "stellarator_symmetric": bool(surface.stellarator_symmetric),
        "source_path": "-" if source_path is None else str(source_path),
        "source_size_bytes": None if source_stat is None else int(source_stat.st_size),
        "source_mtime": None if source_stat is None else float(source_stat.st_mtime),
    }
    if isinstance(surface, BoozerSurface):
        common.update(
            {
                "family": "boozer",
                "nfp": surface.nfp,
                "iota": float(surface.iota),
                "psi_p": float(surface.psi_p),
                "chi_p": None if surface.chi_p is None else float(surface.chi_p),
                "b_theta": float(surface.b_theta),
                "b_zeta": float(surface.b_zeta),
                "b0": None if surface.b0 is None else float(surface.b0),
            }
        )
        return common
    common.update(
        {
            "family": "vmec",
            "requested_psi_n": float(surface.requested_psi_n),
            "selected_psi_n": float(surface.psi_n),
            "nfp": surface.nfp,
            "ns": surface.ns,
            "mpol": surface.mpol,
            "ntor": surface.ntor,
            "total_mode_count": surface.total_mode_count,
            "loaded_mode_count": surface.loaded_mode_count,
            "iota": float(surface.iota),
            "b0": float(surface.b0),
            "phi_edge": float(surface.phi_edge),
            "psi_a_hat": float(surface.psi_a_hat),
            "r_n": float(surface.r_n),
            "r_hat": float(surface.r_hat),
            "dpsi_hat_dr_hat": float(surface.dpsi_hat_dr_hat),
            "dr_hat_dpsi_hat": float(surface.dr_hat_dpsi_hat),
            "aminor_p": None if surface.aminor_p is None else float(surface.aminor_p),
            "transport_psi_scale": float(surface.transport_psi_scale),
        }
    )
    return common


def _geometry_metadata(geom) -> dict[str, Any]:
    return {
        "surface_type": geom.surface_type,
        "n_theta": int(geom.grid.theta.size),
        "n_zeta": int(geom.grid.zeta.size),
        "n_fs": int(geom.grid.theta.size * geom.grid.zeta.size),
        "b_min": float(np.min(np.asarray(geom.b))),
        "b_max": float(np.max(np.asarray(geom.b))),
        "jacobian_min": float(np.min(np.asarray(geom.jacobian))),
        "jacobian_max": float(np.max(np.asarray(geom.jacobian))),
        "radial_drift_min": float(np.min(np.asarray(geom.radial_drift_spatial))),
        "radial_drift_max": float(np.max(np.asarray(geom.radial_drift_spatial))),
        "volume_prime": float(geom.volume_prime),
        "b2_mean": float(geom.b2_mean),
        "transport_psi_scale": float(geom.transport_psi_scale),
        "coefficient_psi_scale": float(geom.coefficient_psi_scale),
    }


def _algorithm_metadata(config: RunConfig, geom) -> dict[str, Any]:
    return {
        "solver": "dense_block_tridiagonal_schur",
        "block_storage": "dense",
        "legendre_modes_retained_for_outputs": 3,
        "nullspace_constraint": "f^(0)(theta=0,zeta=0)=0",
        "jax_backend": jax.default_backend(),
        "jax_device_count": len(jax.devices()),
        "dtype": config.grid.dtype,
        "x64": bool(config.grid.x64),
        "surface_type": geom.surface_type,
    }


def _mode_count(surface: BoozerSurface | VmecSurface) -> int:
    return int(len(surface.m))


def _surface_source_path(surface: BoozerSurface | VmecSurface) -> Path | None:
    if isinstance(surface, BoozerSurface):
        return surface.source_path
    return surface.path


def _source_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _surface_source_text(surface: BoozerSurface | VmecSurface, path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    if isinstance(surface, VmecSurface):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
