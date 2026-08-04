"""Reading, running, and reporting SFINCS-style input files.

The input-file model, the run driver built on it, and the output and reporting
writers that render results back out.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import enable_x64
from .geometry import BoozerSurface, VmecSurface, example_surface, geometry_on_grid
from .grids import GridSpec
from .io import load_dkes_surface, load_vmec_surface
from .solver import (
    MonoenergeticCase,
    TransportResult,
    prepare_monoenergetic_system,
    solve_prepared,
)

__all__ = [
    "build_run_payload",
    "infer_run_output_format",
    "load_run_output",
    "save_run_hdf5",
    "save_run_netcdf",
    "save_run_npz",
    "save_run_output",
]


# --- _inputfiles_model: Input-file configuration models and parsing helpers. ---

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 environments
    import tomli as tomllib


@dataclass(frozen=True)
class SurfaceSpec:
    type: str
    path: Path | None = None
    psi_n: float | None = None
    vmec_radial_option: int = 0
    vmec_nyquist_option: int = 1
    vmec_mode_convention: str = "reduced"
    min_bmn_to_load: float = 0.0


@dataclass(frozen=True, init=False)
class OutputSpec:
    path: Path
    include_modes: bool

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        npz: str | Path | None = None,
        include_modes: bool = True,
    ) -> None:
        """Output-file configuration.

        `path` is the preferred spelling. The legacy `npz` keyword remains
        accepted so existing TOML files and tests keep their public contract.
        """

        if path is not None and npz is not None:
            msg = "set only one of output.path or output.npz"
            raise ValueError(msg)
        selected = path if path is not None else npz
        if selected is None:
            msg = "an output path is required"
            raise ValueError(msg)
        object.__setattr__(self, "path", Path(selected))
        object.__setattr__(self, "include_modes", bool(include_modes))

    @property
    def npz(self) -> Path:
        """Backward-compatible alias for the selected output path."""

        return self.path


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
    output_path_value = _output_path_value(output_data, input_path)
    output = OutputSpec(
        path=_resolve_relative_path(input_path, Path(str(output_path_value))),
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
            vmec_mode_convention=str(surface_data.get("vmec_mode_convention", "reduced")),
            min_bmn_to_load=float(surface_data.get("min_bmn_to_load", 0.0)),
        ),
        grid=grid,
        case=case,
        output=output,
        verbose=bool(logging_data.get("verbose", True)),
    )


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
            vmec_mode_convention=spec.vmec_mode_convention,
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


def _output_path_value(output_data: dict[str, Any], input_path: Path) -> Any:
    path_keys = ("path", "file", "netcdf", "nc", "hdf5", "h5", "npz")
    selected = [key for key in path_keys if key in output_data]
    if len(selected) > 1:
        msg = f"set only one output path key, got {', '.join(selected)}"
        raise ValueError(msg)
    if selected:
        return output_data[selected[0]]
    return input_path.with_suffix(".nc").name


def _resolve_relative_path(input_path: Path, value: Path) -> Path:
    return value if value.is_absolute() else (input_path.parent / value).resolve()


# --- _inputfiles_reporting: Input-file reporting and metadata helpers. ---


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
    table.add_row("output_path", str(config.output.path))
    table.add_row("output_format", _output_format_label(config.output.path))
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


def _timing_table(timings: dict[str, float]) -> Table:
    table = Table(title="Runtime", show_header=True, header_style="bold green")
    table.add_column("Stage")
    table.add_column("Seconds", justify="right")
    for key in ("prepare", "solve", "write", "plot", "total"):
        if key in timings:
            table.add_row(key, f"{timings[key]:.3f}")
    return table


def _output_table(path: Path, config: RunConfig) -> Table:
    table = Table(title="Output Payload", show_header=True, header_style="bold yellow")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    table.add_row("path", str(path))
    table.add_row("format", _output_format_label(path))
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


def _output_format_label(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".nc", ".netcdf"}:
        return "NetCDF"
    if suffix == ".npz":
        return "compressed NPZ"
    if suffix in {".h5", ".hdf5"}:
        return "HDF5"
    return f"unknown ({suffix or 'no suffix'})"


# --- _inputfiles_output: Output payload builders and file writers for TOML-driven NTX runs. ---

_NETCDF_SUFFIXES = {".nc", ".netcdf"}


_HDF5_SUFFIXES = {".h5", ".hdf5"}


_NPZ_SUFFIXES = {".npz"}


def infer_run_output_format(path: str | Path) -> str:
    """Infer an NTX run-output writer from a filename suffix."""

    suffix = Path(path).suffix.lower()
    if suffix in _NETCDF_SUFFIXES:
        return "netcdf"
    if suffix in _NPZ_SUFFIXES:
        return "npz"
    if suffix in _HDF5_SUFFIXES:
        return "hdf5"
    msg = "output path must end in .nc, .netcdf, .npz, .h5, or .hdf5"
    raise ValueError(msg)


def save_run_output(
    path: str | Path,
    config: RunConfig,
    surface: BoozerSurface | VmecSurface,
    result: TransportResult,
    *,
    geometry=None,
) -> Path:
    """Save run inputs, outputs, and resolved geometry using the path suffix."""

    output_path = Path(path).expanduser().resolve()
    output_format = infer_run_output_format(output_path)
    if output_format == "npz":
        return save_run_npz(output_path, config, surface, result, geometry=geometry)
    if output_format == "netcdf":
        return save_run_netcdf(output_path, config, surface, result, geometry=geometry)
    return save_run_hdf5(output_path, config, surface, result, geometry=geometry)


def save_run_npz(
    path: str | Path,
    config: RunConfig,
    surface: BoozerSurface | VmecSurface,
    result: TransportResult,
    *,
    geometry=None,
) -> Path:
    """Save run inputs, outputs, and resolved geometry to `.npz`."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_run_payload(config, surface, result, geometry=geometry)
    np.savez_compressed(output_path, **data)  # type: ignore[arg-type]
    return output_path


def save_run_netcdf(
    path: str | Path,
    config: RunConfig,
    surface: BoozerSurface | VmecSurface,
    result: TransportResult,
    *,
    geometry=None,
) -> Path:
    """Save run inputs, outputs, and resolved geometry to an uncompressed NetCDF file."""

    from netCDF4 import Dataset

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_run_payload(config, surface, result, geometry=geometry)
    with Dataset(output_path, "w", format="NETCDF4") as handle:
        handle.setncattr("ntx_format", "run_output")
        handle.setncattr("ntx_format_version", 1)
        handle.setncattr("ntx_output_format", "netcdf")
        for key, value in data.items():
            array = np.asarray(value)
            if _is_string_array(array):
                handle.setncattr(key, _string_array_value(array))
                continue
            stored = _netcdf_numeric_array(array)
            dims = _netcdf_dims_for(key, stored.shape, data)
            for dim_name, dim_size in zip(dims, stored.shape, strict=True):
                if dim_name not in handle.dimensions:
                    handle.createDimension(dim_name, dim_size)
            variable = handle.createVariable(key, stored.dtype, dims)
            if stored.shape:
                variable[:] = stored
            else:
                variable.assignValue(stored.item())
    return output_path


def save_run_hdf5(
    path: str | Path,
    config: RunConfig,
    surface: BoozerSurface | VmecSurface,
    result: TransportResult,
    *,
    geometry=None,
) -> Path:
    """Save run inputs, outputs, and resolved geometry to an uncompressed HDF5 file."""

    import h5py

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_run_payload(config, surface, result, geometry=geometry)
    with h5py.File(output_path, "w") as handle:
        handle.attrs["ntx_format"] = "run_output"
        handle.attrs["ntx_format_version"] = 1
        handle.attrs["ntx_output_format"] = "hdf5"
        for key, value in data.items():
            array = np.asarray(value)
            if _is_string_array(array):
                handle.attrs[key] = _string_array_value(array)
                continue
            handle.create_dataset(key, data=np.asarray(array), track_times=False)
    return output_path


def load_run_output(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NTX run-output file written as `.npz`, `.nc`, or `.h5`."""

    output_path = Path(path).expanduser().resolve()
    output_format = infer_run_output_format(output_path)
    if output_format == "npz":
        with np.load(output_path, allow_pickle=False) as handle:
            return {key: np.asarray(handle[key]) for key in handle.files}
    if output_format == "netcdf":
        from netCDF4 import Dataset

        with Dataset(output_path, "r") as handle:
            data = {key: np.asarray(variable[()]) for key, variable in handle.variables.items()}
            for key in handle.ncattrs():
                if key.startswith("ntx_"):
                    continue
                data[key] = np.asarray(handle.getncattr(key))
            return data

    import h5py

    with h5py.File(output_path, "r") as handle:
        data = {key: np.asarray(handle[key][()]) for key in handle.keys()}
        for key, value in handle.attrs.items():
            if key.startswith("ntx_"):
                continue
            data[key] = np.asarray(value)
        return data


def build_run_payload(
    config: RunConfig,
    surface: BoozerSurface | VmecSurface,
    result: TransportResult,
    *,
    geometry=None,
) -> dict[str, np.ndarray]:
    """Build the file-backed run payload shared by all output formats."""

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
                        "path": str(config.output.path),
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
    return {key: np.asarray(value) for key, value in data.items()}


def _is_string_array(array: np.ndarray) -> bool:
    return array.dtype.kind in {"U", "S", "O"}


def _string_array_value(array: np.ndarray) -> str:
    if array.shape == ():
        return str(array.item())
    return json.dumps(array.tolist())


def _netcdf_numeric_array(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.dtype("bool"):
        return array.astype(np.int8)
    return np.asarray(array)


def _netcdf_dims_for(
    key: str,
    shape: tuple[int, ...],
    data: dict[str, np.ndarray],
) -> tuple[str, ...]:
    if not shape:
        return ()
    n_theta = int(np.asarray(data["n_theta"]))
    n_zeta = int(np.asarray(data["n_zeta"]))
    surface_mode_count = int(np.asarray(data["surface_mode_count"]))

    if key == "theta_grid" and shape == (n_theta,):
        return ("theta",)
    if key == "zeta_grid" and shape == (n_zeta,):
        return ("zeta",)
    if shape == (n_theta, n_zeta):
        return ("theta", "zeta")
    if key in {"f1_modes", "f3_modes"} and len(shape) == 3:
        return ("xi_mode", "theta", "zeta")
    if key.startswith("surface_modes_") and shape == (surface_mode_count,):
        return ("surface_mode",)
    if key.startswith("vmec_") and len(shape) == 1:
        return (f"{key}_dim0",)
    return tuple(f"{key}_dim{idx}" for idx in range(len(shape)))


# --- _inputfiles_run: Execution and artifact writing for TOML-driven NTX runs. ---


def run_from_input_file(
    path: str | Path,
    *,
    console: Console | None = None,
    output_path: str | Path | None = None,
    plot: bool = False,
    plot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute an NTX run from a TOML input file and save a file-backed payload."""

    config = load_run_config(path)
    if output_path is not None:
        config = RunConfig(
            input_path=config.input_path,
            surface=config.surface,
            grid=config.grid,
            case=config.case,
            output=type(config.output)(
                path=Path(output_path).expanduser(),
                include_modes=config.output.include_modes,
            ),
            verbose=config.verbose,
        )
    console = Console() if console is None else console

    if config.verbose:
        console.print(
            Panel.fit(
                f"[bold]NTX[/bold]\nInput file: [cyan]{config.input_path}[/cyan]",
                title="Run",
            )
        )

    t0 = time.perf_counter()
    enable_x64(config.grid.x64)
    surface = _load_surface(config.surface)
    prepared = prepare_monoenergetic_system(surface, config.grid)
    geom = prepared.geometry
    t_prepared = time.perf_counter()
    if config.verbose:
        console.print(_surface_table(surface, config))
        console.print(_surface_metadata_table(surface))
        console.print(_geometry_table(geom))
        console.print(_case_table(config, surface))
        console.print(_algorithm_table(config, geom))
        console.print("[bold green]Solving monoenergetic system...[/bold green]")

    result = solve_prepared(prepared, config.case)
    t_solved = time.perf_counter()
    result_dict = result.as_dict()
    written_path = save_run_output(config.output.path, config, surface, result, geometry=geom)
    t_written = time.perf_counter()
    plot_pdf = None
    if plot:
        from .plotting import plot_run_output

        plot_outputs = plot_run_output(
            written_path,
            output_prefix=None if plot_path is None else Path(plot_path).with_suffix(""),
            formats=("pdf",),
        )
        plot_pdf = str(plot_outputs[0])
    t_done = time.perf_counter()
    timings = {
        "prepare": t_prepared - t0,
        "solve": t_solved - t_prepared,
        "write": t_written - t_solved,
        "total": t_done - t0,
    }
    if plot:
        timings["plot"] = t_done - t_written

    if config.verbose:
        console.print(_result_table(result_dict))
        console.print(_timing_table(timings))
        console.print(_output_table(written_path, config))
        console.print(f"[bold green]Wrote[/bold green] [cyan]{written_path}[/cyan]")
        if plot_pdf is not None:
            console.print(f"[bold green]Wrote plot[/bold green] [cyan]{plot_pdf}[/cyan]")

    payload = {
        "result": result_dict,
        "output_path": str(written_path),
        "output_format": infer_run_output_format(written_path),
        "timing_seconds": timings,
    }
    if plot_pdf is not None:
        payload["plot_pdf"] = plot_pdf
    if written_path.suffix.lower() == ".npz":
        payload["output_npz"] = str(written_path)
    return payload
