"""Output payload builders and file writers for TOML-driven NTX runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._inputfiles_model import RunConfig
from ._inputfiles_reporting import (
    _algorithm_metadata,
    _geometry_metadata,
    _mode_count,
    _source_sha256,
    _surface_metadata,
    _surface_source_path,
    _surface_source_text,
)
from .geometry import BoozerSurface, VmecSurface, geometry_on_grid
from .solver import TransportResult

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
            data = {
                key: np.asarray(variable[()])
                for key, variable in handle.variables.items()
            }
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


__all__ = [
    "build_run_payload",
    "infer_run_output_format",
    "load_run_output",
    "save_run_hdf5",
    "save_run_netcdf",
    "save_run_npz",
    "save_run_output",
]
