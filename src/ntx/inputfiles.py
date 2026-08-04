"""TOML-driven NTX run configuration and execution helpers."""

from __future__ import annotations

from ._inputfiles import (
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
from ._inputfiles import (
    build_run_payload,
    infer_run_output_format,
    load_run_output,
    save_run_hdf5,
    save_run_netcdf,
    save_run_npz,
    save_run_output,
)
from ._inputfiles import (
    _algorithm_metadata,
    _algorithm_table,
    _case_table,
    _geometry_metadata,
    _geometry_table,
    _mode_count,
    _output_table,
    _source_sha256,
    _surface_metadata,
    _surface_metadata_table,
    _surface_source_path,
    _surface_source_text,
    _surface_table,
)
from ._inputfiles import run_from_input_file

__all__ = [
    "OutputSpec",
    "RunConfig",
    "SurfaceSpec",
    "load_run_config",
    "infer_run_output_format",
    "load_run_output",
    "run_from_input_file",
    "save_run_output",
    "save_run_npz",
    "save_run_netcdf",
    "save_run_hdf5",
    "build_run_payload",
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
