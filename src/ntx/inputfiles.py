"""TOML-driven NTX run configuration and execution helpers."""

from __future__ import annotations

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
    _source_sha256,
    _surface_metadata,
    _surface_metadata_table,
    _surface_source_path,
    _surface_source_text,
    _surface_table,
)
from ._inputfiles_run import run_from_input_file, save_run_npz

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
