from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from ntx.geometry import BoozerSurface, VmecSurface, example_surface, geometry_on_grid
from ntx.grids import GridSpec
from ntx.inputfiles import (
    OutputSpec,
    RunConfig,
    SurfaceSpec,
    _algorithm_metadata,
    _case_table,
    _geometry_metadata,
    _get_optional_table,
    _get_table,
    _load_surface,
    _mode_count,
    _optional_float,
    _output_table,
    _resolve_relative_path,
    _source_sha256,
    _surface_metadata,
    _surface_source_path,
    _surface_source_text,
    _surface_table,
    run_from_input_file,
    save_run_npz,
)
from ntx.solver import MonoenergeticCase, solve_monoenergetic

ROOT = Path(__file__).resolve().parents[1]
DKES = ROOT / "tests" / "fixtures" / "w7x_eim_sample.ddkes2.data"
VMEC = ROOT / "tests" / "fixtures" / "wout_w7x_standardConfig.nc"


def _example_config(
    tmp_path: Path,
    *,
    verbose: bool = False,
    include_modes: bool = False,
) -> RunConfig:
    return RunConfig(
        input_path=tmp_path / "input.toml",
        surface=SurfaceSpec(type="example"),
        grid=GridSpec(5, 5, 4),
        case=MonoenergeticCase(1e-2, epsi_hat=0.0),
        output=OutputSpec(npz=tmp_path / "result.npz", include_modes=include_modes),
        verbose=verbose,
    )


def test_inputfile_helpers_and_tables(tmp_path):
    assert _get_table({"surface": {"type": "example"}}, "surface") == {"type": "example"}
    with pytest.raises(ValueError, match="missing \\[surface\\]"):
        _get_table({}, "surface")
    assert _get_optional_table({}, "logging") == {}
    with pytest.raises(ValueError, match="must be a TOML table"):
        _get_optional_table({"logging": 1}, "logging")
    assert _optional_float(None) is None
    assert _optional_float("1.5") == 1.5
    rel = _resolve_relative_path(tmp_path / "inputs" / "run.toml", Path("x.nc"))
    assert rel == (tmp_path / "inputs" / "x.nc").resolve()
    absolute = Path(tmp_path.anchor) / "example"
    assert _resolve_relative_path(tmp_path / "run.toml", absolute) == absolute


def test_load_surface_branches_and_metadata(tmp_path):
    example = _load_surface(SurfaceSpec(type="example"))
    dkes = _load_surface(SurfaceSpec(type="dkes", path=DKES))
    vmec = _load_surface(SurfaceSpec(type="vmec", path=VMEC, psi_n=0.25))
    assert isinstance(example, BoozerSurface)
    assert isinstance(dkes, BoozerSurface)
    assert isinstance(vmec, VmecSurface)
    with pytest.raises(ValueError, match="unsupported surface.type"):
        _load_surface(SurfaceSpec(type="unsupported"))

    geom = geometry_on_grid(example, GridSpec(5, 5, 4))
    config = _example_config(tmp_path)
    assert _mode_count(example) == len(example.m)
    assert _surface_source_path(example) is None
    assert _surface_source_path(vmec) == vmec.path
    assert _surface_metadata(example)["family"] == "boozer"
    assert _surface_metadata(vmec)["family"] == "vmec"
    assert _geometry_metadata(geom)["surface_type"] == "boozer"
    assert _algorithm_metadata(config, geom)["solver"] == "dense_block_tridiagonal_schur"
    assert _surface_table(example, config).row_count > 0
    assert _surface_table(vmec, config).row_count > 0
    assert _case_table(config, example).row_count > 0
    assert _output_table(tmp_path / "out.npz", config).row_count > 0


def test_source_text_and_hash_helpers(tmp_path):
    text_path = tmp_path / "surface.txt"
    text_path.write_text("hello", encoding="utf-8")
    binary_path = tmp_path / "surface.bin"
    binary_path.write_bytes(b"\xff\xfe\x00")
    surface = example_surface()
    object.__setattr__(surface, "source_path", text_path)

    assert _source_sha256(None) is None
    assert _source_sha256(text_path) == hashlib.sha256(b"hello").hexdigest()
    assert _surface_source_text(surface, text_path) == "hello"
    assert _surface_source_text(surface, binary_path) is None

    vmec_surface = _load_surface(SurfaceSpec(type="vmec", path=VMEC, psi_n=0.25))
    assert _surface_source_text(vmec_surface, VMEC) is None


def test_save_run_npz_without_modes_and_run_from_input_file_verbose(tmp_path):
    config = _example_config(tmp_path, include_modes=False)
    config.input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "example"',
                "",
                "[grid]",
                "n_theta = 5",
                "n_zeta = 5",
                "n_xi = 4",
                "",
                "[case]",
                "nu_hat = 1e-2",
                "epsi_hat = 0.0",
                "",
                "[output]",
                'npz = "result.npz"',
                "include_modes = false",
                "",
                "[logging]",
                "verbose = true",
            ]
        ),
        encoding="utf-8",
    )
    surface = example_surface()
    result = solve_monoenergetic(surface, config.grid, config.case)
    save_run_npz(config.output.npz, config, surface, result)
    with np.load(config.output.npz) as data:
        assert "f1_modes" not in data
        assert data["surface_type"] == "example"

    console = Console(record=True)
    payload = run_from_input_file(config.input_path, console=console)
    assert payload["result"]["D11"] > 0.0
    output = console.export_text()
    assert "NTX" in output
    assert "Solving monoenergetic system" in output
