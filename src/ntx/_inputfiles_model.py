"""Input-file configuration models and parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 environments
    import tomli as tomllib

from .geometry import BoozerSurface, VmecSurface, example_surface
from .grids import GridSpec
from .io import load_dkes_surface, load_vmec_surface
from .solver import MonoenergeticCase


@dataclass(frozen=True)
class SurfaceSpec:
    type: str
    path: Path | None = None
    psi_n: float | None = None
    vmec_radial_option: int = 0
    vmec_nyquist_option: int = 1
    vmec_mode_convention: str = "reduced"
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


def _resolve_relative_path(input_path: Path, value: Path) -> Path:
    return value if value.is_absolute() else (input_path.parent / value).resolve()
