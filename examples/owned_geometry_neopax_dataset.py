#!/usr/bin/env python3
"""Build owned NTX+NEOPAX scans from JAX-native VMEC geometry inputs.

The goal of this example is provenance control.  It generates NTX surfaces,
monoenergetic coefficients, NEOPAX-style HDF5 tables, and an interpolation-path
audit from one owned input/wout pair at a time.  It also stores compact
profile flux/current responses from the same scan tables.  The resulting
artifacts are not SFINCS parity claims; they are self-contained datasets on
which SFINCS, Redl, and NTX+NEOPAX comparisons can later be run without mixing
unrelated geometry, normalization, or interpolation conventions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    build_ntx_neopax_scan_from_surfaces,
    surface_from_vmec_jax_vmec_wout_file,
    surface_from_vmec_jax_wout,
    write_neopax_scan_hdf5,
)
from ntx._checkout_paths import find_vmec_jax_root  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_geometry_neopax_dataset"
DATABASE_DIR = ROOT / "docs" / "_static" / "owned_geometry_neopax_database"
DEFAULT_RHO = (0.30, 0.50, 0.70)
DEFAULT_NU_V = (1.0e-3, 1.0e-2)
DEFAULT_ES = (0.0,)
DEFAULT_GRID = GridSpec(9, 11, 8)
COEFFICIENTS = ("D11", "D13", "D33")
EPS = 1.0e-30
FINITE_BETA_ROOT_CANDIDATES = (
    Path("/Users/rogeriojorge/local/single_stage_finite_beta"),
    Path("/Users/rogeriojorge/local/single_stage_optimization_finite_beta"),
    Path("/Users/rogeriojorge/local/tests/single_stage_optimization_finite_beta"),
)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class OwnedJaxGeometryCase:
    """One local VMEC input/wout pair with explicit provenance."""

    id: str
    label: str
    family: str
    source: str
    input_path: Path
    wout_path: Path
    notes: str = ""
    primary_geometry_path: str = "booz_xform_jax"

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["wout_path"] = str(self.wout_path)
        payload["wout_metadata"] = _wout_metadata(self.wout_path)
        return payload


def find_single_stage_finite_beta_root() -> Path | None:
    """Return the local finite-beta example checkout requested by this lane."""

    for candidate in FINITE_BETA_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _scalar_from_dataset(path: Path, name: str) -> float | None:
    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError:
        return None
    try:
        with Dataset(path) as dataset:
            if name not in dataset.variables:
                return None
            values = np.asarray(dataset.variables[name][:], dtype=float).reshape(-1)
    except Exception:
        return None
    if values.size == 0:
        return None
    return float(values[-1])


def _wout_metadata(path: Path) -> dict[str, float | int | str | bool | None]:
    metadata: dict[str, float | int | str | bool | None] = {
        "path": str(path),
        "exists": path.exists(),
    }
    for name in ("nfp", "ns", "mpol", "ntor"):
        value = _scalar_from_dataset(path, name)
        metadata[name] = None if value is None else int(round(value))
    for name in ("beta_vol", "betatotal", "Aminor_p", "Rmajor_p", "b0"):
        metadata[name] = _scalar_from_dataset(path, name)
    return metadata


def _case_minor_radius_for_drds(case: OwnedJaxGeometryCase) -> float:
    """Return the minor radius used for the NEOPAX `dr/ds` channel."""

    value = _wout_metadata(case.wout_path).get("Aminor_p")
    if value is None or not np.isfinite(float(value)) or float(value) <= 0.0:
        return 1.0
    return float(value)


def _case_psi_p_for_boozer(case: OwnedJaxGeometryCase) -> float:
    """Return the physical edge toroidal flux scale for Boozer-surface solves."""

    value = _scalar_from_dataset(case.wout_path, "phi")
    if value is None or not np.isfinite(float(value)) or float(value) == 0.0:
        return 1.0
    return abs(float(value)) / (2.0 * np.pi)


def _drds_from_minor_radius(rho: np.ndarray, a_b: float) -> np.ndarray:
    """Map normalized toroidal flux `s=rho^2` to minor radius `r=a_b*rho`."""

    rho_arr = np.asarray(rho, dtype=float)
    if np.any(rho_arr <= 0.0):
        raise ValueError("rho values must be positive when building NEOPAX drds")
    return float(a_b) / (2.0 * rho_arr)


def _add_case(
    cases: list[OwnedJaxGeometryCase],
    *,
    root: Path | None,
    input_name: str,
    wout_name: str,
    case_id: str,
    label: str,
    family: str,
    notes: str,
) -> None:
    if root is None:
        return
    data = root / "examples" / "data"
    input_path = data / input_name
    wout_path = data / wout_name
    if input_path.exists() and wout_path.exists():
        cases.append(
            OwnedJaxGeometryCase(
                id=case_id,
                label=label,
                family=family,
                source="vmec_jax examples/data",
                input_path=input_path.resolve(),
                wout_path=wout_path.resolve(),
                notes=notes,
            )
        )


def _add_local_case(
    cases: list[OwnedJaxGeometryCase],
    *,
    root: Path | None,
    input_relative: str,
    wout_relative: str,
    case_id: str,
    label: str,
    family: str,
    source: str,
    notes: str,
    primary_geometry_path: str = "booz_xform_jax",
) -> None:
    if root is None:
        return
    input_path = root / input_relative
    wout_path = root / wout_relative
    if input_path.exists() and wout_path.exists():
        cases.append(
            OwnedJaxGeometryCase(
                id=case_id,
                label=label,
                family=family,
                source=source,
                input_path=input_path.resolve(),
                wout_path=wout_path.resolve(),
                notes=notes,
                primary_geometry_path=primary_geometry_path,
            )
        )


def discover_owned_case_specs() -> tuple[OwnedJaxGeometryCase, ...]:
    """Return local input/wout pairs suitable for JAX-native scan generation."""

    vmec_jax_root = find_vmec_jax_root()
    finite_beta_root = find_single_stage_finite_beta_root()
    cases: list[OwnedJaxGeometryCase] = []
    _add_local_case(
        cases,
        root=finite_beta_root,
        input_relative="test/input.LandremanPaul2021_QA_lowres_pressure_current",
        wout_relative="test/wout_LandremanPaul2021_QA_lowres_pressure_current.nc",
        case_id="finite_beta_qa_pressure_current",
        label="Finite-beta QA pressure/current",
        family="QA finite beta",
        source="single-stage finite-beta test case",
        notes=(
            "finite-beta QA stellarator with pressure and toroidal current profiles "
            "representable by the current vmec_jax input reader"
        ),
    )
    _add_local_case(
        cases,
        root=finite_beta_root,
        input_relative="optimization_finitebeta_nfp3_QH_stage1/input.final",
        wout_relative="optimization_finitebeta_nfp3_QH_stage1/wout_final.nc",
        case_id="finite_beta_nfp3_qh_stage1",
        label="Finite-beta optimized NFP3 QH",
        family="QH finite beta",
        source="single-stage finite-beta optimized case",
        notes=(
            "optimized finite-beta QH stellarator; direct wout-harmonic NTX path is "
            "usable, while the current vmec_jax Boozer path reports the unresolved "
            "cubic-spline current-profile support gap"
        ),
        primary_geometry_path="vmec_jax_wout_cubic",
    )
    _add_local_case(
        cases,
        root=finite_beta_root,
        input_relative="optimization_finitebeta_nfp3_QI_stage1/input.final",
        wout_relative="optimization_finitebeta_nfp3_QI_stage1/wout_final.nc",
        case_id="finite_beta_nfp3_qi_stage1",
        label="Finite-beta optimized NFP3 QI",
        family="QI finite beta",
        source="single-stage finite-beta optimized case",
        notes=(
            "optimized finite-beta QI stellarator for broader family sweeps; same "
            "current-profile support limitation as the optimized QH case"
        ),
        primary_geometry_path="vmec_jax_wout_cubic",
    )
    _add_case(
        cases,
        root=vmec_jax_root,
        input_name="input.circular_tokamak",
        wout_name="wout_circular_tokamak.nc",
        case_id="circular_tokamak",
        label="Circular tokamak",
        family="tokamak",
        notes="axisymmetric smoke baseline with a matching VMEC input and wout",
    )
    _add_case(
        cases,
        root=vmec_jax_root,
        input_name="input.shaped_tokamak_pressure",
        wout_name="wout_shaped_tokamak_pressure.nc",
        case_id="shaped_tokamak_pressure",
        label="Shaped tokamak",
        family="tokamak",
        notes="axisymmetric shaped-pressure baseline",
    )
    _add_case(
        cases,
        root=vmec_jax_root,
        input_name="input.LandremanPaul2021_QA_lowres",
        wout_name="wout_LandremanPaul2021_QA_lowres.nc",
        case_id="precise_qs_qa_lowres",
        label="Precise-QS QA",
        family="QA",
        notes="low-resolution precise-QS QA public example",
    )
    _add_case(
        cases,
        root=vmec_jax_root,
        input_name="input.nfp4_QH_warm_start",
        wout_name="wout_nfp4_QH_warm_start.nc",
        case_id="nfp4_qh_warm_start",
        label="NFP4 QH",
        family="QH",
        notes="QH warm-start example with owned input/wout provenance",
    )
    _add_case(
        cases,
        root=vmec_jax_root,
        input_name="input.nfp3_QI_fixed_resolution_final",
        wout_name="wout_nfp3_QI_fixed_resolution_final.nc",
        case_id="nfp3_qi_fixed_resolution",
        label="NFP3 QI",
        family="QI",
        notes="fixed-resolution QI-family public example",
    )
    return tuple(cases)


def _grid_payload(grid: GridSpec) -> dict[str, int]:
    return {
        "n_theta": int(grid.n_theta),
        "n_zeta": int(grid.n_zeta),
        "n_xi": int(grid.n_xi),
    }


def _coefficient_summary(scan) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name in COEFFICIENTS:
        values = np.asarray(getattr(scan, name), dtype=float)
        summary[name] = {
            "shape": list(values.shape),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "first": float(values.reshape(-1)[0]),
            "values": values.tolist(),
        }
    if scan.iota is not None:
        summary["iota"] = np.asarray(scan.iota, dtype=float).tolist()
    if scan.b00 is not None:
        summary["B00"] = np.asarray(scan.b00, dtype=float).tolist()
    return summary


def _profile_response_summary(scan) -> dict[str, object]:
    """Return compact profile responses from the raw scan coefficients."""

    rho = np.asarray(scan.rho, dtype=float)
    density = 1.0 - rho**8 + 0.2
    temperature = 1.0 - rho**2 + 0.2
    a1 = (-8.0 * rho**7) / density
    a3 = (-2.0 * rho) / temperature
    d11 = np.asarray(scan.D11, dtype=float)[:, 0, 0]
    d13 = np.asarray(scan.D13, dtype=float)[:, 0, 0]
    d31 = (
        np.asarray(scan.D31, dtype=float)[:, 0, 0]
        if scan.D31 is not None
        else -np.asarray(scan.D13, dtype=float)[:, 0, 0]
    )
    d33 = np.asarray(scan.D33, dtype=float)[:, 0, 0]

    electron_particle = -(d11 * a1 + d13 * a3)
    ion_particle = -0.5 * (d11 * a1 + d13 * a3)
    electron_current = d31 * a1 + d33 * a3
    ion_current = -0.5 * (d31 * a1 + d33 * a3)
    ambipolar_residual_response = -electron_particle + ion_particle
    bootstrap_current_response = electron_current + ion_current
    scale = max(float(np.nanmax(np.abs(bootstrap_current_response))), EPS)
    normalized_current = bootstrap_current_response / scale

    return {
        "profile_model": (
            "dimensionless two-species response using analytic n=1-rho^8+0.2 and "
            "T=1-rho^2+0.2 gradients on the first scan collisionality and field point"
        ),
        "density": density.tolist(),
        "temperature": temperature.tolist(),
        "A1": a1.tolist(),
        "A3": a3.tolist(),
        "electron_particle_flux_response": electron_particle.tolist(),
        "ion_particle_flux_response": ion_particle.tolist(),
        "ambipolar_residual_response": ambipolar_residual_response.tolist(),
        "electron_current_response": electron_current.tolist(),
        "ion_current_response": ion_current.tolist(),
        "bootstrap_current_response": bootstrap_current_response.tolist(),
        "bootstrap_current_response_normalized": normalized_current.tolist(),
        "current_response_objective": float(np.trapezoid(normalized_current**2, rho)),
    }


def _max_relative_difference(reference, candidate) -> float:
    reference_arr = np.asarray(reference, dtype=float)
    candidate_arr = np.asarray(candidate, dtype=float)
    if reference_arr.shape != candidate_arr.shape:
        return float("nan")
    denominator = np.maximum(np.abs(reference_arr), EPS)
    difference = np.abs(candidate_arr - reference_arr) / denominator
    return float(np.nanmax(difference))


def _scan_path_metadata(path_key: str) -> dict[str, str]:
    if path_key == "booz_xform_jax":
        return {
            "geometry_path": ("vmec_jax.read_wout -> booz_xform_jax -> NTX BoozerSurface"),
            "interpolation_owner": (
                "The requested radial points are transformed directly from the matching "
                "VMEC input/wout pair; no external profile or database interpolation is used."
            ),
            "normalization_owner": (
                "NTX owns the raw monoenergetic coefficients; NEOPAX export uses "
                "D11*drds^2, D13*drds, and nu_v*D33."
            ),
        }
    return {
        "geometry_path": "vmec_jax read_wout -> NTX VmecSurface",
        "interpolation_owner": (
            "NTX cubic interpolation in VMEC radial coordinate s on the wout harmonic "
            "tables; this is an interpolation-path audit, not the promoted Boozer path."
        ),
        "normalization_owner": (
            "Same NTX raw coefficient and NEOPAX export convention as the Boozer path."
        ),
    }


def _build_surfaces_for_path(
    case: OwnedJaxGeometryCase,
    *,
    rho: np.ndarray,
    path_key: str,
    mboz: int,
    nboz: int,
    min_bmn_to_load: float,
) -> tuple[object, ...]:
    surfaces: list[object] = []
    for rho_value in rho:
        s_value = float(rho_value**2)
        if path_key == "booz_xform_jax":
            surface = surface_from_vmec_jax_wout(
                input_path=case.input_path,
                wout_path=case.wout_path,
                s=s_value,
                mboz=mboz,
                nboz=nboz,
                psi_p=_case_psi_p_for_boozer(case),
                min_bmn_to_load=min_bmn_to_load,
            )
        elif path_key == "vmec_jax_wout_cubic":
            surface = surface_from_vmec_jax_vmec_wout_file(
                case.wout_path,
                s=s_value,
                min_bmn_to_load=min_bmn_to_load,
            )
        else:
            raise ValueError(f"unknown geometry path {path_key!r}")
        surfaces.append(surface)
    return tuple(surfaces)


def _build_scan_for_path(
    case: OwnedJaxGeometryCase,
    *,
    rho: np.ndarray,
    nu_v: np.ndarray,
    es_values: np.ndarray,
    drds: np.ndarray,
    grid: GridSpec,
    path_key: str,
    mboz: int,
    nboz: int,
    min_bmn_to_load: float,
):
    surfaces = _build_surfaces_for_path(
        case,
        rho=rho,
        path_key=path_key,
        mboz=mboz,
        nboz=nboz,
        min_bmn_to_load=min_bmn_to_load,
    )
    es = np.tile(es_values[None, :], (rho.size, 1))
    return build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=jnp.asarray(rho),
        nu_v=jnp.asarray(nu_v),
        Es=jnp.asarray(es),
        drds=jnp.asarray(drds),
        grid=grid,
        source_name=f"owned:{case.id}:{path_key}",
    )


def _write_scan_file(scan, output_dir: Path, case_id: str, path_key: str) -> dict[str, object]:
    path = output_dir / f"{case_id}_{path_key}.h5"
    try:
        written = write_neopax_scan_hdf5(scan, path)
    except ModuleNotFoundError as exc:
        return {
            "status": "skipped",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "written",
        "path": _display_path(Path(written)),
    }


def _solve_case(
    case: OwnedJaxGeometryCase,
    *,
    rho: np.ndarray,
    nu_v: np.ndarray,
    es_values: np.ndarray,
    drds: np.ndarray,
    grid: GridSpec,
    output_dir: Path,
    compare_vmec_harmonic: bool,
    write_hdf5: bool,
    mboz: int,
    nboz: int,
    min_bmn_to_load: float,
) -> dict[str, object]:
    paths = ["booz_xform_jax"]
    if compare_vmec_harmonic:
        paths.append("vmec_jax_wout_cubic")

    case_payload: dict[str, object] = {
        **case.as_payload(),
        "status": "complete",
        "requested_primary_geometry_path": case.primary_geometry_path,
        "primary_scan_path": None,
        "scan_paths": {},
        "interpolation_audit": {},
        "geometry_path_blockers": [],
        "drds": drds.tolist(),
    }
    scans: dict[str, object] = {}
    for path_key in paths:
        start = time.perf_counter()
        try:
            scan = _build_scan_for_path(
                case,
                rho=rho,
                nu_v=nu_v,
                es_values=es_values,
                drds=drds,
                grid=grid,
                path_key=path_key,
                mboz=mboz,
                nboz=nboz,
                min_bmn_to_load=min_bmn_to_load,
            )
        except Exception as exc:  # pragma: no cover - depends on optional external stacks.
            error = f"{type(exc).__name__}: {exc}"
            case_payload["scan_paths"][path_key] = {
                **_scan_path_metadata(path_key),
                "status": "skipped",
                "error": error,
            }
            case_payload["status"] = "partial"
            case_payload["geometry_path_blockers"].append(
                {
                    "path": path_key,
                    "error": error,
                    "reason": (
                        "geometry backend could not build this path for the "
                        "same finite-beta input/wout pair"
                    ),
                }
            )
            continue
        scans[path_key] = scan
        path_payload = {
            **_scan_path_metadata(path_key),
            "status": "complete",
            "seconds": float(time.perf_counter() - start),
            "coefficients": _coefficient_summary(scan),
            "profile_responses": _profile_response_summary(scan),
        }
        if write_hdf5:
            path_payload["hdf5"] = _write_scan_file(scan, output_dir, case.id, path_key)
        case_payload["scan_paths"][path_key] = path_payload

    if case.primary_geometry_path in scans:
        case_payload["primary_scan_path"] = case.primary_geometry_path
    elif "booz_xform_jax" in scans:
        case_payload["primary_scan_path"] = "booz_xform_jax"
    elif "vmec_jax_wout_cubic" in scans:
        case_payload["primary_scan_path"] = "vmec_jax_wout_cubic"

    if "booz_xform_jax" in scans and "vmec_jax_wout_cubic" in scans:
        reference = scans["booz_xform_jax"]
        candidate = scans["vmec_jax_wout_cubic"]
        case_payload["interpolation_audit"] = {
            "reference_path": "booz_xform_jax",
            "candidate_path": "vmec_jax_wout_cubic",
            "claim_scope": (
                "same input/wout, rho, collisionality, electric field, grid, and "
                "normalization; differences isolate geometry/interpolation-path effects "
                "and should not be mixed with external reference datasets"
            ),
            "max_relative_difference": {
                name: _max_relative_difference(getattr(reference, name), getattr(candidate, name))
                for name in COEFFICIENTS
            },
        }
    elif "booz_xform_jax" not in scans and "vmec_jax_wout_cubic" in scans:
        case_payload["interpolation_audit"] = {
            "status": "blocked",
            "reference_path": "booz_xform_jax",
            "candidate_path": "vmec_jax_wout_cubic",
            "claim_scope": (
                "direct VMEC-harmonic NTX scan completed, but the JAX Boozer "
                "reference path did not; this is a geometry-backend support lane, "
                "not a transport-coefficient parity result"
            ),
        }
        case_payload["status"] = "partial"
    elif "booz_xform_jax" not in scans:
        case_payload["status"] = "skipped"

    return case_payload


def build_payload(
    *,
    case_specs: tuple[OwnedJaxGeometryCase, ...] | None = None,
    case_ids: tuple[str, ...] = (),
    case_limit: int | None = 2,
    rho: tuple[float, ...] = DEFAULT_RHO,
    nu_v: tuple[float, ...] = DEFAULT_NU_V,
    es_values: tuple[float, ...] = DEFAULT_ES,
    grid: GridSpec = DEFAULT_GRID,
    output_dir: Path = DATABASE_DIR,
    compare_vmec_harmonic: bool = True,
    write_hdf5: bool = True,
    mboz: int = 4,
    nboz: int = 4,
    min_bmn_to_load: float = 1.0e-5,
) -> dict[str, object]:
    """Generate owned NTX+NEOPAX scans and return a JSON-serializable payload."""

    selected_cases = list(case_specs if case_specs is not None else discover_owned_case_specs())
    if case_ids:
        requested = set(case_ids)
        selected_cases = [case for case in selected_cases if case.id in requested]
    if case_limit is not None and case_limit > 0:
        selected_cases = selected_cases[:case_limit]

    rho_arr = np.asarray(rho, dtype=float)
    nu_arr = np.asarray(nu_v, dtype=float)
    es_arr = np.asarray(es_values, dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)

    drds_by_case: dict[str, list[float]] = {}
    psi_p_by_case: dict[str, float] = {}
    cases: list[dict[str, object]] = []
    for case in selected_cases:
        drds = _drds_from_minor_radius(rho_arr, _case_minor_radius_for_drds(case))
        drds_by_case[case.id] = drds.tolist()
        psi_p_by_case[case.id] = float(_case_psi_p_for_boozer(case))
        cases.append(
            _solve_case(
                case,
                rho=rho_arr,
                nu_v=nu_arr,
                es_values=es_arr,
                drds=drds,
                grid=grid,
                output_dir=output_dir,
                compare_vmec_harmonic=compare_vmec_harmonic,
                write_hdf5=write_hdf5,
                mboz=mboz,
                nboz=nboz,
                min_bmn_to_load=min_bmn_to_load,
            )
        )
    complete_cases = [case for case in cases if case["status"] == "complete"]
    usable_cases = [case for case in cases if case.get("primary_scan_path")]
    skipped_cases = [case for case in cases if case["status"] == "skipped"]
    max_path_difference = max(
        (
            max(
                float(value)
                for value in case["interpolation_audit"]["max_relative_difference"].values()
            )
            for case in cases
            if case.get("interpolation_audit", {}).get("max_relative_difference")
        ),
        default=float("nan"),
    )
    return {
        "benchmark": "owned_geometry_neopax_dataset",
        "classification": "owned JAX-native NTX+NEOPAX dataset and interpolation-path audit",
        "claim_scope": (
            "Generates self-contained NTX+NEOPAX scan artifacts from matching "
            "input/wout geometry pairs. This establishes owned normalization and "
            "interpolation provenance; it is not an external-code parity claim."
        ),
        "comparison_policy": (
            "Do not compare W7-X, QA, QH, Redl, SFINCS, and NTX+NEOPAX curves unless "
            "they are generated from the same geometry, profile, collisionality, radial "
            "electric-field, interpolation, and normalization contract."
        ),
        "normalization_contract": {
            "raw_scan": "D11, D13, and D33 are stored as NTX monoenergetic coefficients.",
            "neopax_export": "D11*drds^2, D13*drds, and nu_v*D33.",
            "booz_xform_psi_p": (
                "The Boozer-coordinate NTX path passes abs(phi_edge)/(2*pi) "
                "from the matching VMEC wout as psi_p, so the Boozer transform "
                "and direct VMEC-harmonic audit use the same physical edge-flux scale."
            ),
        },
        "inputs": {
            "rho": rho_arr.tolist(),
            "s": (rho_arr**2).tolist(),
            "nu_v": nu_arr.tolist(),
            "Es": es_arr.tolist(),
            "drds_definition": "dr/ds = a_b/(2*rho), with s=rho^2 and r=a_b*rho",
            "drds_by_case": drds_by_case,
            "booz_xform_psi_p_by_case": psi_p_by_case,
            "grid": _grid_payload(grid),
            "mboz": int(mboz),
            "nboz": int(nboz),
            "min_bmn_to_load": float(min_bmn_to_load),
        },
        "cases": cases,
        "summary_metrics": {
            "case_count": len(cases),
            "complete_case_count": len(complete_cases),
            "usable_case_count": len(usable_cases),
            "skipped_case_count": len(skipped_cases),
            "max_geometry_path_relative_difference": float(max_path_difference),
        },
        "downstream_interpolation_mode_audit": {
            "status": "blocked_pending_stable_api",
            "required_modes": ["general", "legacy_ntss"],
            "reason": (
                "The downstream NEOPAX interpolation-mode selector is not yet "
                "exposed through a stable public interface. Once it is, this same "
                "owned finite-beta dataset should run both modes on the same "
                "geometry/profile grids and store the difference here."
            ),
        },
        "open_work": [
            (
                "run SFINCS and the analytic bootstrap-current formula on these same "
                "owned geometry/profile grids before promoting new parity figures"
            ),
            (
                "add the NEOPAX general and legacy interpolation modes to this audit "
                "once both are available through a stable public interface"
            ),
            (
                "extend vmec_jax input reconstruction to optimized finite-beta "
                "cubic-spline current profiles so those cases can use the same "
                "JAX Boozer path as the power-series finite-beta QA case"
            ),
            (
                "expand the default case set to production-resolution QA, QH, QI, and W7-X "
                "inputs after owned SFINCS-generation scripts have completed runs"
            ),
        ],
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
        "database_dir": _display_path(output_dir),
    }


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.8, 6.8),
            "figure.dpi": 220,
            "font.size": 10.2,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def build_figure(payload: dict[str, object], output_prefix: Path = OUTPUT_PREFIX) -> None:
    """Write a compact figure from an owned-scan payload."""

    plot_cases = [
        case
        for case in payload["cases"]
        if case["status"] in {"complete", "partial"} and case.get("primary_scan_path")
    ]
    if not plot_cases:
        raise ValueError("no completed owned JAX geometry scans to plot")

    _configure_style()
    fig, ((ax_geom, ax_coeff), (ax_current, ax_audit)) = plt.subplots(
        2,
        2,
        figsize=(13.6, 8.6),
        gridspec_kw={"width_ratios": [1.0, 1.15], "height_ratios": [1.0, 1.0]},
    )

    coefficient_colors = {"D11": "#2878b5", "D13": "#c85200", "D33": "#2ca02c"}
    case_colors = ("#111111", "#0072b2", "#d55e00", "#009e73")
    rho = np.asarray(payload["inputs"]["rho"], dtype=float)

    for case_index, case in enumerate(plot_cases):
        path_key = str(case["primary_scan_path"])
        path_payload = case["scan_paths"][path_key]
        color = case_colors[case_index % len(case_colors)]
        b00 = np.asarray(path_payload["coefficients"].get("B00", []), dtype=float)
        iota = np.asarray(path_payload["coefficients"].get("iota", []), dtype=float)
        if b00.size:
            ax_geom.plot(
                rho,
                b00 / max(abs(float(b00[0])), EPS),
                marker="o",
                lw=1.8,
                color=color,
                label=f"{case['label']} $B_{{00}}$",
            )
        if iota.size:
            ax_geom.plot(
                rho,
                np.abs(iota) / max(abs(float(iota[0])), EPS),
                marker="s",
                lw=1.4,
                ls="--",
                color=color,
                alpha=0.78,
                label=f"{case['label']} $|\\iota|$",
            )
    ax_geom.set_xlabel(r"$\rho$")
    ax_geom.set_ylabel("normalized geometry scalar")
    ax_geom.set_title("(a) Finite-beta geometry inputs")
    ax_geom.legend(loc="best", fontsize=7.8)

    for case_index, case in enumerate(plot_cases):
        path_key = str(case["primary_scan_path"])
        path_payload = case["scan_paths"][path_key]
        for coefficient in COEFFICIENTS:
            values = np.asarray(path_payload["coefficients"][coefficient]["values"], dtype=float)
            line = values[:, 0, 0]
            ax_coeff.plot(
                rho,
                np.abs(line),
                marker="o",
                lw=1.8,
                color=coefficient_colors[coefficient],
                alpha=max(0.48, 1.0 - 0.18 * case_index),
                label=f"{coefficient}" if case_index == 0 else None,
            )
        ax_coeff.text(
            rho[-1] + 0.012,
            max(
                float(
                    np.nanmax(
                        np.abs(np.asarray(path_payload["coefficients"]["D11"]["values"])[:, 0, 0])
                    )
                ),
                EPS,
            ),
            str(case["label"]),
            fontsize=8.5,
            va="center",
            color="0.25",
        )
    ax_coeff.set_yscale("log")
    ax_coeff.set_xlim(float(rho[0]) - 0.02, float(rho[-1]) + 0.08)
    ax_coeff.set_xlabel(r"$\rho$")
    ax_coeff.set_ylabel(r"$|D_{ij}|$ at first $(\nu, E_s)$ point")
    ax_coeff.set_title("(b) Owned finite-beta NTX scan")
    ax_coeff.legend(loc="best")

    for case_index, case in enumerate(plot_cases):
        path_key = str(case["primary_scan_path"])
        path_payload = case["scan_paths"][path_key]
        current = np.asarray(
            path_payload["profile_responses"]["bootstrap_current_response_normalized"],
            dtype=float,
        )
        residual = np.asarray(
            path_payload["profile_responses"]["ambipolar_residual_response"],
            dtype=float,
        )
        residual_scale = max(float(np.nanmax(np.abs(residual))), EPS)
        ax_current.plot(
            rho,
            current,
            marker="o",
            lw=2.0,
            color="#111111",
            alpha=max(0.55, 1.0 - 0.18 * case_index),
            label=str(case["label"]),
        )
        ax_current.plot(
            rho,
            residual / residual_scale,
            lw=1.4,
            ls="--",
            color="0.45",
            alpha=max(0.45, 0.78 - 0.18 * case_index),
        )
    ax_current.axhline(0.0, color="0.2", lw=0.8, alpha=0.55)
    ax_current.set_xlabel(r"$\rho$")
    ax_current.set_ylabel("normalized response")
    ax_current.set_title("(c) Same-grid profile response")
    ax_current.legend(loc="best")

    audited_cases = [
        case
        for case in plot_cases
        if case.get("interpolation_audit", {}).get("max_relative_difference")
    ]
    labels = [str(case["label"]).replace(" ", "\n") for case in audited_cases]
    positions = np.arange(len(audited_cases))
    width = 0.24
    if audited_cases:
        for offset, coefficient in enumerate(COEFFICIENTS):
            values = np.asarray(
                [
                    max(
                        float(case["interpolation_audit"]["max_relative_difference"][coefficient]),
                        1.0e-16,
                    )
                    for case in audited_cases
                ],
                dtype=float,
            )
            ax_audit.bar(
                positions + (offset - 1) * width,
                values,
                width=width,
                label=coefficient,
                color=coefficient_colors[coefficient],
                alpha=0.86,
            )
        ax_audit.set_yscale("log")
        ax_audit.set_xticks(positions)
        ax_audit.set_xticklabels(labels, rotation=25, ha="right")
        ax_audit.set_ylabel("max relative difference")
        ax_audit.set_title("(d) Boozer path vs VMEC-harmonic path")
        ax_audit.legend(loc="best")
    else:
        ax_audit.text(0.5, 0.5, "Interpolation audit disabled", ha="center", va="center")
        ax_audit.set_axis_off()
    blocker_lines = [
        f"{case['label']}: {case['geometry_path_blockers'][0]['path']} blocked"
        for case in plot_cases
        if case.get("geometry_path_blockers")
    ]
    if blocker_lines:
        ax_audit.text(
            0.02,
            0.98,
            "\n".join(blocker_lines[:3]),
            transform=ax_audit.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            color="0.25",
        )

    fig.suptitle("Owned finite-beta NTX+NEOPAX geometry dataset provenance", fontsize=13.5)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_payload(payload: dict[str, object], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], help="case id to include")
    parser.add_argument("--case-limit", type=int, default=2, help="limit discovered cases")
    parser.add_argument("--rho", nargs="+", type=float, default=list(DEFAULT_RHO))
    parser.add_argument("--nu-v", nargs="+", type=float, default=list(DEFAULT_NU_V))
    parser.add_argument("--es", nargs="+", type=float, default=list(DEFAULT_ES))
    parser.add_argument("--n-theta", type=int, default=DEFAULT_GRID.n_theta)
    parser.add_argument("--n-zeta", type=int, default=DEFAULT_GRID.n_zeta)
    parser.add_argument("--n-xi", type=int, default=DEFAULT_GRID.n_xi)
    parser.add_argument("--mboz", type=int, default=4)
    parser.add_argument("--nboz", type=int, default=4)
    parser.add_argument("--min-bmn-to-load", type=float, default=1.0e-5)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--database-dir", type=Path, default=DATABASE_DIR)
    parser.add_argument(
        "--no-vmec-harmonic-audit",
        action="store_true",
        help="skip the direct VMEC-harmonic interpolation-path audit",
    )
    parser.add_argument(
        "--no-hdf5",
        action="store_true",
        help="skip NEOPAX-style HDF5 table writes",
    )
    args = parser.parse_args()

    payload = build_payload(
        case_ids=tuple(args.case),
        case_limit=args.case_limit,
        rho=tuple(args.rho),
        nu_v=tuple(args.nu_v),
        es_values=tuple(args.es),
        grid=GridSpec(args.n_theta, args.n_zeta, args.n_xi),
        output_dir=args.database_dir,
        compare_vmec_harmonic=not args.no_vmec_harmonic_audit,
        write_hdf5=not args.no_hdf5,
        mboz=args.mboz,
        nboz=args.nboz,
        min_bmn_to_load=args.min_bmn_to_load,
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(
        f"wrote {args.output_prefix.with_suffix('.json')}, "
        f"{args.output_prefix.with_suffix('.png')}, and "
        f"{args.output_prefix.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()
