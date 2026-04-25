#!/usr/bin/env python3
"""Audit fixed-field precise-QS parallel-flow closure against SFINCS-JAX RHSMode=2."""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import f90nml
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.constants import elementary_charge
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import (
    GridSpec,
    build_ntx_neopax_scan_from_surfaces,
    load_neopax_reference_scan,
    load_vmec_surface,
    neopax_scan_requires_rebuild,
    to_neopax_monoenergetic,
    write_neopax_scan_hdf5,
)
from ntx._checkout_paths import (
    find_booz_xform_jax_root,
    find_neopax_root,
    find_qs_zenodo_root,
    find_sfincs_jax_root,
)

OUTPUT_DIR = ROOT / "examples" / "outputs" / "fixed_field_parallel_flow_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = OUTPUT_DIR / "fixed_field_parallel_flow_audit"

_rho_env = os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_RHO", "").strip()
if _rho_env:
    RHO_VALUES = np.array([float(value) for value in _rho_env.split(",")], dtype=float)
else:
    RHO_VALUES = np.array([0.25, 0.50, 0.75], dtype=float)
CASE_FILTER = tuple(
    value.strip().lower()
    for value in os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_CASES", "qa,qh").split(",")
    if value.strip()
)
SFINCS_JAX_SAMPLE_COUNT = 9
NTX_SURFACE_GRID = GridSpec(
    n_theta=int(os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_NTX_NTHETA", "25")),
    n_zeta=int(os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_NTX_NZETA", "25")),
    n_xi=int(os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_NTX_NXI", "31")),
)
NTX_NEOPAX_RADIAL_POINTS = int(os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_NTX_NR", "17"))
ER_AXIS_FACTORS = np.array([0.5, 1.0, 2.0], dtype=float)
SFINCS_SOLVE_METHOD = (
    os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_SOLVE_METHOD", "auto").strip() or "auto"
)
RHSMODE2_SPECIES = (
    os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_SPECIES", "ion").strip().lower() or "ion"
)
RHSMODE2_NTHETA = int(os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_NTHETA", "0") or 0)
RHSMODE2_NZETA = int(os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_NZETA", "0") or 0)
RHSMODE2_NXI = int(os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_NXI", "0") or 0)
RHSMODE2_NX = int(os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_NX", "0") or 0)
D33_MODE = os.environ.get(
    "NTX_FIXED_FIELD_PARALLEL_FLOW_D33_MODE",
    "raw",
).strip().lower()
PRECISE_QS_PROFILE_MODE = os.environ.get(
    "NTX_FIXED_FIELD_PROFILE_MODE",
    "analytic",
).strip().lower()
POSTPROCESS_PROFILE_INTERP = os.environ.get(
    "NTX_FIXED_FIELD_POSTPROCESS_INTERP",
    "pchip",
).strip().lower()
RECOMPUTE = os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_AUDIT_RECOMPUTE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SFINCS_JHAT_TO_AM2 = 437695.0 * 1.0e20 * elementary_charge


@dataclass(frozen=True)
class FixedFieldCase:
    name: str
    label: str
    helicity_n: int
    wout_path: Path
    sfincs_scan_path: Path

    @property
    def output_dir(self) -> Path:
        path = OUTPUT_DIR / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def boozmn_path(self) -> Path:
        return self.output_dir / f"boozmn_{self.name}.nc"

    @property
    def sfincs_scan_dir(self) -> Path:
        return self.sfincs_scan_path.parent


@dataclass(frozen=True)
class ArchivedProfiles:
    psi_n: np.ndarray
    rho: np.ndarray
    n_hat: np.ndarray
    t_hat: np.ndarray
    dn_hat_drhat: np.ndarray
    dT_hat_drhat: np.ndarray
    er: np.ndarray
    alpha: np.ndarray
    a_hat: float

    @property
    def density_si(self) -> np.ndarray:
        return self.n_hat * 1.0e20

    @property
    def temperature_ev(self) -> np.ndarray:
        return self.t_hat * 1.0e3

    @property
    def density_rho_derivative_si(self) -> np.ndarray:
        # SFINCS stores d/d rHat, with rHat = aHat * rho for this archive.
        # Cubic-Hermite interpolation below is parameterized in rho, so the
        # supplied slope must be converted by drHat/drho = aHat.
        return self.dn_hat_drhat * self.a_hat * 1.0e20

    @property
    def temperature_rho_derivative_ev(self) -> np.ndarray:
        # Same coordinate conversion as density_rho_derivative_si().
        return self.dT_hat_drhat * self.a_hat * 1.0e3

    @property
    def electric_field_kv_per_m(self) -> np.ndarray:
        # Archived precise-QS SFINCS inputs store Er = -dPhiHat/drHat, where
        # PhiHat = Phi / PhiBar and rHat = r / a. NEOPAX expects the physical
        # radial electric field in kV/m. With TBar = 1 keV in the archived
        # normalization, PhiBar = alpha * 1 kV, so:
        #     Er_phys [kV/m] = Er_hat * alpha / aHat.
        return self.er * self.alpha / max(self.a_hat, 1.0e-30)


def _use_exact_precise_qs_profiles(case: FixedFieldCase) -> bool:
    if PRECISE_QS_PROFILE_MODE in {"archive", "archived"}:
        return False
    return True


def _exact_precise_qs_profiles(
    *,
    psi_n: np.ndarray,
    rho: np.ndarray,
    er: np.ndarray,
    alpha: np.ndarray,
    a_hat: float,
) -> ArchivedProfiles:
    rho_arr = np.asarray(rho, dtype=float)
    return ArchivedProfiles(
        psi_n=np.asarray(psi_n, dtype=float),
        rho=rho_arr,
        n_hat=4.13 * (1.0 - rho_arr**10),
        t_hat=12.0 * (1.0 - rho_arr**2),
        dn_hat_drhat=(-41.3 * rho_arr**9) / max(float(a_hat), 1.0e-30),
        dT_hat_drhat=(-24.0 * rho_arr) / max(float(a_hat), 1.0e-30),
        er=np.asarray(er, dtype=float),
        alpha=np.asarray(alpha, dtype=float),
        a_hat=float(a_hat),
    )


def _zenodo_root() -> Path:
    root = find_qs_zenodo_root()
    if root is None:
        raise RuntimeError("fixed-field parallel-flow audit requires the local Zenodo archive")
    return root


def _cases() -> dict[str, FixedFieldCase]:
    root = _zenodo_root()
    calc_root = root / "calculations" / "20211226-01-sfincs_for_precise_QS_for_Redl_benchmark"
    wout_root = root / "codes" / "simsopt" / "tests" / "test_files"
    cases = {
        "qa": FixedFieldCase(
            name="qa",
            label="QA precise-QS fixed-field reference",
            helicity_n=0,
            wout_path=wout_root / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
            sfincs_scan_path=calc_root
            / "20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces"
            / "sfincsScan.dat",
        ),
        "qh": FixedFieldCase(
            name="qh",
            label="QH precise-QS fixed-field reference",
            helicity_n=-1,
            wout_path=wout_root / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc",
            sfincs_scan_path=calc_root
            / "20211226-01-019_QH_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces"
            / "sfincsScan.dat",
        ),
    }
    missing = [
        path
        for case in cases.values()
        for path in (case.wout_path, case.sfincs_scan_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing fixed-field benchmark files: {missing}")
    return cases


def _sfincs_jax_root() -> Path:
    root = find_sfincs_jax_root()
    if root is None:
        raise RuntimeError("fixed-field parallel-flow audit requires the local sfincs_jax checkout")
    return root


def _neopax_root() -> Path:
    root = find_neopax_root()
    if root is None:
        raise RuntimeError("fixed-field parallel-flow audit requires the local NEOPAX checkout")
    return root


def _bootstrap_current_pythonpath() -> None:
    for extra_path in (_sfincs_jax_root(), _neopax_root()):
        if str(extra_path) not in sys.path:
            sys.path.insert(0, str(extra_path))


def _archived_surface_inputs(case: FixedFieldCase) -> list[tuple[float, Path]]:
    surfaces: list[tuple[float, Path]] = []
    for path in sorted(case.sfincs_scan_dir.glob("psiN_*/input.namelist")):
        try:
            psi_n = float(path.parent.name.split("_", 1)[1])
        except ValueError:
            continue
        surfaces.append((psi_n, path))
    if not surfaces:
        raise FileNotFoundError(f"no archived SFINCS input files under {case.sfincs_scan_dir}")
    return surfaces


def _archived_profiles(case: FixedFieldCase) -> ArchivedProfiles:
    psi_n_values: list[float] = []
    n_hat_values: list[float] = []
    t_hat_values: list[float] = []
    dn_hat_values: list[float] = []
    dt_hat_values: list[float] = []
    er_values: list[float] = []
    alpha_values: list[float] = []
    for psi_n, input_path in _archived_surface_inputs(case):
        nml = f90nml.read(input_path)
        species = nml["speciesparameters"]
        physics = nml["physicsparameters"]
        psi_n_values.append(float(psi_n))
        n_hat_values.append(float(np.atleast_1d(np.asarray(species["nhats"], dtype=float))[0]))
        t_hat_values.append(float(np.atleast_1d(np.asarray(species["thats"], dtype=float))[0]))
        dn_hat_values.append(
            float(np.atleast_1d(np.asarray(species["dnhatdrhats"], dtype=float))[0])
        )
        dt_hat_values.append(
            float(np.atleast_1d(np.asarray(species["dthatdrhats"], dtype=float))[0])
        )
        er_values.append(float(physics["er"]))
        alpha_values.append(float(physics.get("alpha", 1.0)))
    psi_n = np.asarray(psi_n_values, dtype=float)
    order = np.argsort(psi_n)
    psi_n = psi_n[order]
    with Dataset(case.wout_path) as ds:
        a_hat = float(np.asarray(ds.variables["Aminor_p"]).reshape(()))
    profiles = ArchivedProfiles(
        psi_n=psi_n,
        rho=np.sqrt(psi_n),
        n_hat=np.asarray(n_hat_values, dtype=float)[order],
        t_hat=np.asarray(t_hat_values, dtype=float)[order],
        dn_hat_drhat=np.asarray(dn_hat_values, dtype=float)[order],
        dT_hat_drhat=np.asarray(dt_hat_values, dtype=float)[order],
        er=np.asarray(er_values, dtype=float)[order],
        alpha=np.asarray(alpha_values, dtype=float)[order],
        a_hat=a_hat,
    )
    if _use_exact_precise_qs_profiles(case):
        return _exact_precise_qs_profiles(
            psi_n=profiles.psi_n,
            rho=profiles.rho,
            er=profiles.er,
            alpha=profiles.alpha,
            a_hat=profiles.a_hat,
        )
    return profiles


def _interp_profile(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        raise ValueError("cannot interpolate a profile with no finite support")
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]
    if POSTPROCESS_PROFILE_INTERP not in {"linear", "pchip"}:
        raise ValueError(
            "POSTPROCESS_PROFILE_INTERP "
            "(NTX_FIXED_FIELD_POSTPROCESS_INTERP) must be one of {'pchip', 'linear'}"
        )
    if POSTPROCESS_PROFILE_INTERP == "linear" or x_unique.size < 3:
        return np.interp(np.asarray(xq, dtype=float), x_unique, y_unique)
    if POSTPROCESS_PROFILE_INTERP == "pchip":
        return PchipInterpolator(x_unique, y_unique)(np.asarray(xq, dtype=float))
    raise AssertionError("unreachable interpolation branch")


def _hermite_values_and_edge(
    rho_nodes: np.ndarray,
    values: np.ndarray,
    rho_derivatives: np.ndarray,
    rho_query: np.ndarray,
) -> tuple[np.ndarray, float]:
    spline = CubicHermiteSpline(
        np.asarray(rho_nodes, dtype=float),
        np.asarray(values, dtype=float),
        np.asarray(rho_derivatives, dtype=float),
    )
    query = np.asarray(rho_query, dtype=float)
    evaluated = spline(query)
    edge = float(spline(1.0))
    return np.asarray(evaluated, dtype=float), edge


def _ensure_boozmn(case: FixedFieldCase, *, nsurfaces: int = 48) -> Path:
    if case.boozmn_path.exists():
        return case.boozmn_path
    booz_root = find_booz_xform_jax_root()
    if booz_root is None:
        raise RuntimeError("fixed-field parallel-flow audit requires booz_xform_jax")
    src = booz_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from booz_xform_jax import Booz_xform

    bx = Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(case.wout_path), flux=True)
    bx.register_surfaces(np.linspace(0.03, 0.97, nsurfaces))
    bx.run(jit=False)
    bx.write_boozmn(str(case.boozmn_path))
    return case.boozmn_path


def _make_species(field: Any, case: FixedFieldCase) -> Any:
    _bootstrap_current_pythonpath()
    import NEOPAX

    rho = np.asarray(field.rho_grid, dtype=float)
    profiles = _archived_profiles(case)
    density, density_edge = _hermite_values_and_edge(
        profiles.rho,
        profiles.density_si,
        profiles.density_rho_derivative_si,
        rho,
    )
    temperature, temperature_edge = _hermite_values_and_edge(
        profiles.rho,
        profiles.temperature_ev,
        profiles.temperature_rho_derivative_ev,
        rho,
    )
    density_rho_derivative = _interp_profile(
        profiles.rho,
        profiles.density_rho_derivative_si,
        rho,
    )
    temperature_rho_derivative = _interp_profile(
        profiles.rho,
        profiles.temperature_rho_derivative_ev,
        rho,
    )
    rho_to_r = max(float(field.a_b), 1.0e-30)
    density_r_derivative = density_rho_derivative / rho_to_r
    temperature_r_derivative = temperature_rho_derivative / rho_to_r
    n_species = 2
    n_r = rho.size
    temperature = np.vstack([temperature, temperature])
    density = np.vstack([density, density])
    dndr_override = np.vstack([density_r_derivative, density_r_derivative])
    dTdr_override = np.vstack([temperature_r_derivative, temperature_r_derivative])
    electric_field = _interp_profile(profiles.rho, profiles.electric_field_kv_per_m, rho)
    mass = np.asarray([1.0 / 1836.15267343, 1.0])
    charge = np.asarray([-1.0, 1.0])
    return NEOPAX.Species(
        n_species,
        n_r,
        np.arange(n_species),
        mass,
        charge,
        temperature,
        density,
        electric_field,
        field.r_grid,
        field.r_grid_half,
        field.dr,
        field.Vprime_half,
        field.overVprime,
        np.asarray([density_edge, density_edge], dtype=float),
        np.asarray([temperature_edge, temperature_edge], dtype=float),
        dTdr_override=np.asarray(dTdr_override, dtype=float),
        dndr_override=np.asarray(dndr_override, dtype=float),
    )


def _adaptive_nu_values(species: Any, grid: Any) -> np.ndarray:
    _bootstrap_current_pythonpath()
    from NEOPAX._species import collisionality

    positive_values: list[np.ndarray] = []
    for species_index in (0, 1):
        thermal = np.asarray(species.v_thermal[species_index], dtype=float)
        for radial_index, v_th in enumerate(thermal):
            velocity = np.asarray(grid.v_norm, dtype=float) * float(v_th)
            nu_v = np.asarray(
                collisionality(species_index, species, velocity, radial_index) / velocity,
                dtype=float,
            )
            finite_positive = nu_v[np.isfinite(nu_v) & (nu_v > 0.0)]
            if finite_positive.size:
                positive_values.append(finite_positive)
    if not positive_values:
        raise ValueError("could not determine a positive collisionality support for NEOPAX")
    merged_all = np.concatenate(positive_values)
    nu_min = max(float(np.min(merged_all)) / 3.0, 1.0e-8)
    nu_max = max(float(np.max(merged_all)) * 3.0, nu_min * 10.0)
    return np.logspace(np.log10(nu_min), np.log10(nu_max), 17)


def _build_ntx_neopax_lij(case: FixedFieldCase) -> dict[str, np.ndarray]:
    _bootstrap_current_pythonpath()
    import NEOPAX

    boozmn = _ensure_boozmn(case)
    n_r = max(int(NTX_NEOPAX_RADIAL_POINTS), 9)
    field = NEOPAX.Field.read_vmec_booz(n_r, str(case.wout_path), str(boozmn))
    species = _make_species(field, case)
    ntx_grid = NEOPAX.Grid.create_standard(n_r, 64, 2)
    nu_values = _adaptive_nu_values(species, ntx_grid)

    profiles = _archived_profiles(case)
    rho_field = np.asarray(field.rho_grid, dtype=float)
    rho_surface = np.clip(rho_field, 0.05, 0.95)
    drds = float(field.a_b) * 0.5 / np.clip(rho_surface, 0.05, None)
    archived_er_hat = _interp_profile(profiles.rho, profiles.er, rho_surface)
    archived_alpha = _interp_profile(profiles.rho, profiles.alpha, rho_surface)
    archived_er = _interp_profile(profiles.rho, profiles.electric_field_kv_per_m, rho_surface)
    er_axis = float(np.median(archived_er)) * ER_AXIS_FACTORS
    er_values = np.repeat(er_axis[None, :], rho_surface.size, axis=0)

    scan_path = case.output_dir / "ntx_scan.h5"
    if scan_path.exists() and not RECOMPUTE and not neopax_scan_requires_rebuild(scan_path):
        scan = load_neopax_reference_scan(scan_path)
    else:
        surfaces = tuple(
            load_vmec_surface(
                case.wout_path,
                psi_n=float(rho_val**2),
                vmec_radial_option=0,
                vmec_nyquist_option=1,
                vmec_mode_convention="filtered_nyquist",
            )
            for rho_val in rho_surface
        )
        scan = build_ntx_neopax_scan_from_surfaces(
            surfaces,
            rho=rho_surface,
            nu_v=np.asarray(nu_values),
            Er=np.asarray(er_values),
            drds=np.asarray(drds),
            grid=NTX_SURFACE_GRID,
            source_name=f"fixed_field_{case.name}",
        )
        write_neopax_scan_hdf5(scan, scan_path)
    database = to_neopax_monoenergetic(scan, a_b=float(field.a_b), d33_mode=D33_MODE)
    lij, gamma, heat, upar = NEOPAX.get_Neoclassical_Fluxes(species, ntx_grid, field, database)
    return {
        "rho": rho_field,
        "L31_electron": np.asarray(lij[0, :, 2, 0], dtype=float),
        "L32_electron": np.asarray(lij[0, :, 2, 1], dtype=float),
        "L32_eff_electron": np.asarray(lij[0, :, 2, 1] - 1.5 * lij[0, :, 2, 0], dtype=float),
        "L33_electron": np.asarray(lij[0, :, 2, 2], dtype=float),
        "L31_ion": np.asarray(lij[1, :, 2, 0], dtype=float),
        "L32_ion": np.asarray(lij[1, :, 2, 1], dtype=float),
        "L32_eff_ion": np.asarray(lij[1, :, 2, 1] - 1.5 * lij[1, :, 2, 0], dtype=float),
        "L33_ion": np.asarray(lij[1, :, 2, 2], dtype=float),
        "upar_electron": np.asarray(upar[0], dtype=float),
        "upar_ion": np.asarray(upar[1], dtype=float),
        "gamma_electron": np.asarray(gamma[0], dtype=float),
        "gamma_ion": np.asarray(gamma[1], dtype=float),
        "heat_electron": np.asarray(heat[0], dtype=float),
        "heat_ion": np.asarray(heat[1], dtype=float),
        "nu_values": np.asarray(nu_values, dtype=float),
        "archived_er_hat_rho": archived_er_hat,
        "archived_alpha_rho": archived_alpha,
        "archived_er_kv_per_m_rho": archived_er,
        "scan_path": str(scan_path),
    }


def _patched_rhsmode2_input(case: FixedFieldCase, psi_n: float, source_input: Path) -> Path:
    nml = f90nml.read(source_input)
    nml.setdefault("general", {})
    nml.setdefault("geometryParameters", {})
    nml["general"]["RHSMode"] = 2
    nml["geometryParameters"]["inputRadialCoordinate"] = 3
    nml["geometryParameters"]["rN_wish"] = float(np.sqrt(psi_n))
    nml["geometryParameters"]["inputRadialCoordinateForGradients"] = 4
    nml["geometryParameters"]["equilibriumFile"] = str(case.wout_path)
    species = nml.setdefault("speciesParameters", {})
    species_index = {"ion": 0, "electron": 1}.get(RHSMODE2_SPECIES)
    if species_index is None:
        raise ValueError(
            "NTX_FIXED_FIELD_PARALLEL_FLOW_SPECIES must be 'ion' or 'electron'"
        )
    for key in ("nHats", "dnHatdrHats", "THats", "dTHatdrHats", "Zs", "mHats"):
        if key not in species:
            continue
        values = np.atleast_1d(np.asarray(species[key], dtype=float))
        pick = min(species_index, values.size - 1)
        species[key] = [float(values[pick])]
    resolution = nml.setdefault("resolutionParameters", {})
    if RHSMODE2_NTHETA > 0:
        resolution["Ntheta"] = RHSMODE2_NTHETA
    if RHSMODE2_NZETA > 0:
        resolution["Nzeta"] = RHSMODE2_NZETA
    if RHSMODE2_NXI > 0:
        resolution["Nxi"] = RHSMODE2_NXI
    if RHSMODE2_NX > 0:
        resolution["Nx"] = RHSMODE2_NX
    target_dir = case.output_dir / "sfincs_jax_rhsmode2" / RHSMODE2_SPECIES / f"psiN_{psi_n:.3f}"
    target_dir.mkdir(parents=True, exist_ok=True)
    input_path = target_dir / "input_rhsmode2.namelist"
    nml.write(input_path, force=True)
    return input_path


def _run_sfincs_jax_rhsmode2(
    case: FixedFieldCase,
    psi_n: float,
    source_input: Path,
) -> dict[str, Any]:
    input_path = _patched_rhsmode2_input(case, psi_n, source_input)
    matrix_path = input_path.with_name("transportMatrix.npy")
    log_path = input_path.with_name("sfincs_jax_rhsmode2.log")
    run_seconds = None
    if RECOMPUTE or not matrix_path.exists():
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join(
            [str(_sfincs_jax_root()), *[path for path in sys.path if path]]
        )
        command = [
            sys.executable,
            "-c",
            "from sfincs_jax.cli import main; raise SystemExit(main())",
            "transport-matrix-v3",
            "--input",
            str(input_path),
            "--out-matrix",
            str(matrix_path),
            "--solve-method",
            SFINCS_SOLVE_METHOD,
            "--tol",
            "1e-10",
        ]
        start = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                cwd=str(_sfincs_jax_root()),
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_path.write_text(result.stdout, encoding="utf-8")
        except subprocess.CalledProcessError as exc:
            log_path.write_text(exc.stdout, encoding="utf-8")
            raise RuntimeError(exc.stdout) from exc
        run_seconds = float(time.perf_counter() - start)
    elif log_path.exists():
        run_seconds = 0.0

    _bootstrap_current_pythonpath()
    from sfincs_jax.io import _conversion_factors_to_from_dpsi_hat
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.transport_matrix import _flux_functions_from_op
    from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist
    from sfincs_jax.v3_system import full_system_operator_from_namelist

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, grids=grids, geom=geom)
    surface = load_vmec_surface(
        case.wout_path,
        psi_n=float(psi_n),
        vmec_radial_option=0,
        vmec_nyquist_option=1,
        vmec_mode_convention="filtered_nyquist",
    )
    b0_over_bbar, g_hat, i_hat = _flux_functions_from_op(op0)
    conv = _conversion_factors_to_from_dpsi_hat(
        psi_a_hat=float(surface.psi_a_hat),
        a_hat=float(surface.aminor_p),
        r_n=float(np.sqrt(psi_n)),
    )
    matrix = np.asarray(np.load(matrix_path), dtype=float)
    return {
        "input_path": str(input_path),
        "log_path": str(log_path),
        "run_seconds": run_seconds,
        "transport_matrix": matrix,
        "meta": {
            "n_hat": float(np.asarray(op0.n_hat[0], dtype=float)),
            "t_hat": float(np.asarray(op0.t_hat[0], dtype=float)),
            "m_hat": float(np.asarray(op0.m_hat[0], dtype=float)),
            "z": float(np.asarray(op0.z_s[0], dtype=float)),
            "delta": float(np.asarray(op0.delta, dtype=float)),
            "alpha": float(np.asarray(op0.alpha, dtype=float)),
            "fsab_hat2": float(np.asarray(op0.fsab_hat2, dtype=float)),
            "g_hat": float(np.asarray(g_hat, dtype=float)),
            "i_hat": float(np.asarray(i_hat, dtype=float)),
            "iota": float(np.asarray(geom.iota, dtype=float)),
            "b0_over_bbar": float(np.asarray(b0_over_bbar, dtype=float)),
            "ddpsiHat2ddrHat": float(conv["ddpsiHat2ddrHat"]),
            "psi_a_hat": float(surface.psi_a_hat),
            "a_hat": float(surface.aminor_p),
        },
    }


def _relative_error(values: np.ndarray, reference: np.ndarray) -> list[float]:
    return (
        np.abs(np.asarray(values, dtype=float) - np.asarray(reference, dtype=float))
        / np.maximum(np.abs(np.asarray(reference, dtype=float)), 1.0e-16)
    ).tolist()


def _rhsmode2_hat_sources(*, which_rhs: int, n_hat: float, t_hat: float) -> tuple[float, float]:
    # Match the reference solver branch used by the precise-QS archive:
    # fortran/version3/solver.F90 sets the thermal column with
    # dnHatdpsiHats = (3/2) * nHats(1) * THats(1), dTHatdpsiHats = 1.
    if which_rhs == 1:
        return 1.0, 0.0
    if which_rhs == 2:
        return 1.5 * float(n_hat) * float(t_hat), 1.0
    if which_rhs == 3:
        return 0.0, 0.0
    raise ValueError("RHSMode=2 expects which_rhs in {1, 2, 3}.")


def _neopax_row3_thermal_bridge(
    *,
    l31: float,
    l32: float,
    sfincs_meta: dict[str, float],
    which_rhs: int,
) -> tuple[float, float, dict[str, float]]:
    n_hat = float(sfincs_meta["n_hat"])
    t_hat = float(sfincs_meta["t_hat"])
    z = float(sfincs_meta["z"])
    delta = float(sfincs_meta["delta"])
    g_hat = float(sfincs_meta["g_hat"])
    ddpsi_hat_to_dr_hat = float(sfincs_meta["ddpsiHat2ddrHat"])
    dn_hat_dpsi_hat, dT_hat_dpsi_hat = _rhsmode2_hat_sources(
        which_rhs=which_rhs,
        n_hat=n_hat,
        t_hat=t_hat,
    )
    a1 = (
        dn_hat_dpsi_hat / max(n_hat, 1.0e-30)
        - 1.5 * dT_hat_dpsi_hat / max(t_hat, 1.0e-30)
    ) * ddpsi_hat_to_dr_hat
    a2 = (dT_hat_dpsi_hat / max(t_hat, 1.0e-30)) * ddpsi_hat_to_dr_hat
    density_si = n_hat * 1.0e20
    upar_density = -density_si * (float(l31) * a1 + float(l32) * a2)
    current_density = z * elementary_charge * upar_density
    # NEOPAX's physical Upar closure matches the SFINCS row-3 thermal channels
    # only after restoring the historical DKES/SFINCS normalization between the
    # physical parallel flow moment and the hat-normalized FSABFlow diagnostic.
    # On the precise-QS archive this is the common factor needed to collapse the
    # density- and thermal-source columns simultaneously.
    flow_bridge = 2.0 * float(sfincs_meta["b0_over_bbar"]) / np.sqrt(np.pi)
    row31 = (
        2.0
        * flow_bridge
        * current_density
        / (SFINCS_JHAT_TO_AM2 * delta * g_hat * t_hat)
    )
    row32 = (
        2.0
        * flow_bridge
        * current_density
        / (SFINCS_JHAT_TO_AM2 * delta * g_hat * n_hat)
    )
    return float(row31), float(row32), {
        "dn_hat_dpsi_hat": float(dn_hat_dpsi_hat),
        "dT_hat_dpsi_hat": float(dT_hat_dpsi_hat),
        "A1": float(a1),
        "A2": float(a2),
        "upar_density": float(upar_density),
        "current_density": float(current_density),
        "ddpsiHat2ddrHat": float(ddpsi_hat_to_dr_hat),
        "flow_bridge": float(flow_bridge),
    }


def _run_case(case: FixedFieldCase) -> dict[str, Any]:
    archived_inputs = dict(_archived_surface_inputs(case))
    lij = _build_ntx_neopax_lij(case)
    rho_grid = np.asarray(lij["rho"], dtype=float)
    rows: list[dict[str, Any]] = []
    for rho in RHO_VALUES:
        psi_n = float(rho**2)
        source_psi_n = min(archived_inputs, key=lambda value: abs(value - psi_n))
        source_input = archived_inputs[source_psi_n]
        sfincs = _run_sfincs_jax_rhsmode2(case, source_psi_n, source_input)
        row3 = np.asarray(sfincs["transport_matrix"][2, :], dtype=float)
        ion_l31 = _interp_profile(rho_grid, lij["L31_ion"], np.asarray([rho]))[0]
        ion_l32 = _interp_profile(rho_grid, lij["L32_ion"], np.asarray([rho]))[0]
        ion_l33 = _interp_profile(rho_grid, lij["L33_ion"], np.asarray([rho]))[0]
        electron_l31 = _interp_profile(rho_grid, lij["L31_electron"], np.asarray([rho]))[0]
        electron_l32 = _interp_profile(rho_grid, lij["L32_electron"], np.asarray([rho]))[0]
        electron_l33 = _interp_profile(rho_grid, lij["L33_electron"], np.asarray([rho]))[0]
        ion_row31, ion_row32, ion_diag_1 = _neopax_row3_thermal_bridge(
            l31=ion_l31,
            l32=ion_l32,
            sfincs_meta=sfincs["meta"],
            which_rhs=1,
        )
        ion_row31_t, ion_row32_t, ion_diag_2 = _neopax_row3_thermal_bridge(
            l31=ion_l31,
            l32=ion_l32,
            sfincs_meta=sfincs["meta"],
            which_rhs=2,
        )
        electron_row31, electron_row32, electron_diag_1 = _neopax_row3_thermal_bridge(
            l31=electron_l31,
            l32=electron_l32,
            sfincs_meta=sfincs["meta"],
            which_rhs=1,
        )
        electron_row31_t, electron_row32_t, electron_diag_2 = _neopax_row3_thermal_bridge(
            l31=electron_l31,
            l32=electron_l32,
            sfincs_meta=sfincs["meta"],
            which_rhs=2,
        )
        ion_row = np.array([ion_row31, ion_row32_t, ion_l33], dtype=float)
        electron_row = np.array([electron_row31, electron_row32_t, electron_l33], dtype=float)
        rows.append(
            {
                "rho": float(rho),
                "psi_n_source": float(source_psi_n),
                "sfincs_jax": {
                    "transport_matrix": sfincs["transport_matrix"].tolist(),
                    "row3": row3.tolist(),
                    "meta": sfincs["meta"],
                    "input_path": sfincs["input_path"],
                },
                "ntx_neopax": {
                    "ion_row3_thermal_bridge": ion_row.tolist(),
                    "electron_row3_thermal_bridge": electron_row.tolist(),
                    "ion_raw_lij": [float(ion_l31), float(ion_l32), float(ion_l33)],
                    "electron_raw_lij": [
                        float(electron_l31),
                        float(electron_l32),
                        float(electron_l33),
                    ],
                    "diagnostics": {
                        "ion_which_rhs_1": ion_diag_1,
                        "ion_which_rhs_2": ion_diag_2,
                        "electron_which_rhs_1": electron_diag_1,
                        "electron_which_rhs_2": electron_diag_2,
                    },
                },
                "relative_error": {
                    "ion_thermal_bridge_vs_sfincs": _relative_error(ion_row[:2], row3[:2]),
                    "electron_thermal_bridge_vs_sfincs": _relative_error(
                        electron_row[:2], row3[:2]
                    ),
                },
            }
        )
    case_payload = asdict(case)
    case_payload["wout_path"] = str(case.wout_path)
    case_payload["sfincs_scan_path"] = str(case.sfincs_scan_path)
    return {"case": case_payload, "rows": rows}


def _plot(summary: dict[str, Any]) -> None:
    case_keys = [key for key in ("qa", "qh") if key in summary]
    if not case_keys:
        return
    fig, axes = plt.subplots(
        2,
        len(case_keys),
        figsize=(5.7 * len(case_keys), 7.0),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    channel_labels = ("row 3, col 1 (density source)", "row 3, col 2 (thermal source)")
    species_label = RHSMODE2_SPECIES

    for col, case_key in enumerate(case_keys):
        rows = summary[case_key]["rows"]
        rho = np.asarray([row["rho"] for row in rows], dtype=float)
        sfincs_row3 = np.asarray([row["sfincs_jax"]["row3"] for row in rows], dtype=float)
        ntx_row3 = np.asarray(
            [row["ntx_neopax"][f"{species_label}_row3_thermal_bridge"][:2] for row in rows],
            dtype=float,
        )
        for row_idx, label in enumerate(channel_labels):
            ax = axes[row_idx, col]
            ax.plot(
                rho,
                sfincs_row3[:, row_idx],
                "o-",
                color="black",
                lw=2.1,
                label="SFINCS-JAX row 3",
            )
            ax.plot(
                rho,
                ntx_row3[:, row_idx],
                "s--",
                color="#1f77b4",
                lw=1.9,
                label=f"NTX+NEOPAX {species_label} thermal bridge",
            )
            ax.grid(alpha=0.25, lw=0.6)
            ax.set_title(f"{summary[case_key]['case']['label']}: {label}")
            if row_idx == 1:
                ax.set_xlabel(r"$\rho$")
            if col == 0:
                ax.set_ylabel("row-3 coefficient")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = {
        "inputs": {
            "rho": RHO_VALUES.tolist(),
            "ntx_surface_grid": {
                "n_theta": NTX_SURFACE_GRID.n_theta,
                "n_zeta": NTX_SURFACE_GRID.n_zeta,
                "n_xi": NTX_SURFACE_GRID.n_xi,
            },
            "ntx_neopax_radial_points": NTX_NEOPAX_RADIAL_POINTS,
            "er_axis_factors": ER_AXIS_FACTORS.tolist(),
            "rhsmode2_species": RHSMODE2_SPECIES,
            "rhsmode2_resolution_override": {
                "n_theta": RHSMODE2_NTHETA,
                "n_zeta": RHSMODE2_NZETA,
                "n_xi": RHSMODE2_NXI,
                "n_x": RHSMODE2_NX,
            },
            "row3_bridge_note": (
                "Columns 1 and 2 are reconstructed from exact RHSMode=2 source gradients in "
                "physical units and converted back to SFINCS row-3 normalization. Column 3 "
                "is excluded here because the current closure does not expose the matching "
                "electric-field source channel."
            ),
            "zenodo_root": str(_zenodo_root()),
        }
    }
    for key, case in _cases().items():
        if CASE_FILTER and key not in CASE_FILTER:
            continue
        summary[key] = _run_case(case)
    _plot(summary)
    OUTPUT_PREFIX.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
