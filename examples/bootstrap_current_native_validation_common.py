#!/usr/bin/env python3
"""Shared helpers for native bootstrap-current validation against SFINCS."""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import h5py
import jax.numpy as jnp
import matplotlib
import numpy as np
from netCDF4 import Dataset
from scipy.constants import elementary_charge

matplotlib.use("Agg")

from ntx import (
    GridSpec,
    PrimitiveSpeciesProfile,
    build_bootstrap_species_profiles,
    build_ntx_neopax_scan_from_surfaces,
    evaluate_bootstrap_current,
    load_vmec_surface,
)
from ntx._checkout_paths import (
    find_booz_xform_jax_root,
    find_sfincs_executable,
    find_sfincs_jax_root,
    find_simsopt_root,
    find_single_stage_finite_beta_root,
)

OUTPUT_ROOT = ROOT / "examples" / "outputs" / "bootstrap_current_native_validation"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

LOCAL_ROOT = ROOT.parent
SFINCS_EXECUTABLE = find_sfincs_executable()
SFINCS_JAX_ROOT = find_sfincs_jax_root()
SIMSOPT_ROOT = find_simsopt_root()
BOOZ_XFORM_JAX_ROOT = find_booz_xform_jax_root()
FINITE_BETA_ROOT = find_single_stage_finite_beta_root()

for extra_path in (
    SFINCS_JAX_ROOT,
    SIMSOPT_ROOT,
    BOOZ_XFORM_JAX_ROOT,
):
    if extra_path is not None and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

SFINCS_JHAT_TO_AM2 = 437695.0 * 1.0e20 * elementary_charge
ION_MASS_MP = 1.0
DEFAULT_NTX_GRID = GridSpec(n_theta=25, n_zeta=25, n_xi=24)
DEFAULT_NTX_NU_VALUES = np.logspace(-4.5, -0.5, 11)
DEFAULT_NTX_ER_AXIS = np.array([-1.0e-8, 0.0, 1.0e-8], dtype=float)
SFINCS_COMPARE_RESOLUTION = {
    "ntheta": 5,
    "nzeta": 9,
    "nxi": 8,
    "nx": 2,
    "solver_tolerance": "5d-6",
}


@dataclass(frozen=True)
class CaseSpec:
    name: str
    label: str
    case_dir: Path
    helicity_n: int

    @property
    def wout_path(self) -> Path:
        return self.case_dir / "wout_final.nc"

    @property
    def input_path(self) -> Path:
        return self.case_dir / "input.final"

    @property
    def output_dir(self) -> Path:
        path = OUTPUT_ROOT / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def boozmn_path(self) -> Path:
        return self.output_dir / f"boozmn_{self.name}.nc"


def available_cases() -> dict[str, CaseSpec]:
    if FINITE_BETA_ROOT is None:
        return {}
    return {
        "qa": CaseSpec(
            name="qa",
            label="QA (nfp = 3)",
            case_dir=FINITE_BETA_ROOT / "optimization_finitebeta_nfp3_QA_stage1",
            helicity_n=0,
        ),
        "qh": CaseSpec(
            name="qh",
            label="QH (nfp = 4)",
            case_dir=FINITE_BETA_ROOT / "optimization_finitebeta_nfp4_QH_stage1",
            helicity_n=-1,
        ),
    }


def finite_beta_profiles(rho: np.ndarray) -> dict[str, np.ndarray]:
    beta = 2.5
    ne0 = 3.0e20 * (beta / 100.0 / 0.05) ** (1.0 / 3.0)
    te0 = 15.0e3 * (beta / 100.0 / 0.05) ** (2.0 / 3.0)
    ni0 = ne0
    ti0 = te0
    zeff = np.ones_like(rho)

    rho = np.asarray(rho, dtype=float)
    n_e = ne0 * (1.0 - 0.99 * rho**10)
    t_e = te0 * (1.0 - 0.99 * rho**2)
    n_i = ni0 * (1.0 - 0.99 * rho**10)
    t_i = ti0 * (1.0 - 0.99 * rho**2)

    dn_drho = -9.9 * ne0 * rho**9
    dT_drho = -1.98 * te0 * rho

    return {
        "rho": rho,
        "n_e": n_e,
        "t_e": t_e,
        "n_i": n_i,
        "t_i": t_i,
        "dn_drho": dn_drho,
        "dT_drho": dT_drho,
        "zeff": zeff,
    }


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _optional_imports():
    if BOOZ_XFORM_JAX_ROOT is None or SIMSOPT_ROOT is None:
        raise RuntimeError("simsopt and booz_xform_jax are required for Redl comparisons")
    from booz_xform_jax import Booz_xform
    from simsopt.mhd import Vmec
    from simsopt.mhd.bootstrap import compute_trapped_fraction, j_dot_B_Redl
    from simsopt.mhd.profiles import ProfilePolynomial

    return Booz_xform, Vmec, compute_trapped_fraction, j_dot_B_Redl, ProfilePolynomial


def ensure_boozmn(case: CaseSpec, *, nsurfaces: int = 32) -> Path:
    if case.boozmn_path.exists():
        return case.boozmn_path
    Booz_xform, *_ = _optional_imports()
    bx = Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(case.wout_path), flux=True)
    bx.register_surfaces(np.linspace(0.03, 0.97, nsurfaces))
    bx.run(jit=False)
    bx.write_boozmn(str(case.boozmn_path))
    return case.boozmn_path


def build_redl_profiles():
    *_, ProfilePolynomial = _optional_imports()
    beta = 2.5
    ne0 = 3.0e20 * (beta / 100.0 / 0.05) ** (1.0 / 3.0)
    te0 = 15.0e3 * (beta / 100.0 / 0.05) ** (2.0 / 3.0)
    ne = ProfilePolynomial(ne0 * np.array([1.0, 0.0, 0.0, 0.0, 0.0, -0.99]))
    te = ProfilePolynomial(te0 * np.array([1.0, -0.99]))
    ti = ProfilePolynomial(te0 * np.array([1.0, -0.99]))
    zeff = ProfilePolynomial([1.0])
    return ne, te, ti, zeff


def nearest_half_grid_surfaces(wout_path: Path, rho: np.ndarray) -> np.ndarray:
    with Dataset(wout_path) as ds:
        ns = int(np.asarray(ds.variables["iotaf"][:]).size)
    s_full = np.linspace(0.0, 1.0, ns)
    ds = s_full[1] - s_full[0]
    s_half = s_full[1:] - 0.5 * ds
    return np.asarray(
        [float(s_half[np.argmin(np.abs(s_half - s))]) for s in np.asarray(rho, dtype=float) ** 2]
    )


def compute_redl_profile(case: CaseSpec, rho: np.ndarray) -> dict[str, np.ndarray]:
    _, _, compute_trapped_fraction, j_dot_B_Redl, _ = _optional_imports()
    s_redl = nearest_half_grid_surfaces(case.wout_path, rho)
    ne, te, ti, zeff = build_redl_profiles()
    boozmn = ensure_boozmn(case)
    with Dataset(case.wout_path) as wout, Dataset(boozmn) as booz:
        phi_wout = np.asarray(wout.variables["phi"][:], dtype=float)
        psi_edge = -float(phi_wout[-1]) / (2.0 * np.pi)
        nfp = int(np.asarray(booz.variables["nfp_b"][:]).reshape(()))
        ixm_b = np.asarray(booz.variables["ixm_b"][:], dtype=int)
        ixn_b = np.asarray(booz.variables["ixn_b"][:], dtype=int)
        bmnc_raw = np.asarray(booz.variables["bmnc_b"][:], dtype=float)
        gmn_raw = np.asarray(booz.variables["gmn_b"][:], dtype=float)
        jlist = np.asarray(booz.variables["jlist"][:], dtype=int)
        phi_b = np.asarray(booz.variables["phi_b"][:], dtype=float)
        iota_b = np.asarray(booz.variables["iota_b"][:], dtype=float)
        bvco_b = np.asarray(booz.variables["bvco_b"][:], dtype=float)
        buco_b = np.asarray(booz.variables["buco_b"][:], dtype=float)

    s_b = phi_b[jlist - 1] / phi_b[-1]
    bmnc_interp = np.vstack(
        [np.interp(s_redl, s_b, bmnc_raw[:, jmn]) for jmn in range(bmnc_raw.shape[1])]
    )
    gmn_interp = np.vstack(
        [np.interp(s_redl, s_b, gmn_raw[:, jmn]) for jmn in range(gmn_raw.shape[1])]
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    modB = np.zeros((theta.size, s_redl.size))
    sqrtg = np.zeros_like(modB)
    symmetry_mask = ixm_b * case.helicity_n * nfp == ixn_b
    for jmn, keep in enumerate(symmetry_mask):
        if not keep:
            continue
        phase = np.cos(ixm_b[jmn] * theta)[:, None]
        modB += phase * bmnc_interp[jmn][None, :]
        sqrtg += phase * gmn_interp[jmn][None, :]

    _, _, epsilon, fsab2, fsa_1overB, f_t = compute_trapped_fraction(modB, sqrtg)
    s_full = phi_b / phi_b[-1]
    iota = np.interp(s_redl, s_full, iota_b)
    G = np.interp(s_redl, s_full, bvco_b)
    i_current = np.interp(s_redl, s_full, buco_b)
    R = (G + iota * i_current) * fsa_1overB
    jdotb, _details = j_dot_B_Redl(
        ne,
        te,
        ti,
        zeff,
        helicity_n=case.helicity_n,
        s=s_redl,
        G=G,
        R=R,
        iota=iota,
        epsilon=epsilon,
        f_t=f_t,
        psi_edge=psi_edge,
        nfp=nfp,
    )
    current = np.asarray(jdotb, dtype=float) / np.sqrt(fsab2)
    return {
        "rho": np.asarray(rho, dtype=float),
        "observable": np.asarray(jdotb, dtype=float),
        "jdotb": np.asarray(jdotb, dtype=float),
        "root_fsab2": np.sqrt(fsab2),
        "current_density": current,
        "current_over_root": current,
    }


def compute_ntx_native_profile(
    case: CaseSpec,
    rho: np.ndarray,
    *,
    grid: GridSpec | None = None,
    nu_values: np.ndarray | None = None,
    n_x: int = 64,
    loader_mode: str = "filtered_nyquist",
    compatibility_mode: bool = False,
) -> dict[str, np.ndarray]:
    grid = grid or DEFAULT_NTX_GRID
    nu_values = np.asarray(
        nu_values if nu_values is not None else DEFAULT_NTX_NU_VALUES,
        dtype=float,
    )
    rho = np.asarray(rho, dtype=float)
    er_axis = np.repeat(DEFAULT_NTX_ER_AXIS[None, :], rho.size, axis=0)

    surfaces = [
        _load_ntx_surface(case, float(rho_value), loader_mode)
        for rho_value in rho
    ]
    a_b = float(surfaces[0].aminor_p if surfaces[0].aminor_p is not None else 1.0)
    drds = a_b * 0.5 / np.clip(rho, 1.0e-6, None)
    scan = build_ntx_neopax_scan_from_surfaces(
        tuple(surfaces),
        rho=jnp.asarray(rho),
        nu_v=jnp.asarray(nu_values),
        Es=jnp.asarray(er_axis),
        Er=jnp.asarray(er_axis),
        drds=jnp.asarray(drds),
        grid=grid,
        source_name=f"ntx_native_{case.name}",
    )
    profiles = finite_beta_profiles(rho)
    species_profiles = build_bootstrap_species_profiles(
        jnp.asarray(rho),
        (
            PrimitiveSpeciesProfile(
                charge=-1.0,
                nu_v=jnp.full(rho.shape, nu_values[len(nu_values) // 2]),
                density=jnp.asarray(profiles["n_e"]),
                temperature=jnp.asarray(profiles["t_e"]),
                name="e",
            ),
            PrimitiveSpeciesProfile(
                charge=1.0,
                nu_v=jnp.full(rho.shape, nu_values[len(nu_values) // 2]),
                density=jnp.asarray(profiles["n_i"]),
                temperature=jnp.asarray(profiles["t_i"]),
                name="i",
            ),
        ),
        mass_mp=(1.0 / 1836.15267343, ION_MASS_MP),
        er_profile=jnp.zeros(rho.shape),
        a_b=a_b,
        smoothing_strength=0.0,
    )
    result = evaluate_bootstrap_current(
        scan,
        species_profiles,
        a_b=a_b,
        er_profile=jnp.zeros(rho.shape),
        n_x=n_x,
        neopax_compat_boundary=compatibility_mode,
    )
    return {
        "rho": rho,
        "observable": np.asarray(result.jdotb, dtype=float),
        "jdotb": np.asarray(result.jdotb, dtype=float),
        "current_density": np.asarray(result.current_density, dtype=float),
        "loader_mode": np.asarray(loader_mode),
        "compatibility_mode": np.asarray(compatibility_mode),
        "a_b": np.asarray(a_b),
    }


def _load_ntx_surface(case: CaseSpec, rho_value: float, loader_mode: str):
    psi_n = float(rho_value**2)
    if loader_mode == "reduced":
        return load_vmec_surface(case.wout_path, psi_n=psi_n)
    if loader_mode == "filtered_nyquist":
        return load_vmec_surface(
            case.wout_path,
            psi_n=psi_n,
            vmec_nyquist_option=1,
            vmec_mode_convention="filtered_nyquist",
        )
    if loader_mode == "full_nyquist":
        return load_vmec_surface(
            case.wout_path,
            psi_n=psi_n,
            vmec_nyquist_option=2,
            vmec_mode_convention="filtered_nyquist",
        )
    raise ValueError("loader_mode must be one of 'reduced', 'filtered_nyquist', or 'full_nyquist'")


def render_sfincs_input(
    *,
    wout_path: Path,
    rho: float,
    collision_operator: int = 0,
    ntheta: int = SFINCS_COMPARE_RESOLUTION["ntheta"],
    nzeta: int = SFINCS_COMPARE_RESOLUTION["nzeta"],
    nxi: int = SFINCS_COMPARE_RESOLUTION["nxi"],
    nx: int = SFINCS_COMPARE_RESOLUTION["nx"],
    solver_tolerance: str = SFINCS_COMPARE_RESOLUTION["solver_tolerance"],
) -> str:
    profiles = finite_beta_profiles(np.asarray([rho]))
    n_hat = profiles["n_i"][0] / 1.0e20
    t_hat = profiles["t_i"][0] / 1.0e3
    d_n_hat = profiles["dn_drho"][0] / 1.0e20
    d_t_hat = profiles["dT_drho"][0] / 1.0e3
    return f"""! Local VMEC bootstrap-current comparison case
&general
  RHSMode = 1
/

&geometryParameters
  geometryScheme = 5
  VMECRadialOption = 0
  inputRadialCoordinate = 3
  rN_wish = {rho:.8f}
  equilibriumFile = "{wout_path}"
/

&speciesParameters
  Zs = 1 -1
  mHats = 1 0.000545509d+0
  nHats = {n_hat:.12f} {n_hat:.12f}
  THats = {t_hat:.12f} {t_hat:.12f}
  dNHatdrHats = {d_n_hat:.12f} {d_n_hat:.12f}
  dTHatdrHats = {d_t_hat:.12f} {d_t_hat:.12f}
/

&physicsParameters
  Er = 0.0
  Delta = 4.5694d-3
  alpha = 1d+0
  nu_n = 0.00831565d+0
  collisionOperator = {collision_operator}
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  useDKESExBDrift = .false.
  includePhi1 = .false.
/

&resolutionParameters
  Ntheta = {ntheta}
  Nzeta = {nzeta}
  Nxi = {nxi}
  Nx = {nx}
  solverTolerance = {solver_tolerance}
/

&otherNumericalParameters
/

&preconditionerOptions
/

&export_f
/
"""


def prepare_sfincs_case(
    case: CaseSpec,
    *,
    rho: float,
    code_name: str,
    collision_operator: int = 0,
) -> Path:
    workdir = case.output_dir / code_name / f"rho_{rho:.3f}"
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = workdir / "input.namelist"
    input_path.write_text(
        render_sfincs_input(
            wout_path=case.wout_path,
            rho=rho,
            collision_operator=collision_operator,
        ),
        encoding="utf-8",
    )
    return workdir


def _read_sfincs_current_observable(out_path: Path, dataset: str) -> float:
    with h5py.File(out_path, "r") as handle:
        return float(np.asarray(handle[dataset][()]).reshape(-1)[-1]) * SFINCS_JHAT_TO_AM2


def compute_sfincs_jax_profile(
    case: CaseSpec,
    rho: np.ndarray,
    *,
    dataset: str = "FSABjHat",
    recompute: bool = False,
) -> dict[str, np.ndarray]:
    if SFINCS_JAX_ROOT is None:
        raise RuntimeError("sfincs_jax is required for the native benchmark")
    from sfincs_jax.io import write_sfincs_jax_output_h5

    current = []
    current_over_root = []
    for rho_value in np.asarray(rho, dtype=float):
        workdir = prepare_sfincs_case(case, rho=rho_value, code_name="sfincs_jax")
        out_path = workdir / "sfincsOutput.h5"
        if recompute or not out_path.exists():
            write_sfincs_jax_output_h5(
                input_namelist=workdir / "input.namelist",
                output_path=out_path,
                wout_path=case.wout_path,
                compute_solution=True,
                return_results=False,
                verbose=False,
            )
        current.append(_read_sfincs_current_observable(out_path, dataset))
        current_over_root.append(_read_sfincs_current_observable(out_path, "FSABjHatOverRootFSAB2"))
    return {
        "rho": np.asarray(rho, dtype=float),
        "observable": np.asarray(current, dtype=float),
        "jdotb": np.asarray(current, dtype=float),
        "current_density": np.asarray(current, dtype=float),
        "current_over_root": np.asarray(current_over_root, dtype=float),
        "dataset": np.asarray(dataset),
    }


def compute_sfincs_profile(
    case: CaseSpec,
    rho: np.ndarray,
    *,
    dataset: str = "FSABjHat",
    recompute: bool = False,
) -> dict[str, np.ndarray]:
    if SFINCS_EXECUTABLE is None:
        raise RuntimeError("Fortran SFINCS executable is required for the native benchmark")
    current = []
    current_over_root = []
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    for rho_value in np.asarray(rho, dtype=float):
        workdir = prepare_sfincs_case(case, rho=rho_value, code_name="sfincs")
        out_path = workdir / "sfincsOutput.h5"
        if recompute or not out_path.exists():
            subprocess.run(
                [str(SFINCS_EXECUTABLE)],
                cwd=workdir,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        current.append(_read_sfincs_current_observable(out_path, dataset))
        current_over_root.append(_read_sfincs_current_observable(out_path, "FSABjHatOverRootFSAB2"))
    return {
        "rho": np.asarray(rho, dtype=float),
        "observable": np.asarray(current, dtype=float),
        "jdotb": np.asarray(current, dtype=float),
        "current_density": np.asarray(current, dtype=float),
        "current_over_root": np.asarray(current_over_root, dtype=float),
        "dataset": np.asarray(dataset),
    }


def summarize_case_results(results: dict[str, dict[str, np.ndarray | float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, values in results.items():
        summary[key] = {
            name: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
            for name, value in values.items()
        }
    return summary


def max_relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    scale = np.maximum(np.abs(reference), 1.0)
    return float(np.max(np.abs(values - reference) / scale))


def update_nested_output_prefix(script_path: Path, output_prefix: Path) -> str:
    text = script_path.read_text(encoding="utf-8")
    return text.replace(
        'OUTPUT_PREFIX = ROOT / "docs" / "_static" / "bootstrap_current_native_validation"',
        f'OUTPUT_PREFIX = Path(r"{output_prefix}")',
    )
