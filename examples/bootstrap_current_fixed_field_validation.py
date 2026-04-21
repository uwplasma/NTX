#!/usr/bin/env python3
"""Compare fixed-field precise-QS bootstrap-current profiles.

This script keeps the archive-backed fixed-field QA/QH benchmark separate from
the finite-beta integrated workflow. It compares:

- archived Fortran SFINCS
- SFINCS-JAX reruns of the archived inputs
- Redl reconstructed on the same reference family
- NTX+NEOPAX on the same reference equilibria and profile family
"""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import f90nml
import h5py
import jax
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
from ntx._checkout_paths import find_booz_xform_jax_root, find_qs_zenodo_root

SCRATCH_ROOT = ROOT / "examples" / "outputs" / "bootstrap_current_fixed_field_validation"
SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "bootstrap_current_fixed_field_validation"

LOCAL_ROOT = Path.home() / "local"
NEOPAX_ROOT = LOCAL_ROOT / "tests" / "NEOPAX"
SFINCS_JAX_ROOT = LOCAL_ROOT / "tests" / "sfincs_jax"
SFINCS_EXECUTABLE = LOCAL_ROOT / "sfincs" / "fortran" / "version3" / "sfincs"
for extra_path in (NEOPAX_ROOT, SFINCS_JAX_ROOT):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import NEOPAX  # noqa: E402
from NEOPAX._neoclassical import (  # noqa: E402
    _replace_nonfinite_radial_boundaries,
    get_Lij_matrix_with_momentum_correction,
    get_momentum_Correction,
    get_Neoclassical_Fluxes,
)
from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5  # noqa: E402

SFINCS_JHAT_TO_AM2 = 437695.0 * 1.0e20 * elementary_charge
NTX_GRID = GridSpec(
    n_theta=int(os.environ.get("NTX_FIXED_FIELD_VALIDATION_NTX_NTHETA", "25")),
    n_zeta=int(os.environ.get("NTX_FIXED_FIELD_VALIDATION_NTX_NZETA", "25")),
    n_xi=int(os.environ.get("NTX_FIXED_FIELD_VALIDATION_NTX_NXI", "31")),
)
ER_AXIS_FACTORS = np.asarray(
    json.loads(os.environ.get("NTX_FIXED_FIELD_VALIDATION_ER_FACTORS", "[0.5, 1.0, 2.0]")),
    dtype=float,
)
RECOMPUTE = False
SFINCS_JAX_SAMPLE_COUNT = 9
NTX_NEOPAX_RADIAL_POINTS = int(os.environ.get("NTX_FIXED_FIELD_VALIDATION_NTX_NR", "17"))
NTX_NEOPAX_N_ORDER = int(os.environ.get("NTX_FIXED_FIELD_VALIDATION_NEOPAX_N_ORDER", "3"))
INTERIOR_RHO_MIN = 0.25
INTERIOR_RHO_MAX = 0.85
ENABLE_SFINCS_JAX = (
    os.environ.get("NTX_FIXED_FIELD_VALIDATION_ENABLE_SFINCS_JAX", "").strip().lower()
    in {"1", "true", "yes", "on"}
)
NTX_NEOPAX_D33_MODE = os.environ.get(
    "NTX_FIXED_FIELD_VALIDATION_D33_MODE",
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


@dataclass(frozen=True)
class FixedFieldCase:
    name: str
    label: str
    helicity_n: int
    wout_path: Path
    sfincs_scan_path: Path

    @property
    def output_dir(self) -> Path:
        path = SCRATCH_ROOT / self.name / "fixed_field"
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
        # The archived gradients are already expressed with respect to rHat=r/a,
        # which is the same spline coordinate used here, so only the density
        # normalization belongs in this conversion.
        return self.dn_hat_drhat * 1.0e20

    @property
    def temperature_rho_derivative_ev(self) -> np.ndarray:
        # Keep the derivative in rho-space; the physical 1/a factor enters later
        # when NEOPAX differentiates the reconstructed profile on the physical
        # radial grid.
        return self.dT_hat_drhat * 1.0e3

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
        dn_hat_drhat=-41.3 * rho_arr**9,
        dT_hat_drhat=-24.0 * rho_arr,
        er=np.asarray(er, dtype=float),
        alpha=np.asarray(alpha, dtype=float),
        a_hat=float(a_hat),
    )


def _zenodo_root() -> Path:
    root = find_qs_zenodo_root()
    if root is None:
        raise RuntimeError("fixed-field validation requires the local Zenodo archive")
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


def _load_archived_sfincs_scan(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    labels = list(payload["ylabels"])
    idx = labels.index("FSABjHat")
    s = np.asarray(payload["xdata"][idx], dtype=float)
    current = np.asarray(payload["ydata"][idx], dtype=float) * SFINCS_JHAT_TO_AM2
    return np.sqrt(s), current


def _load_archived_sfincs_species_flows(case: FixedFieldCase) -> dict[str, np.ndarray]:
    rho_values: list[float] = []
    ion_current: list[float] = []
    electron_current: list[float] = []
    current: list[float] = []
    for psi_n, source_input in _archived_surface_inputs(case):
        output_path = source_input.parent / "sfincsOutput.h5"
        if not output_path.exists():
            continue
        with h5py.File(output_path, "r") as handle:
            flow = np.asarray(handle["FSABFlow"][()], dtype=float).reshape(-1)
            charges = np.asarray(handle["Zs"][()], dtype=float).reshape(-1)
            current_hat = float(np.asarray(handle["FSABjHat"][()]).reshape(-1)[-1])
        if flow.size < 2:
            continue
        rho_values.append(float(np.sqrt(psi_n)))
        ion_current.append(float(charges[0] * flow[0]) * SFINCS_JHAT_TO_AM2)
        electron_current.append(float(charges[1] * flow[1]) * SFINCS_JHAT_TO_AM2)
        current.append(current_hat * SFINCS_JHAT_TO_AM2)
    rho = np.asarray(rho_values, dtype=float)
    order = np.argsort(rho)
    ion_current_array = np.asarray(ion_current, dtype=float)[order]
    electron_current_array = np.asarray(electron_current, dtype=float)[order]
    return {
        "rho": rho[order],
        "ion_current": ion_current_array,
        "electron_current": electron_current_array,
        "ion_flow": ion_current_array,
        "electron_flow": electron_current_array,
        "jdotb": np.asarray(current, dtype=float)[order],
    }


def _ensure_boozmn(case: FixedFieldCase, *, nsurfaces: int = 48) -> Path:
    if case.boozmn_path.exists():
        return case.boozmn_path
    booz_root = find_booz_xform_jax_root()
    if booz_root is not None:
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
        extrapolate=True,
    )
    rho_eval = np.asarray(rho_query, dtype=float)
    return np.asarray(spline(rho_eval), dtype=float), float(spline(1.0))


def _make_redl_profiles(case: FixedFieldCase) -> tuple[Any, Any, Any, int]:
    simsopt_src = _zenodo_root() / "codes" / "simsopt" / "src"
    if str(simsopt_src) not in sys.path:
        sys.path.insert(0, str(simsopt_src))
    from simsopt.mhd.profiles import ProfileSpline

    profiles = _archived_profiles(case)
    ne = ProfileSpline(profiles.psi_n, profiles.density_si, degree=3)
    te = ProfileSpline(profiles.psi_n, profiles.temperature_ev, degree=3)
    ti = ProfileSpline(profiles.psi_n, profiles.temperature_ev, degree=3)
    zeff = 1
    return ne, te, ti, zeff


def _compute_redl_boozer(case: FixedFieldCase, rho: np.ndarray) -> dict[str, np.ndarray]:
    booz_root = find_booz_xform_jax_root()
    if booz_root is not None:
        src = booz_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    simsopt_src = _zenodo_root() / "codes" / "simsopt" / "src"
    if str(simsopt_src) not in sys.path:
        sys.path.insert(0, str(simsopt_src))
    from booz_xform_jax import Booz_xform
    from simsopt.mhd.bootstrap import compute_trapped_fraction, j_dot_B_Redl

    ne, te, ti, zeff = _make_redl_profiles(case)

    bx = Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(case.wout_path))
    bx.mboz = 16
    bx.nboz = 16
    bx.run(jit=False)

    s_values = np.asarray(rho, dtype=float) ** 2
    s_b = np.asarray(bx.s_b, dtype=float)
    bmnc_b = np.asarray(bx.bmnc_b, dtype=float)
    gmnc_b = np.asarray(bx.gmnc_b, dtype=float)
    xm_b = np.asarray(bx.xm_b, dtype=int)
    xn_b = np.asarray(bx.xn_b, dtype=int)
    nfp = int(np.asarray(bx.nfp).reshape(()))
    keep = xm_b * case.helicity_n * nfp == xn_b

    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    bmnc = np.vstack([np.interp(s_values, s_b, row) for row in bmnc_b[keep]])
    gmnc = np.vstack([np.interp(s_values, s_b, row) for row in gmnc_b[keep]])

    mod_b = np.zeros((theta.size, s_values.size))
    sqrtg = np.zeros((theta.size, s_values.size))
    for m, bcoef, gcoef in zip(xm_b[keep], bmnc, gmnc, strict=True):
        phase = np.cos(m * theta)[:, None]
        mod_b += phase * bcoef[None, :]
        sqrtg += phase * gcoef[None, :]

    _, _, epsilon, fsab2, fsa_1overb, f_t = compute_trapped_fraction(mod_b, sqrtg)
    g = np.interp(s_values, s_b, np.asarray(bx.Boozer_G_all, dtype=float))
    i = np.interp(s_values, s_b, np.asarray(bx.Boozer_I_all, dtype=float))
    iota = np.interp(s_values, s_b, np.asarray(bx.iota, dtype=float))
    with Dataset(case.wout_path, "r") as handle:
        psi_edge = -float(np.asarray(handle.variables["phi"][:], dtype=float)[-1]) / (2.0 * np.pi)
    current, _ = j_dot_B_Redl(
        ne,
        te,
        ti,
        zeff,
        case.helicity_n,
        s=s_values,
        G=g,
        R=(g + iota * i) * fsa_1overb,
        iota=iota,
        epsilon=epsilon,
        f_t=f_t,
        psi_edge=psi_edge,
        nfp=nfp,
    )
    return {
        "rho": np.asarray(rho, dtype=float),
        "jdotb": np.asarray(current, dtype=float),
        "current_over_root": np.asarray(current, dtype=float) / np.maximum(np.sqrt(fsab2), 1.0e-30),
        "root_fsab2": np.sqrt(fsab2),
    }


def _make_species(field: NEOPAX.Field, case: FixedFieldCase) -> NEOPAX.Species:
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


def _adaptive_nu_values(
    species: NEOPAX.Species, grid: NEOPAX.Grid
) -> tuple[np.ndarray, dict[str, Any]]:
    from NEOPAX._species import collisionality

    support: dict[str, Any] = {}
    positive_values: list[np.ndarray] = []
    species_labels = ("electron", "ion")
    for species_index, label in enumerate(species_labels):
        thermal = np.asarray(species.v_thermal[species_index], dtype=float)
        samples: list[np.ndarray] = []
        for radial_index, v_th in enumerate(thermal):
            velocity = np.asarray(grid.v_norm, dtype=float) * float(v_th)
            nu_v = np.asarray(
                collisionality(species_index, species, velocity, radial_index) / velocity,
                dtype=float,
            )
            finite_positive = nu_v[np.isfinite(nu_v) & (nu_v > 0.0)]
            if finite_positive.size:
                positive_values.append(finite_positive)
                samples.append(finite_positive)
        if samples:
            merged = np.concatenate(samples)
            support[label] = {
                "min": float(np.min(merged)),
                "max": float(np.max(merged)),
            }
        else:
            support[label] = {"min": None, "max": None}
    if not positive_values:
        raise ValueError("could not determine a positive collisionality support for NEOPAX")
    merged_all = np.concatenate(positive_values)
    nu_min = max(float(np.min(merged_all)) / 3.0, 1.0e-8)
    nu_max = max(float(np.max(merged_all)) * 3.0, nu_min * 10.0)
    values = np.logspace(np.log10(nu_min), np.log10(nu_max), 17)
    support["axis"] = {
        "min": float(values[0]),
        "max": float(values[-1]),
        "count": int(values.size),
    }
    return values, support


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


def _compute_ntx_neopax_profile(case: FixedFieldCase, rho: np.ndarray) -> dict[str, np.ndarray]:
    boozmn = _ensure_boozmn(case)
    n_r = max(int(NTX_NEOPAX_RADIAL_POINTS), 9)
    timing: dict[str, float] = {}
    profiles = _archived_profiles(case)

    start = time.perf_counter()
    field = NEOPAX.Field.read_vmec_booz(n_r, str(case.wout_path), str(boozmn))
    timing["field_seconds"] = float(time.perf_counter() - start)

    start = time.perf_counter()
    species = _make_species(field, case)
    ntx_grid = NEOPAX.Grid.create_standard(n_r, 64, 2, n_order=NTX_NEOPAX_N_ORDER)
    nu_values, nu_support = _adaptive_nu_values(species, ntx_grid)
    timing["species_seconds"] = float(time.perf_counter() - start)

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
        timing["surface_load_seconds"] = 0.0
        start = time.perf_counter()
        scan = load_neopax_reference_scan(scan_path)
        timing["ntx_scan_seconds"] = float(time.perf_counter() - start)
    else:
        start = time.perf_counter()
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
        timing["surface_load_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        scan = build_ntx_neopax_scan_from_surfaces(
            surfaces,
            rho=rho_surface,
            nu_v=np.asarray(nu_values),
            Er=np.asarray(er_values),
            drds=np.asarray(drds),
            grid=NTX_GRID,
            source_name=f"fixed_field_{case.name}",
        )
        timing["ntx_scan_seconds"] = float(time.perf_counter() - start)
        write_neopax_scan_hdf5(scan, scan_path)

    start = time.perf_counter()
    database = to_neopax_monoenergetic(
        scan,
        a_b=float(field.a_b),
        d33_mode=NTX_NEOPAX_D33_MODE,
    )
    lij_nomom, _, _, upar_nomom = get_Neoclassical_Fluxes(species, ntx_grid, field, database)
    lij_spitzer, eij_spitzer, nu_weighted_average = jax.vmap(
        jax.vmap(
            get_Lij_matrix_with_momentum_correction,
            in_axes=(None, None, None, None, None, 0),
        ),
        in_axes=(None, None, None, None, 0, None),
    )(species, ntx_grid, field, database, species.species_indeces, ntx_grid.full_grid_indeces)
    lij_spitzer = lij_spitzer.at[:, 0, :, :].set(lij_spitzer.at[:, 1, :, :].get())
    eij_spitzer = eij_spitzer.at[:, 0, :, :].set(eij_spitzer.at[:, 1, :, :].get())
    lij_spitzer = _replace_nonfinite_radial_boundaries(lij_spitzer)
    eij_spitzer = _replace_nonfinite_radial_boundaries(eij_spitzer)
    nu_weighted_average = _replace_nonfinite_radial_boundaries(nu_weighted_average)
    _, _, upar_total, _, _ = jax.vmap(
        get_momentum_Correction,
        in_axes=(None, None, None, 0, 1, 1, 1),
    )(
        species,
        ntx_grid,
        field,
        ntx_grid.full_grid_indeces,
        lij_spitzer,
        eij_spitzer,
        nu_weighted_average,
    )
    timing["neopax_closure_seconds"] = float(time.perf_counter() - start)

    upar_nomom = np.asarray(upar_nomom, dtype=float)
    if upar_nomom.shape != (2, rho_field.size):
        raise ValueError(
            f"unexpected no-momentum parallel-flow shape for {case.name}: {upar_nomom.shape}"
        )
    upar_total = np.asarray(upar_total, dtype=float)
    if upar_total.shape != (rho_field.size, 2):
        raise ValueError(
            "unexpected momentum-corrected parallel-flow shape for "
            f"{case.name}: {upar_total.shape}"
        )
    electron_current_nomom = np.asarray(-elementary_charge * upar_nomom[0], dtype=float)
    ion_current_nomom = np.asarray(elementary_charge * upar_nomom[1], dtype=float)
    current_profile_nomom = np.asarray(electron_current_nomom + ion_current_nomom, dtype=float)
    electron_current_total = np.asarray(
        -elementary_charge * upar_total[:, 0],
        dtype=float,
    )
    ion_current_total = np.asarray(elementary_charge * upar_total[:, 1], dtype=float)
    electron_current_correction = np.asarray(
        electron_current_total - electron_current_nomom,
        dtype=float,
    )
    ion_current_correction = np.asarray(ion_current_total - ion_current_nomom, dtype=float)
    electron_current = electron_current_total
    ion_current = ion_current_total
    current_profile = np.asarray(electron_current + ion_current, dtype=float)
    current_profile_correction = np.asarray(
        electron_current_correction + ion_current_correction,
        dtype=float,
    )
    root_fsab2 = np.asarray(np.abs(field.B0) * np.sqrt(np.abs(field.Bsqav)), dtype=float)
    finite_mask = np.isfinite(rho_field) & np.isfinite(current_profile) & np.isfinite(root_fsab2)
    if np.count_nonzero(finite_mask) < 3:
        raise ValueError(
            f"NTX+NEOPAX produced too few finite fixed-field current samples for {case.name}: "
            f"{np.count_nonzero(finite_mask)}"
        )
    return {
        "rho": np.asarray(rho, dtype=float),
        "jdotb": _interp_profile(rho_field, current_profile, rho),
        "electron_A1": _interp_profile(rho_field, np.asarray(species.A1[0], dtype=float), rho),
        "ion_A1": _interp_profile(rho_field, np.asarray(species.A1[1], dtype=float), rho),
        "electron_A2": _interp_profile(rho_field, np.asarray(species.A2[0], dtype=float), rho),
        "ion_A2": _interp_profile(rho_field, np.asarray(species.A2[1], dtype=float), rho),
        "electron_L31": _interp_profile(
            rho_field, np.asarray(lij_nomom[0, :, 2, 0], dtype=float), rho
        ),
        "ion_L31": _interp_profile(
            rho_field, np.asarray(lij_nomom[1, :, 2, 0], dtype=float), rho
        ),
        "electron_L32": _interp_profile(
            rho_field, np.asarray(lij_nomom[0, :, 2, 1], dtype=float), rho
        ),
        "ion_L32": _interp_profile(
            rho_field, np.asarray(lij_nomom[1, :, 2, 1], dtype=float), rho
        ),
        "electron_L33": _interp_profile(
            rho_field, np.asarray(lij_nomom[0, :, 2, 2], dtype=float), rho
        ),
        "ion_L33": _interp_profile(
            rho_field, np.asarray(lij_nomom[1, :, 2, 2], dtype=float), rho
        ),
        "electron_current": _interp_profile(rho_field, electron_current, rho),
        "ion_current": _interp_profile(rho_field, ion_current, rho),
        "electron_current_nomom": _interp_profile(rho_field, electron_current_nomom, rho),
        "ion_current_nomom": _interp_profile(rho_field, ion_current_nomom, rho),
        "electron_current_correction": _interp_profile(rho_field, electron_current_correction, rho),
        "ion_current_correction": _interp_profile(rho_field, ion_current_correction, rho),
        "jdotb_nomom": _interp_profile(rho_field, current_profile_nomom, rho),
        "jdotb_correction": _interp_profile(rho_field, current_profile_correction, rho),
        "electron_flow": _interp_profile(rho_field, electron_current, rho),
        "ion_flow": _interp_profile(rho_field, ion_current, rho),
        "current_over_root": _interp_profile(
            rho_field,
            current_profile / np.maximum(root_fsab2, 1.0e-30),
            rho,
        ),
        "root_fsab2": _interp_profile(rho_field, root_fsab2, rho),
        "rho_field": rho_field,
        "jdotb_grid": current_profile,
        "jdotb_nomom_grid": current_profile_nomom,
        "electron_A1_grid": np.asarray(species.A1[0], dtype=float),
        "ion_A1_grid": np.asarray(species.A1[1], dtype=float),
        "electron_A2_grid": np.asarray(species.A2[0], dtype=float),
        "ion_A2_grid": np.asarray(species.A2[1], dtype=float),
        "electron_L31_grid": np.asarray(lij_nomom[0, :, 2, 0], dtype=float),
        "ion_L31_grid": np.asarray(lij_nomom[1, :, 2, 0], dtype=float),
        "electron_L32_grid": np.asarray(lij_nomom[0, :, 2, 1], dtype=float),
        "ion_L32_grid": np.asarray(lij_nomom[1, :, 2, 1], dtype=float),
        "electron_L33_grid": np.asarray(lij_nomom[0, :, 2, 2], dtype=float),
        "ion_L33_grid": np.asarray(lij_nomom[1, :, 2, 2], dtype=float),
        "electron_L43_grid": np.asarray(lij_spitzer[0, :, 3, 2], dtype=float),
        "ion_L43_grid": np.asarray(lij_spitzer[1, :, 3, 2], dtype=float),
        "electron_L45_grid": np.asarray(lij_spitzer[0, :, 3, 4], dtype=float),
        "ion_L45_grid": np.asarray(lij_spitzer[1, :, 3, 4], dtype=float),
        "electron_L55_grid": np.asarray(lij_spitzer[0, :, 4, 4], dtype=float),
        "ion_L55_grid": np.asarray(lij_spitzer[1, :, 4, 4], dtype=float),
        "electron_current_grid": electron_current,
        "ion_current_grid": ion_current,
        "electron_current_nomom_grid": electron_current_nomom,
        "ion_current_nomom_grid": ion_current_nomom,
        "electron_current_correction_grid": electron_current_correction,
        "ion_current_correction_grid": ion_current_correction,
        "electron_flow_grid": electron_current,
        "ion_flow_grid": ion_current,
        "rho_field_finite": rho_field[finite_mask],
        "jdotb_grid_finite": current_profile[finite_mask],
        "jdotb_nomom_grid_finite": current_profile_nomom[finite_mask],
        "jdotb_correction_grid_finite": current_profile_correction[finite_mask],
        "nu_values": np.asarray(nu_values, dtype=float),
        "nu_support": nu_support,
        "er_axis": np.asarray(er_axis, dtype=float),
        "archived_er_hat_rho": archived_er_hat,
        "archived_alpha_rho": archived_alpha,
        "archived_er_rho": archived_er,
        "scan_path": str(scan_path),
        "timing": timing,
    }


def _patched_sfincs_jax_input(case: FixedFieldCase, psi_n: float, source_input: Path) -> Path:
    workdir = case.output_dir / "sfincs_jax_precise_qs" / f"psiN_{psi_n:.3f}"
    workdir.mkdir(parents=True, exist_ok=True)
    patched = workdir / "input_sfincs_jax.namelist"
    text = source_input.read_text(encoding="utf-8")
    text = text.replace("inputRadialCoordinate = 1", "inputRadialCoordinate = 3")
    text = text.replace("inputRadialCoordinate = 1  ! psiN", "inputRadialCoordinate = 3  ! rN")
    if "rN_wish" not in text:
        text = text.replace(
            "  psiN_wish =",
            f"  rN_wish = {np.sqrt(psi_n):.17g}\n  psiN_wish =",
            1,
        )
    if "inputRadialCoordinateForGradients" not in text:
        text = text.replace(
            "&geometryParameters\n",
            "&geometryParameters\n  inputRadialCoordinateForGradients = 4\n",
            1,
        )
    patched.write_text(text, encoding="utf-8")
    return patched


def _compute_sfincs_jax_profile(case: FixedFieldCase, rho: np.ndarray) -> dict[str, np.ndarray]:
    archived_inputs = _archived_surface_inputs(case)
    if len(archived_inputs) <= SFINCS_JAX_SAMPLE_COUNT:
        sampled_inputs = archived_inputs
    else:
        sample_idx = np.unique(
            np.round(
                np.linspace(0, len(archived_inputs) - 1, SFINCS_JAX_SAMPLE_COUNT)
            ).astype(int)
        )
        sampled_inputs = [archived_inputs[idx] for idx in sample_idx]
    current = []
    current_over_root = []
    rho_sample = []
    for psi_n, source_input in sampled_inputs:
        workdir = case.output_dir / "sfincs_jax_precise_qs" / f"psiN_{psi_n:.3f}"
        out_path = workdir / "sfincsOutput.h5"
        input_path = _patched_sfincs_jax_input(case, psi_n, source_input)
        if RECOMPUTE or not out_path.exists():
            write_sfincs_jax_output_h5(
                input_namelist=input_path,
                output_path=out_path,
                wout_path=case.wout_path,
                compute_solution=True,
                return_results=False,
                verbose=False,
            )
        data = read_sfincs_h5(out_path)
        rho_sample.append(np.sqrt(psi_n))
        current.append(float(np.asarray(data["FSABjHat"][-1], dtype=float)) * SFINCS_JHAT_TO_AM2)
        current_over_root.append(
            float(np.asarray(data["FSABjHatOverRootFSAB2"][-1], dtype=float)) * SFINCS_JHAT_TO_AM2
        )
    rho_scan = np.asarray(rho_sample, dtype=float)
    return {
        "rho": np.asarray(rho, dtype=float),
        "jdotb": _interp_profile(rho_scan, np.asarray(current, dtype=float), rho),
        "current_over_root": _interp_profile(
            rho_scan,
            np.asarray(current_over_root, dtype=float),
            rho,
        ),
        "rho_sample": rho_scan,
        "jdotb_sample": np.asarray(current, dtype=float),
        "current_over_root_sample": np.asarray(current_over_root, dtype=float),
    }


def _compute_archived_sfincs_profile(case: FixedFieldCase) -> dict[str, np.ndarray]:
    rho, current = _load_archived_sfincs_scan(case.sfincs_scan_path)
    payload: dict[str, np.ndarray] = {"rho": rho, "jdotb": current}
    species_flows = _load_archived_sfincs_species_flows(case)
    if species_flows["rho"].size:
        payload["electron_current"] = _interp_profile(
            species_flows["rho"], species_flows["electron_current"], rho
        )
        payload["ion_current"] = _interp_profile(
            species_flows["rho"],
            species_flows["ion_current"],
            rho,
        )
        payload["electron_flow"] = payload["electron_current"]
        payload["ion_flow"] = payload["ion_current"]
        payload["electron_current_sample"] = species_flows["electron_current"]
        payload["ion_current_sample"] = species_flows["ion_current"]
        payload["electron_flow_sample"] = species_flows["electron_current"]
        payload["ion_flow_sample"] = species_flows["ion_current"]
        payload["rho_sample"] = species_flows["rho"]
    return payload


def _display_label(key: str) -> str:
    return {
        "SFINCS": "SFINCS",
        "SFINCS-JAX": "SFINCS-JAX",
        "NTX+NEOPAX": "NTX+NEOPAX",
        "Redl": "Redl (Boozer)",
    }[key]


def _styles() -> dict[str, dict[str, Any]]:
    return {
        "SFINCS": dict(color="#111111", lw=2.8, ls="-"),
        "SFINCS-JAX": dict(color="#d55e00", lw=2.0, ls="--"),
        "NTX+NEOPAX": dict(color="#1f77b4", lw=2.4, ls="-"),
        "Redl": dict(color="#009e73", lw=2.0, ls="-."),
    }


def _plot_order(case_results: dict[str, dict[str, np.ndarray]]) -> tuple[str, ...]:
    order = ["SFINCS"]
    if "SFINCS-JAX" in case_results:
        order.append("SFINCS-JAX")
    order.extend(["NTX+NEOPAX", "Redl"])
    return tuple(order)


def _panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _least_squares_scale(reference: np.ndarray, model: np.ndarray, mask: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)[mask]
    trial = np.asarray(model, dtype=float)[mask]
    denom = float(np.dot(trial, trial))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(ref, trial) / denom)


def _relative_error_array(reference: np.ndarray, model: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    trial = np.asarray(model, dtype=float)
    return np.abs(trial - ref) / np.maximum(np.abs(ref), 1.0)


def _sign_mismatch_count(reference: np.ndarray, model: np.ndarray, mask: np.ndarray) -> int:
    ref = np.asarray(reference, dtype=float)
    trial = np.asarray(model, dtype=float)
    valid = mask & (np.abs(ref) > 1.0e-12)
    return int(np.count_nonzero(np.signbit(ref[valid]) != np.signbit(trial[valid])))


def _closure_diagnostics(
    case: FixedFieldCase,
    case_results: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    ref = np.asarray(case_results["SFINCS"]["jdotb"], dtype=float)
    rho = np.asarray(case_results["SFINCS"]["rho"], dtype=float)
    interior = (rho >= INTERIOR_RHO_MIN) & (rho <= INTERIOR_RHO_MAX)
    ntx = case_results["NTX+NEOPAX"]
    profiles = _archived_profiles(case)
    density = _interp_profile(profiles.rho, profiles.density_si, rho)

    species_scale: dict[str, dict[str, float]] = {}
    density_term = np.zeros_like(ref)
    thermal_term_raw = np.zeros_like(ref)
    thermal_term_eff = np.zeros_like(ref)
    target_rho = 0.5
    mid_index = int(np.argmin(np.abs(rho - target_rho)))
    branch_currents: dict[str, np.ndarray] = {}
    for label, charge_sign in (("electron", -1.0), ("ion", 1.0)):
        ref_species = np.asarray(case_results["SFINCS"][f"{label}_current"], dtype=float)
        model_species = np.asarray(ntx[f"{label}_current"], dtype=float)
        model_species_nomom = np.asarray(ntx[f"{label}_current_nomom"], dtype=float)
        model_species_correction = np.asarray(ntx[f"{label}_current_correction"], dtype=float)
        branch_currents[f"{label}_nomom"] = model_species_nomom
        branch_currents[f"{label}_correction"] = model_species_correction
        species_scale[label] = {
            "current_scale": _least_squares_scale(ref_species, model_species, interior),
            "current_nomom_scale": _least_squares_scale(ref_species, model_species_nomom, interior),
            "current_sign_mismatch_count_interior": _sign_mismatch_count(
                ref_species,
                model_species,
                interior,
            ),
            "current_worst_relative_error_interior": float(
                np.max(_relative_error_array(ref_species, model_species)[interior])
            ),
            "current_worst_rho_interior": float(
                rho[interior][
                    int(np.argmax(_relative_error_array(ref_species, model_species)[interior]))
                ]
            ),
            "midpoint_snapshot": {
                "rho": float(rho[mid_index]),
                "reference_current": float(ref_species[mid_index]),
                "model_current": float(model_species[mid_index]),
                "model_current_nomom": float(model_species_nomom[mid_index]),
                "model_current_correction": float(model_species_correction[mid_index]),
                "A1": float(np.asarray(ntx[f"{label}_A1"], dtype=float)[mid_index]),
                "A2": float(np.asarray(ntx[f"{label}_A2"], dtype=float)[mid_index]),
                "L31": float(np.asarray(ntx[f"{label}_L31"], dtype=float)[mid_index]),
                "L32": float(np.asarray(ntx[f"{label}_L32"], dtype=float)[mid_index]),
                "L33": float(np.asarray(ntx[f"{label}_L33"], dtype=float)[mid_index]),
            },
        }
        a1 = np.asarray(ntx[f"{label}_A1"], dtype=float)
        a2 = np.asarray(ntx[f"{label}_A2"], dtype=float)
        l31 = np.asarray(ntx[f"{label}_L31"], dtype=float)
        l32 = np.asarray(ntx[f"{label}_L32"], dtype=float)
        density_term += charge_sign * elementary_charge * (-density * (l31 * a1))
        thermal_term_raw += charge_sign * elementary_charge * (-density * (l32 * a2))
        thermal_term_eff += charge_sign * elementary_charge * (
            -density * ((l32 - 1.5 * l31) * a2)
        )

    raw_alpha = _least_squares_scale(ref - density_term, thermal_term_raw, interior)
    eff_alpha = _least_squares_scale(ref - density_term, thermal_term_eff, interior)
    raw_fit = density_term + raw_alpha * thermal_term_raw
    eff_fit = density_term + eff_alpha * thermal_term_eff
    ref_interior = np.maximum(np.abs(ref[interior]), 1.0)
    rel_total = _relative_error_array(ref, np.asarray(ntx["jdotb"], dtype=float))
    hybrid_currents = {
        "nomom": branch_currents["electron_nomom"] + branch_currents["ion_nomom"],
        "electron_total_ion_nomom": (
            branch_currents["electron_nomom"]
            + branch_currents["electron_correction"]
            + branch_currents["ion_nomom"]
        ),
        "electron_nomom_ion_total": (
            branch_currents["electron_nomom"]
            + branch_currents["ion_nomom"]
            + branch_currents["ion_correction"]
        ),
        "total": (
            branch_currents["electron_nomom"]
            + branch_currents["electron_correction"]
            + branch_currents["ion_nomom"]
            + branch_currents["ion_correction"]
        ),
    }
    return {
        "current_scale": _least_squares_scale(ref, np.asarray(ntx["jdotb"], dtype=float), interior),
        "current_nomom_scale": _least_squares_scale(
            ref,
            np.asarray(ntx["jdotb_nomom"], dtype=float),
            interior,
        ),
        "hybrid_current_max_relative_error_interior": {
            name: float(np.max(_relative_error_array(ref, current)[interior]))
            for name, current in hybrid_currents.items()
        },
        "current_sign_mismatch_count_interior": _sign_mismatch_count(
            ref,
            np.asarray(ntx["jdotb"], dtype=float),
            interior,
        ),
        "current_worst_relative_error_interior": float(np.max(rel_total[interior])),
        "current_worst_rho_interior": float(rho[interior][int(np.argmax(rel_total[interior]))]),
        "species_scale": species_scale,
        "thermal_raw_best_alpha": float(raw_alpha),
        "thermal_eff_best_alpha": float(eff_alpha),
        "thermal_raw_fit_max_relative_error": float(
            np.max(np.abs(raw_fit[interior] - ref[interior]) / ref_interior)
        ),
        "thermal_eff_fit_max_relative_error": float(
            np.max(np.abs(eff_fit[interior] - ref[interior]) / ref_interior)
        ),
        "midpoint_snapshot": {
            "rho": float(rho[mid_index]),
            "reference_current": float(ref[mid_index]),
            "model_current": float(np.asarray(ntx["jdotb"], dtype=float)[mid_index]),
            "model_current_nomom": float(np.asarray(ntx["jdotb_nomom"], dtype=float)[mid_index]),
            "model_current_correction": float(
                np.asarray(ntx["jdotb_correction"], dtype=float)[mid_index]
            ),
        },
    }


def _summary_payload(
    results: dict[str, dict[str, dict[str, np.ndarray]]],
    cases: dict[str, FixedFieldCase],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "cases": {},
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png")),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf")),
    }
    for key, case_results in results.items():
        ref = np.asarray(case_results["SFINCS"]["jdotb"], dtype=float)
        ref_scale = np.maximum(np.abs(ref), 1.0)
        rho_ref = np.asarray(case_results["SFINCS"]["rho"], dtype=float)
        interior = (rho_ref >= INTERIOR_RHO_MIN) & (rho_ref <= INTERIOR_RHO_MAX)
        out["cases"][key] = {
            name: {subkey: _jsonify(value) for subkey, value in payload.items()}
            for name, payload in case_results.items()
        }
        out["cases"][key]["max_relative_error_vs_sfincs"] = {
            name: float(np.max(np.abs(np.asarray(payload["jdotb"], dtype=float) - ref) / ref_scale))
            for name, payload in case_results.items()
            if name != "SFINCS"
        }
        out["cases"][key]["max_relative_error_vs_sfincs_interior"] = {
            name: float(
                np.max(
                    np.abs(np.asarray(payload["jdotb"], dtype=float)[interior] - ref[interior])
                    / np.maximum(np.abs(ref[interior]), 1.0)
                )
            )
            for name, payload in case_results.items()
            if name != "SFINCS"
        }
        out["cases"][key]["closure_diagnostics"] = _closure_diagnostics(cases[key], case_results)
    return out


def _plot(
    results: dict[str, dict[str, dict[str, np.ndarray]]],
    cases: dict[str, FixedFieldCase],
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 8.4),
        constrained_layout=True,
        sharex="col",
        gridspec_kw={"height_ratios": (1.0, 0.55)},
    )
    styles = _styles()
    for col, key in enumerate(("qa", "qh")):
        case_results = results[key]
        ref = np.asarray(case_results["SFINCS"]["jdotb"], dtype=float)
        ax = axes[0, col]
        ax.axvspan(INTERIOR_RHO_MIN, INTERIOR_RHO_MAX, color="#f0f0f0", alpha=0.5, zorder=0)
        for name in _plot_order(case_results):
            payload = case_results[name]
            ax.plot(
                np.asarray(payload["rho"], dtype=float),
                np.asarray(payload["jdotb"], dtype=float) / 1.0e6,
                label=_display_label(name),
                **styles[name],
            )
            if name == "SFINCS-JAX" and "rho_sample" in payload:
                ax.plot(
                    np.asarray(payload["rho_sample"], dtype=float),
                    np.asarray(payload["jdotb_sample"], dtype=float) / 1.0e6,
                    marker="o",
                    ms=4.2,
                    lw=0,
                    color=styles[name]["color"],
                )
        ax.set_title(cases[key].label)
        ax.set_ylabel(r"$\langle \mathbf{J}\cdot\mathbf{B}\rangle$ [MA T A m$^{-2}$]")
        ax.grid(alpha=0.24, lw=0.6)
        _panel_label(ax, f"({chr(ord('a') + col)})")

        ax_err = axes[1, col]
        ref_scale = np.maximum(np.abs(ref), 1.0)
        ax_err.axvspan(INTERIOR_RHO_MIN, INTERIOR_RHO_MAX, color="#f0f0f0", alpha=0.5, zorder=0)
        err_order = [name for name in ("SFINCS-JAX", "NTX+NEOPAX", "Redl") if name in case_results]
        for name in err_order:
            payload = case_results[name]
            rel = np.abs(np.asarray(payload["jdotb"], dtype=float) - ref) / ref_scale
            ax_err.plot(
                np.asarray(payload["rho"], dtype=float),
                rel,
                label=f"{_display_label(name)} vs SFINCS",
                **styles[name],
            )
            if name == "SFINCS-JAX" and "rho_sample" in payload:
                sample_ref = _interp_profile(
                    np.asarray(case_results["SFINCS"]["rho"], dtype=float),
                    ref,
                    np.asarray(payload["rho_sample"], dtype=float),
                )
                sample_rel = np.abs(
                    np.asarray(payload["jdotb_sample"], dtype=float) - sample_ref
                ) / np.maximum(np.abs(sample_ref), 1.0)
                ax_err.plot(
                    np.asarray(payload["rho_sample"], dtype=float),
                    sample_rel,
                    marker="o",
                    ms=4.2,
                    lw=0,
                    color=styles[name]["color"],
                )
        ax_err.set_xlabel(r"$\rho$")
        ax_err.set_ylabel("relative error")
        ax_err.set_yscale("log")
        ax_err.grid(alpha=0.24, lw=0.6)
        _panel_label(ax_err, f"({chr(ord('c') + col)})")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_case(case: FixedFieldCase) -> dict[str, dict[str, np.ndarray]]:
    sfincs = _compute_archived_sfincs_profile(case)
    rho_ref = np.asarray(sfincs["rho"], dtype=float)
    ntx = _compute_ntx_neopax_profile(case, rho_ref)
    redl = _compute_redl_boozer(case, rho_ref)
    out = {
        "SFINCS": sfincs,
        "NTX+NEOPAX": ntx,
        "Redl": redl,
    }
    if ENABLE_SFINCS_JAX:
        out["SFINCS-JAX"] = _compute_sfincs_jax_profile(case, rho_ref)
    return out


def main() -> None:
    cases = _cases()
    results = {key: run_case(case) for key, case in cases.items()}
    _plot(results, cases)
    summary = _summary_payload(results, cases)
    summary["enable_sfincs_jax"] = ENABLE_SFINCS_JAX
    summary["sfincs_jax_sample_count"] = SFINCS_JAX_SAMPLE_COUNT
    summary["ntx_neopax_radial_points"] = NTX_NEOPAX_RADIAL_POINTS
    summary["ntx_neopax_n_order"] = NTX_NEOPAX_N_ORDER
    summary["case_metadata"] = {
        key: {
            **asdict(case),
            "wout_path": str(case.wout_path),
            "sfincs_scan_path": str(case.sfincs_scan_path),
        }
        for key, case in cases.items()
    }
    OUTPUT_PREFIX.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
