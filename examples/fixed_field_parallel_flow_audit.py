#!/usr/bin/env python3
"""Audit fixed-field precise-QS parallel-flow closure against SFINCS-JAX RHSMode=2."""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import f90nml
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import (
    GridSpec,
    build_ntx_neopax_scan_from_surfaces,
    load_vmec_surface,
    to_neopax_monoenergetic,
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
NTX_SURFACE_GRID = GridSpec(n_theta=25, n_zeta=25, n_xi=31)
NTX_NEOPAX_RADIAL_POINTS = 17
ER_AXIS_FACTORS = np.array([0.5, 1.0, 2.0], dtype=float)
SFINCS_SOLVE_METHOD = (
    os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_SOLVE_METHOD", "auto").strip() or "auto"
)
RECOMPUTE = os.environ.get("NTX_FIXED_FIELD_PARALLEL_FLOW_AUDIT_RECOMPUTE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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
    a_hat: float

    @property
    def density_si(self) -> np.ndarray:
        return self.n_hat * 1.0e20

    @property
    def temperature_ev(self) -> np.ndarray:
        return self.t_hat * 1.0e3

    @property
    def density_rho_derivative_si(self) -> np.ndarray:
        return self.dn_hat_drhat * 1.0e20 * self.a_hat

    @property
    def temperature_rho_derivative_ev(self) -> np.ndarray:
        return self.dT_hat_drhat * 1.0e3 * self.a_hat


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
    psi_n = np.asarray(psi_n_values, dtype=float)
    order = np.argsort(psi_n)
    psi_n = psi_n[order]
    with Dataset(case.wout_path) as ds:
        a_hat = float(np.asarray(ds.variables["Aminor_p"]).reshape(()))
    return ArchivedProfiles(
        psi_n=psi_n,
        rho=np.sqrt(psi_n),
        n_hat=np.asarray(n_hat_values, dtype=float)[order],
        t_hat=np.asarray(t_hat_values, dtype=float)[order],
        dn_hat_drhat=np.asarray(dn_hat_values, dtype=float)[order],
        dT_hat_drhat=np.asarray(dt_hat_values, dtype=float)[order],
        er=np.asarray(er_values, dtype=float)[order],
        a_hat=a_hat,
    )


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
    if x_unique.size >= 3:
        return PchipInterpolator(x_unique, y_unique)(np.asarray(xq, dtype=float))
    return np.interp(np.asarray(xq, dtype=float), x_unique, y_unique)


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
    n_species = 2
    n_r = rho.size
    temperature = np.vstack([temperature, temperature])
    density = np.vstack([density, density])
    electric_field = _interp_profile(profiles.rho, profiles.er, rho)
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
    archived_er = _interp_profile(profiles.rho, profiles.er, rho_surface)
    er_axis = float(np.median(archived_er)) * ER_AXIS_FACTORS
    er_values = np.repeat(er_axis[None, :], rho_surface.size, axis=0)

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
    database = to_neopax_monoenergetic(scan, a_b=float(field.a_b))
    lij, gamma, heat, upar = NEOPAX.get_Neoclassical_Fluxes(species, ntx_grid, field, database)
    return {
        "rho": rho_field,
        "L31_electron": np.asarray(lij[0, :, 2, 0], dtype=float),
        "L32_electron": np.asarray(lij[0, :, 2, 1], dtype=float),
        "L33_electron": np.asarray(lij[0, :, 2, 2], dtype=float),
        "L31_ion": np.asarray(lij[1, :, 2, 0], dtype=float),
        "L32_ion": np.asarray(lij[1, :, 2, 1], dtype=float),
        "L33_ion": np.asarray(lij[1, :, 2, 2], dtype=float),
        "upar_electron": np.asarray(upar[0], dtype=float),
        "upar_ion": np.asarray(upar[1], dtype=float),
        "gamma_electron": np.asarray(gamma[0], dtype=float),
        "gamma_ion": np.asarray(gamma[1], dtype=float),
        "heat_electron": np.asarray(heat[0], dtype=float),
        "heat_ion": np.asarray(heat[1], dtype=float),
        "nu_values": np.asarray(nu_values, dtype=float),
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
    target_dir = case.output_dir / "sfincs_jax_rhsmode2" / f"psiN_{psi_n:.3f}"
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
    state_prefix = input_path.with_name("stateVector")
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
            "--out-state-prefix",
            str(state_prefix),
            "--solve-method",
            SFINCS_SOLVE_METHOD,
            "--tol",
            "1e-10",
        ]
        try:
            subprocess.run(
                command,
                cwd=str(_sfincs_jax_root()),
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stdout) from exc

    _bootstrap_current_pythonpath()
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.transport_matrix import (
        v3_transport_matrix_from_state_vectors,
        v3_transport_output_fields_vm_only,
    )
    from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist
    from sfincs_jax.v3_system import full_system_operator_from_namelist

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, grids=grids, geom=geom)
    state_vectors = {
        which_rhs: np.load(input_path.with_name(f"stateVector.whichRHS{which_rhs}.npy"))
        for which_rhs in (1, 2, 3)
    }
    matrix = np.asarray(
        v3_transport_matrix_from_state_vectors(
            op0=op0,
            geom=geom,
            state_vectors_by_rhs=state_vectors,
        ),
        dtype=float,
    )
    output_fields = v3_transport_output_fields_vm_only(op0=op0, state_vectors_by_rhs=state_vectors)
    fields = {key: np.asarray(value, dtype=float) for key, value in output_fields.items()}
    return {
        "input_path": str(input_path),
        "transport_matrix": matrix,
        "FSABFlow": fields["FSABFlow"],
        "FSABjHat": fields["FSABjHat"],
        "FSABjHatOverRootFSAB2": fields["FSABjHatOverRootFSAB2"],
    }


def _relative_error(values: np.ndarray, reference: np.ndarray) -> list[float]:
    return (
        np.abs(np.asarray(values, dtype=float) - np.asarray(reference, dtype=float))
        / np.maximum(np.abs(np.asarray(reference, dtype=float)), 1.0e-16)
    ).tolist()


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
        ion_row = np.array(
            [
                _interp_profile(rho_grid, lij["L31_ion"], np.asarray([rho]))[0],
                _interp_profile(rho_grid, lij["L32_ion"], np.asarray([rho]))[0],
                _interp_profile(rho_grid, lij["L33_ion"], np.asarray([rho]))[0],
            ],
            dtype=float,
        )
        electron_row = np.array(
            [
                _interp_profile(rho_grid, lij["L31_electron"], np.asarray([rho]))[0],
                _interp_profile(rho_grid, lij["L32_electron"], np.asarray([rho]))[0],
                _interp_profile(rho_grid, lij["L33_electron"], np.asarray([rho]))[0],
            ],
            dtype=float,
        )
        rows.append(
            {
                "rho": float(rho),
                "psi_n_source": float(source_psi_n),
                "sfincs_jax": {
                    "transport_matrix": sfincs["transport_matrix"].tolist(),
                    "row3": row3.tolist(),
                    "FSABFlow": np.asarray(sfincs["FSABFlow"], dtype=float).tolist(),
                    "FSABjHat": np.asarray(sfincs["FSABjHat"], dtype=float).tolist(),
                    "FSABjHatOverRootFSAB2": np.asarray(
                        sfincs["FSABjHatOverRootFSAB2"], dtype=float
                    ).tolist(),
                    "input_path": sfincs["input_path"],
                },
                "ntx_neopax": {
                    "ion_row3": ion_row.tolist(),
                    "ion_row3_sign_flipped": (-ion_row).tolist(),
                    "electron_row3": electron_row.tolist(),
                    "electron_row3_sign_flipped": (-electron_row).tolist(),
                },
                "relative_error": {
                    "ion_raw_vs_sfincs": _relative_error(ion_row, row3),
                    "ion_sign_flipped_vs_sfincs": _relative_error(-ion_row, row3),
                    "electron_raw_vs_sfincs": _relative_error(electron_row, row3),
                    "electron_sign_flipped_vs_sfincs": _relative_error(-electron_row, row3),
                },
            }
        )
    case_payload = asdict(case)
    case_payload["wout_path"] = str(case.wout_path)
    case_payload["sfincs_scan_path"] = str(case.sfincs_scan_path)
    return {"case": case_payload, "rows": rows}


def _plot(summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 9.8), sharex=True, constrained_layout=True)
    channel_labels = ("L31", "L32", "L33")

    for col, case_key in enumerate(("qa", "qh")):
        rows = summary[case_key]["rows"]
        rho = np.asarray([row["rho"] for row in rows], dtype=float)
        sfincs_row3 = np.asarray([row["sfincs_jax"]["row3"] for row in rows], dtype=float)
        ion_row3 = np.asarray([row["ntx_neopax"]["ion_row3"] for row in rows], dtype=float)
        ion_row3_flipped = np.asarray(
            [row["ntx_neopax"]["ion_row3_sign_flipped"] for row in rows],
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
                ion_row3[:, row_idx],
                "s--",
                color="#1f77b4",
                lw=1.9,
                label="NTX+NEOPAX ion raw",
            )
            ax.plot(
                rho,
                ion_row3_flipped[:, row_idx],
                "^:",
                color="#6c757d",
                lw=1.7,
                label="NTX+NEOPAX ion sign-flipped",
            )
            ax.grid(alpha=0.25, lw=0.6)
            ax.set_title(f"{summary[case_key]['case']['label']}: {label}")
            if row_idx == 2:
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
