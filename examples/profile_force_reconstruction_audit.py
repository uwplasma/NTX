#!/usr/bin/env python3
"""Audit primitive-to-force reconstruction on the archived precise-QS profiles."""
# ruff: noqa: E402

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import f90nml
import matplotlib
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import CubicHermiteSpline

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import PrimitiveSpeciesProfile, build_species_profile_from_primitives
from ntx._checkout_paths import find_qs_zenodo_root

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "profile_force_reconstruction_audit"
INTERIOR_RHO_MIN = 0.25
INTERIOR_RHO_MAX = 0.85
DENSE_RHO_COUNT = 41


@dataclass(frozen=True)
class FixedFieldCase:
    name: str
    label: str
    wout_path: Path
    sfincs_scan_path: Path

    @property
    def sfincs_scan_dir(self) -> Path:
        return self.sfincs_scan_path.parent


@dataclass(frozen=True)
class ArchivedProfiles:
    rho: np.ndarray
    n_hat: np.ndarray
    t_hat: np.ndarray
    dn_hat_drhat: np.ndarray
    dT_hat_drhat: np.ndarray
    er: np.ndarray
    alpha: np.ndarray
    a_hat: float


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.0, 7.2),
            "figure.dpi": 220,
            "font.size": 10.5,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _zenodo_root() -> Path:
    root = find_qs_zenodo_root()
    if root is None:
        raise RuntimeError("profile reconstruction audit requires the local precise-QS archive")
    return root


def _cases() -> dict[str, FixedFieldCase]:
    root = _zenodo_root()
    calc_root = root / "calculations" / "20211226-01-sfincs_for_precise_QS_for_Redl_benchmark"
    wout_root = root / "codes" / "simsopt" / "tests" / "test_files"
    cases = {
        "qa": FixedFieldCase(
            name="qa",
            label="QA precise-QS archived profile family",
            wout_path=wout_root / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
            sfincs_scan_path=calc_root
            / "20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces"
            / "sfincsScan.dat",
        ),
        "qh": FixedFieldCase(
            name="qh",
            label="QH precise-QS archived profile family",
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
    with Dataset(case.wout_path) as ds:
        a_hat = float(np.asarray(ds.variables["Aminor_p"]).reshape(()))
    return ArchivedProfiles(
        rho=np.sqrt(psi_n[order]),
        n_hat=np.asarray(n_hat_values, dtype=float)[order],
        t_hat=np.asarray(t_hat_values, dtype=float)[order],
        dn_hat_drhat=np.asarray(dn_hat_values, dtype=float)[order],
        dT_hat_drhat=np.asarray(dt_hat_values, dtype=float)[order],
        er=np.asarray(er_values, dtype=float)[order],
        alpha=np.asarray(alpha_values, dtype=float)[order],
        a_hat=a_hat,
    )


def _reference_force_profiles(
    profiles: ArchivedProfiles,
    charge: float,
) -> tuple[np.ndarray, np.ndarray]:
    density = np.maximum(profiles.n_hat, 1.0e-30)
    temperature = np.maximum(profiles.t_hat, 1.0e-30)
    a3 = profiles.dT_hat_drhat / temperature
    a1 = profiles.dn_hat_drhat / density - 1.5 * a3 + charge * profiles.alpha * profiles.er
    return a1, a3


def _resample_profiles(profiles: ArchivedProfiles, rho_eval: np.ndarray) -> ArchivedProfiles:
    rho_nodes = np.asarray(profiles.rho, dtype=float)
    rho_query = np.asarray(rho_eval, dtype=float)
    rhat_nodes = rho_nodes * float(profiles.a_hat)
    rhat_query = rho_query * float(profiles.a_hat)

    def hermite_pair(values: np.ndarray, derivatives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spline = CubicHermiteSpline(
            rhat_nodes,
            np.asarray(values, dtype=float),
            np.asarray(derivatives, dtype=float),
            extrapolate=True,
        )
        return (
            np.asarray(spline(rhat_query), dtype=float),
            np.asarray(spline.derivative()(rhat_query), dtype=float),
        )

    n_hat, dn_hat = hermite_pair(profiles.n_hat, profiles.dn_hat_drhat)
    t_hat, dt_hat = hermite_pair(profiles.t_hat, profiles.dT_hat_drhat)
    er = np.interp(rho_query, rho_nodes, np.asarray(profiles.er, dtype=float))
    alpha = np.interp(rho_query, rho_nodes, np.asarray(profiles.alpha, dtype=float))
    return ArchivedProfiles(
        rho=rho_query,
        n_hat=n_hat,
        t_hat=t_hat,
        dn_hat_drhat=dn_hat,
        dT_hat_drhat=dt_hat,
        er=er,
        alpha=alpha,
        a_hat=float(profiles.a_hat),
    )


def _reconstructed_force_profiles(
    profiles: ArchivedProfiles,
    charge: float,
) -> tuple[np.ndarray, np.ndarray]:
    primitive = PrimitiveSpeciesProfile(
        charge=charge,
        nu_v=np.full_like(profiles.rho, 1.0e-3),
        density=profiles.n_hat,
        temperature=profiles.t_hat,
        electrostatic_prefactor=profiles.alpha,
    )
    species = build_species_profile_from_primitives(
        profiles.rho * profiles.a_hat,
        primitive,
        er_profile=profiles.er,
    )
    return np.asarray(species.A1, dtype=float), np.asarray(species.A3, dtype=float)


def _interior_mask(rho: np.ndarray) -> np.ndarray:
    return (rho >= INTERIOR_RHO_MIN) & (rho <= INTERIOR_RHO_MAX)


def _max_relative_error(reference: np.ndarray, reconstructed: np.ndarray, rho: np.ndarray) -> float:
    mask = _interior_mask(rho)
    denom = np.maximum(np.abs(reference[mask]), 1.0e-12)
    return float(np.max(np.abs(reconstructed[mask] - reference[mask]) / denom))


def build_report() -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for name, case in _cases().items():
        coarse_profiles = _archived_profiles(case)
        dense_rho = np.linspace(
            float(coarse_profiles.rho[0]),
            float(coarse_profiles.rho[-1]),
            DENSE_RHO_COUNT,
        )
        profiles = _resample_profiles(coarse_profiles, dense_rho)
        electron_ref_a1, ref_a3 = _reference_force_profiles(profiles, charge=-1.0)
        ion_ref_a1, _ = _reference_force_profiles(profiles, charge=1.0)
        electron_a1, electron_a3 = _reconstructed_force_profiles(profiles, charge=-1.0)
        ion_a1, ion_a3 = _reconstructed_force_profiles(profiles, charge=1.0)
        report[name] = {
            "label": case.label,
            "rho": profiles.rho.tolist(),
            "electron_A1_reference": electron_ref_a1.tolist(),
            "electron_A1_reconstructed": electron_a1.tolist(),
            "ion_A1_reference": ion_ref_a1.tolist(),
            "ion_A1_reconstructed": ion_a1.tolist(),
            "A3_reference": ref_a3.tolist(),
            "electron_A3_reconstructed": electron_a3.tolist(),
            "ion_A3_reconstructed": ion_a3.tolist(),
            "metrics": {
                "electron_A1_interior_max_relative_error": _max_relative_error(
                    electron_ref_a1,
                    electron_a1,
                    profiles.rho,
                ),
                "ion_A1_interior_max_relative_error": _max_relative_error(
                    ion_ref_a1,
                    ion_a1,
                    profiles.rho,
                ),
                "A3_interior_max_relative_error": _max_relative_error(
                    ref_a3,
                    electron_a3,
                    profiles.rho,
                ),
            },
        }
    return report


def plot_report(report: dict[str, dict[str, object]], output_prefix: Path) -> None:
    _configure_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    case_order = ("qa", "qh")
    for row, case_key in enumerate(case_order):
        payload = report[case_key]
        rho = np.asarray(payload["rho"], dtype=float)
        ax_a1 = axes[row, 0]
        ax_a3 = axes[row, 1]
        electron_ref = np.asarray(payload["electron_A1_reference"], dtype=float)
        electron_rec = np.asarray(payload["electron_A1_reconstructed"], dtype=float)
        ion_ref = np.asarray(payload["ion_A1_reference"], dtype=float)
        ion_rec = np.asarray(payload["ion_A1_reconstructed"], dtype=float)
        a3_ref = np.asarray(payload["A3_reference"], dtype=float)
        a3_rec = np.asarray(payload["electron_A3_reconstructed"], dtype=float)
        metrics = payload["metrics"]

        ax_a1.plot(rho, electron_ref, color="#0072B2", lw=2.2, label="electron ref")
        ax_a1.plot(rho, electron_rec, color="#0072B2", lw=2.0, ls="--", label="electron NTX")
        ax_a1.plot(rho, ion_ref, color="#009E73", lw=2.2, label="ion ref")
        ax_a1.plot(rho, ion_rec, color="#009E73", lw=2.0, ls="--", label="ion NTX")
        ax_a1.axvspan(INTERIOR_RHO_MIN, INTERIOR_RHO_MAX, color="#f3f4f6", zorder=0)
        ax_a1.set_title(f"{payload['label']}: $A_1(\\rho)$")
        ax_a1.set_xlabel(r"$\rho$")
        ax_a1.set_ylabel(r"$A_1$")
        ax_a1.text(
            0.03,
            0.97,
            (
                f"e max rel. err. = {metrics['electron_A1_interior_max_relative_error']:.2e}\n"
                f"i max rel. err. = {metrics['ion_A1_interior_max_relative_error']:.2e}"
            ),
            transform=ax_a1.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
        )
        if row == 0:
            ax_a1.legend(loc="best", ncols=2)

        ax_a3.plot(rho, a3_ref, color="#D55E00", lw=2.2, label="reference")
        ax_a3.plot(rho, a3_rec, color="#D55E00", lw=2.0, ls="--", label="NTX reconstruction")
        ax_a3.axvspan(INTERIOR_RHO_MIN, INTERIOR_RHO_MAX, color="#f3f4f6", zorder=0)
        ax_a3.set_title(f"{payload['label']}: $A_3(\\rho)$")
        ax_a3.set_xlabel(r"$\rho$")
        ax_a3.set_ylabel(r"$A_3$")
        ax_a3.text(
            0.03,
            0.97,
            f"max rel. err. = {metrics['A3_interior_max_relative_error']:.2e}",
            transform=ax_a3.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
        )
        if row == 0:
            ax_a3.legend(loc="best")

    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    report = build_report()
    plot_report(report, output_prefix)
    output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
