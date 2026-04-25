#!/usr/bin/env python3
"""Audit precise-QS Redl benchmarks from the archived Zenodo bundle."""
# ruff: noqa: E402

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx._checkout_paths import find_booz_xform_jax_root, find_qs_zenodo_root

OUTPUT_DIR = ROOT / "examples" / "outputs" / "precise_qs_redl_sfincs_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = OUTPUT_DIR / "precise_qs_redl_sfincs_audit"

S_VALUES = np.linspace(0.025, 0.975, 39)
BOOZER_RESOLUTION = {"mboz": 16, "nboz": 16, "n_theta": 256}
SFINCS_SI_FACTOR = 437695 * 1.0e20 * 1.602177e-19


@dataclass(frozen=True)
class PreciseQSCase:
    name: str
    label: str
    helicity_n: int
    wout_path: Path
    sfincs_scan_path: Path


def _zenodo_root() -> Path:
    root = find_qs_zenodo_root()
    if root is None:
        raise RuntimeError("precise-QS audit requires the local Zenodo archive")
    return root


def _load_redl_symbols() -> tuple[Any, Any, Any, Any]:
    simsopt_src = _zenodo_root() / "codes" / "simsopt" / "src"
    if str(simsopt_src) not in sys.path:
        sys.path.insert(0, str(simsopt_src))
    from simsopt.mhd.bootstrap import RedlGeomVmec, compute_trapped_fraction, j_dot_B_Redl
    from simsopt.mhd.profiles import ProfilePolynomial
    from simsopt.mhd.vmec import Vmec

    return ProfilePolynomial, RedlGeomVmec, Vmec, (compute_trapped_fraction, j_dot_B_Redl)


def _load_booz_xform() -> Any:
    booz_root = find_booz_xform_jax_root()
    if booz_root is not None:
        src = booz_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    from booz_xform_jax import Booz_xform

    return Booz_xform


def _precise_qs_cases() -> dict[str, PreciseQSCase]:
    root = _zenodo_root()
    calc_root = (
        root / "calculations" / "20211226-01-sfincs_for_precise_QS_for_Redl_benchmark"
    )
    wout_root = root / "codes" / "simsopt" / "tests" / "test_files"
    cases = {
        "qa": PreciseQSCase(
            name="qa",
            label="Precise QA fixed-field benchmark",
            helicity_n=0,
            wout_path=wout_root / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
            sfincs_scan_path=calc_root
            / "20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces"
            / "sfincsScan.dat",
        ),
        "qh": PreciseQSCase(
            name="qh",
            label="Precise QH fixed-field benchmark",
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
        raise FileNotFoundError(f"missing precise-QS audit files: {missing}")
    return cases


def _reference_profiles() -> tuple[Any, Any, Any, int]:
    ProfilePolynomial, _, _, _ = _load_redl_symbols()
    ne = ProfilePolynomial(4.13e20 * np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
    te = ProfilePolynomial(12.0e3 * np.array([1.0, -1.0]))
    ti = te
    zeff = 1
    return ne, te, ti, zeff


def _load_archived_sfincs_current(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    labels = list(payload["ylabels"])
    try:
        idx = labels.index("FSABjHat")
    except ValueError as exc:
        raise ValueError(f"FSABjHat not found in {path}") from exc
    s = np.asarray(payload["xdata"][idx], dtype=float)
    current = np.asarray(payload["ydata"][idx], dtype=float) * SFINCS_SI_FACTOR
    return s, current


def _compute_redl_vmec(case: PreciseQSCase) -> np.ndarray:
    _, RedlGeomVmec, Vmec, (_, j_dot_B_Redl) = _load_redl_symbols()
    ne, te, ti, zeff = _reference_profiles()
    vmec = Vmec(str(case.wout_path))
    geom = RedlGeomVmec(vmec, S_VALUES)
    current, _ = j_dot_B_Redl(ne, te, ti, zeff, case.helicity_n, geom=geom, plot=False)
    return np.asarray(current, dtype=float)


def _compute_redl_boozer(case: PreciseQSCase) -> np.ndarray:
    Booz_xform = _load_booz_xform()
    _, _, _, (compute_trapped_fraction, j_dot_B_Redl) = _load_redl_symbols()
    ne, te, ti, zeff = _reference_profiles()

    bx = Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(case.wout_path))
    bx.mboz = BOOZER_RESOLUTION["mboz"]
    bx.nboz = BOOZER_RESOLUTION["nboz"]
    bx.run(jit=False)

    s_b = np.asarray(bx.s_b, dtype=float)
    bmnc_b = np.asarray(bx.bmnc_b, dtype=float)
    gmnc_b = np.asarray(bx.gmnc_b, dtype=float)
    xm_b = np.asarray(bx.xm_b, dtype=int)
    xn_b = np.asarray(bx.xn_b, dtype=int)
    nfp = int(np.asarray(bx.nfp).reshape(()))
    keep = xm_b * case.helicity_n * nfp == xn_b

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        BOOZER_RESOLUTION["n_theta"],
        endpoint=False,
    )
    bmnc = interp1d(s_b, bmnc_b, axis=1, fill_value="extrapolate")(S_VALUES)
    gmnc = interp1d(s_b, gmnc_b, axis=1, fill_value="extrapolate")(S_VALUES)

    mod_b = np.zeros((theta.size, S_VALUES.size))
    sqrtg = np.zeros((theta.size, S_VALUES.size))
    for m, bcoef, gcoef in zip(xm_b[keep], bmnc[keep], gmnc[keep], strict=True):
        cos_mtheta = np.cos(m * theta)[:, None]
        mod_b += cos_mtheta * bcoef[None, :]
        sqrtg += cos_mtheta * gcoef[None, :]

    _, _, epsilon, _, fsa_1overb, f_t = compute_trapped_fraction(mod_b, sqrtg)
    g = interp1d(s_b, np.asarray(bx.Boozer_G_all, dtype=float), fill_value="extrapolate")(S_VALUES)
    i = interp1d(s_b, np.asarray(bx.Boozer_I_all, dtype=float), fill_value="extrapolate")(S_VALUES)
    iota = interp1d(s_b, np.asarray(bx.iota, dtype=float), fill_value="extrapolate")(S_VALUES)
    with Dataset(case.wout_path, "r") as handle:
        psi_edge = -float(np.asarray(handle.variables["phi"][:], dtype=float)[-1]) / (2.0 * np.pi)
    r_eff = (g + iota * i) * fsa_1overb
    current, _ = j_dot_B_Redl(
        ne,
        te,
        ti,
        zeff,
        case.helicity_n,
        s=S_VALUES,
        G=g,
        R=r_eff,
        iota=iota,
        epsilon=epsilon,
        f_t=f_t,
        psi_edge=psi_edge,
        nfp=nfp,
    )
    return np.asarray(current, dtype=float)


def _max_relative_error(values: np.ndarray, reference: np.ndarray, *, interior: bool) -> float:
    if interior:
        values = values[1:-1]
        reference = reference[1:-1]
    return float(np.max(np.abs(values - reference) / np.maximum(np.abs(reference), 1.0e-12)))


def _case_summary(case: PreciseQSCase) -> dict[str, Any]:
    surfaces, sfincs_current = _load_archived_sfincs_current(case.sfincs_scan_path)
    redl_vmec = _compute_redl_vmec(case)
    redl_boozer = _compute_redl_boozer(case)
    case_payload = asdict(case)
    case_payload["wout_path"] = str(case.wout_path)
    case_payload["sfincs_scan_path"] = str(case.sfincs_scan_path)
    return {
        "case": case_payload,
        "surfaces": surfaces.tolist(),
        "sfincs_jdotb": sfincs_current.tolist(),
        "redl_vmec_jdotb": redl_vmec.tolist(),
        "redl_boozer_jdotb": redl_boozer.tolist(),
        "metrics": {
            "redl_vmec_max_rel_error_full": _max_relative_error(
                redl_vmec, sfincs_current, interior=False
            ),
            "redl_vmec_max_rel_error_interior": _max_relative_error(
                redl_vmec, sfincs_current, interior=True
            ),
            "redl_boozer_max_rel_error_full": _max_relative_error(
                redl_boozer, sfincs_current, interior=False
            ),
            "redl_boozer_max_rel_error_interior": _max_relative_error(
                redl_boozer, sfincs_current, interior=True
            ),
            "boozer_vs_vmec_max_rel_error": _max_relative_error(
                redl_boozer, redl_vmec, interior=False
            ),
        },
    }


def _plot(summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True, constrained_layout=True)
    for col, key in enumerate(("qa", "qh")):
        payload = summary[key]
        s = np.asarray(payload["surfaces"], dtype=float)
        sfincs = np.asarray(payload["sfincs_jdotb"], dtype=float) / 1.0e6
        vmec = np.asarray(payload["redl_vmec_jdotb"], dtype=float) / 1.0e6
        booz = np.asarray(payload["redl_boozer_jdotb"], dtype=float) / 1.0e6

        ax = axes[col]
        ax.plot(s, sfincs, color="black", lw=2.0, label="SFINCS")
        ax.plot(s, vmec, color="#1f77b4", lw=1.8, label="Redl (VMEC)")
        ax.plot(s, booz, color="#d95f02", lw=1.8, ls="--", label="Redl (Boozer)")
        ax.set_title(payload["case"]["label"])
        if col == 0:
            ax.set_ylabel(r"$\langle J \cdot B \rangle$ [MA T m$^{-2}$]")
        ax.set_xlabel("Normalized toroidal flux $s$")
        ax.grid(alpha=0.25)
        if col == 0:
            ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Precise-QS Redl benchmark against archived SFINCS profiles", fontsize=14)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cases = _precise_qs_cases()
    summary = {
        "cases": {key: _case_summary(case) for key, case in cases.items()},
        "booz_resolution": BOOZER_RESOLUTION,
        "sfincs_si_factor": SFINCS_SI_FACTOR,
        "comparison_window": "interior surfaces s[1:-1] match the archived Simsopt regression gate",
    }
    _plot(summary["cases"])
    OUTPUT_PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    qa = summary["cases"]["qa"]["metrics"]
    qh = summary["cases"]["qh"]["metrics"]
    print(
        "precise-QS Redl audit:"
        f" QA vmec={qa['redl_vmec_max_rel_error_interior']:.3e}"
        f" QA booz={qa['redl_boozer_max_rel_error_interior']:.3e}"
        f" QH vmec={qh['redl_vmec_max_rel_error_interior']:.3e}"
        f" QH booz={qh['redl_boozer_max_rel_error_interior']:.3e}"
    )


if __name__ == "__main__":
    main()
