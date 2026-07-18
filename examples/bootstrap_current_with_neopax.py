#!/usr/bin/env python3
"""Compute a radial bootstrap-current profile with NTX + NEOPAX.

Edit the configuration block below, run the script, and inspect the figure and
JSON summary written next to ``OUTPUT_PREFIX``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    to_neopax_monoenergetic,
)
from ntx._checkout_paths import find_neopax_root  # noqa: E402
from ntx.vmex_vmec import surface_from_vmex_vmec_wout_file  # noqa: E402

NEOPAX_ROOT = find_neopax_root()
if NEOPAX_ROOT is not None and str(NEOPAX_ROOT) not in sys.path:
    sys.path.insert(0, str(NEOPAX_ROOT))

try:
    import NEOPAX  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised in CI import path
    NEOPAX = None

# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
WOUT_PATH = (
    NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
    if NEOPAX_ROOT is not None
    else Path("/missing/wout_W7-X_standard_configuration.nc")
)
BOOZMN_PATH = (
    NEOPAX_ROOT / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
    if NEOPAX_ROOT is not None
    else Path("/missing/boozmn_wout_W7-X_standard_configuration.nc")
)
REFERENCE_PATH = (
    NEOPAX_ROOT / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
    if NEOPAX_ROOT is not None
    else Path("/missing/Dij_NEOPAX_FULL_S_NEW_W7X.h5")
)
GRID = GridSpec(n_theta=17, n_zeta=25, n_xi=32)
NU_INDICES = np.array([0, 3, 7, 11], dtype=int)
ER_INDICES = np.array([0, 3, 7, 11], dtype=int)
USE_MOMENTUM_CORRECTION = False
D33_MODE = os.environ.get("NTX_BOOTSTRAP_EXAMPLE_D33_MODE", "raw")
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "bootstrap_current_with_neopax"


def _require_neopax_runtime() -> None:
    if NEOPAX_ROOT is None or NEOPAX is None:
        raise SystemExit(
            "This example requires a local NEOPAX checkout with the W7-X reference inputs."
        )


def _surface_loader(rho_value: float):
    return surface_from_vmex_vmec_wout_file(WOUT_PATH, s=float(rho_value**2))


def _build_species_and_field():
    _require_neopax_runtime()
    field = NEOPAX.Field.read_vmec_booz(51, str(WOUT_PATH), str(BOOZMN_PATH))
    grid = NEOPAX.Grid.create_standard(51, 48, 3)
    r = np.asarray(field.r_grid, dtype=float)

    te0, teb = 17.8e3, 0.7e3
    ne0, neb = 4.21e20, 0.6e20
    te = (te0 - teb) * (1.0 - (r / field.a_b) ** 2) + teb
    ne = (ne0 - neb) * (1.0 - (r / field.a_b) ** 10) + neb
    ti = te
    nd = 0.5 * ne
    nt = nd

    species = NEOPAX.Species(
        3,
        51,
        grid.species_indeces,
        np.array([1.0 / 1836.15267343, 2.0, 3.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.stack([te, ti, ti]),
        np.stack([ne, nd, nt]),
        np.zeros_like(r),
        field.r_grid,
        field.r_grid_half,
        field.dr,
        field.Vprime_half,
        field.overVprime,
        np.array([neb, 0.5 * neb, 0.5 * neb]),
        np.array([teb, teb, teb]),
    )
    return field, grid, species, ne, te


def _bootstrap_current_profile(database, grid, field, species):
    lij, gamma, heat, upar_nomom = NEOPAX.get_Neoclassical_Fluxes(species, grid, field, database)
    charges = np.asarray(species.charge, dtype=float)[:, None]
    upar_nomom = np.asarray(upar_nomom, dtype=float)
    current_nomom = np.sum(charges * upar_nomom, axis=0)

    current_total = current_nomom
    correction_current = np.zeros_like(current_nomom)
    if USE_MOMENTUM_CORRECTION:
        _, _, upar_total, _, _ = NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction(
            species,
            grid,
            field,
            database,
        )
        upar_total = np.asarray(upar_total, dtype=float).T
        current_total = np.sum(charges * upar_total, axis=0)
        correction_current = current_total - current_nomom

    return {
        "lij": np.asarray(lij, dtype=float),
        "gamma": np.asarray(gamma, dtype=float),
        "heat": np.asarray(heat, dtype=float),
        "upar_nomom": upar_nomom,
        "current_nomom": current_nomom,
        "current_correction": correction_current,
        "current_total": current_total,
    }


def solve_profiles() -> dict[str, np.ndarray | float | bool]:
    _require_neopax_runtime()
    reference = load_neopax_reference_scan(REFERENCE_PATH)
    rho = np.asarray(reference.rho, dtype=float)
    nu_v = np.asarray(reference.nu_v, dtype=float)[NU_INDICES]
    er = np.asarray(reference.Er, dtype=float)[:, ER_INDICES]
    es = np.asarray(reference.Es, dtype=float)[:, ER_INDICES]
    drds = np.asarray(reference.drds, dtype=float)

    scan = build_ntx_neopax_scan(
        _surface_loader,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=GRID,
        source_name="w7x_bootstrap_current_example",
    )
    field, neopax_grid, species, ne, te = _build_species_and_field()
    database = to_neopax_monoenergetic(scan, a_b=1.0, d33_mode=D33_MODE)
    closure = _bootstrap_current_profile(database, neopax_grid, field, species)

    return {
        "rho": np.asarray(field.rho_grid, dtype=float),
        "ne": np.asarray(ne, dtype=float),
        "te": np.asarray(te, dtype=float),
        "b0": (
            np.asarray(field.B_ref, dtype=float)
            if hasattr(field, "B_ref")
            else np.full_like(field.rho_grid, field.B0)
        ),
        "iota": np.asarray(field.iota, dtype=float),
        "current_nomom": np.asarray(closure["current_nomom"], dtype=float),
        "current_total": np.asarray(closure["current_total"], dtype=float),
        "current_correction": np.asarray(closure["current_correction"], dtype=float),
        "d33_electron": np.asarray(closure["lij"][0, :, 2, 2], dtype=float),
        "use_momentum_correction": USE_MOMENTUM_CORRECTION,
        "nu_v": np.asarray(nu_v, dtype=float),
        "er_axis": np.asarray(er, dtype=float),
        "grid": np.array([GRID.n_theta, GRID.n_zeta, GRID.n_xi], dtype=int),
    }


def plot_profiles(data: dict[str, np.ndarray | float | bool]) -> None:
    rho = np.asarray(data["rho"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), constrained_layout=True)

    axes[0, 0].plot(rho, np.asarray(data["ne"], dtype=float) / 1.0e20, lw=2.3, color="#1f77b4")
    axes[0, 0].set_ylabel(r"$n_e$ [$10^{20}\,\mathrm{m}^{-3}$]")
    ax_te = axes[0, 0].twinx()
    ax_te.plot(rho, np.asarray(data["te"], dtype=float) / 1.0e3, lw=2.0, ls="--", color="#d62728")
    ax_te.set_ylabel(r"$T_e$ [keV]")
    axes[0, 0].set_title("Profile inputs")

    axes[0, 1].plot(rho, np.asarray(data["iota"], dtype=float), lw=2.3, color="#2ca02c")
    axes[0, 1].set_title("Geometry")
    axes[0, 1].set_ylabel(r"$\iota$")

    axes[1, 0].plot(rho, np.asarray(data["d33_electron"], dtype=float), lw=2.3, color="#9467bd")
    axes[1, 0].set_title(r"Electron $L_{33}$")
    axes[1, 0].set_xlabel(r"$\rho$")
    axes[1, 0].set_ylabel("SI normalization")

    axes[1, 1].plot(
        rho,
        np.asarray(data["current_nomom"], dtype=float) / 1.0e6,
        lw=2.3,
        color="#111111",
        label="NTX+NEOPAX no-momentum",
    )
    if bool(data["use_momentum_correction"]):
        axes[1, 1].plot(
            rho,
            np.asarray(data["current_total"], dtype=float) / 1.0e6,
            lw=2.0,
            color="#ff7f0e",
            label="NTX+NEOPAX total",
        )
    axes[1, 1].set_title("Bootstrap current")
    axes[1, 1].set_xlabel(r"$\rho$")
    axes[1, 1].set_ylabel(r"$j\cdot B$ [$10^6\,\mathrm{A\,m}^{-2}$]")
    axes[1, 1].legend(frameon=False, loc="best")

    for ax in axes.flat:
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_xlim(rho[0], rho[-1])

    fig.suptitle("Radial bootstrap-current profile from NTX + NEOPAX", fontsize=14, y=1.02)
    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(data: dict[str, np.ndarray | float | bool]) -> None:
    try:
        figure_png = str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT))
        figure_pdf = str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT))
    except ValueError:
        figure_png = str(OUTPUT_PREFIX.with_suffix(".png"))
        figure_pdf = str(OUTPUT_PREFIX.with_suffix(".pdf"))

    summary = {
        "wout": WOUT_PATH.name,
        "boozmn": BOOZMN_PATH.name,
        "reference_scan": REFERENCE_PATH.name,
        "use_momentum_correction": bool(data["use_momentum_correction"]),
        "d33_mode": D33_MODE,
        "grid": {
            "n_theta": int(data["grid"][0]),
            "n_zeta": int(data["grid"][1]),
            "n_xi": int(data["grid"][2]),
        },
        "rho": np.asarray(data["rho"], dtype=float).tolist(),
        "current_nomom_am2": np.asarray(data["current_nomom"], dtype=float).tolist(),
        "current_total_am2": np.asarray(data["current_total"], dtype=float).tolist(),
        "current_correction_am2": np.asarray(data["current_correction"], dtype=float).tolist(),
        "figure_png": figure_png,
        "figure_pdf": figure_pdf,
    }
    OUTPUT_PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    data = solve_profiles()
    plot_profiles(data)
    write_summary(data)
    print(f"bootstrap-current figure: {OUTPUT_PREFIX.with_suffix('.png')}")
    print(f"bootstrap-current summary: {OUTPUT_PREFIX.with_suffix('.json')}")


if __name__ == "__main__":
    main()
