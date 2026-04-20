#!/usr/bin/env python3
"""Rebuild the shipped W7-X NEOPAX database with NTX and compare closures."""
# ruff: noqa: E402, I001

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import interpax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    neopax_scan_requires_rebuild,
    surface_from_vmec_jax_vmec_wout_file,
    to_neopax_monoenergetic,
    write_neopax_scan_hdf5,
)
from ntx._checkout_paths import find_neopax_root


NEOPAX_ROOT = find_neopax_root()
if NEOPAX_ROOT is None:
    raise SystemExit("This audit requires a local NEOPAX checkout with the W7-X reference files.")
if str(NEOPAX_ROOT) not in sys.path:
    sys.path.insert(0, str(NEOPAX_ROOT))

import NEOPAX  # noqa: E402
from NEOPAX._constants import elementary_charge  # noqa: E402


WOUT_PATH = NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
BOOZMN_PATH = NEOPAX_ROOT / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
REFERENCE_PATH = NEOPAX_ROOT / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
NTSS_INITIAL_PATH = NEOPAX_ROOT / "tests" / "inputs" / "NTSS_W7X_Initial.h5"
OUTPUT_DIR = ROOT / "examples" / "outputs" / "bootstrap_current_w7x_rebuild_audit"
OUTPUT_PREFIX = OUTPUT_DIR / "bootstrap_current_w7x_rebuild_audit"
REBUILT_SCAN_PATH = OUTPUT_DIR / "ntx_w7x_scan.h5"
NTX_GRID = GridSpec(n_theta=25, n_zeta=25, n_xi=63)

J_FINAL_REFERENCE = np.array(
    [
        0.0,
        5490.01149652,
        15723.81318418,
        34574.91308781,
        64887.17372981,
        108450.88898812,
        165983.13877999,
        237387.46345921,
        321587.90346994,
        416175.72179143,
        518472.77319291,
        624510.3142208,
        729934.96988671,
        815973.11170985,
        877139.83742598,
        926288.65759796,
        963878.52631876,
        991093.77976222,
        1009832.80846102,
        1022279.55758274,
        1029733.34148606,
        1029899.2568709,
        1021832.29332657,
        1008456.81159438,
        994577.01664654,
        986048.62449312,
        1003396.57863455,
        1038623.31006343,
        1090188.02497005,
        1153088.43944538,
        1220619.5355514,
        1287086.31167344,
        1345705.97875629,
        1398339.01918713,
        1444801.58088483,
        1482720.44647127,
        1511650.04093041,
        1530628.21933747,
        1538721.02794647,
        1538253.40668978,
        1533457.1424207,
        1532368.16711923,
        1546331.62980544,
        1584620.60009108,
        1622778.96964537,
        1579475.24450098,
        1405819.70164514,
        1119183.09254559,
        760928.70423782,
        358191.32235386,
        -13257.68692882,
    ],
    dtype=float,
)


def _surface_loader(rho_value: float):
    return surface_from_vmec_jax_vmec_wout_file(WOUT_PATH, s=float(rho_value**2))


def _max_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs((a - b) / np.maximum(1.0e-12, np.abs(b)))))


def _build_species_and_field():
    with h5py.File(NTSS_INITIAL_PATH, "r") as handle:
        er_initial = interpax.Interpolator1D(handle["r"][()], handle["Er"][()], extrap=True)

    field = NEOPAX.Field.read_vmec_booz(51, str(WOUT_PATH), str(BOOZMN_PATH))
    grid = NEOPAX.Grid.create_standard(51, 64, 3)
    r = field.r_grid

    te0, teb = 17.8e3, 0.7e3
    ne0, neb = 4.21e20, 0.6e20
    ti0, tib = te0, teb
    ni0, nib = ne0, neb
    deuterium_ratio = 0.5
    tritium_ratio = 0.5

    te = (te0 - teb) * (1.0 - (r / field.a_b) ** 2) + teb
    ne = (ne0 - neb) * (1.0 - (r / field.a_b) ** 10) + neb
    ti = (ti0 - tib) * (1.0 - (r / field.a_b) ** 2) + tib
    nd = deuterium_ratio * ((ni0 - nib) * (1.0 - (r / field.a_b) ** 10) + nib)
    nt = tritium_ratio * ((ni0 - nib) * (1.0 - (r / field.a_b) ** 10) + nib)
    er = jnp.asarray(er_initial(np.asarray(r)), dtype=jnp.float64)

    species = NEOPAX.Species(
        3,
        51,
        grid.species_indeces,
        jnp.array([1.0 / 1836.15267343, 2.0, 3.0]),
        jnp.array([-1.0, 1.0, 1.0]),
        jnp.stack([te, ti, ti]),
        jnp.stack([ne, nd, nt]),
        er,
        field.r_grid,
        field.r_grid_half,
        field.dr,
        field.Vprime_half,
        field.overVprime,
        jnp.array([neb, deuterium_ratio * neb, tritium_ratio * neb]),
        jnp.array([teb, teb, teb]),
    )
    return grid, field, species


def _bootstrap_current_profile(database, grid, field, species) -> np.ndarray:
    _gamma, _heat, upar_total, _qpar, _upar2 = (
        NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction(
            species,
            grid,
            field,
            database,
        )
    )
    return (
        -np.asarray(upar_total[:, 0])
        + np.asarray(upar_total[:, 1])
        + np.asarray(upar_total[:, 2])
    ) * elementary_charge


def _rebuild_ntx_scan(reference) -> Path:
    if REBUILT_SCAN_PATH.exists() and not neopax_scan_requires_rebuild(REBUILT_SCAN_PATH):
        return REBUILT_SCAN_PATH
    scan = build_ntx_neopax_scan(
        _surface_loader,
        rho=jnp.asarray(reference.rho),
        nu_v=jnp.asarray(reference.nu_v),
        Es=jnp.asarray(reference.Es),
        Er=jnp.asarray(reference.Er),
        drds=jnp.asarray(reference.drds),
        grid=NTX_GRID,
        source_name="w7x-ntx-rebuilt",
    )
    write_neopax_scan_hdf5(scan, REBUILT_SCAN_PATH)
    return REBUILT_SCAN_PATH


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_scan = load_neopax_reference_scan(REFERENCE_PATH)
    rebuilt_path = _rebuild_ntx_scan(reference_scan)
    rebuilt_scan = load_neopax_reference_scan(rebuilt_path)
    grid, field, species = _build_species_and_field()

    legacy_database = NEOPAX.Monoenergetic.read_monkes(float(field.a_b), str(REFERENCE_PATH))
    rebuilt_spitzer = to_neopax_monoenergetic(
        rebuilt_scan,
        a_b=float(field.a_b),
        d33_mode="spitzer",
    )
    rebuilt_conductivity_difference = to_neopax_monoenergetic(
        rebuilt_scan,
        a_b=float(field.a_b),
        d33_mode="conductivity_difference",
    )

    rho = np.asarray(field.rho_grid, dtype=float)

    profiles = {}
    for label, database in (
        ("legacy_external", legacy_database),
        ("ntx_rebuilt_spitzer", rebuilt_spitzer),
        ("ntx_rebuilt_conductivity_difference", rebuilt_conductivity_difference),
    ):
        start = time.perf_counter()
        current = _bootstrap_current_profile(database, grid, field, species)
        profiles[label] = {
            "j_bootstrap": np.asarray(current, dtype=float),
            "max_relative_error_vs_reference": _max_relative_error(current, J_FINAL_REFERENCE),
            "seconds": float(time.perf_counter() - start),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    axes[0].plot(rho, J_FINAL_REFERENCE, color="black", lw=2.5, label="Reference")
    colors = {
        "legacy_external": "#1f77b4",
        "ntx_rebuilt_spitzer": "#ff7f0e",
        "ntx_rebuilt_conductivity_difference": "#2ca02c",
    }
    for key, payload in profiles.items():
        axes[0].plot(
            rho,
            payload["j_bootstrap"],
            lw=2.0,
            color=colors[key],
            label=f"{key}  (err={payload['max_relative_error_vs_reference']:.2e})",
        )
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel(r"$J_{\mathrm{bs}}$ [A m$^{-2}$]")
    axes[0].set_title("W7-X momentum-corrected bootstrap current")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.3, loc="best")

    for key, payload in profiles.items():
        rel = np.abs(
            (payload["j_bootstrap"] - J_FINAL_REFERENCE)
            / np.maximum(np.abs(J_FINAL_REFERENCE), 1.0)
        )
        axes[1].semilogy(rho, rel, lw=2.0, color=colors[key], label=key)
    axes[1].set_xlabel(r"$\rho$")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("W7-X momentum-corrected error")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=8.3, loc="best")

    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "wout": str(WOUT_PATH),
        "boozmn": str(BOOZMN_PATH),
        "legacy_database": str(REFERENCE_PATH),
        "rebuilt_scan": str(rebuilt_path),
        "ntx_grid": {
            "n_theta": NTX_GRID.n_theta,
            "n_zeta": NTX_GRID.n_zeta,
            "n_xi": NTX_GRID.n_xi,
        },
        "profiles": {
            key: {
                **payload,
                "j_bootstrap": payload["j_bootstrap"].tolist(),
            }
            for key, payload in profiles.items()
        },
        "rho": rho.tolist(),
        "reference_j_bootstrap": J_FINAL_REFERENCE.tolist(),
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png")),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf")),
    }
    OUTPUT_PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {k: v["max_relative_error_vs_reference"] for k, v in profiles.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
