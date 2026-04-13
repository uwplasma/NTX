#!/usr/bin/env python3
"""Audit W7-X bootstrap-current convergence on the bundled benchmark scan."""
# ruff: noqa: I001

from __future__ import annotations

import json
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
    surface_from_vmec_jax_vmec_wout_file,
    to_neopax_monoenergetic,
)
from ntx._checkout_paths import find_neopax_root  # noqa: E402


NEOPAX_ROOT = find_neopax_root()
if NEOPAX_ROOT is None:
    raise SystemExit("This audit requires a local NEOPAX checkout with the W7-X reference files.")

WOUT_PATH = NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
BOOZMN_PATH = NEOPAX_ROOT / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
REFERENCE_PATH = NEOPAX_ROOT / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "bootstrap_current_reference_audit_w7x"
GRID_LEVELS = (
    GridSpec(n_theta=13, n_zeta=17, n_xi=16),
    GridSpec(n_theta=17, n_zeta=25, n_xi=32),
    GridSpec(n_theta=25, n_zeta=25, n_xi=63),
)
NU_INDICES = np.array([0, 3, 7, 11], dtype=int)
ER_INDICES = np.array([0, 3, 7, 11], dtype=int)


def _surface_loader(rho_value: float):
    return surface_from_vmec_jax_vmec_wout_file(WOUT_PATH, s=float(rho_value**2))


def _max_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs((a - b) / np.maximum(1.0e-12, np.abs(b)))))


def _build_species_and_field():
    import NEOPAX

    field = NEOPAX.Field.read_vmec_booz(51, str(WOUT_PATH), str(BOOZMN_PATH))
    grid = NEOPAX.Grid.create_standard(51, 48, 3)
    r = np.asarray(field.r_grid)

    te0, teb = 17.8e3, 0.7e3
    ne0, neb = 4.21e20, 0.6e20
    te = (te0 - teb) * (1.0 - (r / field.a_b) ** 2) + teb
    ne = (ne0 - neb) * (1.0 - (r / field.a_b) ** 10) + neb
    ti = te
    nd = 0.5 * ((ne0 - neb) * (1.0 - (r / field.a_b) ** 10) + neb)
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
    return NEOPAX, grid, field, species


def _bootstrap_current_profile(database, neopax_module, grid, field, species) -> np.ndarray:
    _lij, _gamma, _q, upar = neopax_module.get_Neoclassical_Fluxes(species, grid, field, database)
    charges = np.asarray(species.charge)[:, None]
    return np.sum(charges * np.asarray(upar), axis=0)


def main() -> None:
    reference = load_neopax_reference_scan(REFERENCE_PATH)
    rho = np.asarray(reference.rho)
    nu_v = np.asarray(reference.nu_v)[NU_INDICES]
    er = np.asarray(reference.Er)[:, ER_INDICES]
    es = np.asarray(reference.Es)[:, ER_INDICES]
    drds = np.asarray(reference.drds)

    reference_subset = type(reference)(
        rho=reference.rho,
        nu_v=reference.nu_v[NU_INDICES],
        Er=reference.Er[:, ER_INDICES],
        Es=reference.Es[:, ER_INDICES],
        drds=reference.drds,
        D11=reference.D11[:, NU_INDICES][:, :, ER_INDICES],
        D13=reference.D13[:, NU_INDICES][:, :, ER_INDICES],
        D33=reference.D33[:, NU_INDICES][:, :, ER_INDICES],
        D31=reference.D31[:, NU_INDICES][:, :, ER_INDICES] if reference.D31 is not None else None,
        Er_tilde=reference.Er_tilde[ER_INDICES] if reference.Er_tilde is not None else None,
        Er_to_Ertilde=reference.Er_to_Ertilde[:, ER_INDICES]
        if reference.Er_to_Ertilde is not None
        else None,
        dr_tildedr=reference.dr_tildedr,
        dr_tildeds=reference.dr_tildeds,
        a_b=reference.a_b,
        psia=reference.psia,
        b00=reference.b00,
        r00=reference.r00,
        boozer_i=reference.boozer_i,
        boozer_g=reference.boozer_g,
        iota=reference.iota,
        fac_reference_to_sfincs_11=reference.fac_reference_to_sfincs_11,
        fac_reference_to_sfincs_31=reference.fac_reference_to_sfincs_31,
        fac_reference_to_sfincs_33=reference.fac_reference_to_sfincs_33,
        fac_sfincs_to_dkes_11=reference.fac_sfincs_to_dkes_11,
        fac_sfincs_to_dkes_31=reference.fac_sfincs_to_dkes_31,
        fac_sfincs_to_dkes_33=reference.fac_sfincs_to_dkes_33,
        fac_dkes_to_d11star=reference.fac_dkes_to_d11star,
        fac_dkes_to_d31star=reference.fac_dkes_to_d31star,
        fac_dkes_to_d33star=reference.fac_dkes_to_d33star,
        source_name=reference.source_name,
    )

    neopax_module, transport_grid, transport_field, transport_species = _build_species_and_field()
    reference_database = to_neopax_monoenergetic(reference_subset, a_b=1.0)
    reference_current = _bootstrap_current_profile(
        reference_database,
        neopax_module,
        transport_grid,
        transport_field,
        transport_species,
    )

    convergence_data: list[tuple[GridSpec, np.ndarray, float]] = []
    for grid in GRID_LEVELS:
        scan = build_ntx_neopax_scan(
            _surface_loader,
            rho=rho,
            nu_v=nu_v,
            Es=es,
            Er=er,
            drds=drds,
            grid=grid,
            source_name="w7x-bootstrap-current-audit",
        )
        database = to_neopax_monoenergetic(scan, a_b=1.0)
        current = _bootstrap_current_profile(
            database,
            neopax_module,
            transport_grid,
            transport_field,
            transport_species,
        )
        convergence_data.append((grid, current, _max_relative_error(current, reference_current)))

    figure_scale = np.max(np.abs(reference_current))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)

    axes[0].plot(
        transport_field.rho_grid,
        reference_current / figure_scale,
        color="black",
        lw=2.7,
        label="Reference",
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for color, (grid, current, error) in zip(colors, convergence_data, strict=True):
        label = f"{grid.n_theta}x{grid.n_zeta}x{grid.n_xi + 1}  (err={error:.2e})"
        axes[0].plot(
            transport_field.rho_grid,
            current / figure_scale,
            color=color,
            lw=2.0,
            label=label,
        )
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel("normalized bootstrap current")
    axes[0].set_title("W7-X bootstrap-current profile")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.3, loc="best")

    work = [grid.n_theta * grid.n_zeta * (grid.n_xi + 1) for grid, _, _ in convergence_data]
    errors = [error for _, _, error in convergence_data]
    axes[1].loglog(work, errors, "o-", color="#d62728", lw=2.4)
    axes[1].axhline(1.0e-2, color="0.3", ls="--", lw=1.2)
    axes[1].set_xlabel(r"$N_\theta N_\zeta (N_\xi+1)$")
    axes[1].set_ylabel("max relative error")
    axes[1].set_title("Bootstrap-current convergence")
    axes[1].grid(alpha=0.25, which="both")
    for x_value, y_value, (grid, _, _) in zip(work, errors, convergence_data, strict=True):
        axes[1].annotate(
            f"{grid.n_theta}x{grid.n_zeta}x{grid.n_xi + 1}",
            (x_value, y_value),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )

    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "wout": WOUT_PATH.name,
        "boozmn": BOOZMN_PATH.name,
        "reference_database": REFERENCE_PATH.name,
        "nu_indices": NU_INDICES.tolist(),
        "er_indices": ER_INDICES.tolist(),
        "bootstrap_current_reference_scale": float(figure_scale),
        "bootstrap_current_errors": [
            {
                "grid": [grid.n_theta, grid.n_zeta, grid.n_xi + 1],
                "max_relative_error": float(error),
            }
            for grid, _, error in convergence_data
        ],
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
    }
    OUTPUT_PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["bootstrap_current_errors"], indent=2))


if __name__ == "__main__":
    main()
