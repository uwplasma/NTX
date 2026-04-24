from __future__ import annotations

from typing import Any

import numpy as np
from _fixed_field_validation_metrics import (
    least_squares_scale,
    relative_error_array,
    sign_mismatch_count,
)


def build_closure_diagnostics(
    *,
    case_results: dict[str, dict[str, np.ndarray]],
    density: np.ndarray,
    charge_unit: float,
    interior_rho_min: float,
    interior_rho_max: float,
    target_rho: float = 0.5,
) -> dict[str, Any]:
    ref = np.asarray(case_results["SFINCS"]["jdotb"], dtype=float)
    rho = np.asarray(case_results["SFINCS"]["rho"], dtype=float)
    interior = (rho >= interior_rho_min) & (rho <= interior_rho_max)
    ntx = case_results["NTX+NEOPAX"]
    density_arr = np.asarray(density, dtype=float)

    species_scale: dict[str, dict[str, float | dict[str, float]]] = {}
    density_term = np.zeros_like(ref)
    thermal_term_raw = np.zeros_like(ref)
    thermal_term_eff = np.zeros_like(ref)
    mid_index = int(np.argmin(np.abs(rho - target_rho)))
    branch_currents: dict[str, np.ndarray] = {}
    for label, charge_sign in (("electron", -1.0), ("ion", 1.0)):
        ref_species = np.asarray(case_results["SFINCS"][f"{label}_current"], dtype=float)
        model_species = np.asarray(ntx[f"{label}_current"], dtype=float)
        model_species_nomom = np.asarray(ntx[f"{label}_current_nomom"], dtype=float)
        model_species_correction = np.asarray(ntx[f"{label}_current_correction"], dtype=float)
        branch_currents[f"{label}_nomom"] = model_species_nomom
        branch_currents[f"{label}_correction"] = model_species_correction
        species_relative_error = relative_error_array(ref_species, model_species)
        species_scale[label] = {
            "current_scale": least_squares_scale(ref_species, model_species, interior),
            "current_nomom_scale": least_squares_scale(
                ref_species,
                model_species_nomom,
                interior,
            ),
            "current_sign_mismatch_count_interior": sign_mismatch_count(
                ref_species,
                model_species,
                interior,
            ),
            "current_worst_relative_error_interior": float(
                np.max(species_relative_error[interior])
            ),
            "current_worst_rho_interior": float(
                rho[interior][int(np.argmax(species_relative_error[interior]))]
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
        density_term += charge_sign * charge_unit * (-density_arr * (l31 * a1))
        thermal_term_raw += charge_sign * charge_unit * (-density_arr * (l32 * a2))
        thermal_term_eff += charge_sign * charge_unit * (
            -density_arr * ((l32 - 1.5 * l31) * a2)
        )

    raw_alpha = least_squares_scale(ref - density_term, thermal_term_raw, interior)
    eff_alpha = least_squares_scale(ref - density_term, thermal_term_eff, interior)
    raw_fit = density_term + raw_alpha * thermal_term_raw
    eff_fit = density_term + eff_alpha * thermal_term_eff
    ref_interior = np.maximum(np.abs(ref[interior]), 1.0)
    rel_total = relative_error_array(ref, np.asarray(ntx["jdotb"], dtype=float))
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
        "current_scale": least_squares_scale(ref, np.asarray(ntx["jdotb"], dtype=float), interior),
        "current_nomom_scale": least_squares_scale(
            ref,
            np.asarray(ntx["jdotb_nomom"], dtype=float),
            interior,
        ),
        "hybrid_current_max_relative_error_interior": {
            name: float(np.max(relative_error_array(ref, current)[interior]))
            for name, current in hybrid_currents.items()
        },
        "current_sign_mismatch_count_interior": sign_mismatch_count(
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


__all__ = ["build_closure_diagnostics"]
