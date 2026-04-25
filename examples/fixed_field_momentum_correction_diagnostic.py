#!/usr/bin/env python3
"""Dump fixed-field QA/QH momentum-correction internals on selected radii."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from fixed_field_parallel_flow_audit import (  # type: ignore[import-not-found]
    ER_AXIS_FACTORS,
    NTX_NEOPAX_RADIAL_POINTS,
    NTX_SURFACE_GRID,
    _adaptive_nu_values,
    _archived_profiles,
    _bootstrap_current_pythonpath,
    _cases,
    _ensure_boozmn,
    _interp_profile,
    _make_species,
)

from ntx import (
    build_ntx_neopax_scan_from_surfaces,
    load_neopax_reference_scan,
    load_vmec_surface,
    neopax_scan_requires_rebuild,
    to_neopax_monoenergetic,
    write_neopax_scan_hdf5,
)

_bootstrap_current_pythonpath()

import h5py
import jax
import jax.numpy as jnp
import NEOPAX
from NEOPAX._moments import (
    build_collision_projection,
    build_collision_source_columns,
    build_source_projection,
    build_transport_projection,
    build_transport_source_columns,
    sonine_expansion_factors,
)
from NEOPAX._neoclassical import (
    get_Collision_Operator_terms,
    get_correction_matrix,
    get_Lij_matrix_with_momentum_correction,
    get_Matrix,
    get_momentum_Correction,
    get_Neoclassical_Fluxes,
    get_Neoclassical_Fluxes_With_Momentum_Correction,
    get_rhs,
    get_sum,
)
from scipy.constants import elementary_charge

DEFAULT_OUTPUT_DIR = ROOT / "examples" / "outputs" / "fixed_field_momentum_correction_diagnostic"
SONINE_WEIGHTS = np.array([1.0, 0.4, 8.0 / 35.0], dtype=float)
SFINCS_JHAT_TO_AM2 = 437695.0 * 1.0e20 * elementary_charge
D33_MODE = os.environ.get(
    "NTX_FIXED_FIELD_DIAGNOSTIC_D33_MODE",
    "spitzer",
).strip().lower()
D33_LIJ_MODE = os.environ.get(
    "NTX_FIXED_FIELD_DIAGNOSTIC_D33_LIJ_MODE",
    D33_MODE,
).strip().lower()
D33_EIJ_MODE = os.environ.get(
    "NTX_FIXED_FIELD_DIAGNOSTIC_D33_EIJ_MODE",
    D33_MODE,
).strip().lower()
NEOPAX_N_ORDER = int(os.environ.get("NTX_FIXED_FIELD_DIAGNOSTIC_NEOPAX_N_ORDER", "2"))


def _momentum_blocks(
    species: Any,
    neopax_grid: Any,
    field: Any,
    database: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lij_full, eij_full, nu_weighted_average = jax.vmap(
        jax.vmap(
            get_Lij_matrix_with_momentum_correction,
            in_axes=(None, None, None, None, None, 0),
        ),
        in_axes=(None, None, None, None, 0, None),
    )(
        species,
        neopax_grid,
        field,
        database,
        species.species_indeces,
        neopax_grid.full_grid_indeces,
    )
    lij_full = lij_full.at[:, 0, :, :].set(lij_full.at[:, 1, :, :].get())
    eij_full = eij_full.at[:, 0, :, :].set(eij_full.at[:, 1, :, :].get())
    return (
        np.asarray(lij_full, dtype=float),
        np.asarray(eij_full, dtype=float),
        np.asarray(nu_weighted_average, dtype=float),
    )


def _assemble_dense_species_matrix(blocks: np.ndarray) -> np.ndarray:
    array = np.asarray(blocks, dtype=float)
    return np.transpose(array, (0, 2, 1, 3)).reshape(
        array.shape[0] * array.shape[2],
        array.shape[1] * array.shape[3],
    )


def _relative_residual_norm(matrix: np.ndarray, rhs: np.ndarray, solution: np.ndarray) -> float:
    residual = matrix @ solution - rhs
    return float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-30))


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _candidate_upar_from_solution(solution: np.ndarray, mode: str) -> float:
    coeff = np.asarray(solution, dtype=float)
    if mode == "c0":
        return float(coeff[0])
    if mode == "weighted":
        weights = np.asarray(sonine_expansion_factors(coeff.size), dtype=float)
        return float(np.dot(weights, coeff))
    if mode == "c2":
        if coeff.size < 3:
            raise ValueError("c2 reconstruction requires at least three Sonine moments")
        return float(coeff[2])
    raise ValueError(f"unknown solution reconstruction mode: {mode}")


def _relative_error_scalar(value: float, reference: float) -> float:
    return float(abs(float(value) - float(reference)) / max(abs(float(reference)), 1.0e-30))


def _sign_label(value: float, *, tolerance: float = 1.0e-30) -> int:
    if not np.isfinite(value) or abs(float(value)) <= tolerance:
        return 0
    return 1 if float(value) > 0.0 else -1


def _sign_matches(value: float, reference: float) -> bool:
    reference_sign = _sign_label(reference)
    if reference_sign == 0:
        return True
    return _sign_label(value) == reference_sign


def _dominant_channel(values: dict[str, float]) -> str:
    if not values:
        return "none"
    return max(values, key=lambda key: abs(float(values[key])))


def _source_projection(lij_block: np.ndarray) -> np.ndarray:
    block = np.asarray(lij_block, dtype=float)
    n_order = block.shape[0] - 2
    return np.asarray(build_source_projection(block[2 : 2 + n_order, 0:3], n_order))


def _force_current_contributions(
    *,
    lij_block: np.ndarray,
    forces: np.ndarray,
    density: float,
    charge_sign: float,
) -> dict[str, Any]:
    row3 = np.asarray(lij_block, dtype=float)[2, 0:3]
    drive = np.asarray(forces, dtype=float)
    upar_terms = -float(density) * row3 * drive
    current_terms = float(charge_sign) * elementary_charge * upar_terms
    labels = ("A1_density_electric", "A2_temperature", "A3_parallel_electric")
    current_by_force = {
        label: float(value) for label, value in zip(labels, current_terms, strict=False)
    }
    upar_by_force = {
        label: float(value) for label, value in zip(labels, upar_terms, strict=False)
    }
    density_electric_force = float(drive[0] + 1.5 * drive[1])
    thermal_effective_coefficient = float(row3[1] - 1.5 * row3[0])
    effective_upar_terms = -float(density) * np.asarray(
        [
            row3[0] * density_electric_force,
            thermal_effective_coefficient * drive[1],
            row3[2] * drive[2],
        ],
        dtype=float,
    )
    effective_current_terms = float(charge_sign) * elementary_charge * effective_upar_terms
    effective_labels = (
        "density_electric_force",
        "effective_temperature_force",
        "parallel_electric_force",
    )
    effective_current_by_force = {
        label: float(value)
        for label, value in zip(
            effective_labels,
            effective_current_terms,
            strict=False,
        )
    }
    return {
        "row3": row3.tolist(),
        "forces": drive.tolist(),
        "upar_by_force": upar_by_force,
        "current_by_force": current_by_force,
        "current_sum": float(np.sum(current_terms)),
        "dominant_current_force": _dominant_channel(current_by_force),
        "density_electric_force": density_electric_force,
        "thermal_effective_coefficient": thermal_effective_coefficient,
        "effective_current_by_force": effective_current_by_force,
        "effective_current_sum": float(np.sum(effective_current_terms)),
        "dominant_effective_current_force": _dominant_channel(effective_current_by_force),
    }


def _rhs_force_contributions(lij_block: np.ndarray, forces: np.ndarray) -> dict[str, Any]:
    projection = _source_projection(lij_block)
    drive = np.asarray(forces, dtype=float)
    contributions = -projection * drive[None, :]
    labels = ("A1_density_electric", "A2_temperature", "A3_parallel_electric")
    force_sums = {
        label: float(value)
        for label, value in zip(
            labels,
            np.sum(contributions, axis=0),
            strict=False,
        )
    }
    effective_projection = np.stack(
        [
            projection[:, 0],
            projection[:, 1] - 1.5 * projection[:, 0],
            projection[:, 2],
        ],
        axis=1,
    )
    effective_drive = np.asarray([drive[0] + 1.5 * drive[1], drive[1], drive[2]], dtype=float)
    effective_contributions = -effective_projection * effective_drive[None, :]
    effective_labels = (
        "density_electric_force",
        "effective_temperature_force",
        "parallel_electric_force",
    )
    effective_force_sums = {
        label: float(value)
        for label, value in zip(
            effective_labels,
            np.sum(effective_contributions, axis=0),
            strict=False,
        )
    }
    return {
        "source_projection": projection.tolist(),
        "contributions_by_moment_and_force": contributions.tolist(),
        "rhs_by_moment": np.sum(contributions, axis=1).tolist(),
        "rhs_by_force": force_sums,
        "dominant_rhs_force": _dominant_channel(force_sums),
        "effective_source_projection": effective_projection.tolist(),
        "effective_forces": effective_drive.tolist(),
        "effective_contributions_by_moment_and_force": effective_contributions.tolist(),
        "effective_rhs_by_moment": np.sum(effective_contributions, axis=1).tolist(),
        "effective_rhs_by_force": effective_force_sums,
        "dominant_effective_rhs_force": _dominant_channel(effective_force_sums),
    }


def _species_forensic_summary(entry: dict[str, Any]) -> dict[str, Any]:
    reference = float(entry["reference_current"])
    candidates = {
        "nomom": float(entry["current_nomom"]),
        "corrected_total": float(entry["current_total"]),
        "solution_c0": float(entry["current_solution_c0"]),
        "solution_weighted": float(entry["current_solution_weighted"]),
        "solution_c0_rhs_flipped": -float(entry["current_solution_c0"]),
        "solution_weighted_rhs_flipped": -float(entry["current_solution_weighted"]),
    }
    if "current_nomom_raw_closure" in entry:
        candidates["nomom_raw_closure"] = float(entry["current_nomom_raw_closure"])
    if "current_total_raw_closure" in entry:
        candidates["corrected_total_raw_closure"] = float(entry["current_total_raw_closure"])
    if "current_solution_c0_sfincs_flow_normalized" in entry:
        candidates["solution_c0_sfincs_flow_normalized"] = float(
            entry["current_solution_c0_sfincs_flow_normalized"]
        )
    if "current_solution_weighted_sfincs_flow_normalized" in entry:
        candidates["solution_weighted_sfincs_flow_normalized"] = float(
            entry["current_solution_weighted_sfincs_flow_normalized"]
        )
    relative_errors = {
        key: _relative_error_scalar(value, reference) for key, value in candidates.items()
    }
    best_candidate = min(relative_errors, key=relative_errors.get)
    return {
        "reference_sign": _sign_label(reference),
        "nomom_sign_matches_reference": _sign_matches(candidates["nomom"], reference),
        "corrected_sign_matches_reference": _sign_matches(candidates["corrected_total"], reference),
        "relative_errors": relative_errors,
        "best_current_candidate": best_candidate,
        "best_current_candidate_relative_error": float(relative_errors[best_candidate]),
        "dominant_no_momentum_force": entry["force_current_contributions"][
            "dominant_current_force"
        ],
        "dominant_effective_no_momentum_force": entry["force_current_contributions"][
            "dominant_effective_current_force"
        ],
        "dominant_rhs_force": entry["rhs_force_contributions"]["dominant_rhs_force"],
        "dominant_effective_rhs_force": entry["rhs_force_contributions"][
            "dominant_effective_rhs_force"
        ],
    }


def _dump_forensic_summary(dump: dict[str, Any]) -> dict[str, Any]:
    electron = _species_forensic_summary(dump["electron"])
    ion = _species_forensic_summary(dump["ion"])
    electron_ref = float(dump["electron"]["reference_current"])
    ion_ref = float(dump["ion"]["reference_current"])
    total_reference = electron_ref + ion_ref
    total_nomom = float(dump["electron"]["current_nomom"]) + float(dump["ion"]["current_nomom"])
    total_corrected = float(dump["electron"]["current_total"]) + float(dump["ion"]["current_total"])
    if (
        "current_nomom_raw_closure" in dump["electron"]
        and "current_nomom_raw_closure" in dump["ion"]
    ):
        total_nomom_raw = float(dump["electron"]["current_nomom_raw_closure"]) + float(
            dump["ion"]["current_nomom_raw_closure"]
        )
    else:
        total_nomom_raw = total_nomom
    if (
        "current_total_raw_closure" in dump["electron"]
        and "current_total_raw_closure" in dump["ion"]
    ):
        total_corrected_raw = float(dump["electron"]["current_total_raw_closure"]) + float(
            dump["ion"]["current_total_raw_closure"]
        )
    else:
        total_corrected_raw = total_corrected
    classes: list[str] = []
    if float(dump["relative_residual_norm"]) > 1.0e-8:
        classes.append("linear_solve_residual")
    if float(dump["matrix_condition_number"]) > 1.0e10:
        classes.append("linear_system_conditioning")
    total_corrected_relative_error = _relative_error_scalar(total_corrected, total_reference)
    if total_corrected_relative_error > 0.1:
        if not (
            electron["nomom_sign_matches_reference"] and ion["nomom_sign_matches_reference"]
        ):
            classes.append("row3_force_or_transport_sign")
        elif not (
            electron["corrected_sign_matches_reference"]
            and ion["corrected_sign_matches_reference"]
        ):
            classes.append("momentum_correction_source_collision_or_observable_sign")
        classes.append("total_current_gap_exceeds_1e-1_gate")
    elif not (
        electron["corrected_sign_matches_reference"] and ion["corrected_sign_matches_reference"]
    ):
        classes.append("species_current_decomposition_stress")
    return {
        "first_failure_class": classes[0] if classes else "no_large_mismatch_detected",
        "failure_classes": classes,
        "total_reference_current": float(total_reference),
        "total_no_momentum_current": float(total_nomom),
        "total_corrected_current": float(total_corrected),
        "total_no_momentum_raw_closure_current": float(total_nomom_raw),
        "total_corrected_raw_closure_current": float(total_corrected_raw),
        "total_no_momentum_relative_error": _relative_error_scalar(total_nomom, total_reference),
        "total_corrected_relative_error": total_corrected_relative_error,
        "total_no_momentum_raw_closure_relative_error": _relative_error_scalar(
            total_nomom_raw,
            total_reference,
        ),
        "total_corrected_raw_closure_relative_error": _relative_error_scalar(
            total_corrected_raw,
            total_reference,
        ),
        "electron": electron,
        "ion": ion,
    }


def _matrix_coefficients(
    lij_block: np.ndarray,
    eij_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror NEOPAX get_Matrix() Sonine coefficients for one species/radius."""
    lij = np.asarray(lij_block, dtype=float)
    eij = np.asarray(eij_block, dtype=float)
    n_order = lij.shape[0] - 2
    coeff = np.asarray(build_transport_projection(lij[2 : 2 + n_order, 2 : 2 + n_order], n_order))
    nucoeff = np.asarray(
        build_collision_projection(eij[2 : 2 + n_order, 2 : 2 + n_order], n_order)
    )
    return coeff, nucoeff


def _observable_coefficients(
    lij_block: np.ndarray,
    eij_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror NEOPAX get_corrected_fluxes() Sonine coefficients for one species/radius."""
    lij = np.asarray(lij_block, dtype=float)
    eij = np.asarray(eij_block, dtype=float)
    n_order = lij.shape[0] - 2
    coeff_columns = np.stack(
        [
            -lij[0, 2 : 2 + n_order],
            -lij[1, 2 : 2 + n_order],
        ],
        axis=1,
    )
    nu_columns = np.stack(
        [
            -eij[0, 2 : 2 + n_order],
            -eij[1, 2 : 2 + n_order],
        ],
        axis=1,
    )
    coeff = np.asarray(build_transport_source_columns(coeff_columns, n_order))
    nucoeff = np.asarray(build_collision_source_columns(nu_columns, n_order))
    return coeff, nucoeff


def _full_observable_coefficient_matrix(
    lij_block: np.ndarray,
    eij_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full matrices used by NEOPAX get_corrected_fluxes()."""
    coeff_columns, nucoeff_columns = _observable_coefficients(lij_block, eij_block)
    n_order = coeff_columns.shape[0]
    coeff = np.zeros((n_order, n_order), dtype=float)
    nucoeff = np.zeros((n_order, n_order), dtype=float)
    coeff[:, : coeff_columns.shape[1]] = coeff_columns
    nucoeff[:, : nucoeff_columns.shape[1]] = nucoeff_columns
    return coeff, nucoeff


def _load_archived_sfincs_species_currents(case: Any) -> dict[str, np.ndarray]:
    rho_values: list[float] = []
    electron_current: list[float] = []
    ion_current: list[float] = []
    for path in sorted(case.sfincs_scan_dir.glob("psiN_*/sfincsOutput.h5")):
        try:
            psi_n = float(path.parent.name.split("_", 1)[1])
        except ValueError:
            continue
        with h5py.File(path, "r") as handle:
            flow = np.asarray(handle["FSABFlow"][()], dtype=float).reshape(-1)
            charges = np.asarray(handle["Zs"][()], dtype=float).reshape(-1)
        if flow.size < 2 or charges.size < 2:
            continue
        rho_values.append(float(np.sqrt(psi_n)))
        ion_current.append(float(charges[0] * flow[0]) * SFINCS_JHAT_TO_AM2)
        electron_current.append(float(charges[1] * flow[1]) * SFINCS_JHAT_TO_AM2)
    rho = np.asarray(rho_values, dtype=float)
    order = np.argsort(rho)
    return {
        "rho": rho[order],
        "electron_current": np.asarray(electron_current, dtype=float)[order],
        "ion_current": np.asarray(ion_current, dtype=float)[order],
    }


def _load_archived_b0_over_bbar(case: Any) -> dict[str, np.ndarray]:
    rho_values: list[float] = []
    b0_values: list[float] = []
    for path in sorted(case.sfincs_scan_dir.glob("psiN_*/sfincsOutput.h5")):
        try:
            psi_n = float(path.parent.name.split("_", 1)[1])
        except ValueError:
            continue
        with h5py.File(path, "r") as handle:
            if "B0OverBBar" not in handle:
                continue
            b0_over_bbar = float(np.asarray(handle["B0OverBBar"][()]).reshape(-1)[0])
        rho_values.append(float(np.sqrt(psi_n)))
        b0_values.append(abs(b0_over_bbar))
    rho = np.asarray(rho_values, dtype=float)
    order = np.argsort(rho)
    return {
        "rho": rho[order],
        "b0_over_bbar": np.asarray(b0_values, dtype=float)[order],
    }


def _build_case_context(case_key: str) -> dict[str, Any]:
    case = _cases()[case_key]
    boozmn = _ensure_boozmn(case)
    n_r = max(int(NTX_NEOPAX_RADIAL_POINTS), 9)
    timings: dict[str, float] = {}

    start = time.perf_counter()
    field = NEOPAX.Field.read_vmec_booz(n_r, str(case.wout_path), str(boozmn))
    timings["field_seconds"] = float(time.perf_counter() - start)

    start = time.perf_counter()
    species = _make_species(field, case)
    neopax_grid = NEOPAX.Grid.create_standard(n_r, 64, 2, n_order=NEOPAX_N_ORDER)
    nu_values = _adaptive_nu_values(species, neopax_grid)
    timings["species_seconds"] = float(time.perf_counter() - start)

    profiles = _archived_profiles(case)
    rho_field = np.asarray(field.rho_grid, dtype=float)
    rho_surface = np.clip(rho_field, 0.05, 0.95)
    drds = float(field.a_b) * 0.5 / np.clip(rho_surface, 0.05, None)
    archived_er = _interp_profile(profiles.rho, profiles.electric_field_kv_per_m, rho_surface)
    er_axis = float(np.median(archived_er)) * ER_AXIS_FACTORS
    er_values = np.repeat(er_axis[None, :], rho_surface.size, axis=0)

    scan_path = case.output_dir / "ntx_scan.h5"
    if scan_path.exists() and not neopax_scan_requires_rebuild(scan_path):
        timings["surface_load_seconds"] = 0.0
        start = time.perf_counter()
        scan = load_neopax_reference_scan(scan_path)
        timings["ntx_scan_seconds"] = float(time.perf_counter() - start)
    else:
        start = time.perf_counter()
        surfaces = tuple(
            load_vmec_surface(
                case.wout_path,
                psi_n=float(rho_value**2),
                vmec_radial_option=0,
                vmec_nyquist_option=1,
                vmec_mode_convention="filtered_nyquist",
            )
            for rho_value in rho_surface
        )
        timings["surface_load_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        scan = build_ntx_neopax_scan_from_surfaces(
            surfaces,
            rho=jnp.asarray(rho_surface),
            nu_v=jnp.asarray(np.asarray(nu_values)),
            Er=jnp.asarray(np.asarray(er_values)),
            drds=jnp.asarray(np.asarray(drds)),
            grid=NTX_SURFACE_GRID,
            source_name=f"fixed_field_{case.name}",
        )
        timings["ntx_scan_seconds"] = float(time.perf_counter() - start)
        write_neopax_scan_hdf5(scan, scan_path)

    start = time.perf_counter()
    database = to_neopax_monoenergetic(scan, a_b=float(field.a_b), d33_mode=D33_MODE)
    timings["database_seconds"] = float(time.perf_counter() - start)

    closure_start = time.perf_counter()
    lij_nomom, gamma_nomom, heat_nomom, upar_nomom = get_Neoclassical_Fluxes(
        species,
        neopax_grid,
        field,
        database,
    )
    mixed_d33_blocks = (D33_LIJ_MODE != D33_MODE) or (D33_EIJ_MODE != D33_MODE)
    if mixed_d33_blocks:
        database_lij = to_neopax_monoenergetic(
            scan,
            a_b=float(field.a_b),
            d33_mode=D33_LIJ_MODE,
        )
        database_eij = to_neopax_monoenergetic(
            scan,
            a_b=float(field.a_b),
            d33_mode=D33_EIJ_MODE,
        )
        block_start = time.perf_counter()
        lij_full, _, nu_weighted_average_lij = _momentum_blocks(
            species,
            neopax_grid,
            field,
            database_lij,
        )
        _, eij_full, nu_weighted_average_eij = _momentum_blocks(
            species,
            neopax_grid,
            field,
            database_eij,
        )
        nu_weighted_average = nu_weighted_average_lij
        if not np.allclose(nu_weighted_average_lij, nu_weighted_average_eij):
            raise RuntimeError("nu-weighted averages unexpectedly depend on D33 mode")
        timings["momentum_blocks_seconds"] = float(time.perf_counter() - block_start)
        correction_gamma, correction_heat, upar_total, qpar, upar2 = jax.vmap(
            get_momentum_Correction,
            in_axes=(None, None, None, 0, 1, 1, 1),
        )(
            species,
            neopax_grid,
            field,
            neopax_grid.full_grid_indeces,
            jnp.asarray(lij_full),
            jnp.asarray(eij_full),
            jnp.asarray(nu_weighted_average),
        )
    else:
        correction_gamma, correction_heat, upar_total, qpar, upar2 = (
            get_Neoclassical_Fluxes_With_Momentum_Correction(
                species,
                neopax_grid,
                field,
                database,
            )
        )
        block_start = time.perf_counter()
        lij_full, eij_full, nu_weighted_average = _momentum_blocks(
            species,
            neopax_grid,
            field,
            database,
        )
        timings["momentum_blocks_seconds"] = float(time.perf_counter() - block_start)
    jax.block_until_ready(lij_nomom)
    jax.block_until_ready(upar_total)
    timings["neopax_closure_seconds"] = float(time.perf_counter() - closure_start)

    archived_b0 = _load_archived_b0_over_bbar(case)
    if archived_b0["rho"].size:
        b0_over_bbar = _interp_profile(
            archived_b0["rho"],
            archived_b0["b0_over_bbar"],
            rho_field,
        )
    else:
        b0_over_bbar = np.asarray(np.abs(field.B0), dtype=float)

    return {
        "case": case,
        "field": field,
        "species": species,
        "neopax_grid": neopax_grid,
        "database": database,
        "rho_grid": rho_field,
        "lij_nomom": np.asarray(lij_nomom, dtype=float),
        "gamma_nomom": np.asarray(gamma_nomom, dtype=float),
        "heat_nomom": np.asarray(heat_nomom, dtype=float),
        "upar_nomom": np.asarray(upar_nomom, dtype=float),
        "gamma_correction": np.asarray(correction_gamma, dtype=float),
        "heat_correction": np.asarray(correction_heat, dtype=float),
        "upar_total": np.asarray(upar_total, dtype=float),
        "qpar_correction": np.asarray(qpar, dtype=float),
        "upar2_correction": np.asarray(upar2, dtype=float),
        "lij_full": np.asarray(lij_full, dtype=float),
        "eij_full": np.asarray(eij_full, dtype=float),
        "nu_weighted_average": np.asarray(nu_weighted_average, dtype=float),
        "archived_currents": _load_archived_sfincs_species_currents(case),
        "b0_over_bbar": np.asarray(np.abs(b0_over_bbar), dtype=float),
        "timings": timings,
    }


def _species_current(charge_sign: float, upar: np.ndarray) -> np.ndarray:
    return np.asarray(charge_sign * elementary_charge * upar, dtype=float)


def _diagnose_case(case_key: str, rho_targets: np.ndarray) -> dict[str, Any]:
    context = _build_case_context(case_key)
    case = context["case"]
    rho_grid = np.asarray(context["rho_grid"], dtype=float)
    species = context["species"]
    field = context["field"]
    neopax_grid = context["neopax_grid"]
    lij_full = jnp.asarray(context["lij_full"])
    eij_full = jnp.asarray(context["eij_full"])
    upar_nomom = np.asarray(context["upar_nomom"], dtype=float)
    upar_total = np.asarray(context["upar_total"], dtype=float)
    b0_over_bbar = np.asarray(context["b0_over_bbar"], dtype=float)
    sfincs_flow_bridge = -b0_over_bbar
    moment_order = int(neopax_grid.n_order)

    electron_current_nomom_raw = _species_current(-1.0, upar_nomom[0])
    ion_current_nomom_raw = _species_current(+1.0, upar_nomom[1])
    electron_current_total_raw = _species_current(-1.0, upar_total[:, 0])
    ion_current_total_raw = _species_current(+1.0, upar_total[:, 1])
    electron_current_nomom = sfincs_flow_bridge * electron_current_nomom_raw
    ion_current_nomom = sfincs_flow_bridge * ion_current_nomom_raw
    electron_current_total = sfincs_flow_bridge * electron_current_total_raw
    ion_current_total = sfincs_flow_bridge * ion_current_total_raw
    electron_current_correction = electron_current_total - electron_current_nomom
    ion_current_correction = ion_current_total - ion_current_nomom
    electron_current_correction_raw = electron_current_total_raw - electron_current_nomom_raw
    ion_current_correction_raw = ion_current_total_raw - ion_current_nomom_raw
    archived_currents = context["archived_currents"]
    reference_electron = _interp_profile(
        np.asarray(archived_currents["rho"], dtype=float),
        np.asarray(archived_currents["electron_current"], dtype=float),
        rho_grid,
    )
    reference_ion = _interp_profile(
        np.asarray(archived_currents["rho"], dtype=float),
        np.asarray(archived_currents["ion_current"], dtype=float),
        rho_grid,
    )

    dumps: list[dict[str, Any]] = []
    for rho_target in rho_targets:
        radial_index = int(np.argmin(np.abs(rho_grid - rho_target)))
        cm_ab, cn_ab, tau = jax.vmap(
            jax.vmap(get_Collision_Operator_terms, in_axes=(None, None, None, 0, None)),
            in_axes=(None, None, 0, None, None),
        )(
            species,
            neopax_grid,
            species.species_indeces,
            species.species_indeces,
            radial_index,
        )
        rhs = jax.vmap(get_rhs, in_axes=(None, None, 0, None, 0))(
            species,
            neopax_grid,
            species.species_indeces,
            radial_index,
            lij_full[:, radial_index, :, :],
        )
        blocks = jax.vmap(
            get_Matrix,
            in_axes=(None, None, None, 0, None, 0, 0, None, None, None),
        )(
            species,
            neopax_grid,
            field,
            species.species_indeces,
            radial_index,
            lij_full[:, radial_index, :, :],
            eij_full[:, radial_index, :, :],
            cm_ab,
            cn_ab,
            tau,
        )
        dense_matrix = _assemble_dense_species_matrix(np.asarray(blocks, dtype=float))
        rhs_vector = np.asarray(jnp.reshape(rhs, rhs.shape[0] * rhs.shape[1]), dtype=float)
        solution = np.linalg.solve(dense_matrix, rhs_vector)
        correction = jnp.asarray(solution.reshape(cm_ab.shape[0], cm_ab.shape[-1]), dtype=float)
        singular_values = np.linalg.svd(dense_matrix, compute_uv=False)
        positive_entries = np.abs(dense_matrix[np.abs(dense_matrix) > 0.0])
        min_entry = float(np.min(positive_entries)) if positive_entries.size else 1.0e-300
        c_vectors: list[np.ndarray] = []
        add_term_sums: list[list[float]] = []
        alt_currents_c2_only: list[float] = []
        alt_currents_c2_total: list[float] = []
        sum_matrices: list[np.ndarray] = []
        matrix_coefficients: list[np.ndarray] = []
        matrix_nucoefficients: list[np.ndarray] = []
        observable_coefficients: list[np.ndarray] = []
        observable_nucoefficients: list[np.ndarray] = []
        force_current_contributions: list[dict[str, Any]] = []
        rhs_force_contributions: list[dict[str, Any]] = []
        collision_factors: list[float] = []
        for species_index in range(cm_ab.shape[0]):
            sum_matrix = jax.vmap(
                jax.vmap(get_sum, in_axes=(None, None, 0, None, None, None)),
                in_axes=(None, 0, None, None, None, None),
            )(
                species_index,
                neopax_grid.sonine_indeces,
                neopax_grid.sonine_indeces,
                cm_ab,
                cn_ab,
                tau,
            )
            sum_matrix_np = np.asarray(sum_matrix, dtype=float)
            species_lij = np.asarray(
                context["lij_full"][species_index, radial_index, :, :],
                dtype=float,
            )
            species_eij = np.asarray(
                context["eij_full"][species_index, radial_index, :, :],
                dtype=float,
            )
            coeff_matrix, nucoeff_matrix = _matrix_coefficients(species_lij, species_eij)
            coeff_upar, nucoeff_upar = _observable_coefficients(species_lij, species_eij)
            coeff_upar_full, nucoeff_upar_full = _full_observable_coefficient_matrix(
                species_lij,
                species_eij,
            )
            factor = (
                2.0
                / float(
                    np.asarray(species.v_thermal[species_index, radial_index], dtype=float) ** 2
                )
                / float(np.asarray(field.Bsqav[radial_index], dtype=float))
            )
            collision_factors.append(float(factor))
            c_terms, add1, add2, add3, add4 = jax.vmap(
                get_correction_matrix,
                in_axes=(None, None, None, 0, None, None, None, None, None, None, None, None, None),
            )(
                species,
                neopax_grid,
                species_index,
                species.species_indeces,
                jnp.asarray(coeff_upar_full),
                jnp.asarray(nucoeff_upar_full),
                cm_ab,
                cn_ab,
                sum_matrix,
                tau,
                factor,
                correction,
                radial_index,
            )
            c_vec = np.asarray(jnp.sum(c_terms, axis=0), dtype=float)
            c_vectors.append(c_vec)
            add_term_sums.append(
                [
                    float(jnp.sum(add1)),
                    float(jnp.sum(add2)),
                    float(jnp.sum(add3)),
                    float(jnp.sum(add4)),
                ]
            )
            sum_matrices.append(sum_matrix_np)
            matrix_coefficients.append(coeff_matrix)
            matrix_nucoefficients.append(nucoeff_matrix)
            observable_coefficients.append(coeff_upar)
            observable_nucoefficients.append(nucoeff_upar)
            row3 = np.asarray(context["lij_full"][species_index, radial_index, 2, :], dtype=float)
            density = float(np.asarray(species.density[species_index, radial_index], dtype=float))
            a1 = float(np.asarray(species.A1[species_index, radial_index], dtype=float))
            a2 = float(np.asarray(species.A2[species_index, radial_index], dtype=float))
            a3 = float(np.asarray(species.A3[radial_index], dtype=float))
            forces = np.asarray([a1, a2, a3], dtype=float)
            charge_sign = -1.0 if species_index == 0 else 1.0
            force_current_contributions.append(
                _force_current_contributions(
                    lij_block=species_lij,
                    forces=forces,
                    density=density,
                    charge_sign=charge_sign,
                )
            )
            rhs_force_contributions.append(
                _rhs_force_contributions(species_lij, forces)
            )
            base_upar = -density * (row3[0] * a1 + row3[1] * a2 + row3[2] * a3)
            if moment_order >= 3:
                alt_currents_c2_total.append(base_upar + density * c_vec[2])
                alt_currents_c2_only.append(density * c_vec[2])
            else:
                alt_currents_c2_total.append(np.nan)
                alt_currents_c2_only.append(np.nan)
        radial_bridge = float(sfincs_flow_bridge[radial_index])
        electron_slice = solution[:moment_order]
        ion_slice = solution[moment_order : 2 * moment_order]
        weighted_solution_raw = {
            "electron": float(
                -elementary_charge
                * float(np.asarray(species.density[0, radial_index], dtype=float))
                * _candidate_upar_from_solution(electron_slice, "weighted")
            ),
            "ion": float(
                elementary_charge
                * float(np.asarray(species.density[1, radial_index], dtype=float))
                * _candidate_upar_from_solution(ion_slice, "weighted")
            ),
        }
        c0_solution_raw = {
            "electron": float(
                -elementary_charge
                * float(np.asarray(species.density[0, radial_index], dtype=float))
                * _candidate_upar_from_solution(electron_slice, "c0")
            ),
            "ion": float(
                elementary_charge
                * float(np.asarray(species.density[1, radial_index], dtype=float))
                * _candidate_upar_from_solution(ion_slice, "c0")
            ),
        }
        weighted_solution = {
            label: radial_bridge * value for label, value in weighted_solution_raw.items()
        }
        c0_solution = {label: radial_bridge * value for label, value in c0_solution_raw.items()}
        dump = {
                "rho_target": float(rho_target),
                "rho_grid_value": float(rho_grid[radial_index]),
                "radial_index": radial_index,
                "b0_over_bbar": float(b0_over_bbar[radial_index]),
                "sfincs_flow_bridge": float(radial_bridge),
                "matrix_condition_number": float(np.linalg.cond(dense_matrix)),
                "matrix_log10_abs_min": float(np.log10(max(min_entry, 1.0e-300))),
                "matrix_log10_abs_max": float(np.log10(np.max(np.abs(dense_matrix)))),
                "singular_values": singular_values.tolist(),
                "rhs_vector": rhs_vector.tolist(),
                "rhs_norm": float(np.linalg.norm(rhs_vector)),
                "solution_vector": solution.tolist(),
                "relative_residual_norm": _relative_residual_norm(
                    dense_matrix,
                    rhs_vector,
                    solution,
                ),
                "electron": {
                    "reference_current": float(reference_electron[radial_index]),
                    "A1": float(np.asarray(species.A1[0], dtype=float)[radial_index]),
                    "A2": float(np.asarray(species.A2[0], dtype=float)[radial_index]),
                    "A3": float(np.asarray(species.A3, dtype=float)[radial_index]),
                    "current_nomom": float(electron_current_nomom[radial_index]),
                    "current_nomom_raw_closure": float(electron_current_nomom_raw[radial_index]),
                    "current_correction": float(electron_current_correction[radial_index]),
                    "current_correction_raw_closure": float(
                        electron_current_correction_raw[radial_index]
                    ),
                    "current_total": float(
                        electron_current_nomom[radial_index]
                        + electron_current_correction[radial_index]
                    ),
                    "current_total_raw_closure": float(electron_current_total_raw[radial_index]),
                    "current_alt_c2_only": float(-elementary_charge * alt_currents_c2_only[0]),
                    "current_alt_c2_total": float(-elementary_charge * alt_currents_c2_total[0]),
                    "current_solution_c0": c0_solution["electron"],
                    "current_solution_c0_raw_closure": c0_solution_raw["electron"],
                    "current_solution_c0_sfincs_flow_normalized": c0_solution["electron"],
                    "current_solution_weighted": weighted_solution["electron"],
                    "current_solution_weighted_raw_closure": weighted_solution_raw["electron"],
                    "current_solution_weighted_sfincs_flow_normalized": weighted_solution[
                        "electron"
                    ],
                    "c_vector": c_vectors[0].tolist(),
                    "add_terms_sum": add_term_sums[0],
                    "rhs": rhs_vector[:moment_order].tolist(),
                    "solution": electron_slice.tolist(),
                    "collision_factor": collision_factors[0],
                    "sum_matrix": sum_matrices[0].tolist(),
                    "matrix_coefficients": matrix_coefficients[0].tolist(),
                    "matrix_nucoefficients": matrix_nucoefficients[0].tolist(),
                    "observable_coefficients": observable_coefficients[0].tolist(),
                    "observable_nucoefficients": observable_nucoefficients[0].tolist(),
                    "force_current_contributions": force_current_contributions[0],
                    "rhs_force_contributions": rhs_force_contributions[0],
                    "Lij_full": np.asarray(
                        context["lij_full"][0, radial_index, :, :],
                        dtype=float,
                    ).tolist(),
                    "Eij_full": np.asarray(
                        context["eij_full"][0, radial_index, :, :],
                        dtype=float,
                    ).tolist(),
                    "CM_ab": np.asarray(cm_ab[0], dtype=float).tolist(),
                    "CN_ab": np.asarray(cn_ab[0], dtype=float).tolist(),
                    "tau": np.asarray(tau[0], dtype=float).tolist(),
                    "Lij_rows_3_to_5_cols_1_to_3": np.asarray(
                        context["lij_full"][0, radial_index, 2 : 2 + moment_order, 0:3],
                        dtype=float,
                    ).tolist(),
                    "Eij_rows_3_to_5_cols_1_to_3": np.asarray(
                        context["eij_full"][0, radial_index, 2 : 2 + moment_order, 0:3],
                        dtype=float,
                    ).tolist(),
                },
                "ion": {
                    "reference_current": float(reference_ion[radial_index]),
                    "A1": float(np.asarray(species.A1[1], dtype=float)[radial_index]),
                    "A2": float(np.asarray(species.A2[1], dtype=float)[radial_index]),
                    "A3": float(np.asarray(species.A3, dtype=float)[radial_index]),
                    "current_nomom": float(ion_current_nomom[radial_index]),
                    "current_nomom_raw_closure": float(ion_current_nomom_raw[radial_index]),
                    "current_correction": float(ion_current_correction[radial_index]),
                    "current_correction_raw_closure": float(
                        ion_current_correction_raw[radial_index]
                    ),
                    "current_total": float(
                        ion_current_nomom[radial_index] + ion_current_correction[radial_index]
                    ),
                    "current_total_raw_closure": float(ion_current_total_raw[radial_index]),
                    "current_alt_c2_only": float(elementary_charge * alt_currents_c2_only[1]),
                    "current_alt_c2_total": float(elementary_charge * alt_currents_c2_total[1]),
                    "current_solution_c0": c0_solution["ion"],
                    "current_solution_c0_raw_closure": c0_solution_raw["ion"],
                    "current_solution_c0_sfincs_flow_normalized": c0_solution["ion"],
                    "current_solution_weighted": weighted_solution["ion"],
                    "current_solution_weighted_raw_closure": weighted_solution_raw["ion"],
                    "current_solution_weighted_sfincs_flow_normalized": weighted_solution["ion"],
                    "c_vector": c_vectors[1].tolist(),
                    "add_terms_sum": add_term_sums[1],
                    "rhs": rhs_vector[moment_order : 2 * moment_order].tolist(),
                    "solution": ion_slice.tolist(),
                    "collision_factor": collision_factors[1],
                    "sum_matrix": sum_matrices[1].tolist(),
                    "matrix_coefficients": matrix_coefficients[1].tolist(),
                    "matrix_nucoefficients": matrix_nucoefficients[1].tolist(),
                    "observable_coefficients": observable_coefficients[1].tolist(),
                    "observable_nucoefficients": observable_nucoefficients[1].tolist(),
                    "force_current_contributions": force_current_contributions[1],
                    "rhs_force_contributions": rhs_force_contributions[1],
                    "Lij_full": np.asarray(
                        context["lij_full"][1, radial_index, :, :],
                        dtype=float,
                    ).tolist(),
                    "Eij_full": np.asarray(
                        context["eij_full"][1, radial_index, :, :],
                        dtype=float,
                    ).tolist(),
                    "CM_ab": np.asarray(cm_ab[1], dtype=float).tolist(),
                    "CN_ab": np.asarray(cn_ab[1], dtype=float).tolist(),
                    "tau": np.asarray(tau[1], dtype=float).tolist(),
                    "Lij_rows_3_to_5_cols_1_to_3": np.asarray(
                        context["lij_full"][1, radial_index, 2 : 2 + moment_order, 0:3],
                        dtype=float,
                    ).tolist(),
                    "Eij_rows_3_to_5_cols_1_to_3": np.asarray(
                        context["eij_full"][1, radial_index, 2 : 2 + moment_order, 0:3],
                        dtype=float,
                    ).tolist(),
                },
            }
        dump["forensic_summary"] = _dump_forensic_summary(dump)
        dumps.append(dump)
    return {
        "case": {
            "name": case.name,
            "label": case.label,
            "helicity_n": case.helicity_n,
            "wout_path": str(case.wout_path),
            "sfincs_scan_path": str(case.sfincs_scan_path),
        },
        "timings": context["timings"],
        "rho_grid": rho_grid.tolist(),
        "d33_mode": D33_MODE,
        "d33_lij_mode": D33_LIJ_MODE,
        "d33_eij_mode": D33_EIJ_MODE,
        "neopax_n_order": NEOPAX_N_ORDER,
        "current_grid": {
            "electron_reference": reference_electron.tolist(),
            "electron_nomom": electron_current_nomom.tolist(),
            "electron_correction": electron_current_correction.tolist(),
            "electron_total": (electron_current_nomom + electron_current_correction).tolist(),
            "electron_nomom_raw_closure": electron_current_nomom_raw.tolist(),
            "electron_correction_raw_closure": electron_current_correction_raw.tolist(),
            "electron_total_raw_closure": (
                electron_current_nomom_raw + electron_current_correction_raw
            ).tolist(),
            "ion_reference": reference_ion.tolist(),
            "ion_nomom": ion_current_nomom.tolist(),
            "ion_correction": ion_current_correction.tolist(),
            "ion_total": (ion_current_nomom + ion_current_correction).tolist(),
            "ion_nomom_raw_closure": ion_current_nomom_raw.tolist(),
            "ion_correction_raw_closure": ion_current_correction_raw.tolist(),
            "ion_total_raw_closure": (
                ion_current_nomom_raw + ion_current_correction_raw
            ).tolist(),
            "b0_over_bbar": b0_over_bbar.tolist(),
        },
        "dumps": dumps,
    }


def _plot(payload: dict[str, Any], output_prefix: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), constrained_layout=True, sharex=True)
    rho = np.asarray(payload["rho_grid"], dtype=float)
    currents = payload["current_grid"]
    axes[0].plot(rho, currents["electron_nomom"], "o-", label="electron nomom", color="#1f77b4")
    axes[0].plot(
        rho,
        currents["electron_total"],
        "s--",
        label="electron total",
        color="#d62728",
    )
    axes[0].plot(rho, currents["ion_nomom"], "o-", label="ion nomom", color="#2ca02c")
    axes[0].plot(rho, currents["ion_total"], "s--", label="ion total", color="#9467bd")
    axes[0].set_ylabel("current [A/m$^2$]")
    axes[0].grid(alpha=0.25, lw=0.6)
    axes[0].legend(frameon=False, ncol=2)

    dump_rho = np.asarray([item["rho_grid_value"] for item in payload["dumps"]], dtype=float)
    condition = np.asarray(
        [item["matrix_condition_number"] for item in payload["dumps"]],
        dtype=float,
    )
    residual = np.asarray(
        [item["relative_residual_norm"] for item in payload["dumps"]],
        dtype=float,
    )
    axes[1].plot(dump_rho, condition, "o-", color="black", label="cond(M)")
    axes[1].plot(dump_rho, residual, "s--", color="#ff7f0e", label="relative residual")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\rho$")
    axes[1].set_ylabel("matrix diagnostic")
    axes[1].grid(alpha=0.25, lw=0.6)
    axes[1].legend(frameon=False)

    fig.savefig(output_prefix.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("qa", "qh"), default="qa")
    parser.add_argument(
        "--rho",
        default="0.50,0.61,0.71",
        help="comma-separated target rho values for matrix dumps",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="optional output prefix for JSON/PNG/PDF diagnostics",
    )
    args = parser.parse_args(argv)

    rho_targets = np.asarray(
        [float(value) for value in args.rho.split(",") if value.strip()],
        dtype=float,
    )
    output_prefix = args.output_prefix or (
        DEFAULT_OUTPUT_DIR / f"fixed_field_momentum_correction_diagnostic_{args.case}"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    payload = _diagnose_case(args.case, rho_targets)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(_jsonify(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _plot(payload, output_prefix)
    print(json.dumps(_jsonify(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
