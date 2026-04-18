#!/usr/bin/env python3
"""Dump fixed-field QA/QH momentum-correction internals on selected radii."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
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

from ntx import build_ntx_neopax_scan_from_surfaces, load_vmec_surface, to_neopax_monoenergetic

_bootstrap_current_pythonpath()

import jax
import jax.numpy as jnp
import NEOPAX
from NEOPAX._neoclassical import (
    get_Collision_Operator_terms,
    get_correction_matrix,
    get_Lij_matrix_with_momentum_correction,
    get_Matrix,
    get_Neoclassical_Fluxes,
    get_Neoclassical_Fluxes_With_Momentum_Correction,
    get_rhs,
    get_sum,
)
from scipy.constants import elementary_charge

DEFAULT_OUTPUT_DIR = ROOT / "examples" / "outputs" / "fixed_field_momentum_correction_diagnostic"


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
    neopax_grid = NEOPAX.Grid.create_standard(n_r, 64, 2)
    nu_values = _adaptive_nu_values(species, neopax_grid)
    timings["species_seconds"] = float(time.perf_counter() - start)

    profiles = _archived_profiles(case)
    rho_field = np.asarray(field.rho_grid, dtype=float)
    rho_surface = np.clip(rho_field, 0.05, 0.95)
    drds = float(field.a_b) * 0.5 / np.clip(rho_surface, 0.05, None)
    archived_er = _interp_profile(profiles.rho, profiles.electric_field_kv_per_m, rho_surface)
    er_axis = float(np.median(archived_er)) * ER_AXIS_FACTORS
    er_values = np.repeat(er_axis[None, :], rho_surface.size, axis=0)

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

    start = time.perf_counter()
    database = to_neopax_monoenergetic(scan, a_b=float(field.a_b))
    timings["database_seconds"] = float(time.perf_counter() - start)

    start = time.perf_counter()
    lij_nomom, gamma_nomom, heat_nomom, upar_nomom = get_Neoclassical_Fluxes(
        species,
        neopax_grid,
        field,
        database,
    )
    correction_gamma, correction_heat, upar_correction, qpar, upar2 = (
        get_Neoclassical_Fluxes_With_Momentum_Correction(species, neopax_grid, field, database)
    )
    jax.block_until_ready(lij_nomom)
    jax.block_until_ready(upar_correction)
    timings["neopax_closure_seconds"] = float(time.perf_counter() - start)

    start = time.perf_counter()
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
    timings["momentum_blocks_seconds"] = float(time.perf_counter() - start)

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
        "upar_correction": np.asarray(upar_correction, dtype=float),
        "qpar_correction": np.asarray(qpar, dtype=float),
        "upar2_correction": np.asarray(upar2, dtype=float),
        "lij_full": np.asarray(lij_full, dtype=float),
        "eij_full": np.asarray(eij_full, dtype=float),
        "nu_weighted_average": np.asarray(nu_weighted_average, dtype=float),
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
    upar_correction = np.asarray(context["upar_correction"], dtype=float)

    electron_current_nomom = _species_current(-1.0, upar_nomom[0])
    ion_current_nomom = _species_current(+1.0, upar_nomom[1])
    electron_current_correction = _species_current(-1.0, upar_correction[:, 0])
    ion_current_correction = _species_current(+1.0, upar_correction[:, 1])

    dumps: list[dict[str, Any]] = []
    for rho_target in rho_targets:
        radial_index = int(np.argmin(np.abs(rho_grid - rho_target)))
        cm_ab, cn_ab, tau = jax.vmap(
            jax.vmap(get_Collision_Operator_terms, in_axes=(None, None, 0, None)),
            in_axes=(None, 0, None, None),
        )(
            species,
            species.species_indeces,
            species.species_indeces,
            radial_index,
        )
        rhs = jax.vmap(get_rhs, in_axes=(None, 0, None, 0))(
            species,
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
        alt_currents_c2_only: list[float] = []
        alt_currents_c2_total: list[float] = []
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
            factor = (
                2.0
                / float(np.asarray(species.v_thermal[species_index, radial_index], dtype=float) ** 2)
                / float(np.asarray(field.Bsqav[radial_index], dtype=float))
            )
            c_terms, *_ = jax.vmap(
                get_correction_matrix,
                in_axes=(None, None, None, 0, None, None, None, None, None, None, None, None, None),
            )(
                species,
                neopax_grid,
                species_index,
                species.species_indeces,
                jnp.zeros((3, 3)),
                jnp.zeros((3, 3)),
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
            row3 = np.asarray(context["lij_full"][species_index, radial_index, 2, :], dtype=float)
            density = float(np.asarray(species.density[species_index, radial_index], dtype=float))
            a1 = float(np.asarray(species.A1[species_index, radial_index], dtype=float))
            a2 = float(np.asarray(species.A2[species_index, radial_index], dtype=float))
            a3 = float(np.asarray(species.A3[radial_index], dtype=float))
            base_upar = -density * (row3[0] * a1 + row3[1] * a2 + row3[2] * a3)
            alt_currents_c2_total.append(base_upar + density * c_vec[2])
            alt_currents_c2_only.append(density * c_vec[2])
        dumps.append(
            {
                "rho_target": float(rho_target),
                "rho_grid_value": float(rho_grid[radial_index]),
                "radial_index": radial_index,
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
                    "A1": float(np.asarray(species.A1[0], dtype=float)[radial_index]),
                    "A2": float(np.asarray(species.A2[0], dtype=float)[radial_index]),
                    "A3": float(np.asarray(species.A3, dtype=float)[radial_index]),
                    "current_nomom": float(electron_current_nomom[radial_index]),
                    "current_correction": float(electron_current_correction[radial_index]),
                    "current_total": float(
                        electron_current_nomom[radial_index]
                        + electron_current_correction[radial_index]
                    ),
                    "current_alt_c2_only": float(-elementary_charge * alt_currents_c2_only[0]),
                    "current_alt_c2_total": float(-elementary_charge * alt_currents_c2_total[0]),
                    "c_vector": c_vectors[0].tolist(),
                    "rhs": rhs_vector[:3].tolist(),
                    "solution": solution[:3].tolist(),
                    "Lij_rows_3_to_5_cols_1_to_3": np.asarray(
                        context["lij_full"][0, radial_index, 2:5, 0:3],
                        dtype=float,
                    ).tolist(),
                    "Eij_rows_3_to_5_cols_1_to_3": np.asarray(
                        context["eij_full"][0, radial_index, 2:5, 0:3],
                        dtype=float,
                    ).tolist(),
                },
                "ion": {
                    "A1": float(np.asarray(species.A1[1], dtype=float)[radial_index]),
                    "A2": float(np.asarray(species.A2[1], dtype=float)[radial_index]),
                    "A3": float(np.asarray(species.A3, dtype=float)[radial_index]),
                    "current_nomom": float(ion_current_nomom[radial_index]),
                    "current_correction": float(ion_current_correction[radial_index]),
                    "current_total": float(
                        ion_current_nomom[radial_index] + ion_current_correction[radial_index]
                    ),
                    "current_alt_c2_only": float(elementary_charge * alt_currents_c2_only[1]),
                    "current_alt_c2_total": float(elementary_charge * alt_currents_c2_total[1]),
                    "c_vector": c_vectors[1].tolist(),
                    "rhs": rhs_vector[3:].tolist(),
                    "solution": solution[3:].tolist(),
                    "Lij_rows_3_to_5_cols_1_to_3": np.asarray(
                        context["lij_full"][1, radial_index, 2:5, 0:3],
                        dtype=float,
                    ).tolist(),
                    "Eij_rows_3_to_5_cols_1_to_3": np.asarray(
                        context["eij_full"][1, radial_index, 2:5, 0:3],
                        dtype=float,
                    ).tolist(),
                },
            }
        )
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
        "current_grid": {
            "electron_nomom": electron_current_nomom.tolist(),
            "electron_correction": electron_current_correction.tolist(),
            "electron_total": (electron_current_nomom + electron_current_correction).tolist(),
            "ion_nomom": ion_current_nomom.tolist(),
            "ion_correction": ion_current_correction.tolist(),
            "ion_total": (ion_current_nomom + ion_current_correction).tolist(),
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
