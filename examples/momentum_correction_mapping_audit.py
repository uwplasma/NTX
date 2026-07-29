#!/usr/bin/env python3
"""Audit simple momentum-correction reconstruction maps against fixed-field and W7-X references."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

NEOPAX_ROOT = Path.home() / "local" / "tests" / "NEOPAX"
if str(NEOPAX_ROOT) not in sys.path:
    sys.path.insert(0, str(NEOPAX_ROOT))

import jax
import jax.numpy as jnp

from ntx._interp import Interpolator1D

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import NEOPAX
from NEOPAX._constants import elementary_charge
from NEOPAX._neoclassical import (
    get_Collision_Operator_terms,
    get_Lij_matrix_with_momentum_correction,
    get_Matrix,
    get_Neoclassical_Fluxes,
    get_rhs,
)

OUTPUT_DIR = ROOT / "examples" / "outputs" / "momentum_correction_mapping_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = OUTPUT_DIR / "momentum_correction_mapping_audit"

SONINE_WEIGHTS = np.array([1.0, 0.4, 8.0 / 35.0], dtype=float)


@dataclass(frozen=True)
class Sample:
    dataset: str
    case: str
    rho: float
    species: str
    density: float
    solution: np.ndarray
    current_nomom: float
    current_reference: float


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _assemble_dense_species_matrix(blocks: np.ndarray) -> np.ndarray:
    array = np.asarray(blocks, dtype=float)
    return np.transpose(array, (0, 2, 1, 3)).reshape(
        array.shape[0] * array.shape[2],
        array.shape[1] * array.shape[3],
    )


def _candidate_current(
    *,
    density: float,
    solution: np.ndarray,
    charge_sign: float,
    weights: np.ndarray,
) -> float:
    return float(charge_sign * elementary_charge * density * np.dot(weights, solution))


def _fit_species_weights(samples: list[Sample]) -> np.ndarray:
    x = np.stack([sample.density * sample.solution for sample in samples], axis=0)
    y = np.array(
        [
            sample.current_reference
            / (
                elementary_charge
                * (-1.0 if sample.species == "electron" else 1.0)
            )
            for sample in samples
        ],
        dtype=float,
    )
    weights, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.asarray(weights, dtype=float)


def _evaluate_weights(samples: list[Sample], weights: np.ndarray) -> dict[str, Any]:
    rows = []
    errors = []
    for sample in samples:
        charge_sign = -1.0 if sample.species == "electron" else 1.0
        current_total = _candidate_current(
            density=sample.density,
            solution=sample.solution,
            charge_sign=charge_sign,
            weights=weights,
        )
        rel = abs(current_total - sample.current_reference) / max(
            abs(sample.current_reference), 1.0
        )
        errors.append(rel)
        rows.append(
            {
                "dataset": sample.dataset,
                "case": sample.case,
                "rho": sample.rho,
                "species": sample.species,
                "reference_current": sample.current_reference,
                "nomom_current": sample.current_nomom,
                "predicted_correction": current_total - sample.current_nomom,
                "predicted_total": current_total,
                "relative_error": rel,
            }
        )
    return {
        "weights": np.asarray(weights, dtype=float),
        "max_relative_error": float(np.max(np.asarray(errors, dtype=float))),
        "rows": rows,
    }


def _fixed_field_samples() -> list[Sample]:
    samples: list[Sample] = []
    for case in ("qa", "qh"):
        path = (
            ROOT
            / "examples"
            / "outputs"
            / "fixed_field_momentum_correction_diagnostic"
            / f"fixed_field_momentum_correction_diagnostic_{case}.json"
        )
        payload = json.loads(path.read_text())
        for dump in payload["dumps"]:
            rho = float(dump["rho_grid_value"])
            for species in ("electron", "ion"):
                branch = dump[species]
                charge_sign = -1.0 if species == "electron" else 1.0
                solution = np.asarray(branch["solution"], dtype=float)
                density = abs(
                    float(branch["current_solution_c0"])
                    / max(abs(charge_sign * elementary_charge * solution[0]), 1.0e-30)
                )
                samples.append(
                    Sample(
                        dataset="fixed_field",
                        case=case,
                        rho=rho,
                        species=species,
                        density=density,
                        solution=solution,
                        current_nomom=float(branch["current_nomom"]),
                        current_reference=float(branch["reference_current"]),
                    )
                )
    return samples


def _w7x_samples() -> list[Sample]:
    file_initial = h5py.File(NEOPAX_ROOT / "tests" / "inputs" / "NTSS_W7X_Initial.h5", "r")
    er_initial = Interpolator1D(file_initial["r"][()], file_initial["Er"][()], method="akima")
    file_initial.close()

    vmec_file = NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
    boozer_file = NEOPAX_ROOT / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
    neoclassical_file = NEOPAX_ROOT / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"

    n_species = 3
    n_x = 64
    n_radial = 51
    grid = NEOPAX.Grid.create_standard(n_radial, n_x, n_species)
    field = NEOPAX.Field.read_vmec_booz(n_radial, str(vmec_file), str(boozer_file))

    ne0 = 4.21e20
    te0 = 17.8e3
    ni0 = 4.21e20
    ti0 = 17.8e3
    neb = 0.6e20
    teb = 0.7e3
    nib = 0.6e20
    tib = 0.7e3
    deuterium_ratio = 0.5
    tritium_ratio = 0.5

    t_edge = jnp.array([0.7e3, 0.7e3, 0.7e3])
    n_edge = jnp.array([0.6e20, deuterium_ratio * 0.6e20, tritium_ratio * 0.6e20])

    te_initial = (te0 - teb) * (1 - (field.r_grid / field.a_b) ** 2) + teb
    ne_initial = (ne0 - neb) * (1 - (field.r_grid / field.a_b) ** 10.0) + neb
    td_initial = (ti0 - tib) * (1 - (field.r_grid / field.a_b) ** 2) + tib
    nd_initial = deuterium_ratio * ((ni0 - nib) * (1 - (field.r_grid / field.a_b) ** 10.0) + nib)
    tt_initial = (ti0 - tib) * (1 - (field.r_grid / field.a_b) ** 2) + tib
    nt_initial = tritium_ratio * ((ni0 - nib) * (1 - (field.r_grid / field.a_b) ** 10.0) + nib)
    er_profile = er_initial(field.r_grid)

    temperature = jnp.vstack([te_initial, td_initial, tt_initial])
    density = jnp.vstack([ne_initial, nd_initial, nt_initial])
    mass = jnp.array([1 / 1836.15267343, 2.0, 3.0])
    charge = jnp.array([-1.0, 1.0, 1.0])

    species = NEOPAX.Species(
        n_species,
        n_radial,
        grid.species_indeces,
        mass,
        charge,
        temperature,
        density,
        er_profile,
        field.r_grid,
        field.r_grid_half,
        field.dr,
        field.Vprime_half,
        field.overVprime,
        n_edge,
        t_edge,
    )
    database = NEOPAX.Monoenergetic.read_monkes(field.a_b, str(neoclassical_file))
    _, _, _, upar_nomom = get_Neoclassical_Fluxes(species, grid, field, database)

    lij_full, eij_full, _ = jax.vmap(
        jax.vmap(
            get_Lij_matrix_with_momentum_correction,
            in_axes=(None, None, None, None, None, 0),
        ),
        in_axes=(None, None, None, None, 0, None),
    )(
        species,
        grid,
        field,
        database,
        species.species_indeces,
        grid.full_grid_indeces,
    )
    lij_full = lij_full.at[:, 0, :, :].set(lij_full.at[:, 1, :, :].get())
    eij_full = eij_full.at[:, 0, :, :].set(eij_full.at[:, 1, :, :].get())

    solution_vectors = []
    for radial_index in range(n_radial):
        cm_ab, cn_ab, tau = jax.vmap(
            jax.vmap(get_Collision_Operator_terms, in_axes=(None, None, None, 0, None)),
            in_axes=(None, None, 0, None, None),
        )(species, grid, species.species_indeces, species.species_indeces, radial_index)
        rhs = jax.vmap(get_rhs, in_axes=(None, None, 0, None, 0))(
            species,
            grid,
            species.species_indeces,
            radial_index,
            lij_full[:, radial_index, :, :],
        )
        blocks = jax.vmap(
            get_Matrix,
            in_axes=(None, None, None, 0, None, 0, 0, None, None, None),
        )(
            species,
            grid,
            field,
            species.species_indeces,
            radial_index,
            lij_full[:, radial_index, :, :],
            eij_full[:, radial_index, :, :],
            cm_ab,
            cn_ab,
            tau,
        )
        solution = np.linalg.solve(
            _assemble_dense_species_matrix(np.asarray(blocks, dtype=float)),
            np.asarray(jnp.reshape(rhs, rhs.shape[0] * rhs.shape[1]), dtype=float),
        ).reshape(n_species, 3)
        solution_vectors.append(solution)
    solution_vectors = np.asarray(solution_vectors, dtype=float)

    with h5py.File(
        NEOPAX_ROOT / "tests" / "inputs" / "NTSS_W7X_Initial_Momentum.h5",
        "r",
    ) as handle:
        j_bse = np.asarray(handle["J_bse"], dtype=float)
        j_bsi = np.asarray(handle["J_bsi"], dtype=float)
        rho = np.asarray(handle["r"], dtype=float) / float(field.a_b)

    electron_nomom = -elementary_charge * np.asarray(upar_nomom[0], dtype=float)
    deuterium_nomom = elementary_charge * np.asarray(upar_nomom[1], dtype=float)
    tritium_nomom = elementary_charge * np.asarray(upar_nomom[2], dtype=float)

    samples: list[Sample] = []
    for radial_index in range(1, n_radial):
        samples.append(
            Sample(
                dataset="w7x",
                case="w7x",
                rho=float(rho[radial_index]),
                species="electron",
                density=float(np.asarray(species.density[0, radial_index], dtype=float)),
                solution=solution_vectors[radial_index, 0],
                current_nomom=float(electron_nomom[radial_index]),
                current_reference=float(j_bse[radial_index]),
            )
        )
        ion_solution = (
            solution_vectors[radial_index, 1]
            * float(np.asarray(species.density[1, radial_index], dtype=float))
            + solution_vectors[radial_index, 2]
            * float(np.asarray(species.density[2, radial_index], dtype=float))
        )
        ion_nomom = float(deuterium_nomom[radial_index] + tritium_nomom[radial_index])
        samples.append(
            Sample(
                dataset="w7x",
                case="w7x",
                rho=float(rho[radial_index]),
                species="ion",
                density=1.0,
                solution=ion_solution,
                current_nomom=ion_nomom,
                current_reference=float(j_bsi[radial_index]),
            )
        )
    return samples


def _plot(summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True, sharey=True)
    labels = ["c0", "weighted", "fixed_field_fit", "w7x_fit", "combined_fit"]
    species_order = ["electron", "ion"]
    for ax, species in zip(axes, species_order, strict=True):
        values = [
            summary["mappings"][label]["datasets"]["combined"]["species"][species]["max_relative_error"]
            for label in labels
        ]
        ax.bar(
            np.arange(len(labels)),
            values,
            color=["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"],
        )
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(species.capitalize())
        ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.6, axis="y")
        ax.set_ylabel("max relative error")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fixed_field = _fixed_field_samples()
    w7x = _w7x_samples()
    combined = fixed_field + w7x

    mappings: dict[str, dict[str, np.ndarray]] = {
        "c0": {
            "electron": np.array([1.0, 0.0, 0.0], dtype=float),
            "ion": np.array([1.0, 0.0, 0.0], dtype=float),
        },
        "weighted": {
            "electron": SONINE_WEIGHTS.copy(),
            "ion": SONINE_WEIGHTS.copy(),
        },
        "fixed_field_fit": {
            "electron": _fit_species_weights(
                [sample for sample in fixed_field if sample.species == "electron"]
            ),
            "ion": _fit_species_weights(
                [sample for sample in fixed_field if sample.species == "ion"]
            ),
        },
        "w7x_fit": {
            "electron": _fit_species_weights(
                [sample for sample in w7x if sample.species == "electron"]
            ),
            "ion": _fit_species_weights(
                [sample for sample in w7x if sample.species == "ion"]
            ),
        },
        "combined_fit": {
            "electron": _fit_species_weights(
                [sample for sample in combined if sample.species == "electron"]
            ),
            "ion": _fit_species_weights(
                [sample for sample in combined if sample.species == "ion"]
            ),
        },
    }

    summary: dict[str, Any] = {
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png")),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf")),
        "mappings": {},
    }
    datasets = {"fixed_field": fixed_field, "w7x": w7x, "combined": combined}
    for label, species_weights in mappings.items():
        mapping_payload: dict[str, Any] = {
            "weights": {species: weights for species, weights in species_weights.items()},
            "datasets": {},
        }
        for dataset_name, dataset_samples in datasets.items():
            dataset_payload = {"species": {}}
            for species in ("electron", "ion"):
                evaluated = _evaluate_weights(
                    [sample for sample in dataset_samples if sample.species == species],
                    species_weights[species],
                )
                dataset_payload["species"][species] = {
                    "weights": evaluated["weights"],
                    "max_relative_error": evaluated["max_relative_error"],
                    "rows": evaluated["rows"],
                }
            dataset_payload["max_relative_error"] = float(
                max(
                    dataset_payload["species"]["electron"]["max_relative_error"],
                    dataset_payload["species"]["ion"]["max_relative_error"],
                )
            )
            mapping_payload["datasets"][dataset_name] = dataset_payload
        summary["mappings"][label] = mapping_payload

    _plot(summary)
    OUTPUT_PREFIX.with_suffix(".json").write_text(
        json.dumps(_jsonify(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_jsonify(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
