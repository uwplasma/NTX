#!/usr/bin/env python3
"""Decompose the finite-beta profile-current stress point by source channel.

The finite-beta current stress artifact has already separated coefficient
normalization, profile/current conditioning, and velocity-quadrature aliasing.
This diagnostic freezes that same owned finite-beta contract and decomposes the
momentum-restoring linear system into density/electric, temperature-gradient,
and parallel-electric source channels at the inner stress radius.

It is a closure-localization audit, not a fitted correction and not a parity
claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.constants import elementary_charge  # noqa: E402

from examples.owned_finite_beta_bootstrap_comparison import (  # noqa: E402
    DEFAULT_CASE,
    DEFAULT_MBOZ,
    DEFAULT_NBOZ,
    DEFAULT_REDL_NTHETA,
    EPS,
    ProfileContract,
    _build_scan_for_path,
    _build_species,
    _case_by_id,
    _drds_from_minor_radius,
    _evaluate_neopax_currents,
    _interp,
    _require_external_stacks,
    _to_jsonable,
    _write_boozmn,
)
from ntx import (  # noqa: E402
    GridSpec,
    load_neopax_reference_scan,
    neopax_scan_requires_rebuild,
    to_neopax_monoenergetic,
    write_neopax_scan_hdf5,
)

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_finite_beta_source_channel_audit"
BOOTSTRAP_JSON = ROOT / "docs" / "_static" / "owned_finite_beta_bootstrap_comparison.json"
WORKDIR = ROOT / "examples" / "outputs" / "owned_finite_beta_source_channel_audit"
DEFAULT_SETTINGS = ((10, 12), (18, 18))
PROFILE_CURRENT_GATE = 1.0e-1
SOURCE_RECONSTRUCTION_GATE = 1.0e-8
TRANSPORT_LABELS = (
    "A1_density_electric",
    "A2_temperature",
    "A3_parallel_electric",
)
EFFECTIVE_LABELS = (
    "density_electric_force",
    "effective_temperature_force",
    "parallel_electric_force",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _contract_from_payload(payload: dict[str, Any]) -> ProfileContract:
    values = dict(payload.get("profile_contract", {}))
    default = ProfileContract()
    return ProfileContract(
        density_core_m3=float(values.get("density_core_m3", default.density_core_m3)),
        density_edge_m3=float(values.get("density_edge_m3", default.density_edge_m3)),
        temperature_core_ev=float(
            values.get("temperature_core_ev", default.temperature_core_ev)
        ),
        temperature_edge_ev=float(
            values.get("temperature_edge_ev", default.temperature_edge_ev)
        ),
        density_power=int(values.get("density_power", default.density_power)),
        temperature_power=int(
            values.get("temperature_power", default.temperature_power)
        ),
        zeff=float(values.get("zeff", default.zeff)),
    )


def _grid_from_payload(payload: dict[str, Any]) -> GridSpec:
    grid = payload.get("inputs", {}).get("ntx_grid", {})
    return GridSpec(
        int(grid.get("n_theta", 25)),
        int(grid.get("n_zeta", 31)),
        int(grid.get("n_xi", 24)),
    )


def _stress_rho_from_reference(payload: dict[str, Any]) -> float:
    comparison = payload["comparison"]
    rho = np.asarray(comparison["rho"], dtype=float)
    error = np.asarray(comparison["relative_error_total_vs_redl"], dtype=float)
    return float(rho[int(np.nanargmax(error))])


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _dominant_channel(values: dict[str, float]) -> str:
    if not values:
        return "none"
    return max(values, key=lambda key: abs(float(values[key])))


def _assemble_dense_species_matrix(blocks: np.ndarray) -> np.ndarray:
    array = np.asarray(blocks, dtype=float)
    return np.transpose(array, (0, 2, 1, 3)).reshape(
        array.shape[0] * array.shape[2],
        array.shape[1] * array.shape[3],
    )


def _relative_scalar_error(candidate: float, reference: float) -> float:
    return float(abs(float(candidate) - float(reference)) / max(abs(float(reference)), EPS))


def _relative_scalar_error_or_none(
    candidate: float | None,
    reference: float | None,
) -> float | None:
    if candidate is None or reference is None:
        return None
    if not np.isfinite(float(candidate)) or not np.isfinite(float(reference)):
        return None
    return _relative_scalar_error(float(candidate), float(reference))


def _redl_effective_channel_targets(
    payload: dict[str, Any],
    rho: float,
) -> dict[str, float]:
    """Return Redl density/effective-temperature/parallel target channels."""

    redl = payload.get("redl", {})
    redl_rho = np.asarray(redl.get("rho", []), dtype=float)
    if redl_rho.size == 0:
        return {}

    def interp_key(key: str) -> float | None:
        values = redl.get(key)
        if values is None:
            return None
        array = np.asarray(values, dtype=float)
        if array.size != redl_rho.size:
            return None
        return float(_interp(redl_rho, array, np.asarray([rho], dtype=float))[0])

    density = interp_key("density_gradient_term_over_root_fsab2")
    electron_temperature = interp_key("electron_temperature_gradient_term_over_root_fsab2")
    ion_temperature = interp_key("ion_temperature_gradient_term_over_root_fsab2")
    temperature = interp_key("temperature_gradient_term_over_root_fsab2")
    if temperature is None and electron_temperature is not None and ion_temperature is not None:
        temperature = electron_temperature + ion_temperature
    channels = {
        "density_electric_force": density,
        "effective_temperature_force": temperature,
        "parallel_electric_force": 0.0,
    }
    return {
        label: float(value)
        for label, value in channels.items()
        if value is not None and np.isfinite(float(value))
    }


def _channel_response_ratios(
    candidate_by_channel: dict[str, float],
    target_by_channel: dict[str, float],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    multipliers: dict[str, float | None] = {}
    relative_errors: dict[str, float | None] = {}
    for label in EFFECTIVE_LABELS:
        candidate = candidate_by_channel.get(label)
        target = target_by_channel.get(label)
        if (
            candidate is None
            or target is None
            or not np.isfinite(float(candidate))
            or not np.isfinite(float(target))
            or abs(float(candidate)) <= EPS
        ):
            multipliers[label] = None
        else:
            multipliers[label] = float(target) / float(candidate)
        relative_errors[label] = _relative_scalar_error_or_none(candidate, target)
    return multipliers, relative_errors


def _effective_projection_and_drives(
    projection: np.ndarray,
    drives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rewrite A1/A2 source columns into physical density/electric and T channels."""

    source_projection = np.asarray(projection, dtype=float)
    force = np.asarray(drives, dtype=float)
    effective_projection = np.stack(
        [
            source_projection[:, 0],
            source_projection[:, 1] - 1.5 * source_projection[:, 0],
            source_projection[:, 2],
        ],
        axis=1,
    )
    effective_drives = np.asarray(
        [force[0] + 1.5 * force[1], force[1], force[2]],
        dtype=float,
    )
    return effective_projection, effective_drives


def _source_contributions_by_channel(
    projection_by_species: np.ndarray,
    drives_by_species: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Return RHS contributions shaped `(species, moment, channel)`."""

    projection = np.asarray(projection_by_species, dtype=float)
    drives = np.asarray(drives_by_species, dtype=float)
    if mode == "transport":
        labels = TRANSPORT_LABELS
        active_projection = projection
        active_drives = drives
    elif mode == "effective":
        labels = EFFECTIVE_LABELS
        pieces = [
            _effective_projection_and_drives(projection[index], drives[index])
            for index in range(projection.shape[0])
        ]
        active_projection = np.stack([item[0] for item in pieces])
        active_drives = np.stack([item[1] for item in pieces])
    else:
        raise ValueError(f"unknown source-decomposition mode {mode!r}")
    rhs = -active_projection * active_drives[:, None, :]
    return rhs, labels, active_drives


def _copy_nonfinite_radial_boundaries(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    if array.shape[1] < 2:
        return array
    first = np.where(np.isfinite(array[:, 0, ...]), array[:, 0, ...], array[:, 1, ...])
    last = np.where(np.isfinite(array[:, -1, ...]), array[:, -1, ...], array[:, -2, ...])
    array[:, 0, ...] = first
    array[:, -1, ...] = last
    return array


def _load_or_build_scan(
    *,
    bootstrap_payload: dict[str, Any],
    case: Any,
    field: Any,
    scan_grid: GridSpec,
    output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    inputs = bootstrap_payload["inputs"]
    hdf5_payload = bootstrap_payload.get("ntx_neopax", {}).get("hdf5") or {}
    preferred_path = Path(str(hdf5_payload.get("path", ""))).expanduser()
    if preferred_path.exists() and not neopax_scan_requires_rebuild(preferred_path):
        start = time.perf_counter()
        scan = load_neopax_reference_scan(preferred_path)
        return scan, {
            "scan_source": "cached_hdf5",
            "scan_path": str(preferred_path),
            "scan_load_seconds": float(time.perf_counter() - start),
        }

    scan_rho = np.asarray(inputs["scan_rho"], dtype=float)
    nu_v = np.asarray(inputs["nu_v"], dtype=float)
    es_values = np.asarray(inputs["Es"], dtype=float)
    drds = _drds_from_minor_radius(scan_rho, float(field.a_b))
    start = time.perf_counter()
    scan = _build_scan_for_path(
        case,
        rho=scan_rho,
        nu_v=nu_v,
        es_values=es_values,
        drds=drds,
        grid=scan_grid,
        path_key="booz_xform_jax",
        mboz=int(inputs.get("mboz", DEFAULT_MBOZ)),
        nboz=int(inputs.get("nboz", DEFAULT_NBOZ)),
        min_bmn_to_load=float(inputs.get("min_bmn_to_load", 1.0e-5)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_path = output_dir / f"{case.id}_source_channel_scan.h5"
    write_neopax_scan_hdf5(scan, scan_path)
    return scan, {
        "scan_source": "rebuilt",
        "scan_path": str(scan_path),
        "scan_build_seconds": float(time.perf_counter() - start),
    }


def _momentum_blocks(
    species: Any,
    neopax_grid: Any,
    field: Any,
    database: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import jax
    from NEOPAX._neoclassical import get_Lij_matrix_with_momentum_correction

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
    return (
        _copy_nonfinite_radial_boundaries(np.asarray(lij_full, dtype=float)),
        _copy_nonfinite_radial_boundaries(np.asarray(eij_full, dtype=float)),
        _copy_nonfinite_radial_boundaries(np.asarray(nu_weighted_average, dtype=float)),
    )


def _solve_channels(
    *,
    species: Any,
    neopax_grid: Any,
    field: Any,
    radial_index: int,
    lij_full: np.ndarray,
    eij_full: np.ndarray,
    nu_weighted_average: np.ndarray,
    redl_current: float,
    redl_effective_targets: dict[str, float],
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from NEOPAX._moments import build_source_projection
    from NEOPAX._neoclassical import get_Collision_Operator_terms, get_Matrix

    species_count = int(np.asarray(species.species_indeces).size)
    moment_order = int(neopax_grid.n_order)
    cm_ab, cn_ab, tau = jax.vmap(
        jax.vmap(get_Collision_Operator_terms, in_axes=(None, None, None, 0, None)),
        in_axes=(None, None, 0, None, None),
    )(
        species,
        neopax_grid,
        species.species_indeces,
        species.species_indeces,
        int(radial_index),
    )
    blocks = jax.vmap(
        get_Matrix,
        in_axes=(None, None, None, 0, None, 0, 0, None, None, None),
    )(
        species,
        neopax_grid,
        field,
        species.species_indeces,
        int(radial_index),
        jnp.asarray(lij_full[:, radial_index, :, :]),
        jnp.asarray(eij_full[:, radial_index, :, :]),
        cm_ab,
        cn_ab,
        tau,
    )
    dense_matrix = _assemble_dense_species_matrix(np.asarray(blocks, dtype=float))
    projection_by_species = np.stack(
        [
            np.asarray(
                build_source_projection(
                    jnp.asarray(lij_full[index, radial_index, 2 : 2 + moment_order, 0:3]),
                    moment_order,
                ),
                dtype=float,
            )
            for index in range(species_count)
        ],
        axis=0,
    )
    drives_by_species = np.stack(
        [
            np.asarray(
                [
                    species.A1[index, radial_index],
                    species.A2[index, radial_index],
                    species.A3[radial_index],
                ],
                dtype=float,
            )
            for index in range(species_count)
        ],
        axis=0,
    )
    density = np.asarray(species.density[:, radial_index], dtype=float)
    charge_qp = np.asarray(species.charge_qp, dtype=float)
    root_fsab2 = float(
        abs(float(np.asarray(field.B0[radial_index], dtype=float)))
        * np.sqrt(abs(float(np.asarray(field.Bsqav[radial_index], dtype=float))))
    )
    observable_bridge = -abs(float(np.asarray(field.B0[radial_index], dtype=float)))

    def current_from_solution(solution: np.ndarray) -> tuple[np.ndarray, float]:
        solved = np.asarray(solution, dtype=float).reshape(species_count, moment_order)
        raw_species_current = charge_qp * elementary_charge * density * solved[:, 0]
        observable_species = observable_bridge * raw_species_current / max(root_fsab2, EPS)
        return observable_species, float(np.sum(observable_species))

    full_rhs_blocks = -np.sum(projection_by_species * drives_by_species[:, None, :], axis=2)
    full_rhs_vector = full_rhs_blocks.reshape(species_count * moment_order)
    full_solution = np.linalg.solve(dense_matrix, full_rhs_vector)
    full_species_current, full_total_current = current_from_solution(full_solution)

    source_payloads: dict[str, Any] = {}
    for mode in ("transport", "effective"):
        rhs_by_channel, labels, active_drives = _source_contributions_by_channel(
            projection_by_species,
            drives_by_species,
            mode=mode,
        )
        species_currents: list[np.ndarray] = []
        total_currents: list[float] = []
        residual_norms: list[float] = []
        for channel_index in range(rhs_by_channel.shape[2]):
            rhs_vector = rhs_by_channel[:, :, channel_index].reshape(
                species_count * moment_order
            )
            solution = np.linalg.solve(dense_matrix, rhs_vector)
            residual = dense_matrix @ solution - rhs_vector
            residual_norms.append(
                float(np.linalg.norm(residual) / max(np.linalg.norm(rhs_vector), EPS))
            )
            species_current, total_current = current_from_solution(solution)
            species_currents.append(species_current)
            total_currents.append(total_current)
        current_by_channel = {
            label: float(value)
            for label, value in zip(labels, total_currents, strict=True)
        }
        species_by_channel = {
            label: np.asarray(values, dtype=float).tolist()
            for label, values in zip(labels, species_currents, strict=True)
        }
        source_payloads[mode] = {
            "labels": list(labels),
            "drives_by_species": active_drives.tolist(),
            "current_by_channel_over_root_fsab2": current_by_channel,
            "species_current_by_channel_over_root_fsab2": species_by_channel,
            "channel_sum_over_root_fsab2": float(np.sum(total_currents)),
            "dominant_channel": _dominant_channel(current_by_channel),
            "max_channel_residual_norm": float(np.max(residual_norms)),
        }

    effective_currents = source_payloads["effective"][
        "current_by_channel_over_root_fsab2"
    ]
    response_multipliers, response_relative_errors = _channel_response_ratios(
        effective_currents,
        redl_effective_targets,
    )
    no_momentum_effective: dict[str, float] = dict.fromkeys(EFFECTIVE_LABELS, 0.0)
    no_momentum_transport: dict[str, float] = dict.fromkeys(TRANSPORT_LABELS, 0.0)
    for index in range(species_count):
        row3 = np.asarray(lij_full[index, radial_index, 2, 0:3], dtype=float)
        drives = drives_by_species[index]
        transport_upar = -density[index] * row3 * drives
        transport_current = (
            observable_bridge
            * charge_qp[index]
            * elementary_charge
            * transport_upar
            / max(root_fsab2, EPS)
        )
        effective_projection, effective_drives = _effective_projection_and_drives(
            row3[None, :],
            drives,
        )
        effective_upar = -density[index] * effective_projection.reshape(3) * effective_drives
        effective_current = (
            observable_bridge
            * charge_qp[index]
            * elementary_charge
            * effective_upar
            / max(root_fsab2, EPS)
        )
        for label, value in zip(TRANSPORT_LABELS, transport_current, strict=True):
            no_momentum_transport[label] += float(value)
        for label, value in zip(EFFECTIVE_LABELS, effective_current, strict=True):
            no_momentum_effective[label] += float(value)

    channel_sum = float(source_payloads["effective"]["channel_sum_over_root_fsab2"])
    species_l1 = float(np.sum(np.abs(full_species_current)))
    net_current = float(full_total_current)
    return {
        "radial_index": int(radial_index),
        "rho": float(np.asarray(field.rho_grid, dtype=float)[radial_index]),
        "moment_order": int(moment_order),
        "matrix_condition_number": float(np.linalg.cond(dense_matrix)),
        "full_rhs_norm": float(np.linalg.norm(full_rhs_vector)),
        "full_solution_residual_norm": float(
            np.linalg.norm(dense_matrix @ full_solution - full_rhs_vector)
            / max(np.linalg.norm(full_rhs_vector), EPS)
        ),
        "redl_current_over_root_fsab2": float(redl_current),
        "full_solve_current_over_root_fsab2": net_current,
        "full_solve_species_current_over_root_fsab2": full_species_current.tolist(),
        "full_solve_relative_error_vs_redl": _relative_scalar_error(
            net_current,
            redl_current,
        ),
        "species_cancellation_factor": float(species_l1 / max(abs(net_current), EPS)),
        "source_decomposition": source_payloads,
        "redl_effective_channel_current_by_channel_over_root_fsab2": {
            label: float(redl_effective_targets[label])
            for label in EFFECTIVE_LABELS
            if label in redl_effective_targets
        },
        "effective_channel_response_multiplier_to_redl": response_multipliers,
        "effective_channel_relative_error_vs_redl": response_relative_errors,
        "effective_temperature_response_multiplier_to_redl": _finite_or_none(
            response_multipliers.get("effective_temperature_force")
        ),
        "effective_temperature_channel_relative_error_vs_redl": _finite_or_none(
            response_relative_errors.get("effective_temperature_force")
        ),
        "redl_effective_temperature_fraction_of_total": (
            abs(float(redl_effective_targets["effective_temperature_force"]))
            / max(abs(float(redl_current)), EPS)
            if "effective_temperature_force" in redl_effective_targets
            else None
        ),
        "redl_dominant_effective_channel": _dominant_channel(redl_effective_targets),
        "no_momentum_transport_current_by_channel_over_root_fsab2": no_momentum_transport,
        "no_momentum_effective_current_by_channel_over_root_fsab2": no_momentum_effective,
        "source_channel_superposition_relative_residual": _relative_scalar_error(
            channel_sum,
            net_current,
        ),
        "effective_temperature_fraction_of_total": float(
            abs(effective_currents["effective_temperature_force"])
            / max(abs(net_current), EPS)
        ),
        "density_electric_fraction_of_total": float(
            abs(effective_currents["density_electric_force"])
            / max(abs(net_current), EPS)
        ),
        "parallel_electric_fraction_of_total": float(
            abs(effective_currents["parallel_electric_force"])
            / max(abs(net_current), EPS)
        ),
        "dominant_effective_channel": _dominant_channel(effective_currents),
    }


def _evaluate_setting(
    *,
    NEOPAX: Any,
    species: Any,
    field: Any,
    database: Any,
    neopax_x: int,
    n_order: int,
    radial_index: int,
    redl_current: float,
    redl_effective_targets: dict[str, float],
) -> dict[str, Any]:
    start = time.perf_counter()
    neopax_grid = NEOPAX.Grid.create_standard(
        int(field.n_r),
        int(neopax_x),
        2,
        n_order=int(n_order),
    )
    lij_full, eij_full, nu_weighted_average = _momentum_blocks(
        species,
        neopax_grid,
        field,
        database,
    )
    block_seconds = float(time.perf_counter() - start)

    solve_start = time.perf_counter()
    decomposition = _solve_channels(
        species=species,
        neopax_grid=neopax_grid,
        field=field,
        radial_index=radial_index,
        lij_full=lij_full,
        eij_full=eij_full,
        nu_weighted_average=nu_weighted_average,
        redl_current=redl_current,
        redl_effective_targets=redl_effective_targets,
    )
    solve_seconds = float(time.perf_counter() - solve_start)

    public_start = time.perf_counter()
    public_closure = _evaluate_neopax_currents(
        NEOPAX,
        species=species,
        field=field,
        database=database,
        neopax_x=int(neopax_x),
        n_order=int(n_order),
    )
    public_seconds = float(time.perf_counter() - public_start)
    public_total = float(
        np.asarray(public_closure["current_total_over_root_fsab2"], dtype=float)[
            radial_index
        ]
    )
    public_nomom = float(
        np.asarray(public_closure["current_nomom_over_root_fsab2"], dtype=float)[
            radial_index
        ]
    )
    public_correction = float(public_total - public_nomom)

    return {
        "neopax_x": int(neopax_x),
        "n_order": int(n_order),
        "x_to_order_ratio": float(neopax_x / max(n_order, 1)),
        **decomposition,
        "public_neopax_current_over_root_fsab2": public_total,
        "public_neopax_nomom_over_root_fsab2": public_nomom,
        "public_neopax_correction_over_root_fsab2": public_correction,
        "public_neopax_relative_error_vs_redl": _relative_scalar_error(
            public_total,
            redl_current,
        ),
        "full_vs_public_relative_difference": _relative_scalar_error(
            decomposition["full_solve_current_over_root_fsab2"],
            public_total,
        ),
        "timings": {
            "momentum_blocks_seconds": block_seconds,
            "source_solve_seconds": solve_seconds,
            "public_closure_seconds": public_seconds,
        },
    }


def _summary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("source-channel audit requires at least one row")
    high_stable = max(
        rows,
        key=lambda row: (
            row["x_to_order_ratio"] >= 1.0,
            row["neopax_x"],
            row["n_order"],
        ),
    )
    max_reconstruction_residual = max(
        float(row["source_channel_superposition_relative_residual"]) for row in rows
    )
    max_public_difference = max(float(row["full_vs_public_relative_difference"]) for row in rows)
    stress_errors = [float(row["public_neopax_relative_error_vs_redl"]) for row in rows]
    best_row = rows[int(np.argmin(stress_errors))]
    return {
        "row_count": int(len(rows)),
        "source_reconstruction_gate": SOURCE_RECONSTRUCTION_GATE,
        "source_channel_superposition_gate_pass": bool(
            max_reconstruction_residual <= SOURCE_RECONSTRUCTION_GATE
        ),
        "max_source_channel_superposition_relative_residual": float(
            max_reconstruction_residual
        ),
        "max_full_vs_public_relative_difference": float(max_public_difference),
        "best_public_relative_error_vs_redl": float(
            best_row["public_neopax_relative_error_vs_redl"]
        ),
        "best_public_neopax_x": int(best_row["neopax_x"]),
        "best_public_n_order": int(best_row["n_order"]),
        "profile_current_gate": PROFILE_CURRENT_GATE,
        "best_public_current_gate_pass": bool(
            float(best_row["public_neopax_relative_error_vs_redl"]) <= PROFILE_CURRENT_GATE
        ),
        "high_stable_neopax_x": int(high_stable["neopax_x"]),
        "high_stable_n_order": int(high_stable["n_order"]),
        "high_stable_public_relative_error_vs_redl": float(
            high_stable["public_neopax_relative_error_vs_redl"]
        ),
        "high_stable_dominant_effective_channel": str(
            high_stable["dominant_effective_channel"]
        ),
        "high_stable_effective_temperature_fraction_of_total": float(
            high_stable["effective_temperature_fraction_of_total"]
        ),
        "high_stable_density_electric_fraction_of_total": float(
            high_stable["density_electric_fraction_of_total"]
        ),
        "high_stable_parallel_electric_fraction_of_total": float(
            high_stable["parallel_electric_fraction_of_total"]
        ),
        "high_stable_species_cancellation_factor": float(
            high_stable["species_cancellation_factor"]
        ),
        "high_stable_effective_temperature_response_multiplier_to_redl": (
            _finite_or_none(
                high_stable.get("effective_temperature_response_multiplier_to_redl")
            )
        ),
        "high_stable_effective_temperature_channel_relative_error_vs_redl": (
            _finite_or_none(
                high_stable.get("effective_temperature_channel_relative_error_vs_redl")
            )
        ),
        "high_stable_redl_effective_temperature_fraction_of_total": (
            _finite_or_none(high_stable.get("redl_effective_temperature_fraction_of_total"))
        ),
        "best_effective_temperature_response_multiplier_to_redl": _finite_or_none(
            best_row.get("effective_temperature_response_multiplier_to_redl")
        ),
    }


def build_payload(
    *,
    bootstrap_json: Path = BOOTSTRAP_JSON,
    settings: tuple[tuple[int, int], ...] = DEFAULT_SETTINGS,
    output_dir: Path = WORKDIR,
) -> dict[str, Any]:
    *_, NEOPAX = _require_external_stacks()
    bootstrap_payload = _load_json(bootstrap_json)
    inputs = bootstrap_payload["inputs"]
    case = _case_by_id(str(bootstrap_payload.get("case", {}).get("id", DEFAULT_CASE)))
    contract = _contract_from_payload(bootstrap_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    mboz = int(inputs.get("mboz", DEFAULT_MBOZ))
    nboz = int(inputs.get("nboz", DEFAULT_NBOZ))
    boozmn_path = _write_boozmn(case, output_dir, mboz=mboz, nboz=nboz)
    field = NEOPAX.Field.read_vmec_booz(
        int(inputs.get("field_radial_points", 15)),
        str(case.wout_path),
        str(boozmn_path),
    )
    species = _build_species(NEOPAX, field, contract)
    scan_grid = _grid_from_payload(bootstrap_payload)
    scan, scan_metadata = _load_or_build_scan(
        bootstrap_payload=bootstrap_payload,
        case=case,
        field=field,
        scan_grid=scan_grid,
        output_dir=output_dir,
    )
    database = to_neopax_monoenergetic(
        scan,
        a_b=float(field.a_b),
        d33_mode=str(inputs.get("d33_mode", "spitzer")),
    )
    stress_rho = _stress_rho_from_reference(bootstrap_payload)
    rho_field = np.asarray(field.rho_grid, dtype=float)
    radial_index = int(np.argmin(np.abs(rho_field - stress_rho)))
    comparison_rho = np.asarray(bootstrap_payload["comparison"]["rho"], dtype=float)
    comparison_redl = np.asarray(
        bootstrap_payload["comparison"]["redl_current_over_root_fsab2"],
        dtype=float,
    )
    redl_current = float(_interp(comparison_rho, comparison_redl, np.asarray([stress_rho]))[0])
    redl_effective_targets = _redl_effective_channel_targets(
        bootstrap_payload,
        stress_rho,
    )

    rows = [
        _evaluate_setting(
            NEOPAX=NEOPAX,
            species=species,
            field=field,
            database=database,
            neopax_x=int(neopax_x),
            n_order=int(n_order),
            radial_index=radial_index,
            redl_current=redl_current,
            redl_effective_targets=redl_effective_targets,
        )
        for neopax_x, n_order in settings
    ]
    metrics = _summary_metrics(rows)
    conclusion = (
        "The finite-beta stress current is linear in the frozen source channels "
        "to numerical precision, so the channel audit checks the actual "
        "momentum-restoring system rather than fitting a residual.  At the "
        "quadrature-stable high-order setting the dominant physical drive is "
        f"{metrics['high_stable_dominant_effective_channel']}; the parallel-"
        "electric channel is zero for this profile contract.  The remaining "
        "current gap is therefore localized to the reduced source-channel "
        "response inside the profile-current closure; the Redl density and "
        "temperature source terms are stored as target channels rather than "
        "used as a fitted runtime correction.  The gap is not a hidden additive "
        "normalization or under-integrated apparent pass."
    )
    return _to_jsonable(
        {
            "benchmark": "owned_finite_beta_source_channel_audit",
            "classification": "owned finite-beta source-channel closure audit",
            "claim_scope": (
                "Freezes the owned finite-beta VMEC/Boozer geometry, analytic "
                "profiles, NTX monoenergetic scan, D33 branch, velocity "
                "quadrature, and Sonine order, then solves the same "
                "momentum-restoring linear system with one physical source "
                "channel at a time.  This is a source-localization stress "
                "diagnostic, not a fitted correction and not a finite-beta "
                "bootstrap-current parity claim."
            ),
            "case": case.as_payload(),
            "profile_contract": contract.as_payload(),
            "inputs": {
                "bootstrap_artifact": str(bootstrap_json),
                "settings": [
                    {"neopax_x": int(neopax_x), "n_order": int(n_order)}
                    for neopax_x, n_order in settings
                ],
                "stress_rho": float(stress_rho),
                "radial_index": int(radial_index),
                "field_radial_points": int(inputs.get("field_radial_points", 15)),
                "mboz": mboz,
                "nboz": nboz,
                "redl_ntheta": int(inputs.get("redl_ntheta", DEFAULT_REDL_NTHETA)),
                "d33_mode": str(inputs.get("d33_mode", "spitzer")),
                "redl_effective_channel_targets_over_root_fsab2": redl_effective_targets,
                "ntx_grid": {
                    "n_theta": int(scan_grid.n_theta),
                    "n_zeta": int(scan_grid.n_zeta),
                    "n_xi": int(scan_grid.n_xi),
                },
                **scan_metadata,
            },
            "rows": rows,
            "summary_metrics": metrics,
            "conclusion": conclusion,
            "open_work": [
                (
                    "derive or import a quadrature-converged profile-current "
                    "closure that improves the dominant source-channel response "
                    "without converting the Redl channel response ratio into a "
                    "runtime fit and without regressing fixed-field QA/QH or "
                    "integrated W7-X"
                ),
                (
                    "connect this source-channel localization to same-grid "
                    "finite-beta profile-current diagnostics before promoting "
                    "a finite-beta bootstrap-current parity claim"
                ),
                (
                    "keep interpolation-mode comparisons planned until the "
                    "downstream interface exposes stable general and legacy "
                    "selectors"
                ),
            ],
            "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
            "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
        }
    )


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    rows = payload["rows"]
    metrics = payload["summary_metrics"]
    setting_labels = [f"X={row['neopax_x']}, P={row['n_order']}" for row in rows]
    channel_values = np.asarray(
        [
            [
                row["source_decomposition"]["effective"][
                    "current_by_channel_over_root_fsab2"
                ][label]
                / 1.0e6
                for label in EFFECTIVE_LABELS
            ]
            for row in rows
        ],
        dtype=float,
    )
    redl_channel_targets = np.asarray(
        [
            [
                row.get(
                    "redl_effective_channel_current_by_channel_over_root_fsab2",
                    {},
                ).get(label, np.nan)
                / 1.0e6
                for label in EFFECTIVE_LABELS
            ]
            for row in rows
        ],
        dtype=float,
    )
    public_total = np.asarray(
        [row["public_neopax_current_over_root_fsab2"] for row in rows],
        dtype=float,
    ) / 1.0e6
    public_nomom = np.asarray(
        [row["public_neopax_nomom_over_root_fsab2"] for row in rows],
        dtype=float,
    ) / 1.0e6
    redl = np.asarray([row["redl_current_over_root_fsab2"] for row in rows], dtype=float) / 1.0e6
    public_error = np.asarray(
        [row["public_neopax_relative_error_vs_redl"] for row in rows],
        dtype=float,
    )
    reconstruction = np.asarray(
        [row["source_channel_superposition_relative_residual"] for row in rows],
        dtype=float,
    )
    condition = np.asarray([row["matrix_condition_number"] for row in rows], dtype=float)

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.size": 10.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    ax_channel, ax_current, ax_check, ax_fraction = axes.ravel()

    x = np.arange(len(rows))
    width = 0.22
    colors = ("#0072b2", "#d55e00", "#009e73")
    display_labels = (
        "density/electric",
        "temperature",
        "parallel electric",
    )
    for index, (label, color) in enumerate(zip(display_labels, colors, strict=True)):
        offset = x + (index - 1) * width
        ax_channel.bar(
            offset,
            channel_values[:, index],
            width=width,
            color=color,
            label=label,
        )
        finite_target = np.isfinite(redl_channel_targets[:, index])
        if np.any(finite_target):
            ax_channel.scatter(
                offset[finite_target],
                redl_channel_targets[finite_target, index],
                marker="x",
                s=54,
                linewidths=1.8,
                color="0.05",
                label="Redl channel target" if index == 0 else None,
                zorder=4,
            )
    ax_channel.axhline(0.0, color="0.35", lw=0.8)
    ax_channel.set_xticks(x, setting_labels)
    ax_channel.set_ylabel(r"current contribution [MA m$^{-2}$]")
    ax_channel.set_title("(a) Corrected source-channel current")
    ax_channel.legend(fontsize=8.2)

    ax_current.plot(x, redl, color="#009e73", marker="o", lw=2.0, label="Redl")
    ax_current.plot(
        x,
        public_nomom,
        color="#0072b2",
        marker="s",
        lw=1.8,
        ls="--",
        label="no momentum",
    )
    ax_current.plot(x, public_total, color="#d55e00", marker="^", lw=1.8, label="corrected")
    ax_current.set_xticks(x, setting_labels)
    ax_current.set_ylabel(r"current [MA m$^{-2}$]")
    ax_current.set_title("(b) Stress-radius current")
    ax_current.legend(fontsize=8.2)

    ax_check.semilogy(
        x,
        public_error,
        color="#d55e00",
        marker="o",
        lw=1.8,
        label="current difference",
    )
    ax_check.semilogy(
        x,
        reconstruction,
        color="#0072b2",
        marker="s",
        lw=1.8,
        label="channel reconstruction",
    )
    ax_check.axhline(PROFILE_CURRENT_GATE, color="0.25", ls="--", lw=1.0)
    ax_check.axhline(SOURCE_RECONSTRUCTION_GATE, color="0.45", ls=":", lw=1.0)
    ax_check.set_xticks(x, setting_labels)
    ax_check.set_ylabel("relative value")
    ax_check.set_title("(c) Physics gate and closure stress")
    ax_check.legend(fontsize=8.0)

    fractions = np.abs(channel_values) / np.maximum(np.abs(public_total[:, None]), EPS)
    bottom = np.zeros(len(rows))
    for index, (label, color) in enumerate(zip(display_labels, colors, strict=True)):
        ax_fraction.bar(x, fractions[:, index], bottom=bottom, color=color, label=label)
        bottom += fractions[:, index]
    ax_fraction_t = ax_fraction.twinx()
    ax_fraction_t.plot(
        x,
        condition,
        color="0.15",
        marker="D",
        lw=1.4,
        label="condition",
    )
    ax_fraction_t.set_yscale("log")
    ax_fraction.set_xticks(x, setting_labels)
    ax_fraction.set_ylabel("|channel| / |net current|")
    ax_fraction_t.set_ylabel("matrix condition number")
    ax_fraction.set_title("(d) Channel leverage and conditioning")

    dominant_display = str(
        metrics["high_stable_dominant_effective_channel"]
    ).replace("_", " ")
    fig.suptitle(
        "Owned finite-beta source-channel closure audit "
        f"(dominant high-order channel: {dominant_display})",
        fontsize=13,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _parse_settings(values: list[str]) -> tuple[tuple[int, int], ...]:
    settings: list[tuple[int, int]] = []
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError("settings must be formatted as X:P")
        x_value, p_value = value.split(":", 1)
        settings.append((int(x_value), int(p_value)))
    return tuple(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-json", type=Path, default=BOOTSTRAP_JSON)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=[f"{x}:{p}" for x, p in DEFAULT_SETTINGS],
        help="Closure settings formatted as X:P, for example 18:18.",
    )
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=WORKDIR)
    args = parser.parse_args()

    payload = build_payload(
        bootstrap_json=args.bootstrap_json,
        settings=_parse_settings([str(value) for value in args.settings]),
        output_dir=args.output_dir,
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(json.dumps(payload["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
