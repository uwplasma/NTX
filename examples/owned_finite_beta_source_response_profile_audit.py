#!/usr/bin/env python3
"""Map finite-beta source-channel response across the radial profile.

The stress-radius source-channel audit localizes the remaining finite-beta
bootstrap-current gap to the effective temperature-gradient source response.
This diagnostic keeps the same owned finite-beta contract and extends that
measurement across the profile.  It is deliberately a physics-localization
artifact: it records how the Redl target response relates to the frozen
momentum-restoring source response as a function of radius, collisionality, and
geometry factors, without applying a fitted runtime correction.
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

from examples.owned_finite_beta_bootstrap_comparison import (  # noqa: E402
    DEFAULT_CASE,
    DEFAULT_MBOZ,
    DEFAULT_NBOZ,
    DEFAULT_REDL_NTHETA,
    EPS,
    _build_species,
    _case_by_id,
    _evaluate_neopax_currents,
    _interp,
    _require_external_stacks,
    _to_jsonable,
    _write_boozmn,
)
from examples.owned_finite_beta_source_channel_audit import (  # noqa: E402
    BOOTSTRAP_JSON,
    PROFILE_CURRENT_GATE,
    SOURCE_RECONSTRUCTION_GATE,
    _contract_from_payload,
    _grid_from_payload,
    _load_json,
    _load_or_build_scan,
    _momentum_blocks,
    _redl_effective_channel_targets,
    _relative_scalar_error,
    _solve_channels,
)
from ntx import to_neopax_monoenergetic  # noqa: E402

OUTPUT_PREFIX = (
    ROOT / "docs" / "_static" / "owned_finite_beta_source_response_profile_audit"
)
WORKDIR = ROOT / "examples" / "outputs" / "owned_finite_beta_source_response_profile_audit"
DEFAULT_SETTINGS = ((18, 18),)


def _parse_settings(values: list[str]) -> tuple[tuple[int, int], ...]:
    settings: list[tuple[int, int]] = []
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError("settings must be formatted as X:P")
        x_value, p_value = value.split(":", 1)
        settings.append((int(x_value), int(p_value)))
    return tuple(settings)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _finite_values(values: list[float | None]) -> np.ndarray:
    array = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )
    return array


def _interp_redl_key(
    payload: dict[str, Any],
    key: str,
    rho: float,
) -> float | None:
    redl = payload.get("redl", {})
    redl_rho = np.asarray(redl.get("rho", []), dtype=float)
    values = redl.get(key)
    if redl_rho.size == 0 or values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.size != redl_rho.size:
        return None
    value = float(_interp(redl_rho, array, np.asarray([rho], dtype=float))[0])
    return value if np.isfinite(value) else None


def _redl_profile_drivers(
    bootstrap_payload: dict[str, Any],
    *,
    rho: float,
) -> dict[str, float | None]:
    drivers: dict[str, float | None] = {}
    for key in (
        "epsilon",
        "trapped_fraction",
        "L31",
        "L32",
        "alpha",
        "nu_e_star",
        "nu_i_star",
        "density_gradient_term_over_root_fsab2",
        "temperature_gradient_term_over_root_fsab2",
        "electron_temperature_gradient_term_over_root_fsab2",
        "ion_temperature_gradient_term_over_root_fsab2",
    ):
        drivers[key] = _interp_redl_key(bootstrap_payload, key, rho)
    nu_e_star = drivers.get("nu_e_star")
    drivers["log10_nu_e_star"] = (
        float(np.log10(nu_e_star))
        if nu_e_star is not None and np.isfinite(nu_e_star) and nu_e_star > 0.0
        else None
    )
    return drivers


def _correlation(
    rows: list[dict[str, Any]],
    *,
    driver_key: str,
    response_key: str = "effective_temperature_response_multiplier_to_redl",
) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        driver = row.get("redl_profile_drivers", {}).get(driver_key)
        response = row.get(response_key)
        if (
            driver is not None
            and response is not None
            and np.isfinite(float(driver))
            and np.isfinite(float(response))
        ):
            pairs.append((float(driver), float(response)))
    if len(pairs) < 3:
        return None
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _high_order_setting(rows: list[dict[str, Any]]) -> tuple[int, int]:
    high = max(
        rows,
        key=lambda row: (
            row["x_to_order_ratio"] >= 1.0,
            row["neopax_x"],
            row["n_order"],
        ),
    )
    return int(high["neopax_x"]), int(high["n_order"])


def _rows_for_setting(
    rows: list[dict[str, Any]],
    *,
    setting: tuple[int, int],
) -> list[dict[str, Any]]:
    x_value, p_value = setting
    return sorted(
        [
            row
            for row in rows
            if int(row["neopax_x"]) == x_value and int(row["n_order"]) == p_value
        ],
        key=lambda row: float(row["rho"]),
    )


def _summary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("profile source-response audit requires at least one row")
    high_setting = _high_order_setting(rows)
    high_rows = _rows_for_setting(rows, setting=high_setting)
    multipliers = _finite_values(
        [
            row.get("effective_temperature_response_multiplier_to_redl")
            for row in high_rows
        ]
    )
    response_errors = _finite_values(
        [
            row.get("effective_temperature_channel_relative_error_vs_redl")
            for row in high_rows
        ]
    )
    reconstruction = _finite_values(
        [
            row.get("source_channel_superposition_relative_residual")
            for row in rows
        ]
    )
    public_difference = _finite_values(
        [row.get("full_vs_public_relative_difference") for row in rows]
    )
    profile_errors = _finite_values(
        [row.get("public_neopax_relative_error_vs_redl") for row in high_rows]
    )
    target_values: list[float] = []
    candidate_values: list[float] = []
    for row in high_rows:
        target = row.get(
            "redl_effective_channel_current_by_channel_over_root_fsab2",
            {},
        ).get("effective_temperature_force")
        candidate = row.get("source_decomposition", {}).get("effective", {}).get(
            "current_by_channel_over_root_fsab2",
            {},
        ).get("effective_temperature_force")
        if (
            target is not None
            and candidate is not None
            and np.isfinite(float(target))
            and np.isfinite(float(candidate))
        ):
            target_values.append(float(target))
            candidate_values.append(float(candidate))
    if profile_errors.size:
        stress_row = high_rows[int(np.nanargmax(profile_errors))]
    else:
        stress_row = high_rows[0]
    sign_agreement = (
        float(
            np.mean(
                np.sign(np.asarray(target_values, dtype=float))
                == np.sign(np.asarray(candidate_values, dtype=float))
            )
        )
        if target_values
        else None
    )
    multiplier_min = float(np.min(multipliers)) if multipliers.size else None
    multiplier_max = float(np.max(multipliers)) if multipliers.size else None
    return {
        "row_count": int(len(rows)),
        "radius_count": int(len({float(row["rho"]) for row in rows})),
        "setting_count": int(len({(int(row["neopax_x"]), int(row["n_order"])) for row in rows})),
        "high_order_neopax_x": int(high_setting[0]),
        "high_order_n_order": int(high_setting[1]),
        "source_reconstruction_gate": SOURCE_RECONSTRUCTION_GATE,
        "source_channel_superposition_gate_pass": bool(
            reconstruction.size > 0
            and float(np.max(reconstruction)) <= SOURCE_RECONSTRUCTION_GATE
        ),
        "max_source_channel_superposition_relative_residual": (
            float(np.max(reconstruction)) if reconstruction.size else None
        ),
        "max_full_vs_public_relative_difference": (
            float(np.max(public_difference)) if public_difference.size else None
        ),
        "profile_current_gate": PROFILE_CURRENT_GATE,
        "high_order_max_public_relative_error_vs_redl": (
            float(np.max(profile_errors)) if profile_errors.size else None
        ),
        "high_order_median_public_relative_error_vs_redl": (
            float(np.median(profile_errors)) if profile_errors.size else None
        ),
        "high_order_temperature_response_multiplier_min": multiplier_min,
        "high_order_temperature_response_multiplier_median": (
            float(np.median(multipliers)) if multipliers.size else None
        ),
        "high_order_temperature_response_multiplier_max": multiplier_max,
        "high_order_temperature_response_multiplier_span": (
            float(multiplier_max - multiplier_min)
            if multiplier_min is not None and multiplier_max is not None
            else None
        ),
        "high_order_temperature_response_multiplier_abs_deviation_from_one_max": (
            float(np.max(np.abs(multipliers - 1.0))) if multipliers.size else None
        ),
        "high_order_temperature_channel_relative_error_max": (
            float(np.max(response_errors)) if response_errors.size else None
        ),
        "high_order_temperature_channel_sign_agreement_fraction": sign_agreement,
        "high_order_stress_rho": float(stress_row["rho"]),
        "high_order_stress_temperature_response_multiplier": _finite_or_none(
            stress_row.get("effective_temperature_response_multiplier_to_redl")
        ),
        "temperature_response_correlation_with_log10_nu_e_star": _correlation(
            high_rows,
            driver_key="log10_nu_e_star",
        ),
        "temperature_response_correlation_with_trapped_fraction": _correlation(
            high_rows,
            driver_key="trapped_fraction",
        ),
        "temperature_response_correlation_with_epsilon": _correlation(
            high_rows,
            driver_key="epsilon",
        ),
        "temperature_response_correlation_with_redl_L32": _correlation(
            high_rows,
            driver_key="L32",
        ),
    }


def _evaluate_setting_profile(
    *,
    NEOPAX: Any,
    species: Any,
    field: Any,
    database: Any,
    bootstrap_payload: dict[str, Any],
    radial_indices: np.ndarray,
    target_rho: np.ndarray,
    redl_current: np.ndarray,
    neopax_x: int,
    n_order: int,
) -> list[dict[str, Any]]:
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
    public_total = np.asarray(
        public_closure["current_total_over_root_fsab2"],
        dtype=float,
    )
    public_nomom = np.asarray(
        public_closure["current_nomom_over_root_fsab2"],
        dtype=float,
    )
    rows: list[dict[str, Any]] = []
    solve_seconds = 0.0
    for radial_index, rho_value, redl_value in zip(
        radial_indices,
        target_rho,
        redl_current,
        strict=True,
    ):
        redl_effective_targets = _redl_effective_channel_targets(
            bootstrap_payload,
            float(rho_value),
        )
        solve_start = time.perf_counter()
        row = _solve_channels(
            species=species,
            neopax_grid=neopax_grid,
            field=field,
            radial_index=int(radial_index),
            lij_full=lij_full,
            eij_full=eij_full,
            nu_weighted_average=nu_weighted_average,
            redl_current=float(redl_value),
            redl_effective_targets=redl_effective_targets,
        )
        solve_seconds += float(time.perf_counter() - solve_start)
        row.update(
            {
                "neopax_x": int(neopax_x),
                "n_order": int(n_order),
                "x_to_order_ratio": float(neopax_x / max(n_order, 1)),
                "target_rho": float(rho_value),
                "public_neopax_current_over_root_fsab2": float(
                    public_total[int(radial_index)]
                ),
                "public_neopax_nomom_over_root_fsab2": float(
                    public_nomom[int(radial_index)]
                ),
                "public_neopax_correction_over_root_fsab2": float(
                    public_total[int(radial_index)] - public_nomom[int(radial_index)]
                ),
                "public_neopax_relative_error_vs_redl": _relative_scalar_error(
                    float(public_total[int(radial_index)]),
                    float(redl_value),
                ),
                "full_vs_public_relative_difference": _relative_scalar_error(
                    row["full_solve_current_over_root_fsab2"],
                    float(public_total[int(radial_index)]),
                ),
                "redl_profile_drivers": _redl_profile_drivers(
                    bootstrap_payload,
                    rho=float(rho_value),
                ),
                "timings": {
                    "momentum_blocks_seconds": block_seconds,
                    "source_solve_cumulative_seconds": solve_seconds,
                    "public_closure_seconds": public_seconds,
                },
            }
        )
        rows.append(row)
    return rows


def build_payload(
    *,
    bootstrap_json: Path = BOOTSTRAP_JSON,
    settings: tuple[tuple[int, int], ...] = DEFAULT_SETTINGS,
    radii: tuple[float, ...] | None = None,
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
    comparison = bootstrap_payload["comparison"]
    comparison_rho = np.asarray(comparison["rho"], dtype=float)
    comparison_redl = np.asarray(
        comparison["redl_current_over_root_fsab2"],
        dtype=float,
    )
    if radii is None:
        target_rho = comparison_rho
    else:
        target_rho = np.asarray(radii, dtype=float)
    rho_field = np.asarray(field.rho_grid, dtype=float)
    radial_indices = np.asarray(
        [int(np.argmin(np.abs(rho_field - rho_value))) for rho_value in target_rho],
        dtype=int,
    )
    redl_current = _interp(comparison_rho, comparison_redl, target_rho)

    rows: list[dict[str, Any]] = []
    for neopax_x, n_order in settings:
        rows.extend(
            _evaluate_setting_profile(
                NEOPAX=NEOPAX,
                species=species,
                field=field,
                database=database,
                bootstrap_payload=bootstrap_payload,
                radial_indices=radial_indices,
                target_rho=target_rho,
                redl_current=redl_current,
                neopax_x=int(neopax_x),
                n_order=int(n_order),
            )
        )
    metrics = _summary_metrics(rows)
    conclusion = (
        "The profile source-response audit extends the stress-radius "
        "decomposition over the finite-beta profile.  It keeps the Redl "
        "density and temperature source targets as observables and records the "
        "effective-temperature response multiplier against collisionality and "
        "geometry factors.  The result is a source-response map for the "
        "reduced closure, not a runtime correction or fitted parity bridge."
    )
    return _to_jsonable(
        {
            "benchmark": "owned_finite_beta_source_response_profile_audit",
            "classification": "owned finite-beta profile source-response audit",
            "claim_scope": (
                "Reuses the owned finite-beta VMEC/Boozer geometry, analytic "
                "profiles, NTX monoenergetic scan, D33 branch, velocity "
                "quadrature, Sonine order, and current normalization, then "
                "solves the same momentum-restoring source-channel system at "
                "multiple profile radii.  This is a profile-wide "
                "source-response stress diagnostic, not a finite-beta parity "
                "claim and not a fitted correction."
            ),
            "case": case.as_payload(),
            "profile_contract": contract.as_payload(),
            "inputs": {
                "bootstrap_artifact": str(bootstrap_json),
                "settings": [
                    {"neopax_x": int(neopax_x), "n_order": int(n_order)}
                    for neopax_x, n_order in settings
                ],
                "radii": target_rho.tolist(),
                "radial_indices": radial_indices.tolist(),
                "field_radial_points": int(inputs.get("field_radial_points", 15)),
                "mboz": mboz,
                "nboz": nboz,
                "redl_ntheta": int(inputs.get("redl_ntheta", DEFAULT_REDL_NTHETA)),
                "d33_mode": str(inputs.get("d33_mode", "spitzer")),
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
                    "use the profile-wide source-response map to derive a "
                    "physics-based reduced closure change, then require no "
                    "regression on fixed-field QA/QH and integrated W7-X"
                ),
                (
                    "extend the same profile-response map to QH/QI and W7-X "
                    "owned geometries before making broad finite-beta closure "
                    "claims"
                ),
                (
                    "keep downstream interpolation-mode comparisons planned "
                    "until stable general and legacy selectors are exposed"
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
    high_setting = (
        int(metrics["high_order_neopax_x"]),
        int(metrics["high_order_n_order"]),
    )
    high_rows = _rows_for_setting(rows, setting=high_setting)
    rho = np.asarray([float(row["rho"]) for row in high_rows], dtype=float)
    redl = (
        np.asarray([float(row["redl_current_over_root_fsab2"]) for row in high_rows])
        / 1.0e6
    )
    public_total = (
        np.asarray(
            [float(row["public_neopax_current_over_root_fsab2"]) for row in high_rows]
        )
        / 1.0e6
    )
    public_nomom = (
        np.asarray(
            [float(row["public_neopax_nomom_over_root_fsab2"]) for row in high_rows]
        )
        / 1.0e6
    )
    rel_error = np.asarray(
        [float(row["public_neopax_relative_error_vs_redl"]) for row in high_rows],
        dtype=float,
    )
    reconstruction = np.asarray(
        [
            float(row["source_channel_superposition_relative_residual"])
            for row in high_rows
        ],
        dtype=float,
    )
    temperature_multiplier = np.asarray(
        [
            row.get("effective_temperature_response_multiplier_to_redl", np.nan)
            for row in high_rows
        ],
        dtype=float,
    )
    density_multiplier = np.asarray(
        [
            row.get("effective_channel_response_multiplier_to_redl", {}).get(
                "density_electric_force",
                np.nan,
            )
            for row in high_rows
        ],
        dtype=float,
    )
    nu_e_star = np.asarray(
        [
            row.get("redl_profile_drivers", {}).get("nu_e_star", np.nan)
            for row in high_rows
        ],
        dtype=float,
    )
    trapped_fraction = np.asarray(
        [
            row.get("redl_profile_drivers", {}).get("trapped_fraction", np.nan)
            for row in high_rows
        ],
        dtype=float,
    )

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
    ax_current, ax_response, ax_gate, ax_driver = axes.ravel()

    ax_current.plot(rho, redl, color="#009e73", lw=2.2, marker="o", label="Redl")
    ax_current.plot(
        rho,
        public_nomom,
        color="#0072b2",
        lw=1.8,
        ls="--",
        marker="s",
        label="no momentum",
    )
    ax_current.plot(
        rho,
        public_total,
        color="#d55e00",
        lw=1.8,
        marker="^",
        label="corrected",
    )
    ax_current.axhline(0.0, color="0.35", lw=0.8)
    ax_current.set_xlabel(r"$\rho$")
    ax_current.set_ylabel(r"current [MA m$^{-2}$]")
    ax_current.set_title("(a) Same-profile current observable")
    ax_current.legend(fontsize=8.0)

    ax_response.plot(
        rho,
        temperature_multiplier,
        color="#d55e00",
        lw=2.0,
        marker="o",
        label="temperature source",
    )
    if np.any(np.isfinite(density_multiplier)):
        ax_response.plot(
            rho,
            density_multiplier,
            color="#0072b2",
            lw=1.6,
            marker="s",
            label="density/electric source",
        )
    ax_response.axhline(1.0, color="0.25", lw=1.0, ls="--", label="unit response")
    ax_response.set_xlabel(r"$\rho$")
    ax_response.set_ylabel("Redl target / corrected source response")
    ax_response.set_title("(b) Effective source-response multiplier")
    ax_response.legend(fontsize=8.0)

    ax_gate.semilogy(
        rho,
        rel_error,
        color="#d55e00",
        lw=1.9,
        marker="o",
        label="current difference",
    )
    ax_gate.semilogy(
        rho,
        reconstruction,
        color="#0072b2",
        lw=1.8,
        marker="s",
        label="channel reconstruction",
    )
    ax_gate.axhline(PROFILE_CURRENT_GATE, color="0.25", lw=1.0, ls="--")
    ax_gate.axhline(SOURCE_RECONSTRUCTION_GATE, color="0.45", lw=1.0, ls=":")
    ax_gate.set_xlabel(r"$\rho$")
    ax_gate.set_ylabel("relative value")
    ax_gate.set_title("(c) Current stress and linearity gate")
    ax_gate.legend(fontsize=8.0)

    finite_driver = (
        np.isfinite(nu_e_star)
        & (nu_e_star > 0.0)
        & np.isfinite(temperature_multiplier)
    )
    if np.any(finite_driver):
        scatter = ax_driver.scatter(
            nu_e_star[finite_driver],
            temperature_multiplier[finite_driver],
            c=rho[finite_driver],
            s=46,
            cmap="viridis",
            edgecolors="0.1",
            linewidths=0.35,
            label="profile radii",
        )
        ax_driver.set_xscale("log")
        ax_driver.axhline(1.0, color="0.25", lw=1.0, ls="--")
        cbar = fig.colorbar(scatter, ax=ax_driver)
        cbar.set_label(r"$\rho$")
    ax_driver_t = ax_driver.twinx()
    ax_driver_t.plot(
        nu_e_star,
        trapped_fraction,
        color="0.35",
        lw=1.3,
        marker="D",
        label=r"$f_t$",
    )
    ax_driver.set_xlabel(r"Redl $\nu^*_e$")
    ax_driver.set_ylabel("temperature response multiplier")
    ax_driver_t.set_ylabel(r"trapped fraction $f_t$")
    ax_driver.set_title("(d) Response against physics drivers")

    multiplier_span = metrics.get("high_order_temperature_response_multiplier_span")
    span_text = (
        f", response span={float(multiplier_span):.2g}"
        if multiplier_span is not None
        else ""
    )
    fig.suptitle(
        "Owned finite-beta profile source-response audit "
        f"(X={high_setting[0]}, P={high_setting[1]}{span_text})",
        fontsize=13,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-json", type=Path, default=BOOTSTRAP_JSON)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=[f"{x}:{p}" for x, p in DEFAULT_SETTINGS],
        help="Closure settings formatted as X:P, for example 18:18.",
    )
    parser.add_argument(
        "--rho",
        nargs="+",
        type=float,
        default=None,
        help="Optional radial values. Defaults to all bootstrap-comparison radii.",
    )
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=WORKDIR)
    args = parser.parse_args()

    payload = build_payload(
        bootstrap_json=args.bootstrap_json,
        settings=_parse_settings([str(value) for value in args.settings]),
        radii=tuple(float(value) for value in args.rho) if args.rho is not None else None,
        output_dir=args.output_dir,
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(json.dumps(payload["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
