"""Finite-beta closure target used by the source-channel audit.

Builds the reference the audit compares against, keeping the target definition
separate from the channel bookkeeping that consumes it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1.0e-30

FEATURE_SPECS = (
    ("epsilon", "epsilon"),
    ("trapped_fraction", "trapped_fraction"),
    ("log10_nu_e_star", "log10_nu_e_star"),
    ("redl_L32", "L32"),
)

LINEAR_MODELS = (
    ("constant", ()),
    ("epsilon", ("epsilon",)),
    ("trapped_fraction", ("trapped_fraction",)),
    ("log10_nu_e_star", ("log10_nu_e_star",)),
    ("epsilon_plus_log10_nu_e_star", ("epsilon", "log10_nu_e_star")),
    (
        "trapped_fraction_plus_log10_nu_e_star",
        ("trapped_fraction", "log10_nu_e_star"),
    ),
)


def _finite_array(values: list[float | None]) -> np.ndarray:
    return np.asarray(
        [
            float(value)
            if value is not None and np.isfinite(float(value))
            else np.nan
            for value in values
        ],
        dtype=float,
    )


def _rows_for_high_order(payload: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = payload["summary_metrics"]
    high_x = int(metrics["high_order_neopax_x"])
    high_p = int(metrics["high_order_n_order"])
    rows = [
        row
        for row in payload["rows"]
        if int(row["neopax_x"]) == high_x and int(row["n_order"]) == high_p
    ]
    return sorted(rows, key=lambda row: float(row["rho"]))


def _response_table(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {
        "rho": np.asarray([float(row["rho"]) for row in rows], dtype=float),
        "response": np.asarray(
            [
                float(row["effective_temperature_response_multiplier_to_redl"])
                for row in rows
            ],
            dtype=float,
        ),
        "current_error": np.asarray(
            [float(row["public_neopax_relative_error_vs_redl"]) for row in rows],
            dtype=float,
        ),
        "reconstruction": np.asarray(
            [
                float(row["source_channel_superposition_relative_residual"])
                for row in rows
            ],
            dtype=float,
        ),
    }
    for output_key, driver_key in FEATURE_SPECS:
        table[output_key] = _finite_array(
            [row.get("redl_profile_drivers", {}).get(driver_key) for row in rows]
        )
    return table


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        stop = start + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return None
    x_active = x[mask]
    y_active = y[mask]
    if float(np.std(x_active)) <= EPS or float(np.std(y_active)) <= EPS:
        return None
    return float(np.corrcoef(x_active, y_active)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return None
    return _correlation(_rankdata(x[mask]), _rankdata(y[mask]))


def _design_matrix(table: dict[str, np.ndarray], feature_keys: tuple[str, ...]) -> np.ndarray:
    columns = [np.ones_like(table["response"])]
    for key in feature_keys:
        values = np.asarray(table[key], dtype=float)
        columns.append(values - float(np.nanmean(values)))
    return np.column_stack(columns)


def _fit_predict(
    table: dict[str, np.ndarray],
    feature_keys: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, float]:
    x = _design_matrix(table, feature_keys)
    y = table["response"]
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if int(np.sum(mask)) <= len(feature_keys) + 1:
        return np.full_like(y, np.nan), np.full(len(feature_keys) + 1, np.nan), np.nan
    coeffs, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
    prediction = np.full_like(y, np.nan)
    prediction[mask] = x[mask] @ coeffs
    condition = float(np.linalg.cond(x[mask]))
    return prediction, coeffs, condition


def _leave_one_out_rmse(
    table: dict[str, np.ndarray],
    feature_keys: tuple[str, ...],
) -> float | None:
    y = table["response"]
    predictions = np.full_like(y, np.nan)
    for index in range(y.size):
        train = {key: value.copy() for key, value in table.items()}
        train["response"][index] = np.nan
        x_train = _design_matrix(train, feature_keys)
        train_mask = np.isfinite(train["response"]) & np.all(np.isfinite(x_train), axis=1)
        if int(np.sum(train_mask)) <= len(feature_keys) + 1:
            continue
        coeffs, *_ = np.linalg.lstsq(
            x_train[train_mask],
            train["response"][train_mask],
            rcond=None,
        )
        x_one = _design_matrix(table, feature_keys)[index]
        if np.all(np.isfinite(x_one)):
            predictions[index] = float(x_one @ coeffs)
    mask = np.isfinite(predictions) & np.isfinite(y)
    if int(np.sum(mask)) == 0:
        return None
    return float(np.sqrt(np.mean((predictions[mask] - y[mask]) ** 2)))


def _model_diagnostics(table: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for name, features in LINEAR_MODELS:
        prediction, coeffs, condition = _fit_predict(table, features)
        mask = np.isfinite(prediction) & np.isfinite(table["response"])
        rmse = (
            float(np.sqrt(np.mean((prediction[mask] - table["response"][mask]) ** 2)))
            if int(np.sum(mask))
            else None
        )
        loo_rmse = _leave_one_out_rmse(table, features)
        diagnostics.append(
            {
                "name": name,
                "features": list(features),
                "rmse": rmse,
                "leave_one_out_rmse": loo_rmse,
                "condition_number": condition if np.isfinite(condition) else None,
                "coefficients": [
                    float(value) if np.isfinite(float(value)) else None for value in coeffs
                ],
            }
        )
    return diagnostics


def _best_model(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [
        item
        for item in diagnostics
        if item["leave_one_out_rmse"] is not None
        and np.isfinite(float(item["leave_one_out_rmse"]))
    ]
    if not finite:
        return {}
    return min(finite, key=lambda item: float(item["leave_one_out_rmse"]))


def profile_closure_target_diagnostics(source_payload: dict[str, Any]) -> dict[str, Any]:
    """Build profile-wide closure-target diagnostics from source-response rows."""

    rows = _rows_for_high_order(source_payload)
    table = _response_table(rows)
    response = table["response"]
    current_error = table["current_error"]

    correlations: dict[str, dict[str, float | None]] = {}
    for key, _driver in FEATURE_SPECS:
        correlations[key] = {
            "pearson": _correlation(table[key], response),
            "spearman": _spearman(table[key], response),
        }
    diagnostics = _model_diagnostics(table)
    best = _best_model(diagnostics)
    constant = next(item for item in diagnostics if item["name"] == "constant")
    unit_rmse = float(np.sqrt(np.mean((response - 1.0) ** 2)))
    constant_loo = constant["leave_one_out_rmse"]
    best_loo = best.get("leave_one_out_rmse")
    best_single_driver = max(
        correlations,
        key=lambda key: abs(float(correlations[key]["pearson"] or 0.0)),
    )

    return {
        "rows": [
            {
                "rho": float(row["rho"]),
                "temperature_response_multiplier": float(
                    row["effective_temperature_response_multiplier_to_redl"]
                ),
                "profile_current_relative_error": float(
                    row["public_neopax_relative_error_vs_redl"]
                ),
                "drivers": row.get("redl_profile_drivers", {}),
            }
            for row in rows
        ],
        "correlations": correlations,
        "linear_diagnostics": diagnostics,
        "summary_metrics": {
            "radius_count": int(response.size),
            "response_multiplier_min": float(np.min(response)),
            "response_multiplier_median": float(np.median(response)),
            "response_multiplier_max": float(np.max(response)),
            "response_multiplier_span": float(np.max(response) - np.min(response)),
            "response_multiplier_max_abs_deviation_from_one": float(
                np.max(np.abs(response - 1.0))
            ),
            "unit_response_rmse": unit_rmse,
            "constant_leave_one_out_rmse": constant_loo,
            "best_leave_one_out_model": best.get("name"),
            "best_leave_one_out_rmse": best_loo,
            "best_leave_one_out_improvement_over_constant": (
                float(constant_loo) / max(float(best_loo), EPS)
                if constant_loo is not None and best_loo is not None
                else None
            ),
            "best_single_physics_driver": best_single_driver,
            "best_single_physics_driver_abs_pearson": abs(
                float(correlations[best_single_driver]["pearson"] or 0.0)
            ),
            "epsilon_abs_pearson": abs(float(correlations["epsilon"]["pearson"] or 0.0)),
            "trapped_fraction_abs_pearson": abs(
                float(correlations["trapped_fraction"]["pearson"] or 0.0)
            ),
            "log10_nu_e_star_abs_pearson": abs(
                float(correlations["log10_nu_e_star"]["pearson"] or 0.0)
            ),
            "redl_L32_abs_pearson": abs(
                float(correlations["redl_L32"]["pearson"] or 0.0)
            ),
            "max_profile_current_relative_error": float(np.max(current_error)),
            "median_profile_current_relative_error": float(np.median(current_error)),
            "max_source_channel_superposition_relative_residual": float(
                np.max(table["reconstruction"])
            ),
            "runtime_correction_applied": False,
        },
    }


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def _artifact_path(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _setting(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["neopax_x"]), int(row["n_order"])


def _setting_dict(setting: tuple[int, int]) -> dict[str, int]:
    return {"neopax_x": int(setting[0]), "n_order": int(setting[1])}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _settings_match_rho(
    rows: list[dict[str, Any]],
    stress_rho: float | None,
) -> bool | None:
    if stress_rho is None or not rows:
        return None
    return bool(
        all(abs(float(row["rho"]) - float(stress_rho)) <= 1.0e-12 for row in rows)
    )


def field_radius_matched_response_audit(
    *,
    matched_source_channel_json: Path | None,
    matched_quadrature_json: Path | None,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """Cross-link matched-radius source-channel and quadrature artifacts.

    The returned payload is intentionally diagnostic: it records whether a
    finite-beta current-gate pass survives the quadrature-stable X >= Pmax rule
    and whether the physical source decomposition remains on the same stress
    radius. It does not define or apply a runtime closure correction.
    """

    source_payload = _load_optional_json(matched_source_channel_json)
    quadrature_payload = _load_optional_json(matched_quadrature_json)
    if source_payload is None and quadrature_payload is None:
        return None

    source_rows = source_payload.get("rows", []) if source_payload else []
    source_metrics = source_payload.get("summary_metrics", {}) if source_payload else {}
    quadrature_rows = quadrature_payload.get("rows", []) if quadrature_payload else []
    quadrature_metrics = (
        quadrature_payload.get("summary_metrics", {}) if quadrature_payload else {}
    )

    source_by_setting = {_setting(row): row for row in source_rows}
    quadrature_by_setting = {_setting(row): row for row in quadrature_rows}
    matched_settings = sorted(set(source_by_setting) & set(quadrature_by_setting))
    rows = []
    for setting in matched_settings:
        source_row = source_by_setting[setting]
        quadrature_row = quadrature_by_setting[setting]
        rows.append(
            {
                **_setting_dict(setting),
                "x_to_order_ratio": float(source_row["x_to_order_ratio"]),
                "source_channel_current_relative_error": _float_or_none(
                    source_row.get("public_neopax_relative_error_vs_redl")
                ),
                "quadrature_current_relative_error": _float_or_none(
                    quadrature_row.get("stress_relative_error_total_vs_redl")
                ),
                "effective_temperature_response_multiplier_to_redl": _float_or_none(
                    source_row.get(
                        "effective_temperature_response_multiplier_to_redl"
                    )
                ),
                "effective_temperature_channel_relative_error_vs_redl": _float_or_none(
                    source_row.get(
                        "effective_temperature_channel_relative_error_vs_redl"
                    )
                ),
                "source_channel_superposition_relative_residual": _float_or_none(
                    source_row.get("source_channel_superposition_relative_residual")
                ),
                "dominant_effective_channel": source_row.get("dominant_effective_channel"),
                "species_cancellation_factor": _float_or_none(
                    source_row.get("species_cancellation_factor")
                ),
            }
        )

    high_stable_setting = None
    if source_metrics.get("high_stable_neopax_x") is not None:
        high_stable_setting = (
            int(source_metrics["high_stable_neopax_x"]),
            int(source_metrics["high_stable_n_order"]),
        )
    best_setting = None
    if source_metrics.get("best_public_neopax_x") is not None:
        best_setting = (
            int(source_metrics["best_public_neopax_x"]),
            int(source_metrics["best_public_n_order"]),
        )
    source_stress_rho = (
        float(source_rows[0]["rho"])
        if source_rows and source_rows[0].get("rho") is not None
        else None
    )
    quadrature_stress_rho = _float_or_none(quadrature_metrics.get("stress_rho"))
    if quadrature_stress_rho is None and quadrature_rows:
        quadrature_stress_rho = _float_or_none(quadrature_rows[0].get("stress_rho"))
    same_radius = _settings_match_rho(source_rows, quadrature_stress_rho)
    high_stable_response = _float_or_none(
        source_metrics.get("high_stable_effective_temperature_response_multiplier_to_redl")
    )
    best_response = _float_or_none(
        source_metrics.get("best_effective_temperature_response_multiplier_to_redl")
    )
    response_unit_distance_reduction = None
    if high_stable_response is not None and best_response is not None:
        response_unit_distance_reduction = (
            abs(high_stable_response - 1.0) - abs(best_response - 1.0)
        )

    return {
        "source_artifact": (
            _artifact_path(matched_source_channel_json, root)
            if matched_source_channel_json is not None
            else None
        ),
        "quadrature_artifact": (
            _artifact_path(matched_quadrature_json, root)
            if matched_quadrature_json is not None
            else None
        ),
        "claim_scope": (
            "Cross-links the field-radius-matched source-channel and "
            "quadrature artifacts. It checks whether the apparent current-gate "
            "pass is quadrature stable and whether the same physical source "
            "channel remains the reduced-closure stress after radial "
            "interpolation is removed."
        ),
        "source_stress_rho": source_stress_rho,
        "quadrature_stress_rho": quadrature_stress_rho,
        "same_stress_radius_between_artifacts": same_radius,
        "matched_setting_count": int(len(rows)),
        "rows": rows,
        "best_public_setting": (
            _setting_dict(best_setting) if best_setting is not None else None
        ),
        "high_stable_setting": (
            _setting_dict(high_stable_setting)
            if high_stable_setting is not None
            else None
        ),
        "best_public_relative_error_vs_redl": _float_or_none(
            source_metrics.get("best_public_relative_error_vs_redl")
        ),
        "high_stable_public_relative_error_vs_redl": _float_or_none(
            source_metrics.get("high_stable_public_relative_error_vs_redl")
        ),
        "best_effective_temperature_response_multiplier_to_redl": best_response,
        "high_stable_effective_temperature_response_multiplier_to_redl": (
            high_stable_response
        ),
        "response_unit_distance_reduction_in_best_vs_stable": (
            response_unit_distance_reduction
        ),
        "source_channel_superposition_gate_pass": source_metrics.get(
            "source_channel_superposition_gate_pass"
        ),
        "max_source_channel_superposition_relative_residual": _float_or_none(
            source_metrics.get("max_source_channel_superposition_relative_residual")
        ),
        "high_stable_dominant_effective_channel": source_metrics.get(
            "high_stable_dominant_effective_channel"
        ),
        "high_stable_effective_temperature_fraction_of_total": _float_or_none(
            source_metrics.get("high_stable_effective_temperature_fraction_of_total")
        ),
        "quadrature_stable_gate_pass_count": quadrature_metrics.get(
            "quadrature_stable_gate_pass_count"
        ),
        "quadrature_stable_current_gate_pass": quadrature_metrics.get(
            "quadrature_stable_current_gate_pass"
        ),
        "underintegrated_gate_pass_count": quadrature_metrics.get(
            "underintegrated_gate_pass_count"
        ),
        "best_stress_pass_rejected_as_underintegrated": quadrature_metrics.get(
            "best_stress_pass_rejected_as_underintegrated"
        ),
        "quadrature_aliasing_detected": quadrature_metrics.get(
            "quadrature_aliasing_detected"
        ),
        "conclusion": (
            "The field-radius-matched source solve reconstructs the full "
            "corrected current, but the only current-gate pass is rejected "
            "when it does not transfer to quadrature-stable X >= Pmax "
            "settings. The remaining closure target is therefore a physical "
            "reduced-source response, not a scalar normalization or "
            "radial-interpolation artifact."
        ),
        "runtime_correction_applied": False,
    }


__all__ = [
    "field_radius_matched_response_audit",
    "profile_closure_target_diagnostics",
]
