from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1.0e-30
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


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def dominant_channel(values: dict[str, float]) -> str:
    if not values:
        return "none"
    return max(values, key=lambda key: abs(float(values[key])))


def relative_scalar_error(candidate: float, reference: float) -> float:
    return float(abs(float(candidate) - float(reference)) / max(abs(float(reference)), EPS))


def relative_scalar_error_or_none(
    candidate: float | None,
    reference: float | None,
) -> float | None:
    if candidate is None or reference is None:
        return None
    if not np.isfinite(float(candidate)) or not np.isfinite(float(reference)):
        return None
    return relative_scalar_error(float(candidate), float(reference))


def channel_response_ratios(
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
        relative_errors[label] = relative_scalar_error_or_none(candidate, target)
    return multipliers, relative_errors


def effective_projection_and_drives(
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


def source_contributions_by_channel(
    projection_by_species: np.ndarray,
    drives_by_species: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Return RHS contributions shaped ``(species, moment, channel)``."""

    projection = np.asarray(projection_by_species, dtype=float)
    drives = np.asarray(drives_by_species, dtype=float)
    if mode == "transport":
        labels = TRANSPORT_LABELS
        active_projection = projection
        active_drives = drives
    elif mode == "effective":
        labels = EFFECTIVE_LABELS
        pieces = [
            effective_projection_and_drives(projection[index], drives[index])
            for index in range(projection.shape[0])
        ]
        active_projection = np.stack([item[0] for item in pieces])
        active_drives = np.stack([item[1] for item in pieces])
    else:
        raise ValueError(f"unknown source-decomposition mode {mode!r}")
    rhs = -active_projection * active_drives[:, None, :]
    return rhs, labels, active_drives


def source_channel_summary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    max_public_difference = max(
        float(row["full_vs_public_relative_difference"]) for row in rows
    )
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
            finite_or_none(
                high_stable.get("effective_temperature_response_multiplier_to_redl")
            )
        ),
        "high_stable_effective_temperature_channel_relative_error_vs_redl": (
            finite_or_none(
                high_stable.get("effective_temperature_channel_relative_error_vs_redl")
            )
        ),
        "high_stable_redl_effective_temperature_fraction_of_total": (
            finite_or_none(high_stable.get("redl_effective_temperature_fraction_of_total"))
        ),
        "best_effective_temperature_response_multiplier_to_redl": finite_or_none(
            best_row.get("effective_temperature_response_multiplier_to_redl")
        ),
    }


def finite_values(values: list[float | None]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )


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


def high_order_setting(rows: list[dict[str, Any]]) -> tuple[int, int]:
    high = max(
        rows,
        key=lambda row: (
            row["x_to_order_ratio"] >= 1.0,
            row["neopax_x"],
            row["n_order"],
        ),
    )
    return int(high["neopax_x"]), int(high["n_order"])


def rows_for_setting(
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


def profile_source_response_summary_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("profile source-response audit requires at least one row")
    high_setting = high_order_setting(rows)
    high_rows = rows_for_setting(rows, setting=high_setting)
    multipliers = finite_values(
        [
            row.get("effective_temperature_response_multiplier_to_redl")
            for row in high_rows
        ]
    )
    response_errors = finite_values(
        [
            row.get("effective_temperature_channel_relative_error_vs_redl")
            for row in high_rows
        ]
    )
    reconstruction = finite_values(
        [row.get("source_channel_superposition_relative_residual") for row in rows]
    )
    public_difference = finite_values(
        [row.get("full_vs_public_relative_difference") for row in rows]
    )
    profile_errors = finite_values(
        [row.get("public_neopax_relative_error_vs_redl") for row in high_rows]
    )
    target_values: list[float] = []
    candidate_values: list[float] = []
    for row in high_rows:
        target = row.get(
            "redl_effective_channel_current_by_channel_over_root_fsab2",
            {},
        ).get("effective_temperature_force")
        candidate = (
            row.get("source_decomposition", {})
            .get("effective", {})
            .get("current_by_channel_over_root_fsab2", {})
            .get("effective_temperature_force")
        )
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
        "setting_count": int(
            len({(int(row["neopax_x"]), int(row["n_order"])) for row in rows})
        ),
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
        "high_order_stress_temperature_response_multiplier": finite_or_none(
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


__all__ = [
    "EFFECTIVE_LABELS",
    "EPS",
    "PROFILE_CURRENT_GATE",
    "SOURCE_RECONSTRUCTION_GATE",
    "TRANSPORT_LABELS",
    "channel_response_ratios",
    "dominant_channel",
    "effective_projection_and_drives",
    "finite_or_none",
    "finite_values",
    "high_order_setting",
    "profile_source_response_summary_metrics",
    "relative_scalar_error",
    "relative_scalar_error_or_none",
    "rows_for_setting",
    "source_channel_summary_metrics",
    "source_contributions_by_channel",
]
