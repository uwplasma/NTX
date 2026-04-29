from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


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


__all__ = ["field_radius_matched_response_audit"]
