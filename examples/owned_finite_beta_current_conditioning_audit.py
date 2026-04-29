#!/usr/bin/env python3
"""Audit finite-beta current conditioning from committed sidecars.

The finite-beta bootstrap-current stress case is a net-current observable built
from large species-flow contributions.  This script connects the completed
same-grid coefficient ladder to the profile-current observable by estimating
the coefficient precision required for a `1e-1` net-current gate once the
species-current cancellation scale is included.
"""

from __future__ import annotations

import argparse
import json
import sys
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

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_finite_beta_current_conditioning_audit"
CLOSURE_JSON = ROOT / "docs" / "_static" / "owned_finite_beta_closure_localization.json"
OBSERVABLE_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_profile_current_observable_audit.json"
)
PROFILE_CURRENT_GATE = 1.0e-1
EPS = 1.0e-30


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _coefficient_profile(closure_payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        row
        for row in closure_payload.get("coefficient_by_rho", [])
        if row.get("max_raw_relative_difference") is not None
    ]
    if not rows:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    rho = np.asarray([float(row["rho"]) for row in rows], dtype=float)
    error = np.asarray(
        [float(row["max_raw_relative_difference"]) for row in rows],
        dtype=float,
    )
    order = np.argsort(rho)
    return rho[order], error[order]


def _interpolate_if_in_range(
    rho_source: np.ndarray,
    values: np.ndarray,
    rho: float,
) -> float | None:
    if rho_source.size == 0 or values.size == 0:
        return None
    if float(rho) < float(np.min(rho_source)) or float(rho) > float(np.max(rho_source)):
        return None
    return float(np.interp(float(rho), rho_source, values))


def _row_with_max(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    finite_rows = [row for row in rows if row.get(key) is not None]
    if not finite_rows:
        return None
    return max(finite_rows, key=lambda row: float(row[key]))


def _row_nearest(rows: list[dict[str, Any]], rho: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["rho"]) - float(rho)))


def build_payload(
    *,
    closure_json: Path = CLOSURE_JSON,
    observable_json: Path = OBSERVABLE_JSON,
) -> dict[str, Any]:
    closure_payload = _load_json(closure_json)
    observable_payload = _load_json(observable_json)
    coeff_rho, coeff_error = _coefficient_profile(closure_payload)

    rows: list[dict[str, Any]] = []
    for row in observable_payload.get("rows", []):
        rho = float(row["rho"])
        redl_abs = abs(float(row["redl_current_over_root_fsab2"]))
        total_error = float(row["relative_error_total_vs_redl"])
        species_l1 = float(row["species_momentum_correction_l1_over_root_fsab2"])
        applied_abs = abs(float(row["applied_momentum_correction_over_root_fsab2"]))
        observed_residual_species = float(
            row["residual_after_correction_over_species_correction_l1"]
        )
        coefficient_error = _interpolate_if_in_range(coeff_rho, coeff_error, rho)
        current_condition = species_l1 / max(redl_abs, EPS)
        correction_condition = species_l1 / max(applied_abs, EPS)
        required_coefficient_error = PROFILE_CURRENT_GATE / max(current_condition, EPS)
        coefficient_limited_error = (
            None
            if coefficient_error is None
            else coefficient_error * current_condition
        )
        precision_gap = (
            None
            if coefficient_error is None or required_coefficient_error <= 0.0
            else coefficient_error / required_coefficient_error
        )
        observed_over_bound = (
            None
            if coefficient_limited_error is None or coefficient_limited_error <= 0.0
            else total_error / coefficient_limited_error
        )
        observed_species_over_coefficient = (
            None
            if coefficient_error is None or coefficient_error <= 0.0
            else observed_residual_species / coefficient_error
        )
        rows.append(
            {
                "rho": rho,
                "same_grid_coefficient_relative_difference": _finite_or_none(
                    coefficient_error
                )
                if coefficient_error is not None
                else None,
                "redl_current_abs_over_root_fsab2": redl_abs,
                "profile_current_relative_difference": total_error,
                "species_correction_l1_over_root_fsab2": species_l1,
                "current_condition_number_species_l1_over_redl": current_condition,
                "correction_cancellation_number_species_l1_over_applied": (
                    correction_condition
                ),
                "required_coefficient_relative_difference_for_current_gate": (
                    required_coefficient_error
                ),
                "coefficient_limited_current_relative_error_bound": (
                    _finite_or_none(coefficient_limited_error)
                    if coefficient_limited_error is not None
                    else None
                ),
                "coefficient_precision_gap_to_current_gate": (
                    _finite_or_none(precision_gap) if precision_gap is not None else None
                ),
                "observed_current_error_over_coefficient_bound": (
                    _finite_or_none(observed_over_bound)
                    if observed_over_bound is not None
                    else None
                ),
                "observed_residual_over_species_l1": observed_residual_species,
                "observed_residual_species_l1_over_coefficient_error": (
                    _finite_or_none(observed_species_over_coefficient)
                    if observed_species_over_coefficient is not None
                    else None
                ),
            }
        )

    stress_rho = float(observable_payload["summary_metrics"]["stress_rho"])
    stress_row = _row_nearest(rows, stress_rho)
    max_condition_row = _row_with_max(
        rows,
        "current_condition_number_species_l1_over_redl",
    )
    max_gap_row = _row_with_max(rows, "coefficient_precision_gap_to_current_gate")
    max_bound_row = _row_with_max(
        rows,
        "coefficient_limited_current_relative_error_bound",
    )

    return {
        "benchmark": "owned_finite_beta_current_conditioning_audit",
        "classification": "owned finite-beta current-conditioning audit",
        "claim_scope": (
            "Combines the completed same-grid finite-beta coefficient ladder "
            "with the profile-current observable sidecar.  The diagnostic "
            "quantifies how species-flow cancellation amplifies coefficient "
            "uncertainty in the net bootstrap-current observable.  It is a "
            "conditioning audit, not a fitted closure correction and not a "
            "bootstrap-current parity claim."
        ),
        "inputs": {
            "closure_localization_artifact": str(closure_json),
            "profile_current_observable_artifact": str(observable_json),
            "profile_current_gate": PROFILE_CURRENT_GATE,
            "current_units": "<J dot B> / sqrt(<B^2>)",
        },
        "rows": rows,
        "stress_radius": stress_row,
        "summary_metrics": {
            "stress_rho": stress_row["rho"] if stress_row is not None else None,
            "stress_current_condition_number": (
                stress_row["current_condition_number_species_l1_over_redl"]
                if stress_row is not None
                else None
            ),
            "stress_same_grid_coefficient_relative_difference": (
                stress_row["same_grid_coefficient_relative_difference"]
                if stress_row is not None
                else None
            ),
            "stress_required_coefficient_relative_difference_for_current_gate": (
                stress_row["required_coefficient_relative_difference_for_current_gate"]
                if stress_row is not None
                else None
            ),
            "stress_coefficient_precision_gap_to_current_gate": (
                stress_row["coefficient_precision_gap_to_current_gate"]
                if stress_row is not None
                else None
            ),
            "stress_coefficient_limited_current_relative_error_bound": (
                stress_row["coefficient_limited_current_relative_error_bound"]
                if stress_row is not None
                else None
            ),
            "stress_observed_current_relative_difference": (
                stress_row["profile_current_relative_difference"]
                if stress_row is not None
                else None
            ),
            "stress_observed_current_error_over_coefficient_bound": (
                stress_row["observed_current_error_over_coefficient_bound"]
                if stress_row is not None
                else None
            ),
            "max_current_condition_number": (
                max_condition_row["current_condition_number_species_l1_over_redl"]
                if max_condition_row is not None
                else None
            ),
            "max_condition_rho": (
                max_condition_row["rho"] if max_condition_row is not None else None
            ),
            "max_coefficient_precision_gap_to_current_gate": (
                max_gap_row["coefficient_precision_gap_to_current_gate"]
                if max_gap_row is not None
                else None
            ),
            "max_coefficient_precision_gap_rho": (
                max_gap_row["rho"] if max_gap_row is not None else None
            ),
            "max_coefficient_limited_current_relative_error_bound": (
                max_bound_row["coefficient_limited_current_relative_error_bound"]
                if max_bound_row is not None
                else None
            ),
            "max_coefficient_limited_bound_rho": (
                max_bound_row["rho"] if max_bound_row is not None else None
            ),
        },
        "conclusion": (
            "The same-grid coefficient ladder passes the order-1e-1 "
            "normalization gate, but the finite-beta stress radius is "
            "cancellation-conditioned.  A net-current 1e-1 gate requires "
            "coefficient precision closer to the current-conditioned threshold "
            "reported here before coefficient error can be ruled out as a "
            "source of the remaining profile-current gap."
        ),
        "open_work": [
            (
                "run a production-resolution same-grid coefficient ladder until "
                "the current-conditioned coefficient-precision threshold is "
                "resolved at the stress radius"
            ),
            (
                "only then use the profile-current residual to motivate a "
                "reduced momentum-closure change"
            ),
            (
                "preserve the fixed-field QA/QH total-current gate and W7-X "
                "integrated transfer when testing any closure change"
            ),
        ],
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
    }


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def _array_or_nan(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [np.nan if row.get(key) is None else float(row[key]) for row in rows],
        dtype=float,
    )


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    rows = payload["rows"]
    rho = _array_or_nan(rows, "rho")
    coefficient_error = _array_or_nan(rows, "same_grid_coefficient_relative_difference")
    required_error = _array_or_nan(
        rows,
        "required_coefficient_relative_difference_for_current_gate",
    )
    current_condition = _array_or_nan(
        rows,
        "current_condition_number_species_l1_over_redl",
    )
    correction_condition = _array_or_nan(
        rows,
        "correction_cancellation_number_species_l1_over_applied",
    )
    current_error = _array_or_nan(rows, "profile_current_relative_difference")
    coefficient_bound = _array_or_nan(
        rows,
        "coefficient_limited_current_relative_error_bound",
    )
    residual_species = _array_or_nan(rows, "observed_residual_over_species_l1")
    residual_over_coefficient = _array_or_nan(
        rows,
        "observed_residual_species_l1_over_coefficient_error",
    )
    stress_rho = payload["summary_metrics"]["stress_rho"]

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    ax_precision, ax_condition, ax_bound, ax_residual = axes.ravel()

    ax_precision.semilogy(
        rho,
        coefficient_error,
        marker="o",
        label="same-grid coefficient error",
    )
    ax_precision.semilogy(
        rho,
        required_error,
        marker="s",
        label="required for 1e-1 current",
    )
    ax_precision.axhline(PROFILE_CURRENT_GATE, color="0.35", linestyle=":")
    if stress_rho is not None:
        ax_precision.axvline(float(stress_rho), color="0.35", linestyle=":")
    ax_precision.set_xlabel(r"$\rho$")
    ax_precision.set_ylabel("relative coefficient scale")
    ax_precision.set_title("(a) Current-conditioned coefficient precision")
    ax_precision.grid(alpha=0.25, which="both")
    ax_precision.legend(fontsize=8)

    ax_condition.semilogy(
        rho,
        current_condition,
        marker="o",
        label=r"species L1 / $|J_{Redl}|$",
    )
    ax_condition.semilogy(
        rho,
        correction_condition,
        marker="s",
        label="species L1 / applied correction",
    )
    if stress_rho is not None:
        ax_condition.axvline(float(stress_rho), color="0.35", linestyle=":")
    ax_condition.set_xlabel(r"$\rho$")
    ax_condition.set_ylabel("condition number")
    ax_condition.set_title("(b) Net-current cancellation")
    ax_condition.grid(alpha=0.25, which="both")
    ax_condition.legend(fontsize=8)

    ax_bound.semilogy(rho, current_error, marker="o", label="observed net-current error")
    ax_bound.semilogy(
        rho,
        coefficient_bound,
        marker="s",
        label="coefficient-conditioned bound",
    )
    ax_bound.axhline(PROFILE_CURRENT_GATE, color="0.35", linestyle=":")
    if stress_rho is not None:
        ax_bound.axvline(float(stress_rho), color="0.35", linestyle=":")
    ax_bound.set_xlabel(r"$\rho$")
    ax_bound.set_ylabel("relative current scale")
    ax_bound.set_title("(c) Error budget")
    ax_bound.grid(alpha=0.25, which="both")
    ax_bound.legend(fontsize=8)

    ax_residual.semilogy(
        rho,
        residual_species,
        marker="o",
        label="residual / species L1",
    )
    ax_residual.semilogy(
        rho,
        residual_over_coefficient,
        marker="s",
        label="residual/species L1 per coeff. error",
    )
    if stress_rho is not None:
        ax_residual.axvline(float(stress_rho), color="0.35", linestyle=":")
    ax_residual.set_xlabel(r"$\rho$")
    ax_residual.set_ylabel("dimensionless")
    ax_residual.set_title("(d) Species-flow residual scale")
    ax_residual.grid(alpha=0.25, which="both")
    ax_residual.legend(fontsize=8)

    fig.suptitle("Owned finite-beta current-conditioning audit", fontsize=13)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-json", type=Path, default=CLOSURE_JSON)
    parser.add_argument("--observable-json", type=Path, default=OBSERVABLE_JSON)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    args = parser.parse_args()

    payload = build_payload(
        closure_json=args.closure_json,
        observable_json=args.observable_json,
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(json.dumps(payload["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
