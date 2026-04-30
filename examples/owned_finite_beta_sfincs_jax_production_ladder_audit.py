#!/usr/bin/env python3
"""Audit the owned finite-beta production SFINCS-JAX coefficient ladder.

This diagnostic consumes the committed smoke and production same-grid
SFINCS-JAX/NTX transport-matrix artifacts.  It closes the numerical part of the
finite-beta QA lane by checking whether the coefficient floor persists across
the profile-relevant radius/collisionality ladder, before any remaining
bootstrap-current residual is assigned to the reduced profile-current closure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT_PREFIX = (
    ROOT / "docs" / "_static" / "owned_finite_beta_sfincs_jax_production_ladder_audit"
)
SMOKE_JSON = ROOT / "docs" / "_static" / "owned_finite_beta_sfincs_jax_inputs.json"
PRODUCTION_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_sfincs_jax_production_ladder.json"
)
CONDITIONING_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_current_conditioning_audit.json"
)
OBSERVABLE_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_profile_current_observable_audit.json"
)
COEFFICIENT_GATE = 1.0e-1
EPS = 1.0e-30


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _grid_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    grid = payload["inputs"]["grid"]
    return {
        "Ntheta": int(grid["Ntheta"]),
        "Nzeta": int(grid["Nzeta"]),
        "Nxi": int(grid["Nxi"]),
        "Nx": int(grid.get("Nx", 1)),
    }


def _relative_row(deck: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    summary = deck.get("transport_summary")
    if not isinstance(summary, dict) or summary.get("status") != "complete":
        return None
    same_grid = summary.get("ntx_same_grid")
    if not isinstance(same_grid, dict) or same_grid.get("status") != "complete":
        return None
    relative = same_grid.get("relative_difference", {})
    required = (
        "L13_bridge_vs_sfincs",
        "L31_bridge_vs_sfincs",
        "L33_bridge_vs_sfincs",
    )
    if any(key not in relative for key in required):
        return None
    channel_values = {key: float(relative[key]) for key in required}
    sidecar = relative.get("L33_spitzer_bridge_vs_sfincs")
    return {
        "case_id": str(deck.get("case_id", "")),
        "rho": float(deck["rho"]),
        "nu_prime": float(deck["nu_prime"]),
        "e_star": float(deck["e_star"]),
        "status": str(deck.get("status", "")),
        "seconds": None if deck.get("seconds") is None else float(deck["seconds"]),
        "grid": _grid_from_payload(payload),
        "relative_difference": {
            **channel_values,
            "L33_spitzer_bridge_vs_sfincs": (
                None if sidecar is None else float(sidecar)
            ),
        },
        "max_transport_relative_difference": float(max(channel_values.values())),
    }


def _completed_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for deck in payload.get("decks", []):
        row = _relative_row(deck, payload)
        if row is not None:
            rows.append(row)
    return rows


def _key(row: dict[str, Any]) -> tuple[str, float, float, float]:
    return (
        str(row["case_id"]),
        round(float(row["rho"]), 12),
        round(float(row["nu_prime"]), 12),
        round(float(row["e_star"]), 12),
    )


def _interp_metric(rows: list[dict[str, Any]], key: str, rho: float) -> float | None:
    source = [
        (float(row["rho"]), float(row[key]))
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    if not source:
        return None
    source.sort(key=lambda item: item[0])
    source_rho = np.asarray([item[0] for item in source], dtype=float)
    values = np.asarray([item[1] for item in source], dtype=float)
    if float(rho) < float(source_rho[0]) or float(rho) > float(source_rho[-1]):
        return None
    return float(np.interp(float(rho), source_rho, values))


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _rows(
    *,
    smoke_payload: dict[str, Any],
    production_payload: dict[str, Any],
    conditioning_payload: dict[str, Any],
    observable_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    smoke_by_key = {_key(row): row for row in _completed_rows(smoke_payload)}
    conditioning_rows = conditioning_payload.get("rows", [])
    observable_rows = observable_payload.get("rows", [])

    rows: list[dict[str, Any]] = []
    for production in _completed_rows(production_payload):
        smoke = smoke_by_key.get(_key(production))
        rho = float(production["rho"])
        required = _interp_metric(
            conditioning_rows,
            "required_coefficient_relative_difference_for_current_gate",
            rho,
        )
        profile_error = _interp_metric(
            observable_rows,
            "relative_error_total_vs_redl",
            rho,
        )
        production_max = float(production["max_transport_relative_difference"])
        smoke_max = (
            None
            if smoke is None
            else float(smoke["max_transport_relative_difference"])
        )
        precision_gap = None if required is None else production_max / max(required, EPS)
        rows.append(
            {
                "case_id": production["case_id"],
                "rho": rho,
                "nu_prime": float(production["nu_prime"]),
                "e_star": float(production["e_star"]),
                "production_seconds": production["seconds"],
                "smoke_grid": None if smoke is None else smoke["grid"],
                "production_grid": production["grid"],
                "smoke_max_transport_relative_difference": _finite_or_none(smoke_max),
                "production_max_transport_relative_difference": production_max,
                "production_change_vs_smoke": (
                    None
                    if smoke_max is None
                    else (production_max - smoke_max) / max(abs(smoke_max), EPS)
                ),
                "required_coefficient_relative_difference_for_current_gate": (
                    _finite_or_none(required)
                ),
                "production_precision_gap_to_current_gate": _finite_or_none(
                    precision_gap
                ),
                "profile_current_relative_difference": _finite_or_none(profile_error),
                "relative_difference": production["relative_difference"],
            }
        )
    return sorted(rows, key=lambda row: (row["rho"], row["nu_prime"], row["e_star"]))


def _row_with_max(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    finite_rows = [row for row in rows if row.get(key) is not None]
    if not finite_rows:
        return None
    return max(finite_rows, key=lambda row: float(row[key]))


def build_payload(
    *,
    smoke_json: Path = SMOKE_JSON,
    production_json: Path = PRODUCTION_JSON,
    conditioning_json: Path = CONDITIONING_JSON,
    observable_json: Path = OBSERVABLE_JSON,
) -> dict[str, Any]:
    smoke_payload = _load_json(smoke_json)
    production_payload = _load_json(production_json)
    conditioning_payload = _load_json(conditioning_json)
    observable_payload = _load_json(observable_json)
    rows = _rows(
        smoke_payload=smoke_payload,
        production_payload=production_payload,
        conditioning_payload=conditioning_payload,
        observable_payload=observable_payload,
    )
    if not rows:
        raise ValueError("production ladder has no completed same-grid rows")

    max_production = _row_with_max(rows, "production_max_transport_relative_difference")
    max_gap = _row_with_max(rows, "production_precision_gap_to_current_gate")
    max_profile = _row_with_max(rows, "profile_current_relative_difference")
    completed_count = len(rows)
    production_seconds = [
        float(row["production_seconds"])
        for row in rows
        if row.get("production_seconds") is not None
    ]
    precision_gaps = [
        float(row["production_precision_gap_to_current_gate"])
        for row in rows
        if row.get("production_precision_gap_to_current_gate") is not None
    ]
    production_errors = [
        float(row["production_max_transport_relative_difference"]) for row in rows
    ]

    return {
        "benchmark": "owned_finite_beta_sfincs_jax_production_ladder_audit",
        "classification": "owned finite-beta production same-grid coefficient ladder",
        "claim_scope": (
            "Audits the completed production-resolution finite-beta QA "
            "SFINCS-JAX/NTX same-grid transport-matrix ladder across the "
            "profile-relevant radii and collisionalities. This closes the "
            "coefficient-resolution breadth check for the finite-beta stress "
            "case, while keeping bootstrap-current parity open until the "
            "profile-current closure clears the current-conditioned precision "
            "target."
        ),
        "inputs": {
            "smoke_json": str(Path(smoke_json)),
            "production_json": str(Path(production_json)),
            "conditioning_json": str(Path(conditioning_json)),
            "observable_json": str(Path(observable_json)),
            "coefficient_gate": COEFFICIENT_GATE,
        },
        "rows": rows,
        "stress_row": max_gap,
        "summary_metrics": {
            "completed_production_ladder_count": completed_count,
            "max_production_transport_relative_difference": (
                None
                if max_production is None
                else max_production["production_max_transport_relative_difference"]
            ),
            "max_production_transport_relative_difference_rho": (
                None if max_production is None else max_production["rho"]
            ),
            "max_production_transport_relative_difference_nuPrime": (
                None if max_production is None else max_production["nu_prime"]
            ),
            "max_production_precision_gap_to_current_gate": (
                None
                if max_gap is None
                else max_gap["production_precision_gap_to_current_gate"]
            ),
            "max_production_precision_gap_rho": (
                None if max_gap is None else max_gap["rho"]
            ),
            "max_production_precision_gap_nuPrime": (
                None if max_gap is None else max_gap["nu_prime"]
            ),
            "max_profile_current_relative_difference_on_ladder": (
                None
                if max_profile is None
                else max_profile["profile_current_relative_difference"]
            ),
            "mean_production_seconds": (
                None
                if not production_seconds
                else float(np.mean(np.asarray(production_seconds, dtype=float)))
            ),
            "coefficient_gate": COEFFICIENT_GATE,
            "production_ladder_coefficient_gate_pass": (
                max(production_errors) <= COEFFICIENT_GATE
            ),
            "production_ladder_current_conditioned_gate_pass": (
                bool(precision_gaps) and max(precision_gaps) <= 1.0
            ),
        },
        "conclusion": (
            "The production radial/collisionality ladder keeps the same-grid "
            "finite-beta coefficient differences below the order-1e-1 "
            "coefficient gate at every completed point. The current-conditioned "
            "precision target is still missed at the most cancellation-sensitive "
            "radius, so "
            "the remaining bootstrap-current work stays assigned to the "
            "profile-current observable and reduced closure, not to a broad "
            "production-resolution coefficient failure."
        ),
        "open_work": [
            (
                "apply the same production contract to profile-current closure "
                "diagnostics before promoting finite-beta bootstrap parity"
            ),
            (
                "test any reduced-closure change against the fixed-field QA/QH "
                "current gate and W7-X integrated transfer"
            ),
            (
                "extend the owned production ladder to QH/QI and W7-X-family "
                "inputs only after those independent-reference cases are owned"
            ),
        ],
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
    }


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def _axis_values(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(sorted({float(row["rho"]) for row in rows}), dtype=float)
    nu = np.asarray(sorted({float(row["nu_prime"]) for row in rows}), dtype=float)
    return rho, nu


def _matrix(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    rho, nu = _axis_values(rows)
    values = np.full((nu.size, rho.size), np.nan)
    for row in rows:
        rho_index = int(np.where(np.isclose(rho, float(row["rho"])))[0][0])
        nu_index = int(np.where(np.isclose(nu, float(row["nu_prime"])))[0][0])
        if row.get(key) is not None:
            values[nu_index, rho_index] = float(row[key])
    return values


def _format_axis(values: np.ndarray) -> list[str]:
    return [f"{value:.3g}" for value in values]


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    rows = payload["rows"]
    rho, nu = _axis_values(rows)
    coefficient = _matrix(rows, "production_max_transport_relative_difference")
    smoke = _matrix(rows, "smoke_max_transport_relative_difference")

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    ax_coeff, ax_change, ax_gap, ax_budget = axes.ravel()

    image = ax_coeff.imshow(coefficient, aspect="auto", origin="lower", cmap="viridis")
    ax_coeff.set_xticks(np.arange(rho.size))
    ax_coeff.set_xticklabels(_format_axis(rho))
    ax_coeff.set_yticks(np.arange(nu.size))
    ax_coeff.set_yticklabels(_format_axis(nu))
    ax_coeff.set_xlabel(r"$\rho$")
    ax_coeff.set_ylabel(r"$\nu'$")
    ax_coeff.set_title("(a) Production coefficient difference")
    for (i, j), value in np.ndenumerate(coefficient):
        if np.isfinite(value):
            ax_coeff.text(j, i, f"{value:.2e}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax_coeff, fraction=0.046, pad=0.04)

    finite = np.isfinite(smoke) & np.isfinite(coefficient)
    if np.any(finite):
        ax_change.scatter(smoke[finite], coefficient[finite], s=52, color="#0072b2")
        lower = float(np.nanmin([np.nanmin(smoke[finite]), np.nanmin(coefficient[finite])]))
        upper = float(np.nanmax([np.nanmax(smoke[finite]), np.nanmax(coefficient[finite])]))
        ax_change.plot([lower, upper], [lower, upper], color="0.35", ls="--")
    ax_change.set_xscale("log")
    ax_change.set_yscale("log")
    ax_change.set_xlabel("smoke-grid max relative difference")
    ax_change.set_ylabel("production-grid max relative difference")
    ax_change.set_title("(b) Smoke-to-production transfer")
    ax_change.grid(alpha=0.25, which="both")

    for nu_value in nu:
        subset = [row for row in rows if np.isclose(float(row["nu_prime"]), nu_value)]
        subset.sort(key=lambda row: float(row["rho"]))
        ax_gap.semilogy(
            [float(row["rho"]) for row in subset],
            [float(row["production_precision_gap_to_current_gate"]) for row in subset],
            marker="o",
            label=rf"$\nu'={nu_value:.3g}$",
        )
    ax_gap.axhline(1.0, color="0.35", ls="--")
    ax_gap.set_xlabel(r"$\rho$")
    ax_gap.set_ylabel("coefficient diff / required diff")
    ax_gap.set_title("(c) Current-conditioned precision gap")
    ax_gap.grid(alpha=0.25, which="both")
    ax_gap.legend(fontsize=8)

    max_by_rho: list[float] = []
    required_by_rho: list[float] = []
    profile_by_rho: list[float] = []
    for rho_value in rho:
        subset = [row for row in rows if np.isclose(float(row["rho"]), rho_value)]
        max_by_rho.append(
            max(float(row["production_max_transport_relative_difference"]) for row in subset)
        )
        required_values = [
            float(row["required_coefficient_relative_difference_for_current_gate"])
            for row in subset
            if row.get("required_coefficient_relative_difference_for_current_gate") is not None
        ]
        profile_values = [
            float(row["profile_current_relative_difference"])
            for row in subset
            if row.get("profile_current_relative_difference") is not None
        ]
        required_by_rho.append(max(required_values) if required_values else np.nan)
        profile_by_rho.append(max(profile_values) if profile_values else np.nan)
    ax_budget.semilogy(rho, max_by_rho, marker="o", label="production coefficient")
    ax_budget.semilogy(rho, required_by_rho, marker="s", label="required for current")
    ax_budget.semilogy(rho, profile_by_rho, marker="^", label="profile-current error")
    ax_budget.axhline(COEFFICIENT_GATE, color="0.35", ls=":")
    ax_budget.set_xlabel(r"$\rho$")
    ax_budget.set_ylabel("relative scale")
    ax_budget.set_title("(d) Ladder error budget")
    ax_budget.grid(alpha=0.25, which="both")
    ax_budget.legend(fontsize=8)

    fig.suptitle("Owned finite-beta production SFINCS-JAX coefficient ladder", fontsize=13)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-json", type=Path, default=SMOKE_JSON)
    parser.add_argument("--production-json", type=Path, default=PRODUCTION_JSON)
    parser.add_argument("--conditioning-json", type=Path, default=CONDITIONING_JSON)
    parser.add_argument("--observable-json", type=Path, default=OBSERVABLE_JSON)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    args = parser.parse_args()

    payload = build_payload(
        smoke_json=args.smoke_json,
        production_json=args.production_json,
        conditioning_json=args.conditioning_json,
        observable_json=args.observable_json,
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(json.dumps(payload["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
