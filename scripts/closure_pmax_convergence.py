#!/usr/bin/env python3
"""Run higher-order closure convergence studies against fixed-field and W7-X gates."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SRC = ROOT / "src"
for extra_path in (SRC, EXAMPLES):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import bootstrap_current_fixed_field_validation as fixed_field  # noqa: E402
import bootstrap_current_w7x_rebuild_audit as w7x_audit  # noqa: E402


OUTPUT_PREFIX = ROOT / "docs" / "_static" / "closure_pmax_convergence"
INTERIOR_RHO_MIN = fixed_field.INTERIOR_RHO_MIN
INTERIOR_RHO_MAX = fixed_field.INTERIOR_RHO_MAX


def _parse_pmax_levels() -> list[int]:
    raw = os.environ.get("NTX_CLOSURE_PMAX_LIST", "[3, 4, 5, 6]")
    data = json.loads(raw)
    levels = sorted({int(value) for value in data})
    if not levels:
        raise ValueError("NTX_CLOSURE_PMAX_LIST must not be empty")
    return levels


def _relative_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    return np.abs(arr_a - arr_b) / np.maximum(np.abs(arr_b), 1.0)


def _run_fixed_field(order: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fixed_field.NTX_NEOPAX_N_ORDER = order
    cases = fixed_field._cases()
    results = {key: fixed_field.run_case(case) for key, case in cases.items()}
    summary = fixed_field._summary_payload(results, cases)
    profiles = {
        key: np.asarray(payload["NTX+NEOPAX"]["jdotb"], dtype=float)
        for key, payload in results.items()
    }
    return summary, profiles


def _run_w7x(order: int) -> float:
    w7x_audit.NEOPAX_N_ORDER = order
    grid, field, species = w7x_audit._build_species_and_field()
    database = w7x_audit.NEOPAX.Monoenergetic.read_monkes(
        float(field.a_b),
        str(w7x_audit.REFERENCE_PATH),
    )
    current = w7x_audit._bootstrap_current_profile(database, grid, field, species)
    return w7x_audit._max_relative_error(current, w7x_audit.J_FINAL_REFERENCE)


def main() -> None:
    pmax_levels = _parse_pmax_levels()
    fixed_series: dict[int, dict[str, Any]] = {}
    fixed_profiles: dict[int, dict[str, np.ndarray]] = {}
    w7x_errors: dict[int, float] = {}

    for order in pmax_levels:
        fixed_summary, profiles = _run_fixed_field(order)
        fixed_series[order] = fixed_summary
        fixed_profiles[order] = profiles
        w7x_errors[order] = _run_w7x(order)

    successive_changes: list[float] = []
    for prev, curr in zip(pmax_levels[:-1], pmax_levels[1:], strict=True):
        for case in ("qa", "qh"):
            rho = np.asarray(fixed_series[curr]["cases"][case]["SFINCS"]["rho"], dtype=float)
            interior = (rho >= INTERIOR_RHO_MIN) & (rho <= INTERIOR_RHO_MAX)
            successive_changes.append(
                float(
                    np.max(
                        _relative_change(
                            fixed_profiles[curr][case][interior],
                            fixed_profiles[prev][case][interior],
                        )
                    )
                )
            )

    precise_qs_max_successive_change = (
        max(successive_changes) if successive_changes else 0.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    qa_errors = [
        float(
            fixed_series[order]["cases"]["qa"]["max_relative_error_vs_sfincs_interior"][
                "NTX+NEOPAX"
            ]
        )
        for order in pmax_levels
    ]
    qh_errors = [
        float(
            fixed_series[order]["cases"]["qh"]["max_relative_error_vs_sfincs_interior"][
                "NTX+NEOPAX"
            ]
        )
        for order in pmax_levels
    ]

    axes[0].semilogy(pmax_levels, qa_errors, "o-", lw=2.0, label="QA")
    axes[0].semilogy(pmax_levels, qh_errors, "s-", lw=2.0, label="QH")
    axes[0].set_xlabel(r"$P_{\max}$ moments")
    axes[0].set_ylabel("interior max relative error")
    axes[0].set_title("Precise-QS closure stress")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    axes[1].semilogy(
        pmax_levels,
        [w7x_errors[order] for order in pmax_levels],
        "o-",
        lw=2.0,
        color="#d62728",
    )
    axes[1].axhline(2.0e-2, color="0.3", ls="--", lw=1.2)
    axes[1].set_xlabel(r"$P_{\max}$ moments")
    axes[1].set_ylabel("max relative error")
    axes[1].set_title("Integrated W7-X transfer")
    axes[1].grid(alpha=0.25, which="both")

    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    payload = {
        "pmax_levels": pmax_levels,
        "tail_factor": float(os.environ.get("NEOPAX_COLLISION_TAIL_FACTOR", "1.0")),
        "tail_power": float(os.environ.get("NEOPAX_COLLISION_TAIL_POWER", "1.0")),
        "precise_qs": {
            str(order): {
                case: float(
                    fixed_series[order]["cases"][case]["max_relative_error_vs_sfincs_interior"][
                        "NTX+NEOPAX"
                    ]
                )
                for case in ("qa", "qh")
            }
            for order in pmax_levels
        },
        "precise_qs_max_successive_change": precise_qs_max_successive_change,
        "w7x_max_relative_error": max(w7x_errors.values()),
        "w7x_errors": {str(order): float(error) for order, error in w7x_errors.items()},
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png")),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf")),
    }
    OUTPUT_PREFIX.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
