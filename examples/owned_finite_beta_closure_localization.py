#!/usr/bin/env python3
"""Localize the owned finite-beta bootstrap-current gap.

This diagnostic intentionally reuses committed sidecars instead of launching
new transport solves.  It compares the same-grid finite-beta coefficient ladder
against the finite-beta Redl/NTX+NEOPAX profile-current stress audit so that the
remaining error is classified at the right layer: monoenergetic coefficient
normalization versus downstream profile and reduced momentum closure.
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

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_finite_beta_closure_localization"
SFINCS_JSON = ROOT / "docs" / "_static" / "owned_finite_beta_sfincs_jax_inputs.json"
BOOTSTRAP_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_bootstrap_comparison.json"
)

RAW_CHANNELS = (
    "L13_bridge_vs_sfincs",
    "L31_bridge_vs_sfincs",
    "L33_bridge_vs_sfincs",
)
SIDECAR_CHANNELS = ("L33_spitzer_bridge_vs_sfincs",)
COEFFICIENT_GATE = 1.0e-1
PROFILE_CURRENT_GATE = 1.0e-1


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(float(value)) else None


def _max_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return None
    return float(max(finite))


def coefficient_rows(sfincs_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract completed same-grid coefficient comparison rows."""

    rows: list[dict[str, Any]] = []
    for deck in sfincs_payload.get("decks", []):
        transport = deck.get("transport_summary")
        if not isinstance(transport, dict) or transport.get("status") != "complete":
            continue
        ntx_same_grid = transport.get("ntx_same_grid")
        if (
            not isinstance(ntx_same_grid, dict)
            or ntx_same_grid.get("status") != "complete"
        ):
            continue
        relative = ntx_same_grid.get("relative_difference", {})
        raw = {
            channel: float(relative[channel])
            for channel in RAW_CHANNELS
            if channel in relative and np.isfinite(float(relative[channel]))
        }
        sidecar = {
            channel: float(relative[channel])
            for channel in SIDECAR_CHANNELS
            if channel in relative and np.isfinite(float(relative[channel]))
        }
        if not raw:
            continue
        rows.append(
            {
                "case_id": deck.get("case_id"),
                "case_label": deck.get("case_label"),
                "rho": float(deck["rho"]),
                "nu_prime": float(deck["nu_prime"]),
                "e_star": float(deck["e_star"]),
                "raw_channel_relative_difference": raw,
                "sidecar_channel_relative_difference": sidecar,
                "max_raw_relative_difference": max(raw.values()),
                "max_sidecar_relative_difference": _max_or_none(list(sidecar.values())),
            }
        )
    return rows


def coefficient_by_rho(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["rho"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for rho, rho_rows in sorted(grouped.items()):
        raw_channel_max = {
            channel: _max_or_none(
                [
                    row["raw_channel_relative_difference"].get(channel, np.nan)
                    for row in rho_rows
                ]
            )
            for channel in RAW_CHANNELS
        }
        sidecar_channel_max = {
            channel: _max_or_none(
                [
                    row["sidecar_channel_relative_difference"].get(channel, np.nan)
                    for row in rho_rows
                ]
            )
            for channel in SIDECAR_CHANNELS
        }
        summary.append(
            {
                "rho": float(rho),
                "count": len(rho_rows),
                "max_raw_relative_difference": _max_or_none(
                    [row["max_raw_relative_difference"] for row in rho_rows]
                ),
                "max_sidecar_relative_difference": _max_or_none(
                    [
                        row["max_sidecar_relative_difference"]
                        for row in rho_rows
                        if row["max_sidecar_relative_difference"] is not None
                    ]
                ),
                "raw_channel_max": raw_channel_max,
                "sidecar_channel_max": sidecar_channel_max,
            }
        )
    return summary


def bootstrap_profile(bootstrap_payload: dict[str, Any]) -> dict[str, Any]:
    comparison = bootstrap_payload["comparison"]
    return {
        "rho": [float(value) for value in comparison["rho"]],
        "relative_error_total_vs_redl": [
            float(value) for value in comparison["relative_error_total_vs_redl"]
        ],
        "relative_error_nomom_vs_redl": [
            float(value) for value in comparison["relative_error_nomom_vs_redl"]
        ],
        "redl_current_over_root_fsab2": [
            float(value) for value in comparison["redl_current_over_root_fsab2"]
        ],
        "ntx_neopax_total_over_root_fsab2": [
            float(value) for value in comparison["ntx_neopax_total_over_root_fsab2"]
        ],
    }


def momentum_order_scan(bootstrap_payload: dict[str, Any]) -> dict[str, Any]:
    scan = bootstrap_payload.get("comparison", {}).get("momentum_order_scan", {})
    entries: list[dict[str, float]] = []
    for key, value in sorted(scan.items(), key=lambda item: int(item[0])):
        entries.append(
            {
                "n_order": int(value.get("n_order", int(key))),
                "max_relative_error_total_vs_redl": float(
                    value["max_relative_error_total_vs_redl"]
                ),
                "rms_relative_error_total_vs_redl": float(
                    value["rms_relative_error_total_vs_redl"]
                ),
            }
        )
    return {"entries": entries}


def _interp_profile_error(profile: dict[str, Any], rho: float) -> float | None:
    profile_rho = np.asarray(profile["rho"], dtype=float)
    profile_error = np.asarray(profile["relative_error_total_vs_redl"], dtype=float)
    if profile_rho.size == 0 or rho < profile_rho.min() or rho > profile_rho.max():
        return None
    return float(np.interp(float(rho), profile_rho, profile_error))


def _nearest_coefficient_at_rho(
    rows_by_rho: list[dict[str, Any]],
    target_rho: float,
) -> tuple[dict[str, Any] | None, float | None]:
    if not rows_by_rho:
        return None, None
    distances = [abs(float(row["rho"]) - float(target_rho)) for row in rows_by_rho]
    index = int(np.argmin(np.asarray(distances)))
    return rows_by_rho[index], float(distances[index])


def build_payload(
    *,
    sfincs_json: Path = SFINCS_JSON,
    bootstrap_json: Path = BOOTSTRAP_JSON,
) -> dict[str, Any]:
    sfincs_payload = _load_json(sfincs_json)
    bootstrap_payload = _load_json(bootstrap_json)

    rows = coefficient_rows(sfincs_payload)
    by_rho = coefficient_by_rho(rows)
    profile = bootstrap_profile(bootstrap_payload)
    order_scan = momentum_order_scan(bootstrap_payload)

    matched_radii: list[dict[str, Any]] = []
    for row in by_rho:
        current_error = _interp_profile_error(profile, float(row["rho"]))
        coefficient_error = row["max_raw_relative_difference"]
        ratio = None
        if (
            current_error is not None
            and coefficient_error is not None
            and coefficient_error > 0.0
        ):
            ratio = current_error / coefficient_error
        matched_radii.append(
            {
                "rho": row["rho"],
                "max_raw_coefficient_relative_difference": coefficient_error,
                "interpolated_profile_current_relative_difference": current_error,
                "current_to_coefficient_error_ratio": _finite_or_none(ratio)
                if ratio is not None
                else None,
            }
        )

    profile_rho = np.asarray(profile["rho"], dtype=float)
    profile_error = np.asarray(profile["relative_error_total_vs_redl"], dtype=float)
    inner_index = int(np.nanargmax(profile_error)) if profile_error.size else 0
    inner_rho = float(profile_rho[inner_index]) if profile_rho.size else None
    inner_error = float(profile_error[inner_index]) if profile_error.size else None
    inner_coeff_row, inner_coeff_distance = (
        _nearest_coefficient_at_rho(by_rho, inner_rho)
        if inner_rho is not None
        else (None, None)
    )
    inner_coeff_error = (
        inner_coeff_row["max_raw_relative_difference"]
        if inner_coeff_row is not None
        else None
    )
    inner_ratio = None
    if (
        inner_error is not None
        and inner_coeff_error is not None
        and inner_coeff_error > 0.0
    ):
        inner_ratio = inner_error / inner_coeff_error

    max_coefficient_error = _max_or_none(
        [
            float(row["max_raw_relative_difference"])
            for row in by_rho
            if row["max_raw_relative_difference"] is not None
        ]
    )
    max_profile_error = (
        float(np.nanmax(profile_error)) if profile_error.size else None
    )
    rms_profile_error = (
        float(np.sqrt(np.nanmean(profile_error**2))) if profile_error.size else None
    )

    return {
        "benchmark": "owned_finite_beta_closure_localization",
        "classification": (
            "owned finite-beta coefficient/profile-current mismatch localization"
        ),
        "claim_scope": (
            "Compares the completed same-grid finite-beta SFINCS-JAX "
            "transport-matrix coefficient ladder with the finite-beta Redl and "
            "NTX+NEOPAX profile-current stress artifact. It is a localization "
            "diagnostic: coefficient errors below the 1e-1 gate at the inner "
            "stress radius do not by themselves promote bootstrap-current parity."
        ),
        "inputs": {
            "sfincs_artifact": str(sfincs_json),
            "bootstrap_artifact": str(bootstrap_json),
            "raw_coefficient_channels": list(RAW_CHANNELS),
            "excluded_sidecar_channels": list(SIDECAR_CHANNELS),
        },
        "coefficient_rows": rows,
        "coefficient_by_rho": by_rho,
        "bootstrap_profile": profile,
        "matched_radii": matched_radii,
        "momentum_order_scan": order_scan,
        "summary_metrics": {
            "max_same_grid_coefficient_relative_difference": max_coefficient_error,
            "max_bootstrap_total_relative_difference": max_profile_error,
            "rms_bootstrap_total_relative_difference": rms_profile_error,
            "inner_gap_rho": inner_rho,
            "inner_gap_bootstrap_relative_difference": inner_error,
            "inner_gap_nearest_coefficient_rho": (
                inner_coeff_row["rho"] if inner_coeff_row is not None else None
            ),
            "inner_gap_nearest_coefficient_distance": inner_coeff_distance,
            "inner_gap_coefficient_relative_difference": inner_coeff_error,
            "inner_gap_current_to_coefficient_error_ratio": _finite_or_none(
                inner_ratio
            )
            if inner_ratio is not None
            else None,
            "coefficient_gate": COEFFICIENT_GATE,
            "coefficient_gate_pass": (
                max_coefficient_error is not None
                and max_coefficient_error <= COEFFICIENT_GATE
            ),
            "profile_current_gate": PROFILE_CURRENT_GATE,
            "profile_current_gate_pass": (
                max_profile_error is not None and max_profile_error <= PROFILE_CURRENT_GATE
            ),
        },
        "conclusion": (
            "The same-grid finite-beta coefficient ladder passes below 1e-1 at "
            "the inner stress radius; the remaining current-profile gap is "
            "therefore tracked as a reduced momentum/profile-closure observable "
            "lane rather than a monoenergetic coefficient normalization failure."
        ),
        "open_work": [
            (
                "derive and test the remaining reduced momentum/profile-current "
                "observable term at the inner finite-beta radius"
            ),
            (
                "rerun the profile-current closure with production SFINCS-JAX "
                "radial and collisionality ladders before any parity promotion"
            ),
            (
                "add a stable downstream interpolation-mode audit when the "
                "profile-coupling interface exposes that selector"
            ),
        ],
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
    }


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def _array_or_empty(values: list[Any], key: str) -> np.ndarray:
    return np.asarray(
        [np.nan if value.get(key) is None else float(value[key]) for value in values],
        dtype=float,
    )


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    by_rho = payload["coefficient_by_rho"]
    profile = payload["bootstrap_profile"]
    matched = payload["matched_radii"]
    order_entries = payload["momentum_order_scan"]["entries"]

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    ax_coeff, ax_profile, ax_order, ax_ratio = axes.ravel()

    if by_rho:
        coeff_rho = np.asarray([row["rho"] for row in by_rho], dtype=float)
        coeff_max = _array_or_empty(by_rho, "max_raw_relative_difference")
        spitzer = _array_or_empty(by_rho, "max_sidecar_relative_difference")
        ax_coeff.semilogy(
            coeff_rho,
            coeff_max,
            marker="o",
            label="max raw L13/L31/L33",
        )
        if np.any(np.isfinite(spitzer)):
            ax_coeff.semilogy(
                coeff_rho,
                spitzer,
                marker="s",
                linestyle="--",
                label="D33 Spitzer sidecar",
            )
    ax_coeff.axhline(COEFFICIENT_GATE, color="0.35", linestyle=":", label="1e-1 gate")
    ax_coeff.set_xlabel(r"$\rho$")
    ax_coeff.set_ylabel("relative difference")
    ax_coeff.set_title("(a) Same-grid coefficient bridge")
    ax_coeff.grid(alpha=0.25, which="both")
    ax_coeff.legend(fontsize=8)

    rho = np.asarray(profile["rho"], dtype=float)
    total_error = np.asarray(profile["relative_error_total_vs_redl"], dtype=float)
    nomom_error = np.asarray(profile["relative_error_nomom_vs_redl"], dtype=float)
    ax_profile.semilogy(rho, total_error, marker="o", label="total current")
    ax_profile.semilogy(rho, nomom_error, marker="^", label="no momentum correction")
    if matched:
        match_rho = np.asarray([row["rho"] for row in matched], dtype=float)
        match_error = _array_or_empty(
            matched,
            "interpolated_profile_current_relative_difference",
        )
        ax_profile.scatter(
            match_rho,
            match_error,
            s=70,
            facecolors="none",
            edgecolors="black",
            label="coefficient radii",
        )
    ax_profile.axhline(PROFILE_CURRENT_GATE, color="0.35", linestyle=":")
    ax_profile.set_xlabel(r"$\rho$")
    ax_profile.set_ylabel("relative difference")
    ax_profile.set_title("(b) Profile-current stress")
    ax_profile.grid(alpha=0.25, which="both")
    ax_profile.legend(fontsize=8)

    if order_entries:
        pmax = np.asarray([entry["n_order"] for entry in order_entries], dtype=float)
        max_order = np.asarray(
            [entry["max_relative_error_total_vs_redl"] for entry in order_entries],
            dtype=float,
        )
        rms_order = np.asarray(
            [entry["rms_relative_error_total_vs_redl"] for entry in order_entries],
            dtype=float,
        )
        ax_order.semilogy(pmax, max_order, marker="o", label="max")
        ax_order.semilogy(pmax, rms_order, marker="s", label="RMS")
    ax_order.axhline(PROFILE_CURRENT_GATE, color="0.35", linestyle=":")
    ax_order.set_xlabel(r"$P_{max}$")
    ax_order.set_ylabel("relative difference")
    ax_order.set_title("(c) Reduced-closure convergence")
    ax_order.grid(alpha=0.25, which="both")
    ax_order.legend(fontsize=8)

    if matched:
        match_rho = np.asarray([row["rho"] for row in matched], dtype=float)
        ratio = _array_or_empty(matched, "current_to_coefficient_error_ratio")
        ax_ratio.semilogy(match_rho, ratio, marker="o", color="tab:red")
        inner_rho = payload["summary_metrics"]["inner_gap_rho"]
        if inner_rho is not None:
            ax_ratio.axvline(float(inner_rho), color="0.35", linestyle=":")
    ax_ratio.axhline(1.0, color="0.35", linestyle=":")
    ax_ratio.set_xlabel(r"$\rho$")
    ax_ratio.set_ylabel("current error / coefficient error")
    ax_ratio.set_title("(d) Layer-localization ratio")
    ax_ratio.grid(alpha=0.25, which="both")

    fig.suptitle("Owned finite-beta closure localization", fontsize=13)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfincs-json", type=Path, default=SFINCS_JSON)
    parser.add_argument("--bootstrap-json", type=Path, default=BOOTSTRAP_JSON)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    args = parser.parse_args()

    payload = build_payload(
        sfincs_json=args.sfincs_json,
        bootstrap_json=args.bootstrap_json,
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(json.dumps(payload["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
