#!/usr/bin/env python3
"""Audit the finite-beta profile-current observable at the closure layer.

This script reads committed finite-beta sidecars and does not launch transport
solves.  It answers a narrower question than the coefficient-localization
figure: once the monoenergetic coefficients are on the same grid, does the
profile-current observable fail because the reduced momentum correction has the
wrong sign, insufficient amplitude, poor Pmax convergence, or a suspicious
profile/geometry driver?
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

OUTPUT_PREFIX = (
    ROOT / "docs" / "_static" / "owned_finite_beta_profile_current_observable_audit"
)
BOOTSTRAP_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_bootstrap_comparison.json"
)
CLOSURE_JSON = ROOT / "docs" / "_static" / "owned_finite_beta_closure_localization.json"
PROFILE_CURRENT_GATE = 1.0e-1
EPS = 1.0e-30


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _interp(rho_source: np.ndarray, values: np.ndarray, rho_target: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(rho_target, dtype=float), rho_source, values)


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), EPS)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _profile_drivers(
    bootstrap_payload: dict[str, Any],
    rho: np.ndarray,
) -> dict[str, np.ndarray]:
    redl = bootstrap_payload["redl"]
    ntx = bootstrap_payload["ntx_neopax"]
    redl_rho = np.asarray(redl["rho"], dtype=float)
    ntx_rho = np.asarray(ntx["rho"], dtype=float)
    return {
        "epsilon": _interp(redl_rho, np.asarray(redl["epsilon"], dtype=float), rho),
        "trapped_fraction": _interp(
            redl_rho,
            np.asarray(redl["trapped_fraction"], dtype=float),
            rho,
        ),
        "redl_L31": _interp(redl_rho, np.asarray(redl["L31"], dtype=float), rho),
        "redl_L32": _interp(redl_rho, np.asarray(redl["L32"], dtype=float), rho),
        "redl_alpha": _interp(redl_rho, np.asarray(redl["alpha"], dtype=float), rho),
        "nu_e_star": _interp(redl_rho, np.asarray(redl["nu_e_star"], dtype=float), rho),
        "nu_i_star": _interp(redl_rho, np.asarray(redl["nu_i_star"], dtype=float), rho),
        "density": _interp(ntx_rho, np.asarray(ntx["density"], dtype=float), rho),
        "temperature": _interp(ntx_rho, np.asarray(ntx["temperature"], dtype=float), rho),
        "A1_electron": _interp(ntx_rho, np.asarray(ntx["A1_electron"], dtype=float), rho),
        "A2_electron": _interp(ntx_rho, np.asarray(ntx["A2_electron"], dtype=float), rho),
        "L31_electron": _interp(
            ntx_rho,
            np.asarray(ntx["L31_electron"], dtype=float),
            rho,
        ),
        "L32_electron": _interp(
            ntx_rho,
            np.asarray(ntx["L32_electron"], dtype=float),
            rho,
        ),
    }


def _rows(bootstrap_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    comparison = bootstrap_payload["comparison"]
    rho = np.asarray(comparison["rho"], dtype=float)
    redl_current = np.asarray(comparison["redl_current_over_root_fsab2"], dtype=float)
    nomom_current = np.asarray(
        comparison["ntx_neopax_nomom_over_root_fsab2"],
        dtype=float,
    )
    total_current = np.asarray(
        comparison["ntx_neopax_total_over_root_fsab2"],
        dtype=float,
    )
    applied_correction = total_current - nomom_current
    needed_correction = redl_current - nomom_current
    correction_fraction = applied_correction / np.where(
        np.abs(needed_correction) > EPS,
        needed_correction,
        np.nan,
    )
    residual_after_correction = redl_current - total_current
    residual_fraction = residual_after_correction / np.where(
        np.abs(needed_correction) > EPS,
        needed_correction,
        np.nan,
    )
    total_error = _relative_error(redl_current, total_current)
    nomom_error = _relative_error(redl_current, nomom_current)
    drivers = _profile_drivers(bootstrap_payload, rho)
    ntx = bootstrap_payload["ntx_neopax"]
    ntx_rho = np.asarray(ntx["rho"], dtype=float)
    root_fsab2 = np.asarray(ntx["root_fsab2"], dtype=float)
    species_nomom = np.asarray(ntx["current_nomom_species"], dtype=float)
    species_total = np.asarray(ntx["current_total_species"], dtype=float)
    species_correction = species_total - species_nomom
    species_correction_over_root = np.asarray(
        [
            _interp(
                ntx_rho,
                species_correction[index] / np.maximum(root_fsab2, EPS),
                rho,
            )
            for index in range(species_correction.shape[0])
        ],
        dtype=float,
    )
    species_total_over_root = np.asarray(
        [
            _interp(
                ntx_rho,
                species_total[index] / np.maximum(root_fsab2, EPS),
                rho,
            )
            for index in range(species_total.shape[0])
        ],
        dtype=float,
    )
    species_correction_l1 = np.sum(np.abs(species_correction_over_root), axis=0)
    species_cancellation_amplification = species_correction_l1 / np.maximum(
        np.abs(applied_correction),
        EPS,
    )
    residual_over_species_l1 = np.abs(residual_after_correction) / np.maximum(
        species_correction_l1,
        EPS,
    )
    applied_over_species_l1 = np.abs(applied_correction) / np.maximum(
        species_correction_l1,
        EPS,
    )
    needed_over_species_l1 = np.abs(needed_correction) / np.maximum(
        species_correction_l1,
        EPS,
    )

    rows: list[dict[str, Any]] = []
    for index, rho_value in enumerate(rho):
        rows.append(
            {
                "rho": float(rho_value),
                "redl_current_over_root_fsab2": float(redl_current[index]),
                "ntx_neopax_nomom_over_root_fsab2": float(nomom_current[index]),
                "ntx_neopax_total_over_root_fsab2": float(total_current[index]),
                "applied_momentum_correction_over_root_fsab2": float(
                    applied_correction[index]
                ),
                "needed_momentum_correction_to_redl_over_root_fsab2": float(
                    needed_correction[index]
                ),
                "applied_over_needed_correction": _finite_or_none(
                    correction_fraction[index]
                ),
                "residual_after_correction_over_needed": _finite_or_none(
                    residual_fraction[index]
                ),
                "relative_error_nomom_vs_redl": float(nomom_error[index]),
                "relative_error_total_vs_redl": float(total_error[index]),
                "correction_sign_matches_needed": bool(
                    np.sign(applied_correction[index])
                    == np.sign(needed_correction[index])
                ),
                "species_current_total_over_root_fsab2": [
                    float(value) for value in species_total_over_root[:, index]
                ],
                "species_momentum_correction_over_root_fsab2": [
                    float(value) for value in species_correction_over_root[:, index]
                ],
                "species_momentum_correction_l1_over_root_fsab2": float(
                    species_correction_l1[index]
                ),
                "species_correction_cancellation_amplification": float(
                    species_cancellation_amplification[index]
                ),
                "residual_after_correction_over_species_correction_l1": float(
                    residual_over_species_l1[index]
                ),
                "applied_correction_over_species_correction_l1": float(
                    applied_over_species_l1[index]
                ),
                "needed_correction_over_species_correction_l1": float(
                    needed_over_species_l1[index]
                ),
                "profile_drivers": {
                    key: float(value[index]) for key, value in drivers.items()
                },
            }
        )
    arrays = {
        "rho": rho,
        "redl_current": redl_current,
        "nomom_current": nomom_current,
        "total_current": total_current,
        "applied_correction": applied_correction,
        "needed_correction": needed_correction,
        "correction_fraction": correction_fraction,
        "residual_fraction": residual_fraction,
        "total_error": total_error,
        "nomom_error": nomom_error,
        "species_correction_l1": species_correction_l1,
        "species_cancellation_amplification": species_cancellation_amplification,
        "residual_over_species_l1": residual_over_species_l1,
        **drivers,
    }
    return rows, arrays


def _stress_order_scan(
    bootstrap_payload: dict[str, Any],
    *,
    stress_index: int,
) -> list[dict[str, float]]:
    comparison = bootstrap_payload["comparison"]
    entries: list[dict[str, float]] = []
    for key, value in sorted(
        comparison.get("momentum_order_scan", {}).items(),
        key=lambda item: int(item[0]),
    ):
        errors = np.asarray(value["relative_error_total_vs_redl"], dtype=float)
        currents = np.asarray(value["ntx_neopax_total_over_root_fsab2"], dtype=float)
        entries.append(
            {
                "n_order": int(value.get("n_order", int(key))),
                "stress_relative_error_total_vs_redl": float(errors[stress_index]),
                "stress_current_over_root_fsab2": float(currents[stress_index]),
                "max_relative_error_total_vs_redl": float(
                    value["max_relative_error_total_vs_redl"]
                ),
                "rms_relative_error_total_vs_redl": float(
                    value["rms_relative_error_total_vs_redl"]
                ),
            }
        )
    return entries


def build_payload(
    *,
    bootstrap_json: Path = BOOTSTRAP_JSON,
    closure_json: Path = CLOSURE_JSON,
) -> dict[str, Any]:
    bootstrap_payload = _load_json(bootstrap_json)
    closure_payload = _load_json(closure_json)
    rows, arrays = _rows(bootstrap_payload)
    stress_index = int(np.nanargmax(arrays["total_error"]))
    stress_row = rows[stress_index]
    order_scan = _stress_order_scan(bootstrap_payload, stress_index=stress_index)
    order_errors = np.asarray(
        [entry["stress_relative_error_total_vs_redl"] for entry in order_scan],
        dtype=float,
    )
    correction_fraction = arrays["correction_fraction"]
    residual_fraction = arrays["residual_fraction"]
    sign_matches = [row["correction_sign_matches_needed"] for row in rows]
    pmax_monotone = bool(
        order_errors.size <= 1 or np.all(np.diff(order_errors) <= 1.0e-12)
    )
    pmax_reduction = None
    if order_errors.size >= 2 and order_errors[-1] > 0.0:
        pmax_reduction = float(order_errors[0] / order_errors[-1])

    return {
        "benchmark": "owned_finite_beta_profile_current_observable_audit",
        "classification": "owned finite-beta profile-current observable audit",
        "claim_scope": (
            "Reads the finite-beta Redl and NTX+NEOPAX bootstrap-current stress "
            "sidecar and decomposes the remaining profile-current gap into the "
            "no-momentum current, applied momentum correction, correction needed "
            "to hit the Redl target, local Redl/profile drivers, and Pmax trend. "
            "It is a reduced-closure diagnostic, not a parity claim."
        ),
        "inputs": {
            "bootstrap_artifact": str(bootstrap_json),
            "closure_localization_artifact": str(closure_json),
            "current_units": "<J dot B> / sqrt(<B^2>)",
            "profile_current_gate": PROFILE_CURRENT_GATE,
        },
        "rows": rows,
        "stress_radius": stress_row,
        "momentum_order_at_stress_radius": order_scan,
        "coefficient_localization_summary": closure_payload["summary_metrics"],
        "summary_metrics": {
            "stress_rho": stress_row["rho"],
            "stress_relative_error_total_vs_redl": stress_row[
                "relative_error_total_vs_redl"
            ],
            "stress_relative_error_nomom_vs_redl": stress_row[
                "relative_error_nomom_vs_redl"
            ],
            "stress_applied_over_needed_correction": stress_row[
                "applied_over_needed_correction"
            ],
            "stress_residual_after_correction_over_needed": stress_row[
                "residual_after_correction_over_needed"
            ],
            "stress_species_correction_l1_over_root_fsab2": stress_row[
                "species_momentum_correction_l1_over_root_fsab2"
            ],
            "stress_species_correction_cancellation_amplification": stress_row[
                "species_correction_cancellation_amplification"
            ],
            "stress_residual_after_correction_over_species_correction_l1": stress_row[
                "residual_after_correction_over_species_correction_l1"
            ],
            "stress_applied_correction_over_species_correction_l1": stress_row[
                "applied_correction_over_species_correction_l1"
            ],
            "stress_needed_correction_over_species_correction_l1": stress_row[
                "needed_correction_over_species_correction_l1"
            ],
            "correction_sign_agreement_fraction": float(np.mean(sign_matches)),
            "min_applied_over_needed_correction": float(
                np.nanmin(correction_fraction)
            ),
            "max_applied_over_needed_correction": float(
                np.nanmax(correction_fraction)
            ),
            "max_abs_residual_after_correction_over_needed": float(
                np.nanmax(np.abs(residual_fraction))
            ),
            "max_species_correction_cancellation_amplification": float(
                np.nanmax(arrays["species_cancellation_amplification"])
            ),
            "max_residual_after_correction_over_species_correction_l1": float(
                np.nanmax(arrays["residual_over_species_l1"])
            ),
            "pmax_stress_error_monotone_nonincreasing": pmax_monotone,
            "pmax_stress_error_reduction": pmax_reduction,
            "profile_current_gate": PROFILE_CURRENT_GATE,
            "profile_current_gate_pass": (
                stress_row["relative_error_total_vs_redl"] <= PROFILE_CURRENT_GATE
            ),
        },
        "conclusion": (
            "At the finite-beta stress radius the correction has the right sign "
            "but applies less momentum correction than the Redl target would "
            "require. Increasing Pmax reduces the stress error monotonically, "
            "but the P=12 reduced closure remains above the 1e-1 current gate. "
            "The stress-radius current is also a cancellation-dominated "
            "species-current observable, so small species-flow imbalances are "
            "amplified in the net current."
        ),
        "open_work": [
            (
                "derive the missing reduced momentum/profile-current observable "
                "term that changes correction amplitude without fitting a radius "
                "or device-specific threshold"
            ),
            (
                "validate the correction-amplitude change against production "
                "same-grid SFINCS-JAX profile-current diagnostics"
            ),
            (
                "require no regression of the existing fixed-field total-current "
                "stress gate and integrated W7-X transfer"
            ),
        ],
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
    }


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    rows = payload["rows"]
    rho = np.asarray([row["rho"] for row in rows], dtype=float)
    redl = (
        np.asarray([row["redl_current_over_root_fsab2"] for row in rows], dtype=float)
        / 1.0e6
    )
    nomom = (
        np.asarray(
            [row["ntx_neopax_nomom_over_root_fsab2"] for row in rows],
            dtype=float,
        )
        / 1.0e6
    )
    total = (
        np.asarray(
            [row["ntx_neopax_total_over_root_fsab2"] for row in rows],
            dtype=float,
        )
        / 1.0e6
    )
    correction_fraction = np.asarray(
        [
            np.nan
            if row["applied_over_needed_correction"] is None
            else row["applied_over_needed_correction"]
            for row in rows
        ],
        dtype=float,
    )
    residual_fraction = np.asarray(
        [
            np.nan
            if row["residual_after_correction_over_needed"] is None
            else row["residual_after_correction_over_needed"]
            for row in rows
        ],
        dtype=float,
    )
    stress_rho = float(payload["summary_metrics"]["stress_rho"])
    order = payload["momentum_order_at_stress_radius"]

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    ax_current, ax_correction, ax_order, ax_drivers = axes.ravel()

    ax_current.plot(rho, redl, marker="o", label="Redl")
    ax_current.plot(rho, nomom, marker="^", label="no momentum correction")
    ax_current.plot(rho, total, marker="s", label="corrected")
    ax_current.axvline(stress_rho, color="0.35", linestyle=":")
    ax_current.set_xlabel(r"$\rho$")
    ax_current.set_ylabel(r"current / $10^6$")
    ax_current.set_title("(a) Profile-current observable")
    ax_current.grid(alpha=0.25)
    ax_current.legend(fontsize=8)

    ax_correction.plot(rho, correction_fraction, marker="o", label="applied / needed")
    ax_correction.plot(
        rho,
        np.abs(residual_fraction),
        marker="s",
        label="residual / needed",
    )
    ax_correction.axhline(1.0, color="0.35", linestyle=":")
    ax_correction.axvline(stress_rho, color="0.35", linestyle=":")
    ax_correction.set_xlabel(r"$\rho$")
    ax_correction.set_ylabel("fraction")
    ax_correction.set_title("(b) Momentum-correction amplitude")
    ax_correction.grid(alpha=0.25)
    ax_correction.legend(fontsize=8)
    stress_residual_l1 = payload["summary_metrics"][
        "stress_residual_after_correction_over_species_correction_l1"
    ]
    ax_correction.text(
        0.03,
        0.92,
        rf"stress residual / species L1 = {stress_residual_l1:.2e}",
        transform=ax_correction.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "pad": 2.0},
    )

    if order:
        pmax = np.asarray([entry["n_order"] for entry in order], dtype=float)
        p_error = np.asarray(
            [entry["stress_relative_error_total_vs_redl"] for entry in order],
            dtype=float,
        )
        ax_order.semilogy(pmax, p_error, marker="o", label="stress radius")
    ax_order.axhline(PROFILE_CURRENT_GATE, color="0.35", linestyle=":", label="1e-1 gate")
    ax_order.set_xlabel(r"$P_{max}$")
    ax_order.set_ylabel("relative difference")
    ax_order.set_title("(c) Pmax at stress radius")
    ax_order.grid(alpha=0.25, which="both")
    ax_order.legend(fontsize=8)

    driver_keys = (
        ("epsilon", r"$\epsilon$"),
        ("trapped_fraction", r"$f_t$"),
        ("nu_e_star", r"$\nu_e^*$"),
        ("nu_i_star", r"$\nu_i^*$"),
    )
    for key, label in driver_keys:
        values = np.asarray([row["profile_drivers"][key] for row in rows], dtype=float)
        ax_drivers.plot(rho, values, marker="o", label=label)
    ax_drivers.axvline(stress_rho, color="0.35", linestyle=":")
    ax_drivers.set_xlabel(r"$\rho$")
    ax_drivers.set_ylabel("dimensionless")
    ax_drivers.set_title("(d) Local Redl/profile drivers")
    ax_drivers.grid(alpha=0.25)
    ax_drivers.legend(fontsize=8)

    fig.suptitle("Owned finite-beta profile-current observable audit", fontsize=13)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-json", type=Path, default=BOOTSTRAP_JSON)
    parser.add_argument("--closure-json", type=Path, default=CLOSURE_JSON)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    args = parser.parse_args()

    payload = build_payload(
        bootstrap_json=args.bootstrap_json,
        closure_json=args.closure_json,
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(json.dumps(payload["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
