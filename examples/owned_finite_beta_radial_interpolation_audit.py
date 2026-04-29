#!/usr/bin/env python3
"""Audit radial interpolation sensitivity in the finite-beta current stress.

The finite-beta bootstrap-current comparison uses one owned VMEC/Boozer/profile
contract for Redl and NTX+NEOPAX.  This audit isolates one remaining numerical
choice in that contract: whether the monoenergetic database is built on a sparse
radial scan and interpolated to the profile grid, or built directly on the same
field radii used by the profile-current observable.

The output is a diagnostic.  It does not alter the runtime interpolation policy
and does not apply a fitted closure correction.
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

from examples import owned_finite_beta_bootstrap_comparison as bootstrap  # noqa: E402

BASELINE_JSON = ROOT / "docs" / "_static" / "owned_finite_beta_bootstrap_comparison.json"
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_finite_beta_radial_interpolation_audit"
WORKDIR = ROOT / "examples" / "outputs" / "owned_finite_beta_radial_interpolation_audit"
MATCHED_PAYLOAD = WORKDIR / "field_radius_matched_bootstrap.json"
EPS = 1.0e-30


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _array(payload: dict[str, Any], section: str, key: str) -> np.ndarray:
    return np.asarray(payload[section][key], dtype=float)


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), EPS)


def _stress(values: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    index = int(np.nanargmax(values))
    return {"rho": float(rho[index]), "relative_error": float(values[index])}


def _interp_to_rho(
    source_rho: np.ndarray,
    source_values: np.ndarray,
    target_rho: np.ndarray,
) -> np.ndarray:
    return np.interp(
        np.asarray(target_rho, dtype=float),
        np.asarray(source_rho, dtype=float),
        np.asarray(source_values, dtype=float),
    )


def _field_radius_scan_from_baseline(payload: dict[str, Any]) -> tuple[float, ...]:
    """Return the exact profile-comparison radii used by the baseline artifact."""

    return tuple(float(value) for value in payload["comparison"]["rho"])


def build_field_radius_matched_payload(
    baseline_payload: dict[str, Any],
    *,
    output_dir: Path = WORKDIR,
) -> dict[str, Any]:
    """Rebuild the finite-beta comparison with scan radii equal to field radii."""

    inputs = baseline_payload["inputs"]
    grid = inputs["ntx_grid"]
    contract = bootstrap.ProfileContract(
        **baseline_payload.get("profile_contract", {})
    )
    payload = bootstrap.build_payload(
        case_id=str(baseline_payload["case"]["id"]),
        scan_rho=_field_radius_scan_from_baseline(baseline_payload),
        nu_v=tuple(float(value) for value in inputs.get("nu_v", ())),
        es_values=tuple(float(value) for value in inputs["Es"]),
        contract=contract,
        output_dir=output_dir,
        ntx_grid=bootstrap.GridSpec(
            int(grid["n_theta"]),
            int(grid["n_zeta"]),
            int(grid["n_xi"]),
        ),
        field_radial_points=int(inputs["field_radial_points"]),
        neopax_x=int(inputs["neopax_x"]),
        n_order=int(inputs["n_order"]),
        d33_mode=str(inputs["d33_mode"]),
        mboz=int(inputs["mboz"]),
        nboz=int(inputs["nboz"]),
        redl_ntheta=int(inputs["redl_ntheta"]),
        helicity_n=int(inputs["helicity_n"]),
        min_bmn_to_load=float(inputs["min_bmn_to_load"]),
        write_hdf5=True,
        adaptive_nu=True,
        momentum_orders=tuple(int(value) for value in inputs["momentum_orders"]),
    )
    return payload


def write_matched_payload(payload: dict[str, Any], path: Path = MATCHED_PAYLOAD) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_payload_from_comparisons(
    *,
    baseline_payload: dict[str, Any],
    matched_payload: dict[str, Any],
    baseline_path: Path = BASELINE_JSON,
    matched_path: Path = MATCHED_PAYLOAD,
) -> dict[str, Any]:
    rho = _array(baseline_payload, "comparison", "rho")
    matched_rho = _array(matched_payload, "comparison", "rho")
    if rho.shape != matched_rho.shape or not np.allclose(rho, matched_rho):
        raise ValueError("baseline and field-radius-matched payloads must share rho")

    redl = _array(baseline_payload, "comparison", "redl_current_over_root_fsab2")
    baseline_total = _array(
        baseline_payload,
        "comparison",
        "ntx_neopax_total_over_root_fsab2",
    )
    matched_total = _array(
        matched_payload,
        "comparison",
        "ntx_neopax_total_over_root_fsab2",
    )
    baseline_error = _relative_error(redl, baseline_total)
    matched_error = _relative_error(redl, matched_total)
    baseline_scan_rho = np.asarray(baseline_payload["inputs"]["scan_rho"], dtype=float)
    matched_scan_rho = np.asarray(matched_payload["inputs"]["scan_rho"], dtype=float)

    baseline_stress = _stress(baseline_error, rho)
    matched_stress = _stress(matched_error, rho)
    baseline_stress_error_matched = float(
        _interp_to_rho(rho, matched_error, np.asarray([baseline_stress["rho"]]))[0]
    )
    baseline_stress_reduction = (
        float(baseline_stress["relative_error"] / max(baseline_stress_error_matched, EPS))
        if np.isfinite(baseline_stress_error_matched)
        else None
    )
    max_improvement = float(np.nanmax(baseline_error) / max(float(np.nanmax(matched_error)), EPS))
    rms_baseline = float(np.sqrt(np.nanmean(baseline_error**2)))
    rms_matched = float(np.sqrt(np.nanmean(matched_error**2)))
    current_delta = np.abs(matched_total - baseline_total) / np.maximum(np.abs(redl), EPS)
    improvement = baseline_error - matched_error

    rows = []
    for index, radius in enumerate(rho):
        rows.append(
            {
                "rho": float(radius),
                "redl_current_over_root_fsab2": float(redl[index]),
                "baseline_total_over_root_fsab2": float(baseline_total[index]),
                "field_radius_matched_total_over_root_fsab2": float(matched_total[index]),
                "baseline_relative_error": float(baseline_error[index]),
                "field_radius_matched_relative_error": float(matched_error[index]),
                "relative_error_improvement": float(improvement[index]),
                "current_change_over_redl": float(current_delta[index]),
            }
        )

    summary_metrics = {
        "radius_count": int(rho.size),
        "baseline_scan_rho_count": int(baseline_scan_rho.size),
        "field_radius_matched_scan_rho_count": int(matched_scan_rho.size),
        "baseline_max_relative_error_total_vs_redl": float(np.nanmax(baseline_error)),
        "field_radius_matched_max_relative_error_total_vs_redl": float(
            np.nanmax(matched_error)
        ),
        "max_relative_error_improvement_factor": max_improvement,
        "baseline_rms_relative_error_total_vs_redl": rms_baseline,
        "field_radius_matched_rms_relative_error_total_vs_redl": rms_matched,
        "rms_relative_error_improvement_factor": float(
            rms_baseline / max(rms_matched, EPS)
        ),
        "baseline_stress_rho": float(baseline_stress["rho"]),
        "baseline_stress_relative_error": float(baseline_stress["relative_error"]),
        "field_radius_matched_error_at_baseline_stress_rho": baseline_stress_error_matched,
        "baseline_stress_error_reduction_factor": baseline_stress_reduction,
        "field_radius_matched_stress_rho": float(matched_stress["rho"]),
        "field_radius_matched_stress_relative_error": float(
            matched_stress["relative_error"]
        ),
        "max_current_change_over_redl": float(np.nanmax(current_delta)),
        "median_current_change_over_redl": float(np.nanmedian(current_delta)),
        "profile_current_gate": 1.0e-1,
        "field_radius_matched_current_gate_pass": bool(np.nanmax(matched_error) <= 1.0e-1),
        "runtime_interpolation_policy_changed": False,
        "runtime_correction_applied": False,
    }

    return {
        "benchmark": "owned_finite_beta_radial_interpolation_audit",
        "classification": "finite-beta radial interpolation sensitivity diagnostic",
        "claim_scope": (
            "Compares the committed sparse-radius finite-beta NTX+NEOPAX "
            "bootstrap-current artifact with a rebuild whose monoenergetic "
            "database radii are exactly the profile-current field radii.  This "
            "isolates downstream radial interpolation sensitivity while keeping "
            "the same VMEC/Boozer geometry, profiles, nu/v support policy, "
            "D33_spitzer branch, Sonine order, and Redl observable.  It is a "
            "diagnostic only: no runtime interpolation policy or closure "
            "correction is promoted by this artifact."
        ),
        "source_artifacts": {
            "baseline": _artifact_path(baseline_path),
            "field_radius_matched": _artifact_path(matched_path),
        },
        "radial_contract": {
            "baseline_scan_rho": baseline_scan_rho.tolist(),
            "field_radius_matched_scan_rho": matched_scan_rho.tolist(),
            "comparison_rho": rho.tolist(),
            "interpretation": (
                "A field-radius-matched database removes one interpolation "
                "layer from the profile-current observable.  Improvement at a "
                "single stress point is not sufficient for a runtime change; "
                "the global current gate must also pass and the fixed-field/W7-X "
                "transfer gates must remain unchanged."
            ),
        },
        "rows": rows,
        "summary_metrics": summary_metrics,
        "conclusion": (
            "Building the finite-beta database directly on the field radii "
            "strongly reduces the previous inner stress point, but the profile "
            "maximum remains above the 1e-1 current gate.  Radial interpolation "
            "therefore explains part, not all, of the remaining reduced-closure "
            "stress."
        ),
        "open_work": [
            (
                "repeat this audit with a production dense radial scan and the "
                "downstream general-vs-legacy interpolation selector when that "
                "interface is stable"
            ),
            (
                "require the field-radius-matched and sparse-grid contracts to "
                "clear the finite-beta current gate before promoting a new "
                "finite-beta bootstrap-current parity figure"
            ),
            (
                "preserve the fixed-field QA/QH and integrated W7-X transfer "
                "gates for any future runtime interpolation policy change"
            ),
        ],
        "figure_png": _artifact_path(OUTPUT_PREFIX.with_suffix(".png")),
        "figure_pdf": _artifact_path(OUTPUT_PREFIX.with_suffix(".pdf")),
    }


def build_payload(
    *,
    baseline_json: Path = BASELINE_JSON,
    matched_json: Path | None = None,
    matched_output_dir: Path = WORKDIR,
    rebuild_matched: bool = False,
) -> dict[str, Any]:
    baseline_payload = _load_json(baseline_json)
    matched_path = matched_json or MATCHED_PAYLOAD
    if rebuild_matched or not matched_path.exists():
        matched_payload = build_field_radius_matched_payload(
            baseline_payload,
            output_dir=matched_output_dir,
        )
        write_matched_payload(matched_payload, matched_path)
    else:
        matched_payload = _load_json(matched_path)
    return build_payload_from_comparisons(
        baseline_payload=baseline_payload,
        matched_payload=matched_payload,
        baseline_path=baseline_json,
        matched_path=matched_path,
    )


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    rows = payload["rows"]
    rho = np.asarray([float(row["rho"]) for row in rows], dtype=float)
    redl = (
        np.asarray([float(row["redl_current_over_root_fsab2"]) for row in rows], dtype=float)
        / 1.0e6
    )
    baseline = (
        np.asarray([float(row["baseline_total_over_root_fsab2"]) for row in rows], dtype=float)
        / 1.0e6
    )
    matched = (
        np.asarray(
            [float(row["field_radius_matched_total_over_root_fsab2"]) for row in rows],
            dtype=float,
        )
        / 1.0e6
    )
    baseline_error = np.asarray(
        [float(row["baseline_relative_error"]) for row in rows],
        dtype=float,
    )
    matched_error = np.asarray(
        [float(row["field_radius_matched_relative_error"]) for row in rows],
        dtype=float,
    )
    current_delta = np.asarray(
        [float(row["current_change_over_redl"]) for row in rows],
        dtype=float,
    )
    radial_contract = payload["radial_contract"]
    baseline_scan = np.asarray(radial_contract["baseline_scan_rho"], dtype=float)
    matched_scan = np.asarray(radial_contract["field_radius_matched_scan_rho"], dtype=float)
    metrics = payload["summary_metrics"]

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.size": 10.0,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 7.7), constrained_layout=True)
    ax_current, ax_error, ax_grid, ax_delta = axes.ravel()

    ax_current.plot(rho, redl, color="#009e73", lw=2.2, label="Redl")
    ax_current.plot(rho, baseline, color="#0072b2", lw=1.9, marker="o", label="sparse scan")
    ax_current.plot(
        rho,
        matched,
        color="#d55e00",
        lw=1.9,
        marker="s",
        label="field-radius matched",
    )
    ax_current.axhline(0.0, color="0.3", lw=0.8)
    ax_current.set_title("(a) Current observable")
    ax_current.set_xlabel(r"$\rho$")
    ax_current.set_ylabel(r"$\langle J\cdot B\rangle/\sqrt{\langle B^2\rangle}$ [MA m$^{-2}$]")
    ax_current.legend(fontsize=8.4)

    ax_error.semilogy(rho, baseline_error, color="#0072b2", lw=1.9, marker="o", label="sparse scan")
    ax_error.semilogy(
        rho,
        matched_error,
        color="#d55e00",
        lw=1.9,
        marker="s",
        label="field-radius matched",
    )
    ax_error.axhline(float(metrics["profile_current_gate"]), color="0.25", lw=1.0, ls="--")
    ax_error.set_title("(b) Relative difference to Redl")
    ax_error.set_xlabel(r"$\rho$")
    ax_error.set_ylabel("relative difference")
    ax_error.legend(fontsize=8.4)

    ax_grid.eventplot(
        [baseline_scan, matched_scan, rho],
        colors=["#0072b2", "#d55e00", "#111111"],
        lineoffsets=[3.0, 2.0, 1.0],
        linelengths=0.62,
        linewidths=1.7,
    )
    ax_grid.set_yticks([3.0, 2.0, 1.0])
    ax_grid.set_yticklabels(["sparse scan", "matched scan", "profile grid"])
    ax_grid.set_xlim(0.0, 1.0)
    ax_grid.set_title("(c) Radial database contract")
    ax_grid.set_xlabel(r"$\rho$")

    ax_delta.plot(rho, current_delta, color="#cc79a7", lw=2.0, marker="o")
    ax_delta.axhline(0.0, color="0.3", lw=0.8)
    ax_delta.set_title("(d) Current change from radial contract")
    ax_delta.set_xlabel(r"$\rho$")
    ax_delta.set_ylabel(r"$|\Delta J|/|J_{Redl}|$")
    ax_delta.text(
        0.04,
        0.95,
        (
            "max error "
            f"{metrics['baseline_max_relative_error_total_vs_redl']:.2f}"
            " -> "
            f"{metrics['field_radius_matched_max_relative_error_total_vs_redl']:.2f}\n"
            "old stress "
            f"{metrics['baseline_stress_relative_error']:.2f}"
            " -> "
            f"{metrics['field_radius_matched_error_at_baseline_stress_rho']:.2f}"
        ),
        transform=ax_delta.transAxes,
        va="top",
        ha="left",
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )

    fig.suptitle("Finite-beta radial interpolation sensitivity", fontsize=13)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, default=BASELINE_JSON)
    parser.add_argument("--matched-json", type=Path, default=None)
    parser.add_argument("--matched-output-dir", type=Path, default=WORKDIR)
    parser.add_argument(
        "--rebuild-matched",
        action="store_true",
        help="rebuild the field-radius-matched finite-beta bootstrap payload",
    )
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    args = parser.parse_args()

    payload = build_payload(
        baseline_json=args.baseline_json,
        matched_json=args.matched_json,
        matched_output_dir=args.matched_output_dir,
        rebuild_matched=bool(args.rebuild_matched),
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(
        f"wrote {args.output_prefix.with_suffix('.json')}, "
        f"{args.output_prefix.with_suffix('.png')}, and "
        f"{args.output_prefix.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()
