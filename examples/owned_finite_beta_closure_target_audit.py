#!/usr/bin/env python3
"""Build a physics-driver audit for the finite-beta closure target.

The profile source-response audit shows that the remaining finite-beta
bootstrap-current stress is not a scalar normalization error: the effective
temperature-source response changes across radius. This script turns that
observation into a machine-readable closure-target artifact. It compares the
measured response multiplier with local neoclassical drivers that enter Redl
and related bootstrap-current models: trapped fraction, inverse-aspect-ratio
proxy, and collisionality.

The output is a design diagnostic. It does not modify the runtime closure and
does not prescribe a fitted correction.
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

from examples.owned_finite_beta_bootstrap_comparison import _to_jsonable  # noqa: E402
from ntx.validation._finite_beta_closure_target import (  # noqa: E402
    field_radius_matched_response_audit,
    profile_closure_target_diagnostics,
)

SOURCE_RESPONSE_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_source_response_profile_audit.json"
)
MATCHED_SOURCE_CHANNEL_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_field_radius_matched_source_channel_audit.json"
)
MATCHED_QUADRATURE_JSON = (
    ROOT
    / "docs"
    / "_static"
    / "owned_finite_beta_field_radius_matched_closure_quadrature_audit.json"
)
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_finite_beta_closure_target_audit"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_payload(
    *,
    source_response_json: Path = SOURCE_RESPONSE_JSON,
    matched_source_channel_json: Path | None = MATCHED_SOURCE_CHANNEL_JSON,
    matched_quadrature_json: Path | None = MATCHED_QUADRATURE_JSON,
) -> dict[str, Any]:
    source_payload = _load_json(source_response_json)
    profile = profile_closure_target_diagnostics(source_payload)
    metrics = dict(profile["summary_metrics"])
    matched_audit = field_radius_matched_response_audit(
        matched_source_channel_json=matched_source_channel_json,
        matched_quadrature_json=matched_quadrature_json,
        root=ROOT,
    )
    if matched_audit is not None:
        metrics.update(
            {
                "field_radius_matched_same_stress_radius_between_artifacts": (
                    matched_audit["same_stress_radius_between_artifacts"]
                ),
                "field_radius_matched_best_public_relative_error_vs_redl": (
                    matched_audit["best_public_relative_error_vs_redl"]
                ),
                "field_radius_matched_high_stable_public_relative_error_vs_redl": (
                    matched_audit["high_stable_public_relative_error_vs_redl"]
                ),
                "field_radius_matched_best_effective_temperature_response_multiplier_to_redl": (
                    matched_audit[
                        "best_effective_temperature_response_multiplier_to_redl"
                    ]
                ),
                (
                    "field_radius_matched_high_stable_effective_temperature_"
                    "response_multiplier_to_redl"
                ): matched_audit[
                    "high_stable_effective_temperature_response_multiplier_to_redl"
                ],
                "field_radius_matched_source_channel_superposition_gate_pass": (
                    matched_audit["source_channel_superposition_gate_pass"]
                ),
                "field_radius_matched_quadrature_stable_current_gate_pass": (
                    matched_audit["quadrature_stable_current_gate_pass"]
                ),
                "field_radius_matched_best_pass_rejected_as_underintegrated": (
                    matched_audit["best_stress_pass_rejected_as_underintegrated"]
                ),
                "field_radius_matched_quadrature_aliasing_detected": (
                    matched_audit["quadrature_aliasing_detected"]
                ),
            }
        )
    return _to_jsonable(
        {
            "benchmark": "owned_finite_beta_closure_target_audit",
            "classification": "owned finite-beta closure target physics-driver audit",
            "claim_scope": (
                "Reads the finite-beta profile source-response artifact and "
                "quantifies which local neoclassical drivers explain the "
                "effective-temperature response multiplier. This is a closure "
                "design diagnostic: it applies no runtime correction and does "
                "not promote finite-beta bootstrap-current parity."
            ),
            "source_artifact": _artifact_path(source_response_json),
            "rows": profile["rows"],
            "correlations": profile["correlations"],
            "linear_diagnostics": profile["linear_diagnostics"],
            "field_radius_matched_response_audit": matched_audit,
            "summary_metrics": metrics,
            "closure_requirements": [
                (
                    "any promoted finite-beta profile-current closure must use "
                    "physical local drivers already present in the profile "
                    "equations, such as trapped fraction, inverse-aspect-ratio "
                    "proxy, collisionality, and thermodynamic-force "
                    "coefficients"
                ),
                (
                    "a scalar response multiplier is not sufficient because "
                    "the measured response varies over the committed radial "
                    "profile"
                ),
                (
                    "a runtime closure change must preserve the fixed-field "
                    "QA/QH total-current stress gate, the W7-X transfer gate, "
                    "the source-channel reconstruction gate, and the same-grid "
                    "finite-beta coefficient gate"
                ),
                (
                    "the diagnostic regressions in this artifact are "
                    "model-identification tools only, not production fits"
                ),
                (
                    "a finite-beta current-gate pass at the stress radius must "
                    "survive the field-radius-matched quadrature rule X >= Pmax "
                    "before it can be interpreted as a physical closure "
                    "improvement"
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
    rho = np.asarray([float(row["rho"]) for row in rows], dtype=float)
    response = np.asarray(
        [float(row["temperature_response_multiplier"]) for row in rows],
        dtype=float,
    )
    current_error = np.asarray(
        [float(row["profile_current_relative_error"]) for row in rows],
        dtype=float,
    )
    epsilon = np.asarray([float(row["drivers"]["epsilon"]) for row in rows], dtype=float)
    trapped = np.asarray(
        [float(row["drivers"]["trapped_fraction"]) for row in rows],
        dtype=float,
    )
    correlations = payload["correlations"]
    diagnostics = payload["linear_diagnostics"]

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
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), constrained_layout=True)
    ax_profile, ax_epsilon, ax_corr, ax_models = axes.ravel()

    ax_profile.plot(rho, response, color="#d55e00", marker="o", lw=2.0)
    ax_profile.axhline(1.0, color="0.25", ls="--", lw=1.0)
    ax_profile.set_xlabel(r"$\rho$")
    ax_profile.set_ylabel("temperature response multiplier")
    ax_profile.set_title("(a) Closure target is radial")
    ax_profile_t = ax_profile.twinx()
    ax_profile_t.semilogy(
        rho,
        current_error,
        color="#0072b2",
        marker="s",
        lw=1.6,
        alpha=0.9,
    )
    ax_profile_t.set_ylabel("current relative difference")

    scatter = ax_epsilon.scatter(
        epsilon,
        response,
        c=rho,
        s=52,
        cmap="viridis",
        edgecolors="0.1",
        linewidths=0.35,
    )
    ax_epsilon.axhline(1.0, color="0.25", ls="--", lw=1.0)
    ax_epsilon.set_xlabel(r"Redl $\epsilon$")
    ax_epsilon.set_ylabel("temperature response multiplier")
    ax_epsilon.set_title("(b) Geometry-driver trend")
    ax_trapped = ax_epsilon.twiny()
    ax_trapped.set_xlim(float(np.min(trapped)), float(np.max(trapped)))
    ax_trapped.set_xlabel(r"trapped fraction $f_t$")
    cbar = fig.colorbar(scatter, ax=ax_epsilon)
    cbar.set_label(r"$\rho$")

    labels = list(correlations)
    pearson = [abs(float(correlations[label]["pearson"] or 0.0)) for label in labels]
    spearman = [abs(float(correlations[label]["spearman"] or 0.0)) for label in labels]
    x = np.arange(len(labels))
    ax_corr.bar(x - 0.18, pearson, width=0.36, color="#009e73", label="Pearson")
    ax_corr.bar(x + 0.18, spearman, width=0.36, color="#56b4e9", label="Spearman")
    ax_corr.set_xticks(x)
    ax_corr.set_xticklabels(labels, rotation=25, ha="right")
    ax_corr.set_ylim(0.0, 1.05)
    ax_corr.set_ylabel("absolute correlation")
    ax_corr.set_title("(c) Driver ranking")
    ax_corr.legend(fontsize=8.0)

    model_names = [item["name"] for item in diagnostics]
    loo_values = [
        float(item["leave_one_out_rmse"])
        if item["leave_one_out_rmse"] is not None
        else np.nan
        for item in diagnostics
    ]
    ax_models.bar(np.arange(len(model_names)), loo_values, color="#cc79a7")
    ax_models.axhline(
        float(payload["summary_metrics"]["unit_response_rmse"]),
        color="0.25",
        ls="--",
        lw=1.0,
        label="unit response",
    )
    ax_models.set_xticks(np.arange(len(model_names)))
    ax_models.set_xticklabels(model_names, rotation=30, ha="right")
    ax_models.set_ylabel("leave-one-out RMSE")
    ax_models.set_title("(d) Diagnostic model identifiability")
    ax_models.legend(fontsize=8.0)

    metrics = payload["summary_metrics"]
    fig.suptitle(
        "Finite-beta closure-target audit "
        f"(best driver: {metrics['best_single_physics_driver']}, "
        f"span={metrics['response_multiplier_span']:.2g})",
        fontsize=13,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-response-json", type=Path, default=SOURCE_RESPONSE_JSON)
    parser.add_argument(
        "--matched-source-channel-json",
        type=Path,
        default=MATCHED_SOURCE_CHANNEL_JSON,
        help=(
            "Field-radius-matched source-channel artifact to cross-link with "
            "the profile closure-target audit."
        ),
    )
    parser.add_argument(
        "--matched-quadrature-json",
        type=Path,
        default=MATCHED_QUADRATURE_JSON,
        help=(
            "Field-radius-matched quadrature artifact to cross-link with the "
            "profile closure-target audit."
        ),
    )
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    args = parser.parse_args()

    payload = build_payload(
        source_response_json=args.source_response_json,
        matched_source_channel_json=args.matched_source_channel_json,
        matched_quadrature_json=args.matched_quadrature_json,
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(json.dumps(payload["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
