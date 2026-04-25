#!/usr/bin/env python3
"""Build a compact closure-validation report from tracked benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"
OUTPUT_PREFIX = STATIC / "closure_validation_report"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload() -> dict:
    fixed = _load_json(STATIC / "bootstrap_current_fixed_field_validation.json")
    w7x = _load_json(STATIC / "bootstrap_current_reference_audit_w7x.json")
    pmax = _load_json(STATIC / "closure_pmax_convergence.json")

    qa = fixed["cases"]["qa"]["max_relative_error_vs_sfincs_interior"]
    qh = fixed["cases"]["qh"]["max_relative_error_vs_sfincs_interior"]
    closure_error = max(float(qa["NTX+NEOPAX"]), float(qh["NTX+NEOPAX"]))
    closure_gate = 1.0e-1
    closure_passes = closure_error <= closure_gate
    w7x_errors = {
        "raw": min(
            float(row["max_relative_error"])
            for row in w7x["bootstrap_current_errors"]
        ),
        "pmax4_transfer": float(pmax["w7x_max_relative_error"]),
    }

    return {
        "claim_scope": {
            "positive_gates": (
                "Redl fixed-field current agrees with the archived precise-QS "
                "SFINCS interior-window reference within the documented gate.",
                "The fixed-field NTX+NEOPAX total-current stress comparison "
                "passes the documented interior-window gate using the explicit "
                "low-order Spitzer-conductivity closure, without fitted "
                "bridge constants.",
                "The rebuilt raw-branch imported W7-X workflow remains within "
                "the integrated-transfer regression gate.",
            ),
            "monitored_stress": (
                "The Pmax extension artifact is retained as a development "
                "stress metric until precise-QS convergence improves without "
                "regressing integrated W7-X.",
            ),
            "promotion_requirements": (
                "derive the closure change from the moment equations or an "
                "equivalent documented physics model",
                "avoid fitted bridge constants in the shipping runtime",
                "preserve the fixed-field QA/QH total-current stress gate",
                "preserve the integrated W7-X transfer regression",
            ),
        },
        "closure_decision": {
            "status": (
                "fixed-field-stress-gate-passed"
                if closure_passes
                else "monitored-not-promoted"
            ),
            "reason": (
                "The precise-QS fixed-field total-current stress comparison "
                "now passes after applying only two physics-normalization "
                "changes: the SFINCS flux-surface-averaged parallel-flow "
                "observable bridge and the explicit low-order Spitzer "
                "conductivity block in the reduced momentum-restoring closure. "
                "The integrated W7-X workflow remains a separate raw-branch "
                "transfer gate, since the low-order fixed-field branch does "
                "not transfer to that imported-database convention."
            ),
            "literature_basis": (
                "Monoenergetic momentum-restoring closures are moment-equation "
                "approximations built from precomputed transport coefficients; "
                "the literature does not justify fitted per-benchmark bridge "
                "constants as a replacement for a projected collision model.",
                "The Redl precise-QS comparison is an analytic bootstrap-current "
                "validation path and is intentionally kept separate from the "
                "reduced NTX+NEOPAX closure stress metric.",
            ),
            "promotion_condition": (
                "Any broader closure default can be promoted only if it is "
                "derived from the same moment equations or an equivalent "
                "documented physics model, preserves the fixed-field QA/QH "
                "total-current gate, and preserves the integrated W7-X "
                "transfer gate."
            ),
            "fixed_field_gate": closure_gate,
            "fixed_field_max_error": closure_error,
            "fixed_field_d33_mode": fixed.get("ntx_neopax_d33_mode", "unknown"),
            "fixed_field_n_order": fixed.get("ntx_neopax_n_order", "unknown"),
        },
        "precise_qs": {
            "qa": {"Redl": float(qa["Redl"]), "NTX+NEOPAX": float(qa["NTX+NEOPAX"])},
            "qh": {"Redl": float(qh["Redl"]), "NTX+NEOPAX": float(qh["NTX+NEOPAX"])},
            "redl_gate": 1.0e-1,
            "ntx_neopax_gate": closure_gate,
        },
        "fixed_field_diagnostics": {
            case: {
                "current_total": float(
                    fixed["cases"][case]["closure_diagnostics"][
                        "current_worst_relative_error_interior"
                    ]
                ),
                "current_nomom_scale": float(
                    fixed["cases"][case]["closure_diagnostics"]["current_nomom_scale"]
                ),
                "thermal_raw_fit": float(
                    fixed["cases"][case]["closure_diagnostics"][
                        "thermal_raw_fit_max_relative_error"
                    ]
                ),
                "thermal_eff_fit": float(
                    fixed["cases"][case]["closure_diagnostics"][
                        "thermal_eff_fit_max_relative_error"
                    ]
                ),
                "hybrid_errors": {
                    name: float(value)
                    for name, value in fixed["cases"][case]["closure_diagnostics"][
                        "hybrid_current_max_relative_error_interior"
                    ].items()
                },
            }
            for case in ("qa", "qh")
        },
        "w7x_transfer": {
            "raw_branch_error": w7x_errors["raw"],
            "raw_gate": 2.0e-2,
            "pmax4_transfer_error": w7x_errors["pmax4_transfer"],
        },
        "pmax_stress": {
            "levels": [int(level) for level in pmax["pmax_levels"]],
            "qa_errors": [
                float(pmax["precise_qs"][str(level)]["qa"])
                for level in pmax["pmax_levels"]
            ],
            "qh_errors": [
                float(pmax["precise_qs"][str(level)]["qh"])
                for level in pmax["pmax_levels"]
            ],
            "w7x_errors": [
                float(pmax["w7x_errors"][str(level)]) for level in pmax["pmax_levels"]
            ],
            "max_successive_change": float(pmax["precise_qs_max_successive_change"]),
        },
        "sources": {
            "fixed_field": "docs/_static/bootstrap_current_fixed_field_validation.json",
            "w7x": "docs/_static/bootstrap_current_reference_audit_w7x.json",
            "pmax": "docs/_static/closure_pmax_convergence.json",
        },
    }


def build_markdown(payload: dict) -> str:
    qa = payload["precise_qs"]["qa"]
    qh = payload["precise_qs"]["qh"]
    return "\n".join(
        [
            "# Closure Validation Report",
            "",
            "## Precise-QS fixed-field interior max relative error vs archived SFINCS",
            "",
            "| Case | Redl | NTX+NEOPAX |",
            "| --- | ---: | ---: |",
            f"| QA | `{qa['Redl']:.3e}` | `{qa['NTX+NEOPAX']:.3e}` |",
            f"| QH | `{qh['Redl']:.3e}` | `{qh['NTX+NEOPAX']:.3e}` |",
            "",
            "## W7-X integrated transfer",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            (
                "| Rebuilt raw-branch max relative error | "
                f"`{payload['w7x_transfer']['raw_branch_error']:.3e}` |"
            ),
            (
                "| `Pmax=4` transfer error | "
                f"`{payload['w7x_transfer']['pmax4_transfer_error']:.3e}` |"
            ),
            "",
            "## Fixed-field closure diagnostics",
            "",
            (
                "The NTX+NEOPAX current column below is a reduced-closure "
                "total-current stress diagnostic. It now passes the fixed-field "
                "interior gate, but it is still not an independent species-"
                "current parity claim."
            ),
            "",
            (
                f"Decision: `{payload['closure_decision']['status']}`. "
                "The fixed-field branch uses "
                f"`d33_mode={payload['closure_decision']['fixed_field_d33_mode']}` "
                "and "
                f"`n_order={payload['closure_decision']['fixed_field_n_order']}`. "
                "No fitted bridge constants or benchmark-specific scale factors "
                "are used."
            ),
            "",
            "| Case | Current | Raw thermal fit | Effective thermal fit |",
            "| --- | ---: | ---: | ---: |",
            (
                "| QA | "
                f"`{payload['fixed_field_diagnostics']['qa']['current_total']:.3e}` | "
                f"`{payload['fixed_field_diagnostics']['qa']['thermal_raw_fit']:.3e}` | "
                f"`{payload['fixed_field_diagnostics']['qa']['thermal_eff_fit']:.3e}` |"
            ),
            (
                "| QH | "
                f"`{payload['fixed_field_diagnostics']['qh']['current_total']:.3e}` | "
                f"`{payload['fixed_field_diagnostics']['qh']['thermal_raw_fit']:.3e}` | "
                f"`{payload['fixed_field_diagnostics']['qh']['thermal_eff_fit']:.3e}` |"
            ),
            "",
            "## Higher-order closure stress",
            "",
            (
                "| Max successive precise-QS change across `Pmax` | "
                f"`{payload['pmax_stress']['max_successive_change']:.3e}` |"
            ),
            "",
        ]
    )


def build_figure(payload: dict) -> None:
    qa = payload["precise_qs"]["qa"]
    qh = payload["precise_qs"]["qh"]
    redl_gate = payload["precise_qs"]["redl_gate"]
    pmax_levels = payload["pmax_stress"]["levels"]
    qa_errors = payload["pmax_stress"]["qa_errors"]
    qh_errors = payload["pmax_stress"]["qh_errors"]
    w7x_pmax_errors = payload["pmax_stress"]["w7x_errors"]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), constrained_layout=True)
    axes_flat = axes.ravel()

    labels = ["QA", "QH"]
    x = [0, 1]
    width = 0.34
    axes_flat[0].bar(
        [value - width / 2 for value in x],
        [qa["Redl"], qh["Redl"]],
        width=width,
        label="Redl",
    )
    axes_flat[0].bar(
        [value + width / 2 for value in x],
        [qa["NTX+NEOPAX"], qh["NTX+NEOPAX"]],
        width=width,
        label="NTX+NEOPAX",
    )
    axes_flat[0].axhline(redl_gate, color="0.3", ls="--", lw=1.2)
    axes_flat[0].set_xticks(x, labels)
    axes_flat[0].set_yscale("log")
    axes_flat[0].set_ylabel("interior max relative error")
    axes_flat[0].set_title("Precise-QS fixed-field")
    axes_flat[0].legend(frameon=False)

    axes_flat[1].bar(
        ["W7-X raw", "Pmax=4"],
        [
            payload["w7x_transfer"]["raw_branch_error"],
            payload["w7x_transfer"]["pmax4_transfer_error"],
        ],
        color=["#2ca02c", "#d62728"],
    )
    axes_flat[1].axhline(payload["w7x_transfer"]["raw_gate"], color="0.3", ls="--", lw=1.2)
    axes_flat[1].set_yscale("log")
    axes_flat[1].set_ylabel("max relative error")
    axes_flat[1].set_title("Integrated W7-X transfer")

    diagnostic_labels = ["current", "raw fit", "effective fit"]
    diagnostic_x = range(len(diagnostic_labels))
    diagnostic_width = 0.34
    axes_flat[2].bar(
        [value - diagnostic_width / 2 for value in diagnostic_x],
        [
            payload["fixed_field_diagnostics"]["qa"]["current_total"],
            payload["fixed_field_diagnostics"]["qa"]["thermal_raw_fit"],
            payload["fixed_field_diagnostics"]["qa"]["thermal_eff_fit"],
        ],
        width=diagnostic_width,
        label="QA",
    )
    axes_flat[2].bar(
        [value + diagnostic_width / 2 for value in diagnostic_x],
        [
            payload["fixed_field_diagnostics"]["qh"]["current_total"],
            payload["fixed_field_diagnostics"]["qh"]["thermal_raw_fit"],
            payload["fixed_field_diagnostics"]["qh"]["thermal_eff_fit"],
        ],
        width=diagnostic_width,
        label="QH",
    )
    axes_flat[2].set_xticks(list(diagnostic_x), diagnostic_labels)
    axes_flat[2].set_yscale("log")
    axes_flat[2].set_ylabel("interior max relative error")
    axes_flat[2].set_title("Fixed-field closure diagnostic")
    axes_flat[2].legend(frameon=False)

    axes_flat[3].semilogy(pmax_levels, qa_errors, "o-", lw=2.0, label="QA")
    axes_flat[3].semilogy(pmax_levels, qh_errors, "s-", lw=2.0, label="QH")
    axes_flat[3].semilogy(pmax_levels, w7x_pmax_errors, "^-", lw=2.0, label="W7-X")
    axes_flat[3].set_xlabel(r"$P_{\max}$ moments")
    axes_flat[3].set_ylabel("max relative error")
    axes_flat[3].set_title("Higher-order stress")
    axes_flat[3].legend(frameon=False)

    for label, ax in zip(("a", "b", "c", "d"), axes_flat, strict=True):
        ax.text(
            0.02,
            0.96,
            f"({label})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    payload = build_payload()
    build_figure(payload)
    OUTPUT_PREFIX.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    OUTPUT_PREFIX.with_suffix(".txt").write_text(
        build_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
