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
    w7x_errors = {
        "raw": min(
            float(row["max_relative_error"])
            for row in w7x["bootstrap_current_errors"]
        ),
        "pmax4_transfer": float(pmax["w7x_max_relative_error"]),
    }

    return {
        "precise_qs": {
            "qa": {"Redl": float(qa["Redl"]), "NTX+NEOPAX": float(qa["NTX+NEOPAX"])},
            "qh": {"Redl": float(qh["Redl"]), "NTX+NEOPAX": float(qh["NTX+NEOPAX"])},
            "redl_gate": 1.0e-1,
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

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)

    labels = ["QA", "QH"]
    x = [0, 1]
    width = 0.34
    axes[0].bar(
        [value - width / 2 for value in x],
        [qa["Redl"], qh["Redl"]],
        width=width,
        label="Redl",
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        [qa["NTX+NEOPAX"], qh["NTX+NEOPAX"]],
        width=width,
        label="NTX+NEOPAX",
    )
    axes[0].axhline(redl_gate, color="0.3", ls="--", lw=1.2)
    axes[0].set_xticks(x, labels)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("interior max relative error")
    axes[0].set_title("Precise-QS fixed-field")
    axes[0].legend(frameon=False)

    axes[1].bar(
        ["W7-X raw", "Pmax=4"],
        [
            payload["w7x_transfer"]["raw_branch_error"],
            payload["w7x_transfer"]["pmax4_transfer_error"],
        ],
        color=["#2ca02c", "#d62728"],
    )
    axes[1].axhline(payload["w7x_transfer"]["raw_gate"], color="0.3", ls="--", lw=1.2)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("max relative error")
    axes[1].set_title("Integrated W7-X transfer")

    axes[2].semilogy(pmax_levels, qa_errors, "o-", lw=2.0, label="QA")
    axes[2].semilogy(pmax_levels, qh_errors, "s-", lw=2.0, label="QH")
    axes[2].semilogy(pmax_levels, w7x_pmax_errors, "^-", lw=2.0, label="W7-X")
    axes[2].set_xlabel(r"$P_{\max}$ moments")
    axes[2].set_ylabel("max relative error")
    axes[2].set_title("Higher-order stress")
    axes[2].legend(frameon=False)

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
