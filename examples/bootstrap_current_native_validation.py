#!/usr/bin/env python3
"""Native NTX bootstrap-current validation against Redl, SFINCS-JAX, and SFINCS."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from bootstrap_current_native_validation_common import (
    available_cases,
    compute_ntx_native_profile,
    compute_redl_profile,
    compute_sfincs_jax_profile,
    compute_sfincs_profile,
    max_relative_error,
    summarize_case_results,
    write_metadata,
)

RHO_GRID = np.linspace(0.15, 0.75, 7)
OUTPUT_PREFIX = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "_static"
    / "bootstrap_current_native_validation"
)
SFINCS_DATASET = "FSABjHat"
NTX_LOADER_MODE = "filtered_nyquist"
NTX_COMPAT_BOUNDARY = False
RECOMPUTE_SFINCS = False


def _line_styles():
    return {
        "Redl": dict(color="#222222", lw=2.4, ls="-"),
        "NTX": dict(color="#1f77b4", lw=2.6, ls="-"),
        "SFINCS-JAX": dict(color="#d55e00", lw=2.2, ls="--"),
        "SFINCS": dict(color="#009e73", lw=2.2, ls="-."),
    }


def _panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )


def run_case(case_key: str) -> dict[str, dict[str, np.ndarray | float]]:
    case = available_cases()[case_key]
    return {
        "Redl": compute_redl_profile(case, RHO_GRID),
        "NTX": compute_ntx_native_profile(
            case,
            RHO_GRID,
            loader_mode=NTX_LOADER_MODE,
            compatibility_mode=NTX_COMPAT_BOUNDARY,
        ),
        "SFINCS-JAX": compute_sfincs_jax_profile(
            case,
            RHO_GRID,
            dataset=SFINCS_DATASET,
            recompute=RECOMPUTE_SFINCS,
        ),
        "SFINCS": compute_sfincs_profile(
            case,
            RHO_GRID,
            dataset=SFINCS_DATASET,
            recompute=RECOMPUTE_SFINCS,
        ),
    }


def plot_results(all_results: dict[str, dict[str, dict[str, np.ndarray | float]]]) -> None:
    styles = _line_styles()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 8.4),
        constrained_layout=True,
        sharex="col",
        gridspec_kw={"height_ratios": (1.0, 0.55)},
    )

    cases = available_cases()
    for col, case_key in enumerate(("qa", "qh")):
        case = cases[case_key]
        results = all_results[case_key]
        ax = axes[0, col]
        ref = np.asarray(results["SFINCS"]["observable"], dtype=float)
        for label, values in results.items():
            ax.plot(
                np.asarray(values["rho"], dtype=float),
                np.asarray(values["observable"], dtype=float) / 1.0e6,
                label=label,
                **styles[label],
            )
        ax.set_title(case.label)
        ax.set_ylabel(r"$\langle \mathbf{J}\cdot\mathbf{B}\rangle$ [MA T A m$^{-2}$]")
        ax.grid(alpha=0.24, lw=0.6)
        _panel_label(ax, f"({chr(ord('a') + col)})")

        ax_err = axes[1, col]
        for label in ("Redl", "NTX", "SFINCS-JAX"):
            values = np.asarray(results[label]["observable"], dtype=float)
            rel = np.abs(values - ref) / np.maximum(np.abs(ref), 1.0)
            ax_err.plot(
                np.asarray(results[label]["rho"], dtype=float),
                rel,
                label=f"{label} vs SFINCS",
                **styles[label],
            )
        ax_err.set_xlabel(r"$\rho$")
        ax_err.set_ylabel("relative error")
        ax_err.set_yscale("log")
        ax_err.grid(alpha=0.24, lw=0.6)
        _panel_label(ax_err, f"({chr(ord('c') + col)})")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cases = available_cases()
    if set(cases) != {"qa", "qh"}:
        raise SystemExit("This validation requires the local finite-beta QA/QH case directories.")

    all_results = {case_key: run_case(case_key) for case_key in ("qa", "qh")}
    plot_results(all_results)

    max_errors = {}
    for case_key, case_results in all_results.items():
        ref = np.asarray(case_results["SFINCS"]["observable"], dtype=float)
        max_errors[case_key] = {
            label: max_relative_error(np.asarray(values["observable"], dtype=float), ref)
            for label, values in case_results.items()
            if label != "SFINCS"
        }

    summary = {
        "rho_grid": RHO_GRID.tolist(),
        "case_metadata": {
            case_key: {
                "label": cases[case_key].label,
                "case_dir": str(cases[case_key].case_dir),
                "wout_path": str(cases[case_key].wout_path),
                "input_path": str(cases[case_key].input_path),
                "observable": "jdotb",
                "sfincs_dataset": SFINCS_DATASET,
                "ntx_loader_mode": NTX_LOADER_MODE,
                "ntx_compat_boundary": NTX_COMPAT_BOUNDARY,
            }
            for case_key in ("qa", "qh")
        },
        "cases": {
            case_key: summarize_case_results(case_results)
            for case_key, case_results in all_results.items()
        },
        "max_relative_error_vs_sfincs": max_errors,
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png")),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf")),
    }
    write_metadata(OUTPUT_PREFIX.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
