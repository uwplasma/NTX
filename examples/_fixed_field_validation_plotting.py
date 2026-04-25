from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def display_label(key: str) -> str:
    return {
        "SFINCS": "SFINCS",
        "SFINCS-JAX": "SFINCS-JAX",
        "NTX+NEOPAX": "NTX+NEOPAX",
        "Redl": "Redl (Boozer)",
    }[key]


def plot_styles() -> dict[str, dict[str, Any]]:
    return {
        "SFINCS": dict(color="#111111", lw=2.8, ls="-"),
        "SFINCS-JAX": dict(color="#d55e00", lw=2.0, ls="--"),
        "NTX+NEOPAX": dict(color="#1f77b4", lw=2.4, ls="-"),
        "Redl": dict(color="#009e73", lw=2.0, ls="-."),
    }


def plot_order(case_results: dict[str, dict[str, np.ndarray]]) -> tuple[str, ...]:
    order = ["SFINCS"]
    if "SFINCS-JAX" in case_results:
        order.append("SFINCS-JAX")
    order.extend(["NTX+NEOPAX", "Redl"])
    return tuple(order)


def panel_label(ax: Any, label: str) -> None:
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


def plot_fixed_field_validation(
    *,
    results: dict[str, dict[str, dict[str, np.ndarray]]],
    cases: dict[str, Any],
    output_prefix: Path,
    interior_rho_min: float,
    interior_rho_max: float,
    interp_profile: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.7),
        constrained_layout=True,
        sharey=True,
    )
    styles = plot_styles()
    for col, key in enumerate(("qa", "qh")):
        case_results = results[key]
        ax = axes[col]
        ax.axvspan(interior_rho_min, interior_rho_max, color="#f0f0f0", alpha=0.5, zorder=0)
        for name in plot_order(case_results):
            payload = case_results[name]
            ax.plot(
                np.asarray(payload["rho"], dtype=float),
                np.asarray(payload["jdotb"], dtype=float) / 1.0e6,
                label=display_label(name),
                **styles[name],
            )
            if name == "SFINCS-JAX" and "rho_sample" in payload:
                ax.plot(
                    np.asarray(payload["rho_sample"], dtype=float),
                    np.asarray(payload["jdotb_sample"], dtype=float) / 1.0e6,
                    marker="o",
                    ms=4.2,
                    lw=0,
                    color=styles[name]["color"],
                )
        ax.set_title(cases[key].label)
        if col == 0:
            ax.set_ylabel(r"$\langle \mathbf{J}\cdot\mathbf{B}\rangle$ [MA T m$^{-2}$]")
        ax.set_xlabel(r"$\rho$")
        ax.grid(alpha=0.24, lw=0.6)
        panel_label(ax, f"({chr(ord('a') + col)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.07),
    )
    fig.savefig(output_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "display_label",
    "panel_label",
    "plot_fixed_field_validation",
    "plot_order",
    "plot_styles",
]
