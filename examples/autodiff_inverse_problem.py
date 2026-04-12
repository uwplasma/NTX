#!/usr/bin/env python3
"""Autodiff inverse-problem example for NTX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from ntx import GridSpec, example_inverse_problem  # noqa: E402
from ntx.config import enable_x64  # noqa: E402


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.5, 4.3),
            "figure.dpi": 220,
            "font.size": 10.5,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "autodiff_inverse_problem",
        help="Prefix for PNG and PDF outputs.",
    )
    args = parser.parse_args()
    enable_x64(True)
    _configure_style()
    result = example_inverse_problem(grid=GridSpec(7, 9, 6))
    output_png = args.output_prefix.with_suffix(".png")
    output_pdf = args.output_prefix.with_suffix(".pdf")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, constrained_layout=True)

    iterations = np.arange(1, len(result.loss_history) + 1)
    target_amplitude = float(result.target_amplitude)
    inferred_amplitude = float(result.inferred_amplitude)
    relative_amplitude_error = abs(inferred_amplitude - target_amplitude) / max(
        abs(target_amplitude),
        1e-12,
    )
    colors = {
        "target": "#111827",
        "fit": "#0072B2",
        "initial": "#D55E00",
        "loss": "#009E73",
        "error": "#CC79A7",
    }

    axes[0].plot(
        iterations,
        np.asarray(result.amplitude_history),
        color=colors["fit"],
        lw=2.3,
        label="Recovered iterate",
    )
    axes[0].axhline(
        target_amplitude,
        color=colors["target"],
        ls="--",
        lw=1.4,
        label="Target",
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Boozer coefficient amplitude")
    axes[0].set_title("Inverse solve")
    axes[0].legend(loc="lower right")
    axes[0].text(
        0.03,
        0.96,
        (
            rf"$a^\star={target_amplitude:.4f}$" "\n"
            rf"$\hat a={inferred_amplitude:.4f}$" "\n"
            rf"relative error$={relative_amplitude_error:.2e}$"
        ),
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    axes[1].semilogy(
        iterations,
        np.asarray(result.loss_history),
        color=colors["loss"],
        lw=2.3,
        label="Objective",
    )
    axes[1].semilogy(
        iterations,
        np.abs(np.asarray(result.gradient_history)),
        color=colors["error"],
        lw=1.8,
        ls="--",
        label=r"$|\partial \mathcal{L}/\partial a|$",
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Objective / gradient scale")
    axes[1].set_title("Optimization history")
    axes[1].legend(loc="upper right")

    axes[2].loglog(
        np.asarray(result.nu_hat),
        np.asarray(result.target_response),
        lw=2.4,
        color=colors["target"],
        label="Target",
    )
    axes[2].loglog(
        np.asarray(result.nu_hat),
        np.asarray(result.initial_response),
        lw=1.8,
        ls=":",
        color=colors["initial"],
        label="Initial guess",
    )
    axes[2].loglog(
        np.asarray(result.nu_hat),
        np.asarray(result.fitted_response),
        lw=2.2,
        ls="--",
        color=colors["fit"],
        label="Recovered",
    )
    axes[2].set_xlabel(r"$\hat{\nu}$")
    axes[2].set_ylabel(r"$D_{11}$")
    axes[2].set_title("Recovered transport curve")
    axes[2].legend(loc="lower left")
    axes[2].xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axes[2].xaxis.set_minor_formatter(NullFormatter())

    relative_curve_error = np.abs(
        (np.asarray(result.fitted_response) - np.asarray(result.target_response))
        / np.maximum(np.asarray(result.target_response), 1e-30)
    )
    inset = axes[2].inset_axes([0.56, 0.10, 0.40, 0.34])
    inset.semilogx(np.asarray(result.nu_hat), relative_curve_error, color=colors["error"], lw=1.6)
    inset.set_title("Relative error", fontsize=8.3)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.15)

    for label, ax in zip(("a", "b", "c"), axes, strict=True):
        ax.text(
            -0.14,
            1.02,
            f"({label})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    fig.savefig(output_png)
    fig.savefig(output_pdf)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
