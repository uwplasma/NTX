#!/usr/bin/env python3
"""Autodiff NEOPAX-style profile example for NTX."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    example_neopax_profile_autodiff,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
)
from ntx.config import enable_x64  # noqa: E402


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.0, 8.2),
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
    enable_x64(True)
    _configure_style()
    scan = load_neopax_reference_scan(ROOT / "tests" / "fixtures" / "sample_neopax_scan.h5")
    wout = ROOT / "tests" / "fixtures" / "sample_wout.nc"
    surfaces = tuple(
        surface_from_vmec_jax_vmec_wout_file(wout, s=float(rho_value**2)) for rho_value in scan.rho
    )
    result = example_neopax_profile_autodiff(
        surfaces,
        rho=scan.rho,
        nu_v=scan.nu_v,
        Es=scan.Es,
        Er=scan.Er,
        drds=scan.drds,
        grid=GridSpec(7, 9, 6),
        a_b=1.0,
        learning_rate=0.4,
        steps=128,
    )

    output_dir = ROOT / "docs" / "_static"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "autodiff_neopax_profiles.png"
    output_pdf = output_dir / "autodiff_neopax_profiles.pdf"

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    iterations = np.arange(1, len(result.loss_history) + 1)
    colors = {
        "target": "#111827",
        "fit": "#0072B2",
        "loss": "#009E73",
        "error": "#CC79A7",
    }

    axes[0, 0].plot(
        np.asarray(result.rho),
        np.asarray(result.target_er_profile),
        lw=2.3,
        color=colors["target"],
        label="Target",
    )
    axes[0, 0].plot(
        np.asarray(result.rho),
        np.asarray(result.fitted_er_profile),
        lw=2.2,
        ls="--",
        color=colors["fit"],
        label="Recovered",
    )
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"$E_r$")
    axes[0, 0].set_title("Recovered electric-field profile")
    axes[0, 0].legend(loc="lower right")
    er_rel_error = np.abs(
        (np.asarray(result.fitted_er_profile) - np.asarray(result.target_er_profile))
        / np.maximum(np.abs(np.asarray(result.target_er_profile)), 1e-12)
    )
    axes[0, 0].fill_between(
        np.asarray(result.rho),
        np.asarray(result.target_er_profile),
        np.asarray(result.fitted_er_profile),
        color=colors["fit"],
        alpha=0.12,
    )
    axes[0, 0].text(
        0.03,
        0.96,
        rf"max relative error$={er_rel_error.max():.2e}$",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    axes[0, 1].plot(
        np.asarray(result.rho),
        np.asarray(result.target_d33_profile),
        lw=2.3,
        color=colors["target"],
        label="Target",
    )
    axes[0, 1].plot(
        np.asarray(result.rho),
        np.asarray(result.fitted_d33_profile),
        lw=2.2,
        ls="--",
        color=colors["fit"],
        label="Recovered",
    )
    axes[0, 1].set_xlabel(r"$\rho$")
    axes[0, 1].set_ylabel(r"$D_{33}$")
    axes[0, 1].set_title("Transport profile induced by autodiff fit")
    axes[0, 1].legend(loc="upper right")
    d11_rel_error = np.abs(
        (np.asarray(result.fitted_d11_profile) - np.asarray(result.target_d11_profile))
        / np.maximum(np.abs(np.asarray(result.target_d11_profile)), 1e-30)
    )
    d33_rel_error = np.abs(
        (np.asarray(result.fitted_d33_profile) - np.asarray(result.target_d33_profile))
        / np.maximum(np.abs(np.asarray(result.target_d33_profile)), 1e-30)
    )
    axes[0, 1].text(
        0.03,
        0.96,
        (
            rf"max $D_{{11}}$ relative error$={d11_rel_error.max():.2e}$" "\n"
            rf"max $D_{{33}}$ relative error$={d33_rel_error.max():.2e}$"
        ),
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    axes[1, 0].semilogy(
        iterations,
        np.asarray(result.loss_history),
        color=colors["loss"],
        lw=2.3,
    )
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Profile inversion convergence")
    param_history = np.asarray(result.parameter_history)
    twin = axes[1, 0].twinx()
    twin.plot(iterations, param_history[:, 0], color=colors["fit"], lw=1.5, ls="--")
    twin.plot(iterations, param_history[:, 1], color=colors["error"], lw=1.5, ls=":")
    twin.set_ylabel("Profile parameters")
    twin.grid(False)

    sensitivity = np.asarray(result.sensitivity_matrix)
    im = axes[1, 1].imshow(
        sensitivity,
        aspect="auto",
        origin="lower",
        cmap="cividis",
        extent=[0.5, sensitivity.shape[1] + 0.5, float(result.rho[0]), float(result.rho[-1])],
    )
    axes[1, 1].set_xlabel("Profile parameter index")
    axes[1, 1].set_ylabel(r"$\rho$")
    axes[1, 1].set_title(r"Sensitivity $\partial D_{33}/\partial p_j$")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.86, label=r"$\partial D_{33}/\partial p_j$")

    for label, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
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
