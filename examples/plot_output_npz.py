#!/usr/bin/env python3
"""Open an NTX `.npz` output file and make publication-style summary plots."""

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


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.0, 8.0),
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
    parser.add_argument("npz", type=Path, help="NTX output `.npz` file.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "output_file_summary",
        help="Prefix for PNG and PDF outputs.",
    )
    args = parser.parse_args()

    _configure_style()
    data = np.load(args.npz, allow_pickle=False)
    theta = data["theta_grid"]
    zeta = data["zeta_grid"]
    b = data["b"]
    drift = data["radial_drift_spatial"]
    output_png = args.output_prefix.with_suffix(".png")
    output_pdf = args.output_prefix.with_suffix(".pdf")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    mesh_theta, mesh_zeta = np.meshgrid(theta, zeta, indexing="ij")

    im0 = axes[0, 0].pcolormesh(mesh_theta, mesh_zeta, b, shading="auto", cmap="viridis")
    axes[0, 0].set_xlabel(r"$\theta$")
    axes[0, 0].set_ylabel(r"$\zeta$")
    axes[0, 0].set_title("Magnetic-field strength")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.84, label=r"$B$")

    im1 = axes[0, 1].pcolormesh(mesh_theta, mesh_zeta, drift, shading="auto", cmap="coolwarm")
    axes[0, 1].set_xlabel(r"$\theta$")
    axes[0, 1].set_ylabel(r"$\zeta$")
    axes[0, 1].set_title("Radial-drift source")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.84, label=r"$\omega_d$")

    coeff_labels = [r"$D_{11}$", r"$D_{31}$", r"$D_{13}$", r"$D_{33}$", r"$D_{33}^{\mathrm{Sp}}$"]
    coeff_values = [
        float(data["D11"]),
        float(data["D31"]),
        float(data["D13"]),
        float(data["D33"]),
        float(data["D33_spitzer"]),
    ]
    axes[1, 0].bar(
        coeff_labels,
        coeff_values,
        color=["#0072B2", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"],
    )
    axes[1, 0].set_ylabel("Coefficient value")
    axes[1, 0].set_title("Solved transport coefficients")
    axes[1, 0].tick_params(axis="x", rotation=20)

    summary = (
        rf"$\hat{{\nu}}={float(data['nu_hat']):.2e}$" "\n"
        rf"$\hat{{\epsilon}}={float(data['epsi_hat_resolved']):.2e}$" "\n"
        rf"residual$={float(data['residual_l2']):.2e}$" "\n"
        rf"Onsager$={float(data['onsager_residual']):.2e}$" "\n"
        rf"$N_\theta={int(data['n_theta'])},\;N_\zeta={int(data['n_zeta'])},\;N_\xi={int(data['n_xi'])}$"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        summary,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )
    axes[1, 1].set_title("Run summary")

    for label_text, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
        ax.text(
            -0.14,
            1.02,
            f"({label_text})",
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
