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


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 7),
            "figure.dpi": 180,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def main() -> None:
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
    )

    output_dir = ROOT / "docs" / "_static"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "autodiff_neopax_profiles.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    iterations = np.arange(1, len(result.loss_history) + 1)

    axes[0, 0].plot(
        np.asarray(result.rho),
        np.asarray(result.target_er_profile),
        lw=2,
        label="Target",
    )
    axes[0, 0].plot(
        np.asarray(result.rho),
        np.asarray(result.fitted_er_profile),
        lw=2,
        ls="--",
        label="Recovered",
    )
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"$E_r$")
    axes[0, 0].set_title("Electric-field profile")
    axes[0, 0].legend()

    axes[0, 1].plot(
        np.asarray(result.rho),
        np.asarray(result.target_d33_profile),
        lw=2,
        label="Target",
    )
    axes[0, 1].plot(
        np.asarray(result.rho),
        np.asarray(result.fitted_d33_profile),
        lw=2,
        ls="--",
        label="Recovered",
    )
    axes[0, 1].set_xlabel(r"$\rho$")
    axes[0, 1].set_ylabel(r"$D_{33}$")
    axes[0, 1].set_title("NEOPAX-style transport profile")
    axes[0, 1].legend()

    axes[1, 0].semilogy(iterations, np.asarray(result.loss_history), color="#2ca02c", lw=2)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Profile inversion convergence")

    sensitivity = np.asarray(result.sensitivity_matrix)
    im = axes[1, 1].imshow(
        sensitivity,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0.5, sensitivity.shape[1] + 0.5, float(result.rho[0]), float(result.rho[-1])],
    )
    axes[1, 1].set_xlabel("Profile parameter index")
    axes[1, 1].set_ylabel(r"$\rho$")
    axes[1, 1].set_title(r"Sensitivity $\partial D_{33}/\partial p_j$")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.85)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
