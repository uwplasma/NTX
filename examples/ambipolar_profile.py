#!/usr/bin/env python3
"""Solve a simple ambipolar electric-field profile from an NTX radial scan."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    MonoenergeticSpeciesProfile,
    ambipolar_residual_profile,
    build_ntx_neopax_scan_from_surfaces,
    example_surface,
    solve_ambipolar_er_profile,
)
from ntx.config import enable_x64  # noqa: E402

GRID = GridSpec(7, 9, 6)
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "ambipolar_profile"
HARMONIC_INDEX = 1


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (11.6, 7.0),
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
            "savefig.pad_inches": 0.05,
        }
    )


def _surface_family(rho_values):
    base = example_surface(dtype=GRID.jax_dtype)
    return tuple(
        replace(
            base,
            b_cos=base.b_cos.at[HARMONIC_INDEX].set(
                base.b_cos[HARMONIC_INDEX] * (1.0 + 0.18 * float(rho))
            ),
        )
        for rho in np.asarray(rho_values)
    )


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    enable_x64(True)
    _configure_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rho = jnp.linspace(0.2, 0.8, 6, dtype=GRID.jax_dtype)
    nu_v = jnp.asarray([3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2], dtype=GRID.jax_dtype)
    er_axis = jnp.asarray(
        [-3.0e-3, -1.0e-3, -3.0e-4, 0.0, 3.0e-4, 1.0e-3, 3.0e-3],
        dtype=GRID.jax_dtype,
    )
    er_grid = jnp.tile(er_axis[None, :], (rho.size, 1))
    electron = MonoenergeticSpeciesProfile(
        charge=jnp.asarray(-1.0, dtype=GRID.jax_dtype),
        nu_v=jnp.linspace(4.0e-4, 1.0e-3, rho.size, dtype=GRID.jax_dtype),
        A1=1.1 - 0.25 * rho,
        A3=0.55 - 0.12 * rho,
        current_weight=jnp.asarray(-1.0, dtype=GRID.jax_dtype),
        name="electron",
    )
    ion = MonoenergeticSpeciesProfile(
        charge=jnp.asarray(1.0, dtype=GRID.jax_dtype),
        nu_v=jnp.linspace(2.0e-3, 5.0e-3, rho.size, dtype=GRID.jax_dtype),
        A1=0.7 + 0.35 * rho,
        A3=0.24 + 0.08 * rho,
        particle_weight=jnp.asarray(1.08, dtype=GRID.jax_dtype),
        current_weight=jnp.asarray(1.0, dtype=GRID.jax_dtype),
        name="ion",
    )

    scan = build_ntx_neopax_scan_from_surfaces(
        _surface_family(rho),
        rho=rho,
        nu_v=nu_v,
        Es=er_grid,
        Er=er_grid,
        drds=jnp.ones_like(rho),
        grid=GRID,
        source_name="ambipolar_profile_example",
    )
    result = solve_ambipolar_er_profile(
        scan,
        (electron, ion),
        steps=12,
        damping=0.7,
        smoothing_strength=1.0,
    )

    rho_np = np.asarray(result.rho)
    er_profile = np.asarray(result.er_profile)
    residual = np.asarray(result.ambipolar_residual)
    jbs = np.asarray(result.bootstrap_current_response)
    electron_flux = np.asarray(result.species_particle_flux[0])
    ion_flux = np.asarray(result.species_particle_flux[1])
    electron_current = np.asarray(result.species_current_response[0])
    ion_current = np.asarray(result.species_current_response[1])
    residual_scan = np.asarray(
        [
            ambipolar_residual_profile(
                scan,
                (electron, ion),
                er_profile=jnp.full_like(rho, er_value),
            )
            for er_value in er_axis
        ]
    )
    residual_norm_scan = np.linalg.norm(residual_scan, axis=1)

    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    im = axes[0, 0].imshow(
        np.abs(residual_scan),
        aspect="auto",
        origin="lower",
        extent=(rho_np[0], rho_np[-1], float(er_axis[0]), float(er_axis[-1])),
        cmap="magma_r",
    )
    axes[0, 0].plot(rho_np, er_profile, color="white", lw=2.2, marker="o", ms=4.5)
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"$\hat E_r$")
    axes[0, 0].set_title("Residual landscape and selected profile")
    cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.048, pad=0.03)
    cbar.set_label(r"$|R(\rho,\hat E_r)|$")

    axes[0, 1].plot(rho_np, jbs, color="#D55E00", lw=2.3, marker="s", ms=5, label="total")
    axes[0, 1].plot(rho_np, electron_current, color="#0072B2", lw=1.7, ls="--", label="electron")
    axes[0, 1].plot(rho_np, ion_current, color="#009E73", lw=1.7, ls="--", label="ion")
    axes[0, 1].set_xlabel(r"$\rho$")
    axes[0, 1].set_ylabel("Reduced current response")
    axes[0, 1].set_title("Current-response closure")
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(
        rho_np,
        electron_flux,
        color="#0072B2",
        lw=2.0,
        marker="o",
        ms=4,
        label="electron",
    )
    axes[1, 0].plot(
        rho_np,
        ion_flux,
        color="#009E73",
        lw=2.0,
        marker="s",
        ms=4,
        label="ion",
    )
    axes[1, 0].plot(
        rho_np,
        residual,
        color="#111827",
        lw=1.8,
        ls=":",
        label="charge-weighted sum",
    )
    axes[1, 0].set_xlabel(r"$\rho$")
    axes[1, 0].set_ylabel("Particle-flux response")
    axes[1, 0].set_title("Ambipolar residual closure")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(
        np.asarray(er_axis),
        residual_norm_scan,
        color="#CC79A7",
        lw=2.3,
        marker="D",
        ms=4.5,
    )
    axes[1, 1].axvline(float(np.mean(er_profile)), color="#111827", lw=1.5, ls=":")
    axes[1, 1].set_xlabel(r"Uniform trial $\hat E_r$")
    axes[1, 1].set_ylabel(r"$\|R(\rho,\hat E_r)\|_2$")
    axes[1, 1].set_title("Integrated ambipolar landscape")
    axes[1, 1].text(
        0.04,
        0.95,
        (
            rf"$N_\theta={GRID.n_theta}$, $N_\zeta={GRID.n_zeta}$, $N_\xi={GRID.n_xi}$" "\n"
            + f"$N_\\rho={rho.size}$, $N_\\nu={nu_v.size}$, $N_{{E_r}}={er_axis.size}$"
        ),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
