#!/usr/bin/env python3
"""Solve a small family of ambipolar profiles and compare their current proxies."""

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
    bootstrap_current_objective,
    build_ntx_neopax_scan_from_surfaces,
    example_surface,
    solve_ambipolar_profile_family,
)
from ntx.config import enable_x64  # noqa: E402

GRID = GridSpec(7, 9, 6)
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "ambipolar_profile_family"
HARMONIC_INDEX = 1
CONTROL = jnp.asarray([-0.30, -0.15, 0.0, 0.15, 0.30])


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (11.8, 7.1),
            "figure.dpi": 220,
            "font.size": 10.5,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.alpha": 0.16,
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
                base.b_cos[HARMONIC_INDEX] * (1.0 + 0.16 * float(rho))
            ),
        )
        for rho in np.asarray(rho_values)
    )


def _species_family(rho):
    electron = MonoenergeticSpeciesProfile(
        charge=jnp.asarray(-1.0, dtype=GRID.jax_dtype),
        nu_v=jnp.linspace(4.0e-4, 1.0e-3, rho.size, dtype=GRID.jax_dtype),
        A1=1.05 - 0.20 * rho,
        A3=0.52 - 0.10 * rho,
        current_weight=jnp.asarray(-1.0, dtype=GRID.jax_dtype),
        name="electron",
    )
    ion = MonoenergeticSpeciesProfile(
        charge=jnp.asarray(1.0, dtype=GRID.jax_dtype),
        nu_v=jnp.linspace(2.0e-3, 4.8e-3, rho.size, dtype=GRID.jax_dtype),
        A1=0.72 + 0.30 * rho,
        A3=0.22 + 0.09 * rho,
        particle_weight=jnp.asarray(1.05, dtype=GRID.jax_dtype),
        current_weight=jnp.asarray(1.0, dtype=GRID.jax_dtype),
        name="ion",
    )
    return tuple(
        (
            replace(electron, A3=electron.A3 * (1.0 + control)),
            replace(ion, A1=ion.A1 * (1.0 - 0.6 * control)),
        )
        for control in np.asarray(CONTROL)
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
    scan = build_ntx_neopax_scan_from_surfaces(
        _surface_family(rho),
        rho=rho,
        nu_v=nu_v,
        Es=er_grid,
        Er=er_grid,
        drds=jnp.ones_like(rho),
        grid=GRID,
        source_name="ambipolar_profile_family_example",
    )
    family = solve_ambipolar_profile_family(
        scan,
        _species_family(rho),
        control=CONTROL,
        steps=12,
        damping=0.7,
    )
    objectives = jnp.asarray(
        [
            bootstrap_current_objective(rho, family.bootstrap_current_proxy[index])
            for index in range(CONTROL.size)
        ]
    )
    best = int(jnp.argmin(objectives))
    control_np = np.asarray(family.control)
    rho_np = np.asarray(rho)
    er_profiles = np.asarray(family.er_profile)
    current_profiles = np.asarray(family.bootstrap_current_proxy)
    residual_norm = np.linalg.norm(np.asarray(family.ambipolar_residual), axis=1)
    objectives_np = np.asarray(objectives)

    colors = plt.cm.cividis(np.linspace(0.08, 0.92, control_np.size))
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    for color, control_value, profile in zip(colors, control_np, er_profiles, strict=True):
        axes[0, 0].plot(
            rho_np,
            profile,
            color=color,
            lw=2.0,
            marker="o",
            ms=4.5,
            label=fr"$c={control_value:+.2f}$",
        )
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"Solved $\hat E_r$")
    axes[0, 0].set_title("Ambipolar field family")
    axes[0, 0].legend(loc="best", ncol=2)

    for color, control_value, profile in zip(colors, control_np, current_profiles, strict=True):
        axes[0, 1].plot(
            rho_np,
            profile,
            color=color,
            lw=2.0,
            marker="s",
            ms=4.5,
            label=fr"$c={control_value:+.2f}$",
        )
    axes[0, 1].set_xlabel(r"$\rho$")
    axes[0, 1].set_ylabel("Bootstrap-current proxy")
    axes[0, 1].set_title("Current-profile response")

    axes[1, 0].plot(control_np, objectives_np, color="#D55E00", lw=2.3, marker="D", ms=5)
    axes[1, 0].scatter(
        control_np[best],
        objectives_np[best],
        color="#111827",
        s=36,
        zorder=3,
        label="minimum",
    )
    axes[1, 0].set_xlabel("Profile control")
    axes[1, 0].set_ylabel(r"$\int J_{\mathrm{bs,proxy}}^2\,d\rho$")
    axes[1, 0].set_title("Profile objective landscape")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(control_np, residual_norm, color="#009E73", lw=2.3, marker="^", ms=5)
    axes[1, 1].set_xlabel("Profile control")
    axes[1, 1].set_ylabel(r"$\|R(\rho)\|_2$")
    axes[1, 1].set_title("Ambipolar residual closure")
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
