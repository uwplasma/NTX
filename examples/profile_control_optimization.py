#!/usr/bin/env python3
"""Optimize a scalar profile control against the NTX bootstrap-current objective."""

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
    ProfileControlSpec,
    bootstrap_current_objective,
    build_ntx_neopax_scan_from_surfaces,
    example_surface,
    optimize_profile_control,
)
from ntx.config import enable_x64  # noqa: E402

GRID = GridSpec(7, 9, 6)
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "profile_control_optimization"
HARMONIC_INDEX = 1


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (11.8, 7.2),
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
        base
        if index == 0
        else replace(
            base,
            b_cos=base.b_cos.at[HARMONIC_INDEX].set(
                base.b_cos[HARMONIC_INDEX] * (1.0 + 0.15 * float(rho))
            )
        )
        for index, rho in enumerate(np.asarray(rho_values))
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
        source_name="profile_control_optimization_example",
    )

    species_profiles = (
        MonoenergeticSpeciesProfile(
            charge=jnp.asarray(-1.0, dtype=GRID.jax_dtype),
            nu_v=jnp.linspace(4.0e-4, 1.0e-3, rho.size, dtype=GRID.jax_dtype),
            A1=1.05 - 0.22 * rho,
            A3=0.54 - 0.10 * rho,
            current_weight=jnp.asarray(-1.0, dtype=GRID.jax_dtype),
            name="electron",
        ),
        MonoenergeticSpeciesProfile(
            charge=jnp.asarray(1.0, dtype=GRID.jax_dtype),
            nu_v=jnp.linspace(2.0e-3, 4.8e-3, rho.size, dtype=GRID.jax_dtype),
            A1=0.74 + 0.30 * rho,
            A3=0.22 + 0.08 * rho,
            particle_weight=jnp.asarray(1.05, dtype=GRID.jax_dtype),
            current_weight=jnp.asarray(1.0, dtype=GRID.jax_dtype),
            name="ion",
        ),
    )
    control_spec = ProfileControlSpec(
        a1_response=jnp.asarray([0.0, -0.6], dtype=GRID.jax_dtype),
        a3_response=jnp.asarray([1.0, 0.0], dtype=GRID.jax_dtype),
        control_name="profile control",
    )
    weight = 1.0 + 0.8 * rho
    result = optimize_profile_control(
        scan,
        species_profiles,
        control_spec,
        control_initial=0.20,
        learning_rate=0.10,
        optimization_steps=10,
        solve_steps=12,
        damping=0.7,
        weight=weight,
        residual_penalty=0.5,
        control_bound=0.35,
    )

    control_history = np.asarray(result.control_history)
    objective_history = np.asarray(result.objective_history)
    bootstrap_history = np.asarray(result.bootstrap_objective_history)
    residual_history = np.asarray(result.residual_norm_history)
    best_profile = result.best_profile
    best_current = np.asarray(best_profile.bootstrap_current_proxy)
    best_er = np.asarray(best_profile.er_profile)
    rho_np = np.asarray(rho)
    weight_np = np.asarray(weight)
    best_objective = float(
        bootstrap_current_objective(
            rho,
            best_profile.bootstrap_current_proxy,
            weight=weight,
        )
    )

    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    axes[0, 0].plot(
        np.arange(1, objective_history.size + 1),
        objective_history,
        color="#D55E00",
        lw=2.3,
        marker="o",
        ms=4.5,
        label="total objective",
    )
    axes[0, 0].plot(
        np.arange(1, bootstrap_history.size + 1),
        bootstrap_history,
        color="#0072B2",
        lw=1.8,
        marker="s",
        ms=4.0,
        label="bootstrap term",
    )
    axes[0, 0].set_xlabel("Optimization iteration")
    axes[0, 0].set_ylabel("Objective")
    axes[0, 0].set_title("Control optimization history")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(
        np.arange(1, control_history.size + 1),
        control_history,
        color="#009E73",
        lw=2.3,
        marker="D",
        ms=4.5,
    )
    axes[0, 1].axhline(float(result.best_control), color="#111827", lw=1.2, ls=":")
    axes[0, 1].set_xlabel("Optimization iteration")
    axes[0, 1].set_ylabel(control_spec.control_name)
    axes[0, 1].set_title("Scalar control evolution")

    axes[1, 0].plot(rho_np, best_er, color="#CC79A7", lw=2.3, marker="o", ms=5)
    axes[1, 0].set_xlabel(r"$\rho$")
    axes[1, 0].set_ylabel(r"Solved $\hat E_r$")
    axes[1, 0].set_title("Best ambipolar field profile")

    axes[1, 1].plot(rho_np, best_current, color="#0072B2", lw=2.3, marker="s", ms=5)
    axes[1, 1].fill_between(
        rho_np,
        0.0,
        weight_np * best_current,
        color="#0072B2",
        alpha=0.12,
        linewidth=0.0,
    )
    axes[1, 1].set_xlabel(r"$\rho$")
    axes[1, 1].set_ylabel("Bootstrap-current proxy")
    axes[1, 1].set_title("Best current profile")
    axes[1, 1].text(
        0.04,
        0.95,
        (
            rf"$N_\theta={GRID.n_theta}$, $N_\zeta={GRID.n_zeta}$, $N_\xi={GRID.n_xi}$" "\n"
            + (
                rf"$c_\star={float(result.best_control):+.3f}$, "
                rf"$\mathcal{{J}}_\star={best_objective:.3e}$"
            )
            + "\n"
            + rf"$\|R\|_2={float(residual_history[-1]):.3e}$"
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
