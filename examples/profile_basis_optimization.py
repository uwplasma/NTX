#!/usr/bin/env python3
"""Optimize a low-dimensional radial basis control on top of the NTX profile closure."""

from __future__ import annotations

import json
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
    ProfileBasisControlSpec,
    build_ntx_neopax_scan_from_surfaces,
    current_response_objective,
    example_surface,
    optimize_profile_basis_control,
    solve_ambipolar_er_profile,
)
from ntx.config import enable_x64  # noqa: E402

GRID = GridSpec(9, 11, 8)
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "profile_basis_optimization"
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
                base.b_cos[HARMONIC_INDEX] * (1.0 + 0.14 * float(rho))
            ),
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
        [
            -3.0e-3,
            -2.0e-3,
            -1.2e-3,
            -7.5e-4,
            -3.0e-4,
            -1.0e-4,
            0.0,
            1.0e-4,
            3.0e-4,
            7.5e-4,
            1.2e-3,
            2.0e-3,
            3.0e-3,
        ],
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
        source_name="profile_basis_optimization_example",
    )

    species_profiles = (
        MonoenergeticSpeciesProfile(
            charge=jnp.asarray(-1.0, dtype=GRID.jax_dtype),
            nu_v=jnp.linspace(4.0e-4, 1.0e-3, rho.size, dtype=GRID.jax_dtype),
            A1=1.02 - 0.20 * rho,
            A3=0.54 - 0.10 * rho,
            current_weight=jnp.asarray(-1.0, dtype=GRID.jax_dtype),
            name="electron",
        ),
        MonoenergeticSpeciesProfile(
            charge=jnp.asarray(1.0, dtype=GRID.jax_dtype),
            nu_v=jnp.linspace(2.0e-3, 4.8e-3, rho.size, dtype=GRID.jax_dtype),
            A1=0.74 + 0.28 * rho,
            A3=0.22 + 0.08 * rho,
            particle_weight=jnp.asarray(1.05, dtype=GRID.jax_dtype),
            current_weight=jnp.asarray(1.0, dtype=GRID.jax_dtype),
            name="ion",
        ),
    )
    basis = jnp.asarray(
        [
            jnp.linspace(1.0, 0.0, rho.size, dtype=GRID.jax_dtype),
            1.0 - 4.0 * (rho - 0.5) ** 2,
            jnp.linspace(0.0, 1.0, rho.size, dtype=GRID.jax_dtype),
        ]
    )
    control_spec = ProfileBasisControlSpec(
        basis=basis,
        a1_response=jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [-0.2, -0.6, -0.2],
            ],
            dtype=GRID.jax_dtype,
        ),
        a3_response=jnp.asarray(
            [
                [0.4, 0.8, 0.4],
                [0.0, 0.0, 0.0],
            ],
            dtype=GRID.jax_dtype,
        ),
        control_name="basis control",
    )
    control_initial = jnp.asarray([0.10, -0.08, 0.06], dtype=GRID.jax_dtype)
    weight = 1.0 + 1.2 * rho
    result = optimize_profile_basis_control(
        scan,
        species_profiles,
        control_spec,
        control_initial=control_initial,
        learning_rate=0.08,
        optimization_steps=10,
        solve_steps=12,
        damping=0.7,
        smoothing_strength=0.45,
        weight=weight,
        residual_penalty=0.5,
        control_penalty=1.0e-2,
        control_bound=0.20,
    )
    baseline_profile = solve_ambipolar_er_profile(
        scan,
        species_profiles,
        steps=12,
        damping=0.7,
        smoothing_strength=0.45,
    )
    best_current = np.asarray(result.best_profile.bootstrap_current_response)
    baseline_current = np.asarray(baseline_profile.bootstrap_current_response)
    best_residual = np.asarray(result.best_profile.ambipolar_residual)
    baseline_residual = np.asarray(baseline_profile.ambipolar_residual)
    control_history = np.asarray(result.control_history)
    rho_np = np.asarray(rho)
    basis_np = np.asarray(basis)
    best_control = np.asarray(result.best_control)
    modifier = np.tensordot(best_control, basis_np, axes=1)
    best_objective = float(
        current_response_objective(
            rho,
            result.best_profile.bootstrap_current_response,
            weight=weight,
        )
    )
    baseline_objective = float(
        current_response_objective(rho, baseline_profile.bootstrap_current_response, weight=weight)
    )

    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    for index in range(control_history.shape[1]):
        axes[0, 0].plot(
            np.arange(1, control_history.shape[0] + 1),
            control_history[:, index],
            lw=2.0,
            marker="o",
            ms=4.0,
            label=fr"$c_{index}$",
        )
    axes[0, 0].set_xlabel("Optimization iteration")
    axes[0, 0].set_ylabel("Basis coefficient")
    axes[0, 0].set_title("Basis-control history")
    axes[0, 0].set_ylim(-0.22, 0.22)
    axes[0, 0].legend(loc="best", ncol=3)

    for index in range(basis_np.shape[0]):
        axes[0, 1].plot(
            rho_np,
            basis_np[index],
            lw=1.8,
            label=fr"$\phi_{index}(\rho)$",
        )
    axes[0, 1].plot(rho_np, modifier, color="#111827", lw=2.6, ls="--", label="net modifier")
    axes[0, 1].set_xlabel(r"$\rho$")
    axes[0, 1].set_ylabel("Control amplitude")
    axes[0, 1].set_title("Radial basis and optimized modifier")
    axes[0, 1].legend(loc="best", ncol=2)

    axes[1, 0].plot(
        rho_np,
        baseline_residual,
        color="#9ca3af",
        lw=2.0,
        marker="o",
        ms=4.5,
        label="baseline",
    )
    axes[1, 0].plot(
        rho_np,
        best_residual,
        color="#CC79A7",
        lw=2.3,
        marker="D",
        ms=4.5,
        label="optimized",
    )
    axes[1, 0].set_xlabel(r"$\rho$")
    axes[1, 0].set_ylabel(r"$R(\rho)$")
    axes[1, 0].set_title("Residual-profile reduction")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(
        rho_np,
        baseline_current,
        color="#9ca3af",
        lw=2.0,
        marker="o",
        ms=4.5,
        label="baseline",
    )
    axes[1, 1].plot(
        rho_np,
        best_current,
        color="#0072B2",
        lw=2.3,
        marker="s",
        ms=5,
        label="optimized",
    )
    axes[1, 1].set_xlabel(r"$\rho$")
    axes[1, 1].set_ylabel("Reduced current response")
    axes[1, 1].set_title("Optimized current profile")
    axes[1, 1].legend(loc="best")
    axes[1, 1].text(
        0.04,
        0.95,
        (
            rf"$N_\theta={GRID.n_theta}$, $N_\zeta={GRID.n_zeta}$, $N_\xi={GRID.n_xi}$" "\n"
            + rf"$\mathcal{{J}}_\star={best_objective:.3e}$" "\n"
            + rf"$\|c_\star\|_2={float(np.linalg.norm(best_control)):.3e}$"
        ),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    payload = {
        "artifact": "profile_basis_optimization",
        "claim_scope": (
            "Optimizes a three-function radial basis control on a profile "
            "closure benchmark and records objective, residual, and control "
            "regularity metrics."
        ),
        "grid": {
            "n_theta": GRID.n_theta,
            "n_zeta": GRID.n_zeta,
            "n_xi": GRID.n_xi,
        },
        "rho": rho_np.tolist(),
        "basis_count": int(basis_np.shape[0]),
        "best_control": best_control.tolist(),
        "baseline_objective": baseline_objective,
        "best_objective": best_objective,
        "objective_improvement": best_objective - baseline_objective,
        "objective_gain": best_objective / max(abs(baseline_objective), 1.0e-30),
        "baseline_residual_l2": float(np.linalg.norm(baseline_residual)),
        "best_residual_l2": float(np.linalg.norm(best_residual)),
        "residual_l2_ratio": float(
            np.linalg.norm(best_residual) / max(np.linalg.norm(baseline_residual), 1.0e-30)
        ),
        "max_abs_control": float(np.max(np.abs(best_control))),
        "baseline_current_profile": baseline_current.tolist(),
        "best_current_profile": best_current.tolist(),
        "baseline_residual": baseline_residual.tolist(),
        "best_residual": best_residual.tolist(),
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    plt.close(fig)
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
