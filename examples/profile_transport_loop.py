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

from ntx import (  # noqa: E402
    GridSpec,
    MonoenergeticSpeciesProfile,
    ProfileTransportClosureSpec,
    build_ntx_neopax_scan_from_surfaces,
    example_surface,
    solve_profile_transport_loop,
)
from ntx.config import enable_x64  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "profile_transport_loop"

GRID = GridSpec(5, 7, 5, dtype="float32")
SOLVE_STEPS = 6
TRANSPORT_ITERATIONS = 5


def _surface_family(rho_values):
    base = example_surface(dtype=GRID.jax_dtype)
    return tuple(
        replace(base, b_cos=base.b_cos.at[1].set(base.b_cos[1] * (1.0 + 0.12 * float(rho))))
        for rho in rho_values
    )


def _species_profiles(rho):
    return (
        MonoenergeticSpeciesProfile(
            charge=-1.0,
            nu_v=jnp.linspace(4.0e-4, 1.0e-3, rho.size, dtype=rho.dtype),
            A1=1.10 - 0.25 * rho,
            A3=0.58 - 0.10 * rho,
            current_weight=-1.0,
            name="electron",
        ),
        MonoenergeticSpeciesProfile(
            charge=1.0,
            nu_v=jnp.linspace(2.0e-3, 5.0e-3, rho.size, dtype=rho.dtype),
            A1=0.72 + 0.33 * rho,
            A3=0.24 + 0.05 * rho,
            particle_weight=1.08,
            current_weight=1.0,
            name="ion",
        ),
    )


def _transport_closure(rho):
    return ProfileTransportClosureSpec(
        particle_relaxation=jnp.asarray(
            [
                0.05 + 0.01 * rho,
                0.03 + 0.005 * rho,
            ],
            dtype=rho.dtype,
        ),
        current_relaxation=jnp.asarray(
            [
                0.04 + 0.01 * (1.0 - rho),
                0.02 + 0.005 * rho,
            ],
            dtype=rho.dtype,
        ),
        particle_target=jnp.asarray(
            [
                -0.02 * (1.0 - rho),
                0.015 * (1.0 - 0.5 * rho),
            ],
            dtype=rho.dtype,
        ),
        current_target=jnp.asarray(
            [
                -0.015 * (1.0 - 0.3 * rho),
                0.010 * (1.0 - 0.2 * rho),
            ],
            dtype=rho.dtype,
        ),
        particle_source=jnp.asarray(
            [
                0.004 * jnp.ones_like(rho),
                -0.003 * jnp.ones_like(rho),
            ],
            dtype=rho.dtype,
        ),
        current_source=0.0,
        normalization_floor=0.05,
        max_normalized_update=0.25,
        radial_smoothing_strength=0.35,
        closure_name="radial proxy transport",
    )


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    enable_x64(False)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rho = jnp.linspace(0.2, 0.82, 7, dtype=GRID.jax_dtype)
    nu_v = jnp.asarray([3.0e-4, 1.0e-3, 3.0e-3], dtype=GRID.jax_dtype)
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
    surfaces = _surface_family(rho)
    er_grid = jnp.tile(er_axis[None, :], (rho.size, 1))
    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=er_grid,
        Er=er_grid,
        drds=jnp.ones_like(rho),
        grid=GRID,
        source_name="profile_transport_loop_example",
    )
    result = solve_profile_transport_loop(
        scan,
        _species_profiles(rho),
        _transport_closure(rho),
        iterations=TRANSPORT_ITERATIONS,
        solve_steps=SOLVE_STEPS,
        damping=0.75,
        smoothing_strength=0.55,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    colors = plt.cm.cividis(jnp.linspace(0.15, 0.9, TRANSPORT_ITERATIONS))

    for idx in range(TRANSPORT_ITERATIONS):
        label = f"iter {idx + 1}"
        axes[0, 0].plot(
            rho,
            result.ambipolar_residual_history[idx],
            color=colors[idx],
            lw=2.0,
            label=label,
        )
        axes[0, 1].plot(
            rho,
            result.bootstrap_current_proxy_history[idx],
            color=colors[idx],
            lw=2.0,
        )

    iterations = jnp.arange(1, TRANSPORT_ITERATIONS + 1)
    ambipolar_norm = jnp.linalg.norm(result.ambipolar_residual_history, axis=1)
    relative_transport_loss = result.transport_loss_history / result.transport_loss_history[0]
    relative_ambipolar_norm = ambipolar_norm / ambipolar_norm[0]
    axes[1, 0].plot(
        iterations,
        relative_transport_loss,
        marker="o",
        lw=2.5,
        label="relative transport loss",
    )
    axes[1, 0].plot(
        iterations,
        relative_ambipolar_norm,
        marker="s",
        lw=2.0,
        label="relative ambipolar residual",
    )

    species_names = ("electron A1", "electron A3", "ion A1", "ion A3")
    final_a1 = result.species_a1_history[-1]
    final_a3 = result.species_a3_history[-1]
    axes[1, 1].plot(rho, final_a1[0], lw=2.5, label=species_names[0])
    axes[1, 1].plot(rho, final_a3[0], lw=2.5, label=species_names[1])
    axes[1, 1].plot(rho, final_a1[1], lw=2.5, label=species_names[2])
    axes[1, 1].plot(rho, final_a3[1], lw=2.5, label=species_names[3])

    axes[0, 0].set_title("Ambipolar Residual Evolution")
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"$R(\rho)$")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    axes[0, 1].set_title("Bootstrap-Current Proxy Evolution")
    axes[0, 1].set_xlabel(r"$\rho$")
    axes[0, 1].set_ylabel(r"$J_{\mathrm{bs,proxy}}$")

    axes[1, 0].set_title("Transport Iteration Metrics")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("relative metric")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].set_title("Final Thermodynamic-Force Profiles")
    axes[1, 1].set_xlabel(r"$\rho$")
    axes[1, 1].set_ylabel("profile amplitude")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
