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
    PrimitiveSpeciesProfile,
    ProfileTransportClosureSpec,
    build_ntx_neopax_scan_from_surfaces,
    build_species_profiles_from_primitives,
    example_surface,
    solve_primitive_profile_transport_loop,
)
from ntx.config import enable_x64  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "primitive_profile_transport"
GRID = GridSpec(3, 5, 4, dtype="float32")
ITERATIONS = 3


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (11.5, 8.0),
            "figure.dpi": 220,
            "font.size": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _surface_family(rho):
    base = example_surface(dtype=GRID.jax_dtype)
    return tuple(
        replace(base, b_cos=base.b_cos.at[1].set(base.b_cos[1] * (1.0 + 0.08 * float(rho))))
        for rho in rho
    )


def _primitives(rho):
    return (
        PrimitiveSpeciesProfile(
            charge=-1.0,
            nu_v=jnp.linspace(3.5e-4, 9.0e-4, rho.size, dtype=GRID.jax_dtype),
            density=1.05 - 0.18 * rho + 0.03 * (1.0 - rho) ** 2,
            temperature=1.00 - 0.24 * rho + 0.05 * rho**2,
            electrostatic_prefactor=0.14,
            current_weight=-1.0,
            name="electron",
        ),
        PrimitiveSpeciesProfile(
            charge=1.0,
            nu_v=jnp.linspace(1.8e-3, 4.2e-3, rho.size, dtype=GRID.jax_dtype),
            density=0.96 - 0.10 * rho + 0.04 * rho**2,
            temperature=0.88 - 0.12 * rho + 0.06 * rho**2,
            electrostatic_prefactor=0.10,
            particle_weight=1.05,
            current_weight=1.0,
            name="ion",
        ),
    )


def _closure(rho):
    return ProfileTransportClosureSpec(
        particle_relaxation=jnp.asarray(
            [
                0.040 + 0.005 * rho,
                0.028 + 0.004 * rho,
            ],
            dtype=GRID.jax_dtype,
        ),
        current_relaxation=jnp.asarray(
            [
                0.030 + 0.004 * (1.0 - rho),
                0.018 + 0.003 * rho,
            ],
            dtype=GRID.jax_dtype,
        ),
        particle_target=jnp.asarray(
            [
                -0.012 * (1.0 - 0.4 * rho),
                0.009 * (1.0 - 0.2 * rho),
            ],
            dtype=GRID.jax_dtype,
        ),
        current_target=jnp.asarray(
            [
                -0.010 * (1.0 - 0.3 * rho),
                0.007 * (1.0 - 0.15 * rho),
            ],
            dtype=GRID.jax_dtype,
        ),
        particle_source=jnp.asarray(
            [
                0.003 * jnp.ones_like(rho),
                -0.002 * jnp.ones_like(rho),
            ],
            dtype=GRID.jax_dtype,
        ),
        current_source=0.0,
        normalization_floor=0.05,
        max_normalized_update=0.20,
        density_relaxation=jnp.asarray(
            [
                0.018 + 0.002 * rho,
                0.014 + 0.002 * rho,
            ],
            dtype=GRID.jax_dtype,
        ),
        temperature_relaxation=jnp.asarray(
            [
                0.014 + 0.002 * (1.0 - rho),
                0.010 + 0.001 * rho,
            ],
            dtype=GRID.jax_dtype,
        ),
        density_target=jnp.asarray(
            [
                0.96 - 0.06 * rho + 0.02 * (1.0 - rho) ** 2,
                0.92 - 0.03 * rho + 0.015 * rho**2,
            ],
            dtype=GRID.jax_dtype,
        ),
        temperature_target=jnp.asarray(
            [
                0.94 - 0.18 * rho + 0.04 * rho**2,
                0.86 - 0.09 * rho + 0.04 * rho**2,
            ],
            dtype=GRID.jax_dtype,
        ),
        primitive_normalization_floor=0.03,
        max_primitive_normalized_update=0.12,
        radial_smoothing_strength=0.45,
        closure_name="primitive transport",
    )


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    enable_x64(False)
    _configure_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rho = jnp.linspace(0.15, 0.82, 7, dtype=GRID.jax_dtype)
    nu_v = jnp.asarray([3.0e-4, 2.0e-3], dtype=GRID.jax_dtype)
    er_axis = jnp.asarray(
        [-2.0e-3, -7.5e-4, -2.5e-4, 0.0, 2.5e-4, 7.5e-4, 2.0e-3],
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
        source_name="primitive_profile_transport_example",
    )
    result = solve_primitive_profile_transport_loop(
        scan,
        _primitives(rho),
        _closure(rho),
        iterations=ITERATIONS,
        solve_steps=5,
        damping=0.75,
        smoothing_strength=0.45,
    )

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    colors = plt.cm.cividis(jnp.linspace(0.15, 0.9, ITERATIONS))

    selected = [0, ITERATIONS - 1]
    labels = ("initial", "final")
    for idx, label in zip(selected, labels, strict=True):
        axes[0, 0].plot(
            rho,
            result.ambipolar_residual_history[idx],
            color=colors[idx],
            lw=2.2,
            label=label,
        )
        axes[0, 1].plot(
            rho,
            result.bootstrap_current_proxy_history[idx],
            color=colors[idx],
            lw=2.2,
            label=label,
        )

    final_primitives = tuple(
        PrimitiveSpeciesProfile(
            charge=primitive.charge,
            nu_v=primitive.nu_v,
            density=result.species_density_history[-1, index],
            temperature=result.species_temperature_history[-1, index],
            electrostatic_prefactor=primitive.electrostatic_prefactor,
            particle_weight=primitive.particle_weight,
            current_weight=primitive.current_weight,
            name=primitive.name,
        )
        for index, primitive in enumerate(_primitives(rho))
    )
    final_species = build_species_profiles_from_primitives(
        rho,
        final_primitives,
        er_profile=result.best_profile.er_profile,
    )
    axes[1, 0].plot(rho, final_species[0].A1, lw=2.2, label="electron A1")
    axes[1, 0].plot(rho, final_species[0].A3, lw=2.0, label="electron A3")
    axes[1, 0].plot(rho, final_species[1].A1, lw=2.0, label="ion A1")
    axes[1, 0].plot(rho, final_species[1].A3, lw=1.9, label="ion A3")

    final_density = result.species_density_history[-1]
    final_temperature = result.species_temperature_history[-1]
    axes[1, 1].plot(rho, final_density[0], lw=2.2, label="electron density")
    axes[1, 1].plot(rho, final_temperature[0], lw=2.2, label="electron temperature")
    axes[1, 1].plot(rho, final_density[1], lw=2.2, label="ion density")
    axes[1, 1].plot(rho, final_temperature[1], lw=2.2, label="ion temperature")

    axes[0, 0].set_title("Primitive Ambipolar Residual Evolution")
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"$R(\rho)$")
    axes[0, 0].legend(loc="best")

    axes[0, 1].set_title("Primitive Transport Current Evolution")
    axes[0, 1].set_xlabel(r"$\rho$")
    axes[0, 1].set_ylabel(r"$J_{\mathrm{bs,proxy}}$")
    axes[0, 1].legend(loc="best")

    axes[1, 0].set_title("Derived Monoenergetic Forces")
    axes[1, 0].set_xlabel(r"$\rho$")
    axes[1, 0].set_ylabel("profile amplitude")
    axes[1, 0].legend(loc="best")

    axes[1, 1].set_title("Final Primitive Profiles")
    axes[1, 1].set_xlabel(r"$\rho$")
    axes[1, 1].set_ylabel("normalized amplitude")
    axes[1, 1].legend(loc="best", fontsize=8, ncol=2)

    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
