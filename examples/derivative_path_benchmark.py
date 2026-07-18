#!/usr/bin/env python3
"""Benchmark direct reverse-mode and prepared custom-VJP derivative paths."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    MonoenergeticCase,
    audit_prepared_coefficient_derivative,
    example_surface,
    prepare_monoenergetic_system,
    solve_prepared_coefficient_vector,
    solve_prepared_coefficient_vector_vjp,
)
from ntx.config import enable_x64  # noqa: E402

GRID = GridSpec(7, 9, 6)
NU_HAT = 3.0e-4
SCAN_SIZES = (1, 2, 4, 8, 16, 32)
ER_MIN = 1.0e-6
ER_MAX = 3.0e-3
NU_MIN = 3.0e-5
NU_MAX = 3.0e-3
REPEATS = 3
AUDIT_NU_HAT = (1.0e-3, 3.0e-3, 1.0e-2)
AUDIT_ER_HAT = 1.0e-3
RESOLUTION_GRIDS = ((9, 11, 8), (11, 13, 10), (13, 15, 12))
RESOLUTION_TOLERANCE = 1.0e-1
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "derivative_path_benchmark"


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (17.0, 9.2),
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


def _time_callable(fn, argument) -> float:
    start = time.perf_counter()
    value = fn(argument)
    jax.block_until_ready(value)
    return time.perf_counter() - start


def _memory_stats(compiled) -> dict[str, int]:
    memory = compiled.memory_analysis()
    return {
        name: int(getattr(memory, name, 0) or 0)
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "generated_code_size_in_bytes",
        )
    }


def _profile_compiled_derivative(fn, argument) -> tuple[dict[str, float | int], np.ndarray]:
    compiled = fn.lower(argument).compile()
    values = compiled(argument)
    jax.block_until_ready(values)
    timings = [_time_callable(compiled, argument) for _ in range(REPEATS)]
    return {
        **_memory_stats(compiled),
        "warm_seconds_min": min(timings),
    }, np.asarray(values)


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    enable_x64(True)
    _configure_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    surface = example_surface(dtype=GRID.jax_dtype)
    prepared = prepare_monoenergetic_system(surface, GRID)

    def direct_scalar(er_hat):
        return solve_prepared_coefficient_vector(
            prepared,
            MonoenergeticCase(nu_hat=NU_HAT, er_hat=er_hat),
        )[3]

    def custom_scalar(er_hat):
        return solve_prepared_coefficient_vector_vjp(
            prepared,
            MonoenergeticCase(nu_hat=NU_HAT, er_hat=er_hat),
        )[3]

    def direct_nu_scalar(nu_hat):
        return solve_prepared_coefficient_vector(
            prepared,
            MonoenergeticCase(nu_hat=nu_hat, er_hat=ER_MIN),
        )[0]

    def custom_nu_scalar(nu_hat):
        return solve_prepared_coefficient_vector_vjp(
            prepared,
            MonoenergeticCase(nu_hat=nu_hat, er_hat=ER_MIN),
        )[0]

    direct_er_grad = jax.jit(jax.vmap(jax.grad(direct_scalar)))
    rematerialized_er_grad = jax.jit(jax.vmap(jax.grad(jax.checkpoint(direct_scalar))))
    custom_er_grad = jax.jit(jax.vmap(jax.grad(custom_scalar)))
    direct_nu_grad = jax.jit(jax.vmap(jax.grad(direct_nu_scalar)))
    custom_nu_grad = jax.jit(jax.vmap(jax.grad(custom_nu_scalar)))

    def direct_pair(inputs):
        er_hat_scan, nu_hat_scan = inputs
        return jnp.stack((direct_er_grad(er_hat_scan), direct_nu_grad(nu_hat_scan)))

    def custom_pair(inputs):
        er_hat_scan, nu_hat_scan = inputs
        return jnp.stack((custom_er_grad(er_hat_scan), custom_nu_grad(nu_hat_scan)))

    counts = []
    direct_times = []
    custom_times = []
    max_relative_mismatch = []
    er_d33_relative_mismatch = []
    nu_d11_relative_mismatch = []

    for count in SCAN_SIZES:
        er_hat_scan = jnp.geomspace(ER_MIN, ER_MAX, count)
        nu_hat_scan = jnp.geomspace(NU_MIN, NU_MAX, count)
        inputs = (er_hat_scan, nu_hat_scan)

        _ = direct_pair(inputs)
        _ = custom_pair(inputs)

        direct_measurements = [_time_callable(direct_pair, inputs) for _ in range(REPEATS)]
        custom_measurements = [_time_callable(custom_pair, inputs) for _ in range(REPEATS)]

        direct_er_values = np.asarray(direct_er_grad(er_hat_scan))
        custom_er_values = np.asarray(custom_er_grad(er_hat_scan))
        direct_nu_values = np.asarray(direct_nu_grad(nu_hat_scan))
        custom_nu_values = np.asarray(custom_nu_grad(nu_hat_scan))
        er_mismatch = np.max(
            np.abs(direct_er_values - custom_er_values)
            / np.maximum(np.abs(direct_er_values), 1.0e-30)
        )
        nu_mismatch = np.max(
            np.abs(direct_nu_values - custom_nu_values)
            / np.maximum(np.abs(direct_nu_values), 1.0e-30)
        )
        mismatch = max(er_mismatch, nu_mismatch)

        counts.append(count)
        direct_times.append(min(direct_measurements))
        custom_times.append(min(custom_measurements))
        max_relative_mismatch.append(mismatch)
        er_d33_relative_mismatch.append(er_mismatch)
        nu_d11_relative_mismatch.append(nu_mismatch)

    largest_er_scan = jnp.geomspace(ER_MIN, ER_MAX, SCAN_SIZES[-1])
    direct_profile, direct_profile_values = _profile_compiled_derivative(
        direct_er_grad,
        largest_er_scan,
    )
    rematerialized_profile, rematerialized_profile_values = _profile_compiled_derivative(
        rematerialized_er_grad, largest_er_scan
    )
    prepared_profile, prepared_profile_values = _profile_compiled_derivative(
        custom_er_grad,
        largest_er_scan,
    )
    direct_profile["max_relative_mismatch_direct"] = 0.0
    for profile, values in (
        (rematerialized_profile, rematerialized_profile_values),
        (prepared_profile, prepared_profile_values),
    ):
        profile["max_relative_mismatch_direct"] = float(
            np.max(
                np.abs(values - direct_profile_values)
                / np.maximum(np.abs(direct_profile_values), 1.0e-30)
            )
        )
    derivative_memory = {
        "scan_size": SCAN_SIZES[-1],
        "channel": "dD33_dEr",
        "direct_reverse": direct_profile,
        "selective_recomputation": rematerialized_profile,
        "prepared_adjoint": prepared_profile,
    }
    derivative_audits = [
        audit_prepared_coefficient_derivative(
            prepared,
            MonoenergeticCase(nu_hat=nu_hat, er_hat=AUDIT_ER_HAT),
            coefficient="D11",
            parameter="er_hat",
        ).as_dict()
        for nu_hat in AUDIT_NU_HAT
    ]
    resolution_audits = []
    previous_gradient = None
    for dimensions in RESOLUTION_GRIDS:
        resolution_grid = GridSpec(*dimensions)
        resolution_prepared = prepare_monoenergetic_system(
            example_surface(dtype=resolution_grid.jax_dtype),
            resolution_grid,
        )
        entry = audit_prepared_coefficient_derivative(
            resolution_prepared,
            MonoenergeticCase(nu_hat=1.0e-2, er_hat=AUDIT_ER_HAT),
            coefficient="D33",
            parameter="nu_hat",
        ).as_dict()
        gradient = entry["direct_reverse_gradient"]
        entry["grid"] = {
            "n_theta": dimensions[0],
            "n_zeta": dimensions[1],
            "n_xi": dimensions[2],
        }
        entry["relative_change_previous"] = (
            None
            if previous_gradient is None
            else abs(gradient - previous_gradient)
            / max(abs(gradient), abs(previous_gradient), 1.0e-30)
        )
        resolution_audits.append(entry)
        previous_gradient = gradient

    counts_array = np.asarray(counts, dtype=float)
    direct_times_array = np.asarray(direct_times)
    custom_times_array = np.asarray(custom_times)
    mismatch_array = np.asarray(max_relative_mismatch)
    speedup_array = direct_times_array / np.maximum(custom_times_array, 1.0e-30)

    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    runtime_ax, memory_ax, path_ax, residual_ax, audit_ax, resolution_ax = axes.ravel()

    runtime_ax.loglog(
        counts_array,
        direct_times_array,
        color="#0072B2",
        lw=2.3,
        marker="o",
        ms=5,
        label="Direct reverse-mode",
    )
    runtime_ax.loglog(
        counts_array,
        custom_times_array,
        color="#D55E00",
        lw=2.3,
        marker="s",
        ms=5,
        label="Prepared custom VJP",
    )
    runtime_ax.set_xlabel("Scan size")
    runtime_ax.set_ylabel("Best-of-3 wall time [s]")
    runtime_ax.set_title(r"Two prepared parameter derivatives")
    runtime_ax.legend(loc="upper left")
    runtime_ax.text(
        0.03,
        0.04,
        (
            rf"$N_\theta={GRID.n_theta}$, $N_\zeta={GRID.n_zeta}$, $N_\xi={GRID.n_xi}$"
            "\n"
            rf"$\partial D_{{33}}/\partial \hat E_r$, "
            rf"$\partial D_{{11}}/\partial \hat\nu$"
            "\n32-case temporary memory: "
            f"{direct_profile['temp_size_in_bytes'] / 2**20:.1f} -> "
            f"{prepared_profile['temp_size_in_bytes'] / 2**20:.1f} MiB"
        ),
        transform=runtime_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    memory_labels = ("Direct", "Rematerialized", "Prepared adjoint")
    memory_profiles = (
        direct_profile,
        rematerialized_profile,
        prepared_profile,
    )
    memory_ax.bar(
        memory_labels,
        [profile["temp_size_in_bytes"] / 2**20 for profile in memory_profiles],
        color=("#0072B2", "#9ca3af", "#D55E00"),
    )
    for index, profile in enumerate(memory_profiles):
        memory_ax.text(
            index,
            profile["temp_size_in_bytes"] / 2**20,
            f"{1.0e3 * profile['warm_seconds_min']:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    memory_ax.set_ylabel("XLA temporary memory [MiB]")
    memory_ax.set_title(r"32-case $\partial D_{33}/\partial\hat E_r$")
    memory_ax.tick_params(axis="x", rotation=12)

    path_ax.semilogx(
        counts_array,
        speedup_array,
        color="#009E73",
        lw=2.3,
        marker="D",
        ms=5,
        label="Speedup",
    )
    path_ax.semilogx(
        counts_array,
        mismatch_array,
        color="#111827",
        lw=1.8,
        ls="--",
        marker="^",
        ms=4.5,
        label="Max relative mismatch",
    )
    path_ax.axhline(1.0, color="#9ca3af", lw=1.2, ls=":")
    path_ax.axhline(1.0e-4, color="#C62828", lw=1.2, ls="--", label="gradient gate")
    path_ax.set_yscale("log")
    path_ax.set_xlabel("Scan size")
    path_ax.set_ylabel("Speedup / relative mismatch")
    path_ax.set_title("Efficiency and direct-adjoint agreement")
    path_ax.legend(loc="best")

    audit_nu = np.asarray(AUDIT_NU_HAT)
    primal_residuals = np.asarray(
        [entry["primal_relative_residual"] for entry in derivative_audits]
    )
    transpose_residuals = np.asarray(
        [entry["transpose_relative_residual"] for entry in derivative_audits]
    )
    residual_ax.loglog(
        audit_nu,
        primal_residuals,
        color="#0072B2",
        lw=2.2,
        marker="o",
        label=r"Primal $\|Af-s\|/\|s\|$",
    )
    residual_ax.loglog(
        audit_nu,
        transpose_residuals,
        color="#D55E00",
        lw=2.2,
        marker="s",
        label=r"Adjoint $\|A^T\lambda-g\|/\|g\|$",
    )
    residual_ax.axhline(1.0e-10, color="#C62828", lw=1.2, ls="--", label="residual gate")
    residual_ax.set_xlabel(r"Collisionality $\hat\nu$")
    residual_ax.set_ylabel("Relative residual")
    residual_ax.set_title("Independent full-system residual gates")
    residual_ax.legend(loc="best")

    for key, label, style in (
        (
            "prepared_adjoint_relative_error",
            "Prepared adjoint",
            {"color": "#D55E00", "marker": "s"},
        ),
        ("forward_relative_error", "Forward mode", {"color": "#009E73", "marker": "D"}),
        (
            "finite_difference_relative_error",
            "Centered finite difference",
            {"color": "#111827", "marker": "^"},
        ),
    ):
        audit_ax.loglog(
            audit_nu,
            np.maximum([entry[key] for entry in derivative_audits], 1.0e-16),
            lw=2.0,
            label=label,
            **style,
        )
    audit_ax.axhline(1.0e-4, color="#C62828", lw=1.2, ls="--", label="gradient gate")
    audit_ax.set_xlabel(r"Collisionality $\hat\nu$")
    audit_ax.set_ylabel("Relative error against direct reverse mode")
    audit_ax.set_title(r"$\partial D_{11}/\partial\hat E_r$ derivative audit")
    audit_ax.legend(loc="best")

    resolution_sizes = np.asarray(
        [entry["grid"]["n_theta"] * entry["grid"]["n_zeta"] for entry in resolution_audits]
    )
    resolution_gradients = np.abs([entry["direct_reverse_gradient"] for entry in resolution_audits])
    resolution_ax.loglog(
        resolution_sizes,
        resolution_gradients,
        color="#6A3D9A",
        lw=2.2,
        marker="o",
        label=r"$|\partial D_{33}/\partial\hat\nu|$",
    )
    for size, gradient, entry in zip(
        resolution_sizes,
        resolution_gradients,
        resolution_audits,
        strict=True,
    ):
        grid = entry["grid"]
        resolution_ax.annotate(
            f"{grid['n_theta']}x{grid['n_zeta']}x{grid['n_xi']}",
            (size, gradient),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    changes = [
        entry["relative_change_previous"]
        for entry in resolution_audits
        if entry["relative_change_previous"] is not None
    ]
    resolution_ax.text(
        0.03,
        0.86,
        "successive changes: " + ", ".join(f"{change:.1%}" for change in changes),
        transform=resolution_ax.transAxes,
        fontsize=9,
    )
    resolution_ax.set_xlabel(r"Angular points $N_\theta N_\zeta$")
    resolution_ax.set_ylabel("Absolute derivative")
    resolution_ax.set_title("Two-step derivative refinement")
    resolution_ax.legend(loc="best")

    for label, ax in zip(("a", "b", "c", "d", "e", "f"), axes.ravel(), strict=True):
        ax.text(-0.12, 1.02, f"({label})", transform=ax.transAxes, fontweight="bold")

    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    summary = {
        "grid": {
            "n_theta": GRID.n_theta,
            "n_zeta": GRID.n_zeta,
            "n_xi": GRID.n_xi,
        },
        "nu_hat": NU_HAT,
        "er_min": ER_MIN,
        "er_max": ER_MAX,
        "nu_min": NU_MIN,
        "nu_max": NU_MAX,
        "scan_sizes": counts,
        "direct_times_seconds": direct_times,
        "prepared_times_seconds": custom_times,
        "speedup_prepared_vs_direct": speedup_array.tolist(),
        "max_relative_mismatch": max_relative_mismatch,
        "gradient_channels": {
            "dD33_dEr": {
                "fixed_nu_hat": NU_HAT,
                "er_min": ER_MIN,
                "er_max": ER_MAX,
                "max_relative_mismatch": er_d33_relative_mismatch,
            },
            "dD11_dnu": {
                "fixed_er_hat": ER_MIN,
                "nu_min": NU_MIN,
                "nu_max": NU_MAX,
                "max_relative_mismatch": nu_d11_relative_mismatch,
            },
        },
        "derivative_memory": derivative_memory,
        "validity_audit": {
            "coefficient": "D11",
            "parameter": "er_hat",
            "fixed_er_hat": AUDIT_ER_HAT,
            "entries": derivative_audits,
        },
        "resolution_audit": {
            "coefficient": "D33",
            "parameter": "nu_hat",
            "fixed_nu_hat": 1.0e-2,
            "fixed_er_hat": AUDIT_ER_HAT,
            "relative_change_tolerance": RESOLUTION_TOLERANCE,
            "two_successive_refinements_pass": all(
                entry["valid"] and entry["relative_change_previous"] <= RESOLUTION_TOLERANCE
                for entry in resolution_audits[1:]
            ),
            "entries": resolution_audits,
        },
        "figure_png": str(output_prefix.with_suffix(".png")),
        "figure_pdf": str(output_prefix.with_suffix(".pdf")),
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plt.close(fig)

    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
