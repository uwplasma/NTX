#!/usr/bin/env python3
"""Derivative audit for the differentiable NTX solve lane."""

from __future__ import annotations

import sys
from dataclasses import replace
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
    example_surface,
    prepare_monoenergetic_system,
    solve_prepared_coefficient_vector,
    solve_prepared_coefficient_vector_vjp,
)
from ntx.config import enable_x64  # noqa: E402

GRID = GridSpec(7, 9, 6)
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "derivative_audit"


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (11.8, 7.2),
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


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), 1e-30)


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    enable_x64(True)
    _configure_style()
    output_png = output_prefix.with_suffix(".png")
    output_pdf = output_prefix.with_suffix(".pdf")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    coefficient_index = 1
    amplitude_value = 0.085
    er_reference = 1.0e-3
    nu_reference = 3.0e-4
    fd_step_amplitude = 1.0e-4
    fd_step_er = 1.0e-5
    nu_hat = jnp.logspace(-4.5, -1.5, 9)
    er_hat_scan = jnp.logspace(-6, -2.5, 8)

    base_surface = example_surface(dtype=GRID.jax_dtype)
    trial_surface = replace(
        base_surface,
        b_cos=base_surface.b_cos.at[coefficient_index].set(amplitude_value)
    )
    prepared = prepare_monoenergetic_system(trial_surface, GRID)

    def coefficient_vector_from_amplitude(amplitude):
        surface = replace(
            base_surface,
            b_cos=base_surface.b_cos.at[coefficient_index].set(amplitude)
        )
        trial_prepared = prepare_monoenergetic_system(surface, GRID)
        return jnp.stack(
            [
                solve_prepared_coefficient_vector(
                    trial_prepared,
                    MonoenergeticCase(nu_hat=float(nu_value), er_hat=er_reference),
                )
                for nu_value in nu_hat
            ]
        )

    def coefficient_vector_from_er(er_value):
        case = MonoenergeticCase(nu_hat=nu_reference, er_hat=er_value)
        return solve_prepared_coefficient_vector_vjp(prepared, case)

    baseline_coefficients = jnp.stack(
        [
            solve_prepared_coefficient_vector(
                prepared,
                MonoenergeticCase(nu_hat=float(nu_value), er_hat=er_reference),
            )
            for nu_value in nu_hat
        ]
    )
    autodiff_d11_da = jax.jacrev(
        lambda amplitude: coefficient_vector_from_amplitude(amplitude)[:, 0]
    )(amplitude_value)
    finite_difference_d11_da = (
        coefficient_vector_from_amplitude(amplitude_value + fd_step_amplitude)[:, 0]
        - coefficient_vector_from_amplitude(amplitude_value - fd_step_amplitude)[:, 0]
    ) / (2.0 * fd_step_amplitude)
    autodiff_d33_da = jax.jacrev(
        lambda amplitude: coefficient_vector_from_amplitude(amplitude)[:, 3]
    )(amplitude_value)
    finite_difference_d33_da = (
        coefficient_vector_from_amplitude(amplitude_value + fd_step_amplitude)[:, 3]
        - coefficient_vector_from_amplitude(amplitude_value - fd_step_amplitude)[:, 3]
    ) / (2.0 * fd_step_amplitude)

    autodiff_d11_der = jax.vmap(
        jax.grad(lambda er_value: coefficient_vector_from_er(er_value)[0])
    )(er_hat_scan)
    finite_difference_d11_der = jax.vmap(
        lambda er_value: (
            coefficient_vector_from_er(er_value + fd_step_er)[0]
            - coefficient_vector_from_er(er_value - fd_step_er)[0]
        )
        / (2.0 * fd_step_er)
    )(er_hat_scan)
    autodiff_d33_der = jax.vmap(
        jax.grad(lambda er_value: coefficient_vector_from_er(er_value)[3])
    )(er_hat_scan)
    finite_difference_d33_der = jax.vmap(
        lambda er_value: (
            coefficient_vector_from_er(er_value + fd_step_er)[3]
            - coefficient_vector_from_er(er_value - fd_step_er)[3]
        )
        / (2.0 * fd_step_er)
    )(er_hat_scan)
    nu_hat = np.asarray(nu_hat)
    er_hat_scan = np.asarray(er_hat_scan)
    d11_da = np.asarray(autodiff_d11_da)
    d11_da_fd = np.asarray(finite_difference_d11_da)
    d33_da = np.asarray(autodiff_d33_da)
    d33_da_fd = np.asarray(finite_difference_d33_da)
    d11_der = np.asarray(autodiff_d11_der)
    d11_der_fd = np.asarray(finite_difference_d11_der)
    d33_der = np.asarray(autodiff_d33_der)
    d33_der_fd = np.asarray(finite_difference_d33_der)

    colors = {
        "d11": "#0072B2",
        "d33": "#D55E00",
        "fd": "#111827",
        "err": "#009E73",
    }
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    axes[0, 0].loglog(
        nu_hat,
        np.abs(d11_da),
        color=colors["d11"],
        lw=2.2,
        label=r"AD $\partial D_{11}/\partial a$",
    )
    axes[0, 0].loglog(
        nu_hat,
        np.abs(d11_da_fd),
        color=colors["d11"],
        lw=1.6,
        ls="--",
        label=r"FD $\partial D_{11}/\partial a$",
    )
    axes[0, 0].loglog(
        nu_hat,
        np.abs(d33_da),
        color=colors["d33"],
        lw=2.2,
        label=r"AD $\partial D_{33}/\partial a$",
    )
    axes[0, 0].loglog(
        nu_hat,
        np.abs(d33_da_fd),
        color=colors["d33"],
        lw=1.6,
        ls="--",
        label=r"FD $\partial D_{33}/\partial a$",
    )
    axes[0, 0].set_xlabel(r"$\hat{\nu}$")
    axes[0, 0].set_ylabel("Derivative magnitude")
    axes[0, 0].set_title(r"Harmonic-amplitude sensitivity at fixed $\hat E_r$")
    axes[0, 0].legend(loc="lower left", ncols=2, fontsize=9)
    axes[0, 0].text(
        0.03,
        0.96,
        (
            rf"$a={amplitude_value:.3f}$" "\n"
            rf"$\hat E_r={er_reference:.1e}$" "\n"
            rf"$D_{{11}}(\hat\nu_0)={float(baseline_coefficients[0, 0]):.3e}$"
        ),
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    axes[0, 1].loglog(
        nu_hat,
        _relative_error(d11_da_fd, d11_da),
        color=colors["d11"],
        lw=2.0,
        label=r"$D_{11}$",
    )
    axes[0, 1].loglog(
        nu_hat,
        _relative_error(d33_da_fd, d33_da),
        color=colors["d33"],
        lw=2.0,
        label=r"$D_{33}$",
    )
    axes[0, 1].axhline(1.0e-3, color=colors["fd"], lw=1.2, ls=":")
    axes[0, 1].set_xlabel(r"$\hat{\nu}$")
    axes[0, 1].set_ylabel("Relative AD/FD mismatch")
    axes[0, 1].set_title("Amplitude derivative audit")
    axes[0, 1].legend(loc="lower left")

    axes[1, 0].loglog(
        er_hat_scan,
        np.abs(d11_der),
        color=colors["d11"],
        lw=2.2,
        label=r"AD $\partial D_{11}/\partial \hat E_r$",
    )
    axes[1, 0].loglog(
        er_hat_scan,
        np.abs(d11_der_fd),
        color=colors["d11"],
        lw=1.6,
        ls="--",
        label=r"FD $\partial D_{11}/\partial \hat E_r$",
    )
    axes[1, 0].loglog(
        er_hat_scan,
        np.abs(d33_der),
        color=colors["d33"],
        lw=2.2,
        label=r"AD $\partial D_{33}/\partial \hat E_r$",
    )
    axes[1, 0].loglog(
        er_hat_scan,
        np.abs(d33_der_fd),
        color=colors["d33"],
        lw=1.6,
        ls="--",
        label=r"FD $\partial D_{33}/\partial \hat E_r$",
    )
    axes[1, 0].set_xlabel(r"$\hat E_r$")
    axes[1, 0].set_ylabel("Derivative magnitude")
    axes[1, 0].set_title(r"Electric-field sensitivity at fixed $\hat{\nu}$")
    axes[1, 0].legend(loc="lower left", ncols=2, fontsize=9)

    axes[1, 1].loglog(
        er_hat_scan,
        _relative_error(d11_der_fd, d11_der),
        color=colors["d11"],
        lw=2.0,
        label=r"$D_{11}$",
    )
    axes[1, 1].loglog(
        er_hat_scan,
        _relative_error(d33_der_fd, d33_der),
        color=colors["d33"],
        lw=2.0,
        label=r"$D_{33}$",
    )
    axes[1, 1].axhline(1.0e-3, color=colors["fd"], lw=1.2, ls=":")
    axes[1, 1].set_xlabel(r"$\hat E_r$")
    axes[1, 1].set_ylabel("Relative AD/FD mismatch")
    axes[1, 1].set_title("Electric-field derivative audit")
    axes[1, 1].legend(loc="lower left")

    for label, ax in zip(("a", "b", "c", "d"), axes.flat, strict=True):
        ax.text(
            -0.13,
            1.02,
            f"({label})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    fig.suptitle(
        "Direct JAX sensitivities agree with centered finite differences on the NTX dense solve",
        fontsize=12,
        y=1.02,
    )
    fig.savefig(output_png)
    fig.savefig(output_pdf)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
