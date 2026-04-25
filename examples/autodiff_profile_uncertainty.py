#!/usr/bin/env python3
"""Publication-style uncertainty propagation audit for autodiff profile inversion."""

from __future__ import annotations

import argparse
import json
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
    example_neopax_profile_uncertainty,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
)
from ntx.config import enable_x64  # noqa: E402


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.0, 8.2),
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


def main(
    *,
    output_prefix: Path | None = None,
    monte_carlo_samples: int = 96,
    steps: int = 48,
    basis_size: int = 3,
) -> None:
    enable_x64(True)
    if basis_size < 2:
        raise ValueError("basis_size must be at least 2")
    _configure_style()
    output_prefix = (
        ROOT / "docs" / "_static" / "autodiff_profile_uncertainty"
        if output_prefix is None
        else output_prefix
    )
    scan = load_neopax_reference_scan(ROOT / "tests" / "fixtures" / "sample_neopax_scan.h5")
    wout = ROOT / "tests" / "fixtures" / "sample_wout.nc"
    surfaces = tuple(
        surface_from_vmec_jax_vmec_wout_file(wout, s=float(rho_value**2)) for rho_value in scan.rho
    )
    target_params = np.zeros((basis_size,), dtype=float)
    initial_params = np.zeros((basis_size,), dtype=float)
    parameter_std = np.zeros((basis_size,), dtype=float)
    target_params[:3] = np.asarray([1.4e-3, -6.0e-4, 1.8e-4])[:basis_size]
    initial_params[:3] = np.asarray([5.0e-4, 2.0e-4, -1.0e-4])[:basis_size]
    parameter_std[:3] = np.asarray([5.0e-5, 2.0e-5, 1.0e-5])[:basis_size]
    if basis_size > 3:
        decay = np.arange(3, basis_size, dtype=float) - 2.0
        parameter_std[3:] = 5.0e-6 / decay
    result = example_neopax_profile_uncertainty(
        surfaces,
        rho=scan.rho,
        nu_v=scan.nu_v,
        Es=scan.Es,
        Er=scan.Er,
        drds=scan.drds,
        grid=GridSpec(7, 9, 6),
        a_b=1.0,
        learning_rate=0.35,
        steps=steps,
        target_params=target_params,
        initial_params=initial_params,
        parameter_std=parameter_std,
        monte_carlo_samples=monte_carlo_samples,
    )

    output_png = output_prefix.with_suffix(".png")
    output_pdf = output_prefix.with_suffix(".pdf")
    output_json = output_prefix.with_suffix(".json")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    rho = np.asarray(result.rho)
    fitted = np.asarray(result.fitted_d33_profile)
    target = np.asarray(result.target_d33_profile)
    linear_std = np.asarray(result.linearized_d33_std)
    mc_mean = np.asarray(result.monte_carlo_d33_mean)
    mc_std = np.asarray(result.monte_carlo_d33_std)
    q_low = np.asarray(result.monte_carlo_d33_quantile_low)
    q_high = np.asarray(result.monte_carlo_d33_quantile_high)
    covariance = np.asarray(result.parameter_covariance)
    correlation = np.asarray(result.parameter_correlation)
    fisher = np.asarray(result.fisher_matrix)
    fisher_eigenvalues = np.asarray(result.fisher_eigenvalues)
    hessian_probe = np.asarray(result.hessian_vector_probe)
    gauss_newton_probe = np.asarray(result.gauss_newton_vector_probe)
    hessian_probe_relative_error = float(result.hessian_probe_relative_error)
    std_rel_mismatch = np.abs(linear_std - mc_std) / np.maximum(mc_std, 1.0e-30)
    mean_shift_rel = np.abs(mc_mean - fitted) / np.maximum(np.abs(fitted), 1.0e-30)

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    colors = {
        "target": "#111827",
        "fit": "#0072B2",
        "linear": "#009E73",
        "mc": "#D55E00",
        "error": "#CC79A7",
    }

    axes[0, 0].plot(rho, target, lw=2.2, color=colors["target"], label="Target")
    axes[0, 0].plot(rho, fitted, lw=2.1, color=colors["fit"], label="Fitted")
    axes[0, 0].fill_between(
        rho,
        fitted - linear_std,
        fitted + linear_std,
        color=colors["linear"],
        alpha=0.18,
        label="Linearized $1\\sigma$",
    )
    axes[0, 0].fill_between(
        rho,
        q_low,
        q_high,
        color=colors["mc"],
        alpha=0.12,
        label="Monte Carlo 16-84%",
    )
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"$D_{33}$")
    axes[0, 0].set_title("Profile uncertainty band")
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(rho, linear_std, lw=2.2, color=colors["linear"], label="Linearized")
    axes[0, 1].plot(rho, mc_std, lw=2.2, ls="--", color=colors["mc"], label="Monte Carlo")
    axes[0, 1].set_xlabel(r"$\rho$")
    axes[0, 1].set_ylabel(r"$\sigma(D_{33})$")
    axes[0, 1].set_title("Linearized vs Monte Carlo standard deviation")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].text(
        0.03,
        0.96,
        (
            rf"max std mismatch$={std_rel_mismatch.max():.2e}$" "\n"
            rf"sample count$={int(result.sample_count)}$"
        ),
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    im = axes[1, 0].imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    parameter_labels = [f"$p_{index}$" for index in range(correlation.shape[0])]
    axes[1, 0].set_xticks(range(correlation.shape[0]), labels=parameter_labels)
    axes[1, 0].set_yticks(range(correlation.shape[0]), labels=parameter_labels)
    axes[1, 0].set_title("Gauss-Newton parameter correlation")
    for row in range(correlation.shape[0]):
        for col in range(correlation.shape[1]):
            axes[1, 0].text(
                col,
                row,
                f"{correlation[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=9.0,
                color="white" if abs(correlation[row, col]) > 0.45 else "black",
            )
    fig.colorbar(im, ax=axes[1, 0], shrink=0.86, label="Correlation")

    axes[1, 1].plot(rho, std_rel_mismatch, lw=2.2, color=colors["error"], label="Std mismatch")
    axes[1, 1].plot(
        rho,
        mean_shift_rel,
        lw=2.0,
        ls="--",
        color=colors["fit"],
        label="Mean shift",
    )
    axes[1, 1].set_xlabel(r"$\rho$")
    axes[1, 1].set_ylabel("Relative metric")
    axes[1, 1].set_title("Uncertainty-closure quality checks")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].text(
        0.03,
        0.96,
        (
            rf"basis size$={basis_size}$" "\n"
            rf"$\lambda_{{\max}}(F)={fisher_eigenvalues.max():.2e}$" "\n"
            rf"HVP rel. error$={hessian_probe_relative_error:.2e}$"
        ),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    for label, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
        ax.text(
            -0.14,
            1.02,
            f"({label})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    fig.savefig(output_png)
    fig.savefig(output_pdf)

    payload = {
        "rho": rho.tolist(),
        "fitted_d33_profile": fitted.tolist(),
        "target_d33_profile": target.tolist(),
        "linearized_d33_std": linear_std.tolist(),
        "monte_carlo_d33_mean": mc_mean.tolist(),
        "monte_carlo_d33_std": mc_std.tolist(),
        "parameter_covariance": covariance.tolist(),
        "parameter_std": np.asarray(result.parameter_std).tolist(),
        "parameter_correlation": correlation.tolist(),
        "fisher_matrix": fisher.tolist(),
        "fisher_eigenvalues": fisher_eigenvalues.tolist(),
        "hessian_vector_probe": hessian_probe.tolist(),
        "gauss_newton_vector_probe": gauss_newton_probe.tolist(),
        "hessian_probe_relative_error": hessian_probe_relative_error,
        "basis_size": basis_size,
        "sample_count": int(result.sample_count),
        "max_std_relative_mismatch": float(std_rel_mismatch.max()),
        "max_mean_relative_shift": float(mean_shift_rel.max()),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "autodiff_profile_uncertainty",
        help="Prefix for PNG, PDF, and JSON outputs.",
    )
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=96,
        help="Number of Monte Carlo samples used in the comparison.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=48,
        help="Autodiff profile-fit iterations before uncertainty propagation.",
    )
    parser.add_argument(
        "--basis-size",
        type=int,
        default=3,
        help="Number of odd-power radial electric-field basis functions.",
    )
    cli_args = parser.parse_args()
    main(
        output_prefix=cli_args.output_prefix,
        monte_carlo_samples=cli_args.monte_carlo_samples,
        steps=cli_args.steps,
        basis_size=cli_args.basis_size,
    )
