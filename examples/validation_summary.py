#!/usr/bin/env python3
"""Generate a publication-style validation summary figure for NTX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    MonoenergeticCase,
    enable_x64,
    load_dkes_surface,
    load_vmec_surface,
    onsager_error,
    solve_monoenergetic,
    solve_monoenergetic_scan,
)


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.2, 8.1),
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


def _convergence_metric(
    surface,
    n_theta: int,
    n_zeta: int,
    n_xi_values: list[int],
    case,
) -> np.ndarray:
    reference_grid = GridSpec(n_theta=n_theta, n_zeta=n_zeta, n_xi=n_xi_values[-1])
    reference = solve_monoenergetic(surface, reference_grid, case)
    reference_vector = np.asarray(
        [reference.D11, reference.D13, reference.D33],
        dtype=float,
    )
    metric = []
    for n_xi in n_xi_values:
        grid = GridSpec(n_theta=n_theta, n_zeta=n_zeta, n_xi=n_xi)
        result = solve_monoenergetic(surface, grid, case)
        vector = np.asarray([result.D11, result.D13, result.D33], dtype=float)
        rel = np.abs(vector - reference_vector) / np.maximum(np.abs(reference_vector), 1.0e-30)
        metric.append(float(np.max(rel)))
    return np.asarray(metric, dtype=float)


def _plot_transport_panel(
    ax,
    nu_hat: np.ndarray,
    coeffs: dict[str, np.ndarray],
    title: str,
) -> None:
    colors = {"D11": "#0072B2", "D13": "#D55E00", "D33": "#009E73"}
    labels = {"D11": r"$|D_{11}|$", "D13": r"$|D_{13}|$", "D33": r"$|D_{33}|$"}
    for key in ("D11", "D13", "D33"):
        ax.loglog(
            nu_hat,
            np.abs(np.asarray(coeffs[key])),
            lw=2.2,
            marker="o",
            ms=3.8,
            color=colors[key],
            label=labels[key],
        )
    ax.set_xlabel(r"$\hat{\nu}$")
    ax.set_ylabel("Transport coefficient magnitude")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "validation_summary",
        help="Prefix for PNG and PDF outputs.",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=11,
        help="Poloidal grid points for the validation scans.",
    )
    parser.add_argument(
        "--n-zeta",
        type=int,
        default=13,
        help="Toroidal grid points for the validation scans.",
    )
    parser.add_argument(
        "--n-xi",
        type=int,
        default=10,
        help="Legendre resolution for the transport curves.",
    )
    args = parser.parse_args()

    enable_x64(True)
    _configure_style()

    fixture_dir = ROOT / "tests" / "fixtures"
    dkes_surface = load_dkes_surface(fixture_dir / "sample_surface.ddkes2.data")
    vmec_surface = load_vmec_surface(fixture_dir / "sample_wout.nc", psi_n=0.25)
    grid = GridSpec(args.n_theta, args.n_zeta, args.n_xi)
    nu_hat = np.geomspace(1.0e-4, 1.0e-1, 10)
    er_hat = np.full_like(nu_hat, 1.0e-3)

    dkes_coeffs = solve_monoenergetic_scan(dkes_surface, grid, nu_hat, er_hat=er_hat)
    vmec_coeffs = solve_monoenergetic_scan(vmec_surface, grid, nu_hat, er_hat=er_hat)

    dkes_onsager = np.asarray(onsager_error(dkes_coeffs["D31"], dkes_coeffs["D13"]))
    vmec_onsager = np.asarray(onsager_error(vmec_coeffs["D31"], vmec_coeffs["D13"]))
    dkes_onsager_rel = dkes_onsager / np.maximum(
        np.maximum(np.abs(np.asarray(dkes_coeffs["D31"])), np.abs(np.asarray(dkes_coeffs["D13"]))),
        1.0e-30,
    )
    vmec_onsager_rel = vmec_onsager / np.maximum(
        np.maximum(np.abs(np.asarray(vmec_coeffs["D31"])), np.abs(np.asarray(vmec_coeffs["D13"]))),
        1.0e-30,
    )

    convergence_case = MonoenergeticCase(nu_hat=1.0e-3, er_hat=1.0e-3)
    n_xi_values = [4, 6, 8, 10, 12]
    dkes_convergence = _convergence_metric(
        dkes_surface,
        args.n_theta,
        args.n_zeta,
        n_xi_values,
        convergence_case,
    )
    vmec_convergence = _convergence_metric(
        vmec_surface,
        args.n_theta,
        args.n_zeta,
        n_xi_values,
        convergence_case,
    )
    dkes_convergence = np.maximum(dkes_convergence, 1.0e-15)
    vmec_convergence = np.maximum(vmec_convergence, 1.0e-15)

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    _plot_transport_panel(axes[0, 0], nu_hat, dkes_coeffs, "DKES-style validation scan")
    _plot_transport_panel(axes[0, 1], nu_hat, vmec_coeffs, "VMEC validation scan")

    axes[1, 0].loglog(
        nu_hat,
        dkes_onsager_rel,
        lw=2.2,
        marker="o",
        ms=3.8,
        color="#0072B2",
        label="DKES-style surface",
    )
    axes[1, 0].loglog(
        nu_hat,
        vmec_onsager_rel,
        lw=2.2,
        marker="s",
        ms=3.6,
        color="#CC79A7",
        label="VMEC surface",
    )
    axes[1, 0].set_xlabel(r"$\hat{\nu}$")
    axes[1, 0].set_ylabel(r"$|D_{13} + D_{31}| / \max(|D_{13}|, |D_{31}|)$")
    axes[1, 0].set_title("Onsager residual across the scan")
    axes[1, 0].legend(loc="best")

    axes[1, 1].semilogy(
        n_xi_values,
        dkes_convergence,
        lw=2.2,
        marker="o",
        ms=4.0,
        color="#0072B2",
        label="DKES-style surface",
    )
    axes[1, 1].semilogy(
        n_xi_values,
        vmec_convergence,
        lw=2.2,
        marker="s",
        ms=3.8,
        color="#CC79A7",
        label="VMEC surface",
    )
    axes[1, 1].set_xlabel(r"$N_\xi$")
    axes[1, 1].set_ylabel("Max relative coefficient error")
    axes[1, 1].set_title(r"Low-order coefficient convergence at $\hat{\nu}=10^{-3}$")
    axes[1, 1].legend(loc="best")
    axes[1, 1].text(
        0.03,
        0.96,
        "Reference: finest plotted $N_\\xi$",
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

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_prefix.with_suffix(".png")
    output_pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(output_png)
    fig.savefig(output_pdf)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
