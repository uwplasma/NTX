#!/usr/bin/env python3
"""Publication-style bootstrap-current optimization example for NTX."""

from __future__ import annotations

import argparse
import sys
import time
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
    example_bootstrap_current_optimization,
    solve_monoenergetic_multiprocess_scan,
    solve_monoenergetic_scan,
    surface_from_vmec_jax_vmec_wout_file,
)
from ntx._checkout_paths import find_neopax_root  # noqa: E402
from ntx.config import enable_x64  # noqa: E402


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.3, 8.2),
            "figure.dpi": 240,
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


def _resolve_wout(cli_path: Path | None) -> tuple[Path, str]:
    if cli_path is not None:
        return cli_path.expanduser().resolve(), "User-supplied VMEC equilibrium"
    neopax_root = find_neopax_root()
    if neopax_root is not None:
        candidate = neopax_root / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
        if candidate.exists():
            return candidate, "W7-X standard configuration"
    return ROOT / "tests" / "fixtures" / "sample_wout.nc", "Repository sample VMEC surface"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, default=None, help="Optional VMEC wout file.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "bootstrap_current_optimization",
        help="Prefix for PNG and PDF outputs.",
    )
    parser.add_argument("--steps", type=int, default=48, help="Gradient-ascent iterations.")
    args = parser.parse_args()

    enable_x64(True)
    _configure_style()
    wout_path, label = _resolve_wout(args.wout)
    output_png = args.output_prefix.with_suffix(".png")
    output_pdf = args.output_prefix.with_suffix(".pdf")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    rho = jnp.linspace(0.15, 0.75, 7)
    nu_v = jnp.array([1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2])
    er_scan = jnp.array([1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3])
    Er = jnp.tile(er_scan, (rho.size, 1))
    Es = jnp.zeros_like(Er)
    drds = jnp.ones_like(rho)
    grid = GridSpec(7, 9, 6)
    surfaces = tuple(
        surface_from_vmec_jax_vmec_wout_file(wout_path, s=float(rho_value**2))
        for rho_value in rho
    )

    representative_surface = surfaces[len(surfaces) // 2]
    nu_grid, epsi_grid = jnp.meshgrid(nu_v, er_scan * 0.0, indexing="ij")
    serial_start = time.perf_counter()
    _ = solve_monoenergetic_scan(representative_surface, grid, nu_grid, epsi_hat=epsi_grid)
    serial_seconds = time.perf_counter() - serial_start
    parallel_start = time.perf_counter()
    _ = solve_monoenergetic_multiprocess_scan(
        representative_surface,
        grid,
        nu_grid,
        epsi_hat=epsi_grid,
        backend="cpu",
        workers=2,
    )
    parallel_seconds = time.perf_counter() - parallel_start

    result = example_bootstrap_current_optimization(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=grid,
        a_b=1.0,
        nu_index=2,
        learning_rate=1.2,
        steps=args.steps,
        regularization=1.0,
        serial_seconds=serial_seconds,
        parallel_seconds=parallel_seconds,
    )

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    colors = {
        "baseline": "#111827",
        "optimized": "#0072B2",
        "objective": "#009E73",
        "landscape": "#56B4E9",
        "d13": "#D55E00",
        "d33": "#009E73",
        "history": "#CC79A7",
    }
    rho_np = np.asarray(result.rho)
    weight = np.exp(-0.5 * ((rho_np - 0.45) / 0.16) ** 2)
    baseline_weighted = np.trapezoid(np.asarray(result.baseline_current_profile) * weight, rho_np)
    optimized_weighted = np.trapezoid(np.asarray(result.optimized_current_profile) * weight, rho_np)
    weighted_gain = optimized_weighted / max(baseline_weighted, 1.0e-30)

    axes[0, 0].plot(
        rho_np,
        np.asarray(result.baseline_current_profile) / 1.0e18,
        lw=2.3,
        color=colors["baseline"],
        label="Baseline",
    )
    axes[0, 0].plot(
        rho_np,
        np.asarray(result.optimized_current_profile) / 1.0e18,
        lw=2.3,
        ls="--",
        color=colors["optimized"],
        label="Optimized",
    )
    axes[0, 0].fill_between(
        rho_np,
        0.0,
        weight * np.max(np.asarray(result.optimized_current_profile) / 1.0e18),
        color="#e5f2fb",
        alpha=0.35,
        label="Core weighting",
    )
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"Bootstrap-current proxy [$10^{18}$]")
    axes[0, 0].set_title("Bootstrap-current profile")
    axes[0, 0].legend(loc="upper left", ncol=1)
    axes[0, 0].text(
        0.97,
        0.08,
        (
            f"{label}\n"
            rf"weighted gain$={weighted_gain:.2f}\times$"
        ),
        transform=axes[0, 0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9.1,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    scale_grid = np.asarray(result.scale_grid)
    objective_landscape = np.asarray(result.objective_landscape) / 1.0e18
    scale_history = np.asarray(result.scale_history)
    objective_history = np.asarray(result.objective_history) / 1.0e18
    axes[0, 1].plot(
        scale_grid,
        objective_landscape,
        lw=2.2,
        color=colors["landscape"],
        label="Objective landscape",
    )
    axes[0, 1].plot(
        scale_history,
        objective_history,
        marker="o",
        ms=3.6,
        lw=1.4,
        color=colors["history"],
        alpha=0.9,
        label="Autodiff trajectory",
    )
    axes[0, 1].axvline(float(result.baseline_scale), color=colors["baseline"], lw=1.4, ls=":")
    axes[0, 1].axvline(float(result.optimized_scale), color=colors["optimized"], lw=1.4, ls="--")
    axes[0, 1].set_xlabel("Harmonic scale factor")
    axes[0, 1].set_ylabel(r"Objective [$10^{18}$]")
    axes[0, 1].set_title("Geometry-control landscape")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].text(
        0.03,
        0.96,
        (
            rf"mode $(m,n)=({int(result.harmonic_m)},{int(result.harmonic_n)})$" "\n"
            rf"baseline scale$={float(result.baseline_scale):.2f}$" "\n"
            rf"optimized scale$={float(result.optimized_scale):.2f}$"
        ),
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    axes[1, 0].plot(
        rho_np,
        np.asarray(result.baseline_d13_profile),
        lw=2.2,
        color=colors["baseline"],
        label=r"$D_{13}$ baseline",
    )
    axes[1, 0].plot(
        rho_np,
        np.asarray(result.optimized_d13_profile),
        lw=2.2,
        ls="--",
        color=colors["d13"],
        label=r"$D_{13}$ optimized",
    )
    axes[1, 0].plot(
        rho_np,
        np.asarray(result.baseline_d33_profile),
        lw=1.8,
        color="#94a3b8",
        label=r"$D_{33}$ baseline",
    )
    axes[1, 0].plot(
        rho_np,
        np.asarray(result.optimized_d33_profile),
        lw=1.8,
        ls="--",
        color=colors["d33"],
        label=r"$D_{33}$ optimized",
    )
    axes[1, 0].set_xlabel(r"$\rho$")
    axes[1, 0].set_ylabel("Current-response coefficient")
    axes[1, 0].set_title(r"Current-response change at $\hat{\nu}=10^{-3}$")
    axes[1, 0].legend(loc="best", ncol=2)

    gradient_history = np.asarray(result.gradient_history) / 1.0e18
    iterations = np.arange(1, gradient_history.size + 1)
    axes[1, 1].plot(
        iterations,
        objective_history,
        lw=2.2,
        color=colors["objective"],
        label="Objective",
    )
    twin = axes[1, 1].twinx()
    twin.plot(
        iterations,
        gradient_history,
        lw=1.8,
        ls="--",
        color=colors["history"],
        label="Gradient",
    )
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel(r"Objective [$10^{18}$]")
    twin.set_ylabel(r"Gradient [$10^{18}$]")
    axes[1, 1].set_title("Differentiable optimization history")
    lines = axes[1, 1].get_lines() + twin.get_lines()
    axes[1, 1].legend(lines, [line.get_label() for line in lines], loc="lower right")
    axes[1, 1].text(
        0.03,
        0.96,
        (
            rf"serial coefficient scan$={float(result.serial_seconds):.3f}$ s" "\n"
            rf"2-worker scan$={float(result.parallel_seconds):.3f}$ s" "\n"
            rf"reference amplitude$={float(result.harmonic_reference_value):.3f}$"
        ),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    for label_text, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
        ax.text(
            -0.14,
            1.02,
            f"({label_text})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    fig.savefig(output_png)
    fig.savefig(output_pdf)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
