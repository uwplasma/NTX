#!/usr/bin/env python3
"""Publication-style robust bootstrap-current optimization example for NTX."""

from __future__ import annotations

import argparse
import json
import sys
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
    example_bootstrap_current_robust_optimization,
    surface_from_vmex_vmec_wout_file,
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


def main(
    *,
    output_prefix: Path | None = None,
    steps: int = 40,
    uncertainty_sigma: float = 6.0e-2,
    risk_aversion: float = 0.35,
    wout: Path | None = None,
    radial_points: int = 7,
    grid: GridSpec | None = None,
    scale_grid_size: int = 29,
    quadrature_order: int = 5,
) -> None:
    enable_x64(True)
    _configure_style()
    wout_path, label = _resolve_wout(wout)
    output_prefix = (
        ROOT / "docs" / "_static" / "bootstrap_current_robust_optimization"
        if output_prefix is None
        else output_prefix
    )
    output_png = output_prefix.with_suffix(".png")
    output_pdf = output_prefix.with_suffix(".pdf")
    output_json = output_prefix.with_suffix(".json")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if radial_points < 3:
        raise ValueError("radial_points must be at least 3")
    grid = GridSpec(7, 9, 6) if grid is None else grid

    rho = jnp.linspace(0.15, 0.75, radial_points)
    nu_v = jnp.array([1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2])
    er_scan = jnp.array([1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3])
    Er = jnp.tile(er_scan, (rho.size, 1))
    Es = jnp.zeros_like(Er)
    drds = jnp.ones_like(rho)
    surfaces = tuple(
        surface_from_vmex_vmec_wout_file(wout_path, s=float(rho_value**2))
        for rho_value in rho
    )
    result = example_bootstrap_current_robust_optimization(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=grid,
        learning_rate=1.1,
        steps=steps,
        regularization=1.0,
        uncertainty_sigma=uncertainty_sigma,
        risk_aversion=risk_aversion,
        scale_grid_size=scale_grid_size,
        quadrature_order=quadrature_order,
    )

    rho_np = np.asarray(result.rho)
    scale_grid = np.asarray(result.scale_grid)
    det_land = np.asarray(result.deterministic_objective_landscape) / 1.0e18
    robust_land = np.asarray(result.robust_objective_landscape) / 1.0e18
    scale_history = np.asarray(result.scale_history)
    objective_history = np.asarray(result.objective_history) / 1.0e18
    baseline = np.asarray(result.baseline_current_profile) / 1.0e18
    optimized = np.asarray(result.optimized_current_profile) / 1.0e18
    mean_profile = np.asarray(result.optimized_current_mean) / 1.0e18
    std_profile = np.asarray(result.optimized_current_std) / 1.0e18
    q_low = np.asarray(result.optimized_current_quantile_low) / 1.0e18
    q_high = np.asarray(result.optimized_current_quantile_high) / 1.0e18

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    colors = {
        "baseline": "#111827",
        "optimized": "#0072B2",
        "robust": "#009E73",
        "deterministic": "#56B4E9",
        "history": "#CC79A7",
        "band": "#D55E00",
    }

    axes[0, 0].plot(rho_np, baseline, lw=2.2, color=colors["baseline"], label="Baseline")
    axes[0, 0].plot(rho_np, optimized, lw=2.2, ls="--", color=colors["optimized"], label="Nominal")
    axes[0, 0].plot(rho_np, mean_profile, lw=2.1, color=colors["robust"], label="Uncertain mean")
    axes[0, 0].fill_between(rho_np, q_low, q_high, color=colors["band"], alpha=0.14, label="16-84%")
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"Bootstrap-current response [$10^{18}$]")
    axes[0, 0].set_title("Robust-design current profile")
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].text(
        0.97,
        0.06,
        (
            f"{label}\n"
            rf"$\sigma_s={float(result.uncertainty_sigma):.2f}$, "
            rf"$\lambda={float(result.risk_aversion):.2f}$"
        ),
        transform=axes[0, 0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9.1,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    axes[0, 1].plot(
        scale_grid,
        det_land,
        lw=2.1,
        color=colors["deterministic"],
        label="Deterministic",
    )
    axes[0, 1].plot(scale_grid, robust_land, lw=2.2, color=colors["robust"], label="Robust")
    axes[0, 1].axvline(float(result.baseline_scale), color=colors["baseline"], lw=1.4, ls=":")
    axes[0, 1].axvline(float(result.optimized_scale), color=colors["optimized"], lw=1.4, ls="--")
    axes[0, 1].set_xlabel("Harmonic scale factor")
    axes[0, 1].set_ylabel(r"Objective [$10^{18}$]")
    axes[0, 1].set_title("Deterministic vs robust objective")
    axes[0, 1].legend(loc="lower right")

    axes[1, 0].plot(
        scale_history,
        objective_history,
        color=colors["history"],
        lw=1.8,
        marker="o",
        ms=3.2,
    )
    axes[1, 0].set_xlabel("Scale history")
    axes[1, 0].set_ylabel(r"Robust objective [$10^{18}$]")
    axes[1, 0].set_title("Robust optimization trajectory")

    axes[1, 1].plot(rho_np, std_profile, lw=2.2, color=colors["band"], label=r"$\sigma$")
    axes[1, 1].plot(
        rho_np,
        np.abs(mean_profile - optimized),
        lw=2.0,
        ls="--",
        color=colors["optimized"],
        label="Mean shift",
    )
    axes[1, 1].set_xlabel(r"$\rho$")
    axes[1, 1].set_ylabel(r"Current metric [$10^{18}$]")
    axes[1, 1].set_title("Uncertainty impact after optimization")
    axes[1, 1].legend(loc="upper right")

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

    baseline_weighted = float(np.trapezoid(baseline, rho_np))
    optimized_weighted = float(np.trapezoid(mean_profile, rho_np))
    objective_initial = float(objective_history[0])
    objective_final = float(objective_history[-1])
    current_norm = max(abs(baseline_weighted), 1.0e-30)
    objective_norm = max(abs(objective_initial), 1.0e-30)
    payload = {
        "baseline_scale": float(result.baseline_scale),
        "optimized_scale": float(result.optimized_scale),
        "uncertainty_sigma": float(result.uncertainty_sigma),
        "risk_aversion": float(result.risk_aversion),
        "radial_points": int(rho_np.size),
        "scale_grid_size": int(scale_grid.size),
        "quadrature_order": int(quadrature_order),
        "baseline_weighted_current_response": baseline_weighted,
        "optimized_weighted_current_response": optimized_weighted,
        "weighted_current_ratio": optimized_weighted / current_norm,
        "weighted_current_relative_change": (optimized_weighted - baseline_weighted)
        / current_norm,
        "robust_objective_initial": objective_initial,
        "robust_objective_final": objective_final,
        "robust_objective_relative_change": (objective_final - objective_initial)
        / objective_norm,
        "robust_gain": optimized_weighted / current_norm,
        "robust_gain_definition": (
            "optimized_weighted_current_response / "
            "max(abs(baseline_weighted_current_response), 1e-30)"
        ),
        "max_current_std": float(std_profile.max()),
        "rho": rho_np.tolist(),
        "optimized_current_mean": mean_profile.tolist(),
        "optimized_current_std": std_profile.tolist(),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, default=None, help="Optional VMEC wout file.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "bootstrap_current_robust_optimization",
        help="Prefix for PNG, PDF, and JSON outputs.",
    )
    parser.add_argument("--steps", type=int, default=40, help="Gradient-ascent iterations.")
    parser.add_argument(
        "--radial-points",
        type=int,
        default=7,
        help="Number of radial surfaces used in the example.",
    )
    parser.add_argument("--grid-nalpha", type=int, default=7, help="Alpha grid size.")
    parser.add_argument("--grid-ntheta", type=int, default=9, help="Theta grid size.")
    parser.add_argument("--grid-nzeta", type=int, default=6, help="Zeta grid size.")
    parser.add_argument(
        "--scale-grid-size",
        type=int,
        default=29,
        help="Number of points in the plotted robust-objective landscape.",
    )
    parser.add_argument(
        "--quadrature-order",
        type=int,
        choices=(3, 5),
        default=5,
        help="Gaussian quadrature order for the control-uncertainty moments.",
    )
    parser.add_argument(
        "--uncertainty-sigma",
        type=float,
        default=6.0e-2,
        help="Prescribed Gaussian scale uncertainty in raw-control units.",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=0.35,
        help="Weight on objective standard deviation in the robust objective.",
    )
    cli_args = parser.parse_args()
    main(
        output_prefix=cli_args.output_prefix,
        steps=cli_args.steps,
        uncertainty_sigma=cli_args.uncertainty_sigma,
        risk_aversion=cli_args.risk_aversion,
        wout=cli_args.wout,
        radial_points=cli_args.radial_points,
        grid=GridSpec(cli_args.grid_nalpha, cli_args.grid_ntheta, cli_args.grid_nzeta),
        scale_grid_size=cli_args.scale_grid_size,
        quadrature_order=cli_args.quadrature_order,
    )
