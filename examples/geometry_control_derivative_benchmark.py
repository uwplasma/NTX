#!/usr/bin/env python3
"""Multi-parameter geometry-control derivative benchmark."""

from __future__ import annotations

import argparse
import json
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

from ntx import GridSpec, example_surface, solve_monoenergetic_scan  # noqa: E402
from ntx.config import enable_x64  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "geometry_control_derivative_benchmark"
DEFAULT_GRID = GridSpec(7, 9, 6)
DEFAULT_NU_HAT = jnp.asarray([3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3])
DEFAULT_ER_HAT = 1.0e-3
DEFAULT_FD_STEP = 1.0e-4
CONTROL_INDICES = (1, 2, 3)
COEFFICIENTS = ("D11", "D31", "D33")


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (11.8, 6.2),
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
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), 1.0e-30)


def _controlled_surface(params, *, grid: GridSpec):
    base = example_surface(dtype=grid.jax_dtype)
    b_cos = base.b_cos
    for position, index in enumerate(CONTROL_INDICES):
        b_cos = b_cos.at[index].set(params[position])
    return replace(base, b_cos=b_cos)


def _response(params, *, grid: GridSpec, nu_hat, er_hat: float):
    surface = _controlled_surface(params, grid=grid)
    scan = solve_monoenergetic_scan(
        surface,
        grid,
        nu_hat,
        er_hat=jnp.full_like(nu_hat, er_hat),
    )
    return jnp.stack([scan[name].reshape(-1) for name in COEFFICIENTS])


def _finite_difference_jacobian(params, *, grid: GridSpec, nu_hat, er_hat: float, fd_step: float):
    columns = []
    for index in range(params.size):
        step = jnp.zeros_like(params).at[index].set(fd_step)
        columns.append(
            (
                _response(params + step, grid=grid, nu_hat=nu_hat, er_hat=er_hat)
                - _response(params - step, grid=grid, nu_hat=nu_hat, er_hat=er_hat)
            )
            / (2.0 * fd_step)
        )
    return jnp.stack(columns, axis=-1)


def run_benchmark(
    *,
    grid: GridSpec = DEFAULT_GRID,
    nu_hat=None,
    er_hat: float = DEFAULT_ER_HAT,
    fd_step: float = DEFAULT_FD_STEP,
) -> dict[str, object]:
    enable_x64(grid.x64)
    nu_hat = (
        jnp.asarray(DEFAULT_NU_HAT, dtype=grid.jax_dtype)
        if nu_hat is None
        else jnp.asarray(nu_hat, dtype=grid.jax_dtype)
    )
    base_surface = example_surface(dtype=grid.jax_dtype)
    params = jnp.asarray(base_surface.b_cos[jnp.asarray(CONTROL_INDICES)], dtype=grid.jax_dtype)

    def response_flat(values):
        return _response(
            values,
            grid=grid,
            nu_hat=nu_hat,
            er_hat=er_hat,
        ).reshape(-1)
    direct_jacobian = jax.jacrev(response_flat)(params).reshape(
        len(COEFFICIENTS),
        nu_hat.size,
        params.size,
    )
    finite_difference_jacobian = _finite_difference_jacobian(
        params,
        grid=grid,
        nu_hat=nu_hat,
        er_hat=er_hat,
        fd_step=fd_step,
    )
    baseline = _response(params, grid=grid, nu_hat=nu_hat, er_hat=er_hat)

    direct = np.asarray(direct_jacobian)
    finite_difference = np.asarray(finite_difference_jacobian)
    mismatch = _relative_error(finite_difference, direct)
    return {
        "benchmark": "geometry_control_derivative_benchmark",
        "classification": "artifact-backed autodiff stress benchmark",
        "literature_anchors": [
            {
                "label": "adjoint neoclassical optimization",
                "url": "https://arxiv.org/abs/1904.06430",
            },
            {
                "label": "differentiable programming for plasma workflows",
                "url": "https://arxiv.org/abs/2410.11161",
            },
            {
                "label": "monoenergetic transport formulation",
                "url": "https://arxiv.org/abs/2510.27513",
            },
        ],
        "claim_scope": (
            "Direct geometry-control autodiff agrees with centered finite "
            "differences on a three-harmonic owned surface. This is not yet a "
            "large VMEC/Boozer geometry-control validation."
        ),
        "grid": {
            "n_theta": grid.n_theta,
            "n_zeta": grid.n_zeta,
            "n_xi": grid.n_xi,
        },
        "control_indices": list(CONTROL_INDICES),
        "control_modes": [
            {
                "m": int(base_surface.m[index]),
                "n": int(base_surface.n[index]),
                "baseline_b_cos": float(params[position]),
            }
            for position, index in enumerate(CONTROL_INDICES)
        ],
        "nu_hat": np.asarray(nu_hat).tolist(),
        "er_hat": float(er_hat),
        "fd_step": float(fd_step),
        "coefficients": list(COEFFICIENTS),
        "baseline_response": np.asarray(baseline).tolist(),
        "direct_jacobian": direct.tolist(),
        "finite_difference_jacobian": finite_difference.tolist(),
        "relative_mismatch": mismatch.tolist(),
        "summary_metrics": {
            "max_relative_mismatch": float(np.max(mismatch)),
            "median_relative_mismatch": float(np.median(mismatch)),
            "max_abs_direct_jacobian": float(np.max(np.abs(direct))),
            "max_abs_finite_difference_jacobian": float(np.max(np.abs(finite_difference))),
        },
        "open_work": [
            "extend to reusable VMEC/Boozer geometry-control families",
            (
                "compare against the prepared implicit-adjoint path once "
                "geometry pullbacks are implemented"
            ),
            "measure memory under larger scan/database workloads",
        ],
    }


def write_outputs(payload: dict[str, object], output_prefix: Path) -> None:
    _configure_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    nu_hat = np.asarray(payload["nu_hat"], dtype=float)
    direct = np.asarray(payload["direct_jacobian"], dtype=float)
    finite_difference = np.asarray(payload["finite_difference_jacobian"], dtype=float)
    mismatch = np.asarray(payload["relative_mismatch"], dtype=float)
    controls = payload["control_modes"]

    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]

    coefficient_index = COEFFICIENTS.index("D33")
    for control_index, control in enumerate(controls):
        label = rf"$(m,n)=({control['m']},{control['n']})$"
        axes[0].loglog(
            nu_hat,
            np.abs(direct[coefficient_index, :, control_index]),
            color=colors[control_index],
            lw=2.2,
            marker="o",
            ms=4.5,
            label=rf"AD {label}",
        )
        axes[0].loglog(
            nu_hat,
            np.abs(finite_difference[coefficient_index, :, control_index]),
            color=colors[control_index],
            lw=1.6,
            ls="--",
            marker="s",
            ms=3.8,
            label=rf"FD {label}",
        )
    axes[0].set_xlabel(r"$\hat{\nu}$")
    axes[0].set_ylabel(r"$|\partial D_{33}/\partial B_{mn}|$")
    axes[0].set_title("Three-control geometry derivative")
    axes[0].legend(loc="best", fontsize=8.5, ncols=1)

    heatmap = np.max(mismatch, axis=1)
    im = axes[1].imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm="log",
    )
    axes[1].set_xticks(range(len(controls)))
    axes[1].set_xticklabels(
        [f"({control['m']},{control['n']})" for control in controls],
        rotation=0,
    )
    axes[1].set_yticks(range(len(COEFFICIENTS)))
    axes[1].set_yticklabels(COEFFICIENTS)
    axes[1].set_xlabel("Controlled Boozer harmonic")
    axes[1].set_ylabel("Coefficient")
    axes[1].set_title("Max AD/FD mismatch over collisionality")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("relative mismatch")
    axes[1].text(
        0.02,
        -0.18,
        (
            "Stress benchmark: owned three-harmonic surface; "
            "large VMEC/Boozer controls remain a planned lane."
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )

    figure_png = output_prefix.with_suffix(".png")
    figure_pdf = output_prefix.with_suffix(".pdf")
    figure_json = output_prefix.with_suffix(".json")
    fig.savefig(figure_png)
    fig.savefig(figure_pdf)
    plt.close(fig)

    payload = dict(payload)
    payload["figure_png"] = str(figure_png)
    payload["figure_pdf"] = str(figure_pdf)
    figure_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {figure_png}")
    print(f"Wrote {figure_pdf}")
    print(f"Wrote {figure_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--n-theta", type=int, default=DEFAULT_GRID.n_theta)
    parser.add_argument("--n-zeta", type=int, default=DEFAULT_GRID.n_zeta)
    parser.add_argument("--n-xi", type=int, default=DEFAULT_GRID.n_xi)
    parser.add_argument("--er-hat", type=float, default=DEFAULT_ER_HAT)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    return parser.parse_args()


def main(output_prefix: Path | None = None) -> None:
    args = parse_args() if output_prefix is None else None
    if args is None:
        grid = DEFAULT_GRID
        target = output_prefix if output_prefix is not None else OUTPUT_PREFIX
        er_hat = DEFAULT_ER_HAT
        fd_step = DEFAULT_FD_STEP
    else:
        grid = GridSpec(args.n_theta, args.n_zeta, args.n_xi)
        target = args.output_prefix
        er_hat = args.er_hat
        fd_step = args.fd_step
    payload = run_benchmark(grid=grid, er_hat=er_hat, fd_step=fd_step)
    write_outputs(payload, target)


if __name__ == "__main__":
    main()
