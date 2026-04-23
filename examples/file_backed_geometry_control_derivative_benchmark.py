#!/usr/bin/env python3
"""File-backed geometry-control derivative benchmark on repository sample surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from ntx import GridSpec, load_boozmn_surface, solve_monoenergetic_scan  # noqa: E402
from ntx._checkout_paths import fixture_path  # noqa: E402
from ntx.config import enable_x64  # noqa: E402
from ntx.geometry import BoozerSurface  # noqa: E402
from ntx.vmec_jax_vmec import surface_from_vmec_jax_vmec_wout_file  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "file_backed_geometry_control_derivative_benchmark"
DEFAULT_GRID = GridSpec(7, 9, 6)
DEFAULT_NU_HAT = jnp.asarray([3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3])
DEFAULT_ER_HAT = 1.0e-3
DEFAULT_FD_STEP = 1.0e-4
DEFAULT_MAX_CONTROLS = 3
COEFFICIENTS = ("D11", "D31", "D33")


@dataclass(frozen=True)
class FileBackedCase:
    id: str
    label: str
    source_kind: str
    path: Path
    rho: float | None = None
    s: float | None = None


CASE_SPECS = (
    FileBackedCase(
        id="boozmn_sample",
        label="Sample Boozer file",
        source_kind="boozmn",
        path=fixture_path("sample_boozmn.nc"),
        rho=0.5,
    ),
    FileBackedCase(
        id="vmec_sample",
        label="Sample VMEC-backed surface",
        source_kind="vmec_jax",
        path=fixture_path("sample_wout.nc"),
        s=0.25,
    ),
)


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.2, 8.2),
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


def _load_surface(case: FileBackedCase) -> tuple[BoozerSurface, dict[str, float | str]]:
    if case.source_kind == "boozmn":
        payload = load_boozmn_surface(case.path, rho=float(case.rho))
        return payload.surface, {
            "source_kind": case.source_kind,
            "source_path": str(case.path),
            "rho": float(payload.rho),
            "s": float(payload.s),
        }

    surface = surface_from_vmec_jax_vmec_wout_file(case.path, s=float(case.s))
    s_value = float(case.s if case.s is not None else 0.0)
    return surface, {
        "source_kind": case.source_kind,
        "source_path": str(case.path),
        "rho": float(np.sqrt(max(s_value, 0.0))),
        "s": s_value,
    }


def _select_control_indices(surface: BoozerSurface, *, max_controls: int) -> tuple[int, ...]:
    mask = jnp.logical_not(jnp.logical_and(surface.m == 0, surface.n == 0))
    ranked = jnp.argsort(jnp.where(mask, jnp.abs(surface.b_cos), -1.0))[::-1]
    selected = tuple(int(index) for index in ranked.tolist() if bool(mask[int(index)]))
    if not selected:
        raise ValueError("surface does not contain a non-axisymmetric control mode")
    return selected[:max_controls]


def _scaled_surface(
    surface: BoozerSurface,
    control_indices: tuple[int, ...],
    scales,
) -> BoozerSurface:
    b_cos = surface.b_cos
    for position, mode_index in enumerate(control_indices):
        b_cos = b_cos.at[mode_index].set(surface.b_cos[mode_index] * scales[position])
    return replace(surface, b_cos=b_cos)


def _response(
    surface: BoozerSurface,
    control_indices: tuple[int, ...],
    scales,
    *,
    grid: GridSpec,
    nu_hat,
    er_hat: float,
):
    trial_surface = _scaled_surface(surface, control_indices, scales)
    scan = solve_monoenergetic_scan(
        trial_surface,
        grid,
        nu_hat,
        er_hat=jnp.full_like(nu_hat, er_hat),
    )
    return jnp.stack([scan[name].reshape(-1) for name in COEFFICIENTS])


def _finite_difference_jacobian(
    surface: BoozerSurface,
    control_indices: tuple[int, ...],
    scales,
    *,
    grid: GridSpec,
    nu_hat,
    er_hat: float,
    fd_step: float,
):
    columns = []
    for index in range(len(control_indices)):
        step = jnp.zeros_like(scales).at[index].set(fd_step)
        columns.append(
            (
                _response(
                    surface,
                    control_indices,
                    scales + step,
                    grid=grid,
                    nu_hat=nu_hat,
                    er_hat=er_hat,
                )
                - _response(
                    surface,
                    control_indices,
                    scales - step,
                    grid=grid,
                    nu_hat=nu_hat,
                    er_hat=er_hat,
                )
            )
            / (2.0 * fd_step)
        )
    return jnp.stack(columns, axis=-1)


def _case_payload(
    case: FileBackedCase,
    *,
    grid: GridSpec,
    nu_hat,
    er_hat: float,
    fd_step: float,
    max_controls: int,
) -> dict[str, object]:
    surface, metadata = _load_surface(case)
    control_indices = _select_control_indices(surface, max_controls=max_controls)
    scales = jnp.ones(len(control_indices), dtype=grid.jax_dtype)

    def response_flat(values):
        return _response(
            surface,
            control_indices,
            values,
            grid=grid,
            nu_hat=nu_hat,
            er_hat=er_hat,
        ).reshape(-1)

    direct_jacobian = jax.jacrev(response_flat)(scales).reshape(
        len(COEFFICIENTS),
        nu_hat.size,
        len(control_indices),
    )
    finite_difference_jacobian = _finite_difference_jacobian(
        surface,
        control_indices,
        scales,
        grid=grid,
        nu_hat=nu_hat,
        er_hat=er_hat,
        fd_step=fd_step,
    )
    baseline = _response(
        surface,
        control_indices,
        scales,
        grid=grid,
        nu_hat=nu_hat,
        er_hat=er_hat,
    )

    direct = np.asarray(direct_jacobian)
    finite_difference = np.asarray(finite_difference_jacobian)
    mismatch = _relative_error(finite_difference, direct)
    return {
        "id": case.id,
        "label": case.label,
        **metadata,
        "grid": {
            "n_theta": grid.n_theta,
            "n_zeta": grid.n_zeta,
            "n_xi": grid.n_xi,
        },
        "control_modes": [
            {
                "index": mode_index,
                "m": int(surface.m[mode_index]),
                "n": int(surface.n[mode_index]),
                "baseline_b_cos": float(surface.b_cos[mode_index]),
                "baseline_relative_to_b0": float(abs(surface.b_cos[mode_index]) / abs(surface.b0)),
            }
            for mode_index in control_indices
        ],
        "coefficients": list(COEFFICIENTS),
        "nu_hat": np.asarray(nu_hat).tolist(),
        "er_hat": float(er_hat),
        "fd_step": float(fd_step),
        "baseline_response": np.asarray(baseline).tolist(),
        "direct_jacobian": direct.tolist(),
        "finite_difference_jacobian": finite_difference.tolist(),
        "relative_mismatch": mismatch.tolist(),
        "summary_metrics": {
            "control_count": len(control_indices),
            "max_relative_mismatch": float(np.max(mismatch)),
            "median_relative_mismatch": float(np.median(mismatch)),
            "max_abs_direct_jacobian": float(np.max(np.abs(direct))),
            "max_abs_finite_difference_jacobian": float(np.max(np.abs(finite_difference))),
        },
    }


def run_benchmark(
    *,
    grid: GridSpec = DEFAULT_GRID,
    nu_hat=None,
    er_hat: float = DEFAULT_ER_HAT,
    fd_step: float = DEFAULT_FD_STEP,
    max_controls: int = DEFAULT_MAX_CONTROLS,
) -> dict[str, object]:
    enable_x64(grid.x64)
    nu_hat = (
        jnp.asarray(DEFAULT_NU_HAT, dtype=grid.jax_dtype)
        if nu_hat is None
        else jnp.asarray(nu_hat, dtype=grid.jax_dtype)
    )
    cases = tuple(
        _case_payload(
            case,
            grid=grid,
            nu_hat=nu_hat,
            er_hat=er_hat,
            fd_step=fd_step,
            max_controls=max_controls,
        )
        for case in CASE_SPECS
    )
    max_mismatch = max(case["summary_metrics"]["max_relative_mismatch"] for case in cases)
    median_mismatch = float(
        np.median(
            [
                case["summary_metrics"]["median_relative_mismatch"]
                for case in cases
            ]
        )
    )
    return {
        "benchmark": "file_backed_geometry_control_derivative_benchmark",
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
            "differences on repository-owned file-backed Boozer and VMEC sample "
            "surfaces. This is stronger than the owned analytic audit, but it "
            "is still not a reusable geometry-family validation claim."
        ),
        "cases": list(cases),
        "summary_metrics": {
            "case_count": len(cases),
            "max_relative_mismatch": float(max_mismatch),
            "median_relative_mismatch": median_mismatch,
        },
        "open_work": [
            "transfer the same audit to broader reusable geometry families",
            "compare geometry-control pullbacks against a prepared implicit-adjoint path",
            "measure memory and factorization reuse on larger geometry-control scans",
        ],
    }


def write_outputs(payload: dict[str, object], output_prefix: Path) -> None:
    _configure_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cases = payload["cases"]
    coefficient_index = COEFFICIENTS.index("D33")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(len(cases), 2, constrained_layout=True)
    if len(cases) == 1:
        axes = np.asarray([axes])

    for row, case in enumerate(cases):
        nu_hat = np.asarray(case["nu_hat"], dtype=float)
        direct = np.asarray(case["direct_jacobian"], dtype=float)
        finite_difference = np.asarray(case["finite_difference_jacobian"], dtype=float)
        mismatch = np.asarray(case["relative_mismatch"], dtype=float)
        controls = case["control_modes"]

        for control_index, control in enumerate(controls):
            label = rf"$(m,n)=({control['m']},{control['n']})$"
            axes[row, 0].loglog(
                nu_hat,
                np.abs(direct[coefficient_index, :, control_index]),
                color=colors[control_index],
                lw=2.1,
                marker="o",
                ms=4.2,
                label=rf"AD {label}",
            )
            axes[row, 0].loglog(
                nu_hat,
                np.abs(finite_difference[coefficient_index, :, control_index]),
                color=colors[control_index],
                lw=1.5,
                ls="--",
                marker="s",
                ms=3.6,
                label=rf"FD {label}",
            )
        axes[row, 0].set_xlabel(r"$\hat{\nu}$")
        axes[row, 0].set_ylabel(r"$|\partial D_{33}/\partial \alpha|$")
        axes[row, 0].set_title(case["label"])
        axes[row, 0].legend(loc="best", fontsize=8.2, ncols=1)

        heatmap = np.maximum(np.max(mismatch, axis=1), 1.0e-16)
        vmin = float(np.min(heatmap))
        vmax = float(np.max(heatmap))
        if vmax <= vmin:
            vmax = vmin * 10.0
        im = axes[row, 1].imshow(
            heatmap,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        axes[row, 1].set_xticks(range(len(controls)))
        axes[row, 1].set_xticklabels(
            [f"({control['m']},{control['n']})" for control in controls],
            rotation=0,
        )
        axes[row, 1].set_yticks(range(len(COEFFICIENTS)))
        axes[row, 1].set_yticklabels(COEFFICIENTS)
        axes[row, 1].set_xlabel("Controlled harmonic")
        axes[row, 1].set_ylabel("Coefficient")
        axes[row, 1].set_title(
            "Max AD/FD mismatch over collisionality"
        )
        cbar = fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
        cbar.set_label("relative mismatch")

    axes[-1, 1].text(
        0.02,
        -0.26,
        (
            "Stress benchmark: repository-owned file-backed sample surfaces; "
            "promotion still requires broader reusable geometry families."
        ),
        transform=axes[-1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
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
    parser.add_argument("--max-controls", type=int, default=DEFAULT_MAX_CONTROLS)
    return parser.parse_args()


def main(output_prefix: Path | None = None) -> None:
    args = parse_args() if output_prefix is None else None
    if args is None:
        grid = DEFAULT_GRID
        target = output_prefix if output_prefix is not None else OUTPUT_PREFIX
        er_hat = DEFAULT_ER_HAT
        fd_step = DEFAULT_FD_STEP
        max_controls = DEFAULT_MAX_CONTROLS
    else:
        grid = GridSpec(args.n_theta, args.n_zeta, args.n_xi)
        target = args.output_prefix
        er_hat = args.er_hat
        fd_step = args.fd_step
        max_controls = args.max_controls
    payload = run_benchmark(
        grid=grid,
        er_hat=er_hat,
        fd_step=fd_step,
        max_controls=max_controls,
    )
    write_outputs(payload, target)


if __name__ == "__main__":
    main()
