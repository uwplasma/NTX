#!/usr/bin/env python3
"""Forward-mode boundary-to-output derivative benchmark on projected VMEC geometry."""

from __future__ import annotations

import argparse
import json
import sys
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
    build_differentiable_neopax_field_from_vmex_state,
    build_ntx_neopax_scan_from_vmex_state,
    build_vmex_boundary_context,
    get_differentiable_neopax_fluxes,
    initial_guess_vmex_boundary_state,
    solve_monoenergetic_scan,
    surface_from_vmex_state,
    to_neopax_monoenergetic,
)
from ntx._checkout_paths import (  # noqa: E402
    find_booz_xform_jax_root,
    find_neopax_root,
    find_vmex_example_input,
)
from ntx.config import enable_x64  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "boundary_forward_mode_current_derivative_benchmark"
DEFAULT_GRID = GridSpec(5, 5, 4)
DEFAULT_FD_STEP = 1.0e-4


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.0, 7.8),
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


def _has_boundary_stack() -> bool:
    return (
        find_vmex_example_input() is not None
        and find_booz_xform_jax_root() is not None
        and find_neopax_root() is not None
    )


def _import_neopax():
    root = find_neopax_root()
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import NEOPAX

    if not hasattr(NEOPAX, "Field"):
        from NEOPAX._field import Field

        NEOPAX.Field = Field
    if not hasattr(NEOPAX, "Grid"):
        from NEOPAX._grid import Grid

        NEOPAX.Grid = Grid
    if not hasattr(NEOPAX, "Species"):
        from NEOPAX._species import Species

        NEOPAX.Species = Species
    return NEOPAX


def _make_species(NEOPAX, field):
    rho = jnp.asarray(field.rho_grid)
    te = 1500.0 - 500.0 * rho**2
    ti = 1200.0 - 400.0 * rho**2
    ne = 2.0e19 - 0.5e19 * rho**2
    ni = ne
    return NEOPAX.Species(
        2,
        int(field.n_r),
        jnp.arange(2),
        jnp.asarray([1.0 / 1836.15267343, 1.0]),
        jnp.asarray([-1.0, 1.0]),
        jnp.stack([te, ti]),
        jnp.stack([ne, ni]),
        jnp.zeros_like(field.r_grid),
        field.r_grid,
        field.r_grid_half,
        field.dr,
        field.Vprime_half,
        field.overVprime,
        jnp.asarray([ne[-1], ni[-1]]),
        jnp.asarray([te[-1], ti[-1]]),
    )


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), 1.0e-30)


def _finite_difference_gradient(objective, params, *, fd_step: float) -> jnp.ndarray:
    columns = []
    for index in range(params.shape[0]):
        step = jnp.zeros_like(params).at[index].set(fd_step)
        columns.append((objective(params + step) - objective(params - step)) / (2.0 * fd_step))
    return jnp.asarray(columns)


def _build_payload(*, fd_step: float, grid: GridSpec) -> dict[str, object]:
    if not _has_boundary_stack():
        raise RuntimeError("requires local vmex, booz_xform_jax, and NEOPAX checkouts")

    enable_x64(True)
    NEOPAX = _import_neopax()
    context = build_vmex_boundary_context(
        find_vmex_example_input(),
        max_mode=1,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    if len(context.specs) == 0:
        raise RuntimeError("vmex example input did not expose boundary parameters")

    params0 = jnp.zeros((len(context.specs),), dtype=jnp.float64)
    rho = jnp.asarray([0.25, 0.45, 0.65, 0.85])
    nu_v = jnp.logspace(-4, -2, 4)
    er_row = jnp.asarray([0.0, 1.0e-4, 3.0e-4, 1.0e-3])
    er = jnp.tile(er_row[None, :], (rho.shape[0], 1))

    def _state_from_params(params):
        return initial_guess_vmex_boundary_state(context, params, vmec_project=False)

    def ntx_transport_response(params):
        state = _state_from_params(params)
        surface = surface_from_vmex_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            s=0.25,
            mboz=12,
            nboz=12,
            psi_p=1.0,
        )
        coeffs = solve_monoenergetic_scan(
            surface,
            grid,
            nu_v,
            er_hat=jnp.full_like(nu_v, 1.0e-3),
        )
        return jnp.sum(coeffs["D11"] + coeffs["D33"])

    def integrated_current(params):
        state = _state_from_params(params)
        field = build_differentiable_neopax_field_from_vmex_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            n_r=11,
            mboz=12,
            nboz=12,
        )
        drds = field.a_b * 0.5 / jnp.clip(rho, 0.05, None)
        scan = build_ntx_neopax_scan_from_vmex_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            rho=rho,
            nu_v=nu_v,
            Er=er,
            Es=er,
            drds=drds,
            grid=grid,
            psi_p=field.Psia_value,
            source_name="boundary_projected_forward_mode_benchmark",
        )
        database = to_neopax_monoenergetic(scan, a_b=field.a_b)
        species = _make_species(NEOPAX, field)
        neopax_grid = NEOPAX.Grid.create_standard(int(field.n_r), 12, 2)
        _, _, _, upar = get_differentiable_neopax_fluxes(species, neopax_grid, field, database)
        return jnp.sum(species.charge[:, None] * upar * field.Vprime[None, :] * field.dr)

    objectives = {
        "ntx_transport_response": ntx_transport_response,
        "ntx_neopax_integrated_current": integrated_current,
    }

    parameter_names = [spec.name for spec in context.specs]
    objective_payloads: list[dict[str, object]] = []
    summary_max = 0.0
    summary_median_samples: list[float] = []
    for objective_id, objective in objectives.items():
        value = float(objective(params0))
        direct = np.asarray(jax.jacfwd(objective)(params0), dtype=float)
        finite_difference = np.asarray(
            _finite_difference_gradient(objective, params0, fd_step=fd_step),
            dtype=float,
        )
        mismatch = _relative_error(finite_difference, direct)
        objective_payloads.append(
            {
                "id": objective_id,
                "value": value,
                "parameter_names": parameter_names,
                "direct_forward_mode": direct.tolist(),
                "finite_difference": finite_difference.tolist(),
                "relative_mismatch": mismatch.tolist(),
            }
        )
        summary_max = max(summary_max, float(np.max(mismatch)))
        summary_median_samples.extend(float(x) for x in mismatch.ravel())

    return {
        "benchmark": "boundary_forward_mode_current_derivative_benchmark",
        "classification": "artifact-backed boundary-to-output forward-mode stress benchmark",
        "case": {
            "input_path": str(find_vmex_example_input()),
            "geometry_path": "boundary_projected_initial_guess",
            "parameter_names": parameter_names,
        },
        "grid": {
            "n_theta": grid.n_theta,
            "n_zeta": grid.n_zeta,
            "n_xi": grid.n_xi,
        },
        "fd_step": float(fd_step),
        "objectives": objective_payloads,
        "claim_scope": (
            "Low-dimensional boundary controls propagate through boundary-projected "
            "vmex geometry, booz_xform_jax, NTX coefficients, and an NTX+NEOPAX "
            "integrated-current objective under forward-mode autodiff."
        ),
        "summary_metrics": {
            "max_relative_mismatch": summary_max,
            "median_relative_mismatch": float(np.median(np.asarray(summary_median_samples))),
        },
        "open_work": [
            (
                "transfer from boundary-projected geometry to a self-consistent "
                "equilibrium sensitivity workflow"
            ),
            (
                "validate broader non-axisymmetric benchmark families beyond "
                "the repository sample input"
            ),
            (
                "establish whether reverse-mode can be repaired or whether the "
                "boundary-control lane should stay forward-mode only"
            ),
        ],
    }


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    _configure_style()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(fd_step=DEFAULT_FD_STEP, grid=DEFAULT_GRID)

    objectives = payload["objectives"]
    parameter_names = payload["case"]["parameter_names"]
    colors = {
        "forward": "#0072B2",
        "fd": "#D55E00",
        "heatmap": "#56B4E9",
    }

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    x = np.arange(len(parameter_names))
    width = 0.34
    for axis, objective in zip(axes[0], objectives, strict=True):
        direct = np.asarray(objective["direct_forward_mode"], dtype=float)
        finite_difference = np.asarray(objective["finite_difference"], dtype=float)
        axis.bar(x - width / 2, direct, width=width, color=colors["forward"], label="Forward mode")
        axis.bar(
            x + width / 2,
            finite_difference,
            width=width,
            color=colors["fd"],
            label="Centered FD",
        )
        axis.set_xticks(x, parameter_names)
        axis.set_title(objective["id"].replace("_", " "))
        axis.set_ylabel("Derivative")
        axis.legend(loc="best")

    mismatch_matrix = np.asarray(
        [objective["relative_mismatch"] for objective in objectives],
        dtype=float,
    )
    image = axes[1, 0].imshow(
        mismatch_matrix,
        aspect="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=max(float(np.max(mismatch_matrix)), 1.0e-12),
    )
    axes[1, 0].set_xticks(np.arange(len(parameter_names)), parameter_names)
    axes[1, 0].set_yticks(np.arange(len(objectives)), [obj["id"] for obj in objectives])
    axes[1, 0].set_title("Relative mismatch")
    for row in range(mismatch_matrix.shape[0]):
        for col in range(mismatch_matrix.shape[1]):
            axes[1, 0].text(
                col,
                row,
                f"{mismatch_matrix[row, col]:.1e}",
                ha="center",
                va="center",
                color="#111827",
                fontsize=9.0,
            )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.03,
        0.97,
        (
            "Boundary-projected geometry benchmark\n"
            f"Input: {Path(payload['case']['input_path']).name}\n"
            "Grid: "
            f"({payload['grid']['n_theta']}, {payload['grid']['n_zeta']}, "
            f"{payload['grid']['n_xi']})\n"
            "Max relative mismatch: "
            f"{payload['summary_metrics']['max_relative_mismatch']:.2e}\n"
            "Median relative mismatch: "
            f"{payload['summary_metrics']['median_relative_mismatch']:.2e}\n\n"
            "This validates the fast forward-mode boundary-control lane.\n"
            "It does not yet claim self-consistent equilibrium sensitivity."
        ),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#d1d5db", "alpha": 0.98},
    )

    fig.suptitle("Boundary-to-output forward-mode derivative benchmark", fontsize=13)

    output_png = output_prefix.with_suffix(".png")
    output_pdf = output_prefix.with_suffix(".pdf")
    output_json = output_prefix.with_suffix(".json")
    fig.savefig(output_png)
    fig.savefig(output_pdf)
    plt.close(fig)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUTPUT_PREFIX,
        help="Prefix for the PNG, PDF, and JSON outputs.",
    )
    args = parser.parse_args()
    main(args.output_prefix)
