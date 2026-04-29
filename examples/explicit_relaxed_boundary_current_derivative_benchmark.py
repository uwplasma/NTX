#!/usr/bin/env python3
"""Forward-mode derivative benchmark on explicitly relaxed VMEC equilibria."""

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
    build_differentiable_neopax_field_from_vmec_jax_state,
    build_ntx_neopax_scan_from_vmec_jax_state,
    build_vmec_jax_boundary_context,
    get_differentiable_neopax_fluxes,
    initial_guess_vmec_jax_boundary_state,
    relax_vmec_jax_boundary_state_explicit,
    solve_monoenergetic_scan,
    surface_from_vmec_jax_state,
    to_neopax_monoenergetic,
)
from ntx._checkout_paths import (  # noqa: E402
    find_booz_xform_jax_root,
    find_neopax_root,
    find_vmec_jax_root,
)
from ntx.config import enable_x64  # noqa: E402

OUTPUT_PREFIX = (
    ROOT / "docs" / "_static" / "explicit_relaxed_boundary_current_derivative_benchmark"
)
DEFAULT_GRID = GridSpec(5, 5, 4)
DEFAULT_FD_STEP = 1.0e-5
DEFAULT_MAX_ITER = 10
DEFAULT_STEP_SIZE = 1.0e-8
DEFAULT_MBOZ = 10
DEFAULT_NBOZ = 10
DEFAULT_CASE_SPECS = (
    {
        "id": "qa_lowres",
        "label": "QA low-res",
        "input_name": "input.LandremanPaul2021_QA_lowres",
    },
    {
        "id": "qh_warm_start",
        "label": "QH warm-start",
        "input_name": "input.nfp4_QH_warm_start",
    },
)


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (18.0, 9.5),
            "figure.dpi": 220,
            "font.size": 10.0,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _input_path(name: str) -> Path:
    root = find_vmec_jax_root()
    if root is None:
        raise RuntimeError("requires local vmec_jax checkout")
    path = root / "examples" / "data" / name
    if not path.exists():
        raise RuntimeError(f"missing vmec_jax example input: {path}")
    return path


def _has_boundary_stack() -> bool:
    try:
        inputs_exist = all(_input_path(case["input_name"]).exists() for case in DEFAULT_CASE_SPECS)
    except RuntimeError:
        inputs_exist = False
    return (
        find_vmec_jax_root() is not None
        and find_booz_xform_jax_root() is not None
        and find_neopax_root() is not None
        and inputs_exist
    )


def _import_vmec_jax():
    root = find_vmec_jax_root()
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import vmec_jax

    return vmec_jax


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


def _volume_from_state(vmec_jax, context, state) -> float:
    geom = vmec_jax.eval_geom(state, context.static)
    _dvds, volume = vmec_jax.volume_from_sqrtg(
        geom.sqrtg,
        context.static.s,
        context.static.grid.theta,
        context.static.grid.zeta,
        nfp=int(context.static.cfg.nfp),
    )
    return float(volume[-1])


def _build_case_payload(
    case_spec: dict[str, str],
    *,
    fd_step: float,
    grid: GridSpec,
    vmec_jax,
    NEOPAX,
) -> dict[str, object]:
    input_path = _input_path(case_spec["input_name"])
    context = build_vmec_jax_boundary_context(
        input_path,
        max_mode=1,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    if len(context.specs) == 0:
        raise RuntimeError(f"{case_spec['input_name']} did not expose any boundary parameters")

    params0 = jnp.zeros((len(context.specs),), dtype=jnp.float64)
    parameter_names = [spec.name for spec in context.specs]
    rho = jnp.asarray([0.25, 0.45, 0.65, 0.85])
    nu_v = jnp.logspace(-4, -2, 4)
    er_row = jnp.asarray([0.0, 1.0e-4, 3.0e-4, 1.0e-3])
    er = jnp.tile(er_row[None, :], (rho.shape[0], 1))

    def _state_from_params(params, *, differentiable: bool):
        return relax_vmec_jax_boundary_state_explicit(
            context,
            params,
            vmec_project=False,
            max_iter=DEFAULT_MAX_ITER,
            step_size=DEFAULT_STEP_SIZE,
            differentiable=differentiable,
            stop_grad_in_update=False,
            verbose=False,
        )

    initial_state = initial_guess_vmec_jax_boundary_state(context, params0, vmec_project=False)
    ordinary_state = _state_from_params(params0, differentiable=False)
    explicit_relaxed_state = _state_from_params(params0, differentiable=True)
    initial_volume = _volume_from_state(vmec_jax, context, initial_state)
    ordinary_volume = _volume_from_state(vmec_jax, context, ordinary_state)
    explicit_relaxed_volume = _volume_from_state(vmec_jax, context, explicit_relaxed_state)
    ordinary_explicit_relative_difference = abs(ordinary_volume - explicit_relaxed_volume) / max(
        abs(ordinary_volume),
        1.0e-30,
    )

    def booz_xform_scalar(params):
        state = _state_from_params(params, differentiable=True)
        surface = surface_from_vmec_jax_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            s=0.5,
            mboz=DEFAULT_MBOZ,
            nboz=DEFAULT_NBOZ,
            psi_p=1.0,
        )
        return jnp.sum(surface.b_cos[:4]) + surface.iota

    def ntx_transport_response(params):
        state = _state_from_params(params, differentiable=True)
        surface = surface_from_vmec_jax_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            s=0.5,
            mboz=DEFAULT_MBOZ,
            nboz=DEFAULT_NBOZ,
            psi_p=1.0,
        )
        coeffs = solve_monoenergetic_scan(
            surface,
            grid,
            jnp.logspace(-4, -2, 3),
            er_hat=jnp.full((3,), 1.0e-3),
        )
        return jnp.sum(coeffs["D11"] + coeffs["D33"])

    def integrated_current(params):
        state = _state_from_params(params, differentiable=True)
        field = build_differentiable_neopax_field_from_vmec_jax_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            n_r=9,
            mboz=DEFAULT_MBOZ,
            nboz=DEFAULT_NBOZ,
        )
        drds = field.a_b * 0.5 / jnp.clip(rho, 0.05, None)
        scan = build_ntx_neopax_scan_from_vmec_jax_state(
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
            source_name=f"explicit_relaxed_boundary_benchmark_{case_spec['id']}",
        )
        database = to_neopax_monoenergetic(scan, a_b=field.a_b)
        species = _make_species(NEOPAX, field)
        neopax_grid = NEOPAX.Grid.create_standard(int(field.n_r), 12, 2)
        _, _, _, upar = get_differentiable_neopax_fluxes(species, neopax_grid, field, database)
        return jnp.sum(species.charge[:, None] * upar * field.Vprime[None, :] * field.dr)

    objectives = {
        "booz_xform_scalar": booz_xform_scalar,
        "ntx_transport_response": ntx_transport_response,
        "ntx_neopax_integrated_current": integrated_current,
    }
    objective_payloads: list[dict[str, object]] = []
    objective_summaries: list[dict[str, float | str]] = []
    case_samples: list[float] = []
    case_max = 0.0
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
        case_max = max(case_max, float(np.max(mismatch)))
        case_samples.extend(float(value) for value in mismatch.ravel())
        objective_summaries.append(
            {
                "id": objective_id,
                "max_relative_mismatch": float(np.max(mismatch)),
                "median_relative_mismatch": float(np.median(mismatch)),
            }
        )

    return {
        "id": case_spec["id"],
        "label": case_spec["label"],
        "input_path": str(input_path),
        "geometry_path": "explicit_relaxed_fixed_boundary",
        "parameter_names": parameter_names,
        "volume_metrics": {
            "initial_volume": initial_volume,
            "ordinary_volume": ordinary_volume,
            "explicit_relaxed_volume": explicit_relaxed_volume,
            "ordinary_explicit_relative_difference": ordinary_explicit_relative_difference,
        },
        "objectives": objective_payloads,
        "objective_summaries": objective_summaries,
        "summary_metrics": {
            "max_relative_mismatch": case_max,
            "median_relative_mismatch": float(np.median(np.asarray(case_samples))),
        },
    }


def _build_payload(*, fd_step: float, grid: GridSpec) -> dict[str, object]:
    if not _has_boundary_stack():
        raise RuntimeError("requires local vmec_jax, booz_xform_jax, and NEOPAX checkouts")

    enable_x64(True)
    vmec_jax = _import_vmec_jax()
    NEOPAX = _import_neopax()
    case_payloads = [
        _build_case_payload(case_spec, fd_step=fd_step, grid=grid, vmec_jax=vmec_jax, NEOPAX=NEOPAX)
        for case_spec in DEFAULT_CASE_SPECS
    ]
    summary_max = max(
        case["summary_metrics"]["max_relative_mismatch"] for case in case_payloads
    )
    summary_samples = [
        case["summary_metrics"]["median_relative_mismatch"] for case in case_payloads
    ]
    max_volume_difference = max(
        case["volume_metrics"]["ordinary_explicit_relative_difference"] for case in case_payloads
    )

    return {
        "benchmark": "explicit_relaxed_boundary_current_derivative_benchmark",
        "classification": (
            "artifact-backed explicit-relaxed equilibrium forward-mode "
            "family stress benchmark"
        ),
        "equilibrium_relaxation": {
            "max_iter": DEFAULT_MAX_ITER,
            "step_size": DEFAULT_STEP_SIZE,
            "vmec_project": False,
            "case_ids": [case["id"] for case in case_payloads],
        },
        "grid": {
            "n_theta": grid.n_theta,
            "n_zeta": grid.n_zeta,
            "n_xi": grid.n_xi,
        },
        "fd_step": float(fd_step),
        "objective_ids": [objective["id"] for objective in case_payloads[0]["objectives"]],
        "cases": case_payloads,
        "claim_scope": (
            "Low-dimensional boundary controls propagate through an explicitly "
            "relaxed fixed-boundary vmec_jax solve, booz_xform_jax, NTX "
            "coefficients, and an NTX+NEOPAX integrated-current objective "
            "under forward-mode autodiff on committed QA and QH family cases, "
            "while preserving the ordinary primal volume."
        ),
        "summary_metrics": {
            "max_relative_mismatch": float(summary_max),
            "median_relative_mismatch": float(np.median(np.asarray(summary_samples))),
            "max_ordinary_explicit_volume_relative_difference": float(max_volume_difference),
        },
        "open_work": [
            "widen from the committed QA and QH cases to additional geometry families",
            "recover the same sensitivities on the implicit-equilibrium path",
            "repair reverse mode on the equilibrium-relaxed boundary-control lane",
        ],
    }


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    _configure_style()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(fd_step=DEFAULT_FD_STEP, grid=DEFAULT_GRID)
    objective_ids = payload["objective_ids"]
    colors = {"forward": "#0072B2", "fd": "#D55E00"}

    fig, axes = plt.subplots(len(payload["cases"]), len(objective_ids) + 1, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, case in enumerate(payload["cases"]):
        parameter_names = case["parameter_names"]
        x = np.arange(len(parameter_names))
        width = 0.34
        for col, objective in enumerate(case["objectives"]):
            axis = axes[row, col]
            direct = np.asarray(objective["direct_forward_mode"], dtype=float)
            finite_difference = np.asarray(objective["finite_difference"], dtype=float)
            axis.bar(
                x - width / 2,
                direct,
                width=width,
                color=colors["forward"],
                label="Forward mode",
            )
            axis.bar(
                x + width / 2,
                finite_difference,
                width=width,
                color=colors["fd"],
                label="Centered FD",
            )
            axis.set_xticks(x, parameter_names, rotation=35, ha="right")
            axis.set_title(f"{case['label']}: {objective['id'].replace('_', ' ')}")
            axis.set_ylabel("Derivative")
            axis.legend(loc="best")

        summary_lines = [
            f"{summary['id']}: max={summary['max_relative_mismatch']:.2e}, "
            f"median={summary['median_relative_mismatch']:.2e}"
            for summary in case["objective_summaries"]
        ]
        volume_metrics = case["volume_metrics"]
        axis = axes[row, -1]
        axis.axis("off")
        axis.text(
            0.03,
            0.97,
            (
                f"{case['label']}\n"
                f"Input: {Path(case['input_path']).name}\n"
                f"Initial volume: {volume_metrics['initial_volume']:.6e}\n"
                f"Ordinary volume: {volume_metrics['ordinary_volume']:.6e}\n"
                f"Explicit volume: {volume_metrics['explicit_relaxed_volume']:.6e}\n"
                "Ordinary/explicit rel. diff: "
                f"{volume_metrics['ordinary_explicit_relative_difference']:.2e}\n"
                "Case max mismatch: "
                f"{case['summary_metrics']['max_relative_mismatch']:.2e}\n"
                "Case median mismatch: "
                f"{case['summary_metrics']['median_relative_mismatch']:.2e}\n\n"
                + "\n".join(summary_lines)
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#d1d5db", "alpha": 0.98},
        )

    fig.suptitle(
        "Explicit-relaxed boundary-to-current derivative benchmark family",
        fontsize=13,
    )

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
