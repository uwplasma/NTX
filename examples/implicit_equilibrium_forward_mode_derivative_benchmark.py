#!/usr/bin/env python3
"""Forward-mode diagnostic for the non-shipping implicit fixed-boundary lane."""

from __future__ import annotations

import argparse
import dataclasses
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
    build_vmec_jax_boundary_context,
    initial_guess_vmec_jax_boundary_state,
    solve_monoenergetic_scan,
    solve_vmec_jax_boundary_state,
    surface_from_vmec_jax_state,
)
from ntx._checkout_paths import find_booz_xform_jax_root, find_vmec_jax_root  # noqa: E402
from ntx.config import enable_x64  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "implicit_equilibrium_forward_mode_derivative_benchmark"
DEFAULT_GRID = GridSpec(5, 5, 4)
DEFAULT_FD_STEP = 1.0e-4
DEFAULT_MAX_ITER = 5
DEFAULT_STEP_SIZE = 1.0
DEFAULT_MBOZ = 10
DEFAULT_NBOZ = 10
DEFAULT_INPUT = "input.LandremanPaul2021_QA_lowres"
DEFAULT_PARAMETER_COUNT = 1
DEFAULT_NU_HAT = 1.0e-3
DEFAULT_ER_HAT = 1.0e-3


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (14.0, 8.6),
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
        input_exists = _input_path(DEFAULT_INPUT).exists()
    except RuntimeError:
        input_exists = False
    return (
        find_vmec_jax_root() is not None
        and find_booz_xform_jax_root() is not None
        and input_exists
    )


def _import_vmec_jax():
    root = find_vmec_jax_root()
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import vmec_jax

    return vmec_jax


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), 1.0e-30)


def _finite_difference_gradient(objective, params, *, fd_step: float) -> jnp.ndarray:
    columns = []
    for index in range(params.shape[0]):
        step = jnp.zeros_like(params).at[index].set(fd_step)
        columns.append((objective(params + step) - objective(params - step)) / (2.0 * fd_step))
    return jnp.asarray(columns)


def _infer_signgs(vmec_jax, context, params) -> int:
    state = initial_guess_vmec_jax_boundary_state(context, params, vmec_project=True)
    geom = vmec_jax.eval_geom(state, context.static)
    return int(vmec_jax.signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))


def _residual_history(vmec_jax, context, params) -> list[dict[str, object]]:
    """Return a compact residual-contraction diagnostic for the implicit solver."""

    state_init = initial_guess_vmec_jax_boundary_state(context, params, vmec_project=True)
    geom = vmec_jax.eval_geom(state_init, context.static)
    signgs = int(vmec_jax.signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    rows: list[dict[str, object]] = []
    for max_iter in (5, 20, 50):
        try:
            result = vmec_jax.solve_fixed_boundary_residual_iter(
                state_init,
                context.static,
                indata=context.indata,
                signgs=signgs,
                ftol=None,
                max_iter=int(max_iter),
                step_size=DEFAULT_STEP_SIZE,
                vmec2000_control=True,
                reference_mode=False,
                backtracking=True,
                limit_dt_from_force=True,
                limit_update_rms=True,
                verbose=False,
                verbose_vmec2000_table=False,
                jit_forces="auto",
                use_scan=False,
            )
        except Exception as exc:  # pragma: no cover - optional backend diagnostic
            rows.append(
                {
                    "max_iter": int(max_iter),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc).splitlines()[0],
                }
            )
            continue
        history = np.asarray(getattr(result, "fsqz2_history", []), dtype=float)
        first = None if history.size == 0 else float(history[0])
        last = None if history.size == 0 else float(history[-1])
        rows.append(
            {
                "max_iter": int(max_iter),
                "status": "ok",
                "n_iter": int(getattr(result, "n_iter", -1)),
                "history_size": int(history.size),
                "first_fsqz2": first,
                "last_fsqz2": last,
                "min_fsqz2": None if history.size == 0 else float(np.min(history)),
                "contracts": bool(
                    first is not None and last is not None and float(last) < float(first)
                ),
            }
        )
    return rows


def _build_payload(*, fd_step: float, grid: GridSpec) -> dict[str, object]:
    if not _has_boundary_stack():
        raise RuntimeError("requires local vmec_jax and booz_xform_jax checkouts")

    enable_x64(True)
    vmec_jax = _import_vmec_jax()
    from vmec_jax.implicit import ImplicitFixedBoundaryOptions

    input_path = _input_path(DEFAULT_INPUT)
    context = build_vmec_jax_boundary_context(
        input_path,
        max_mode=1,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    if len(context.specs) == 0:
        raise RuntimeError(f"{DEFAULT_INPUT} did not expose any boundary parameters")
    context = dataclasses.replace(context, specs=tuple(context.specs[:DEFAULT_PARAMETER_COUNT]))

    params0 = jnp.zeros((len(context.specs),), dtype=jnp.float64)
    parameter_names = [spec.name for spec in context.specs]
    inferred_signgs = _infer_signgs(vmec_jax, context, params0)
    if inferred_signgs != context.signgs:
        context = dataclasses.replace(context, signgs=inferred_signgs)
    implicit = ImplicitFixedBoundaryOptions(residual_tangent_mode="auto")

    def _state_from_params(params):
        return solve_vmec_jax_boundary_state(
            context,
            params,
            vmec_project=False,
            max_iter=DEFAULT_MAX_ITER,
            step_size=DEFAULT_STEP_SIZE,
            implicit=implicit,
        )

    def equilibrium_volume(params):
        state = _state_from_params(params)
        geom = vmec_jax.eval_geom(state, context.static)
        _dvds, volume = vmec_jax.volume_from_sqrtg(
            geom.sqrtg,
            context.static.s,
            context.static.grid.theta,
            context.static.grid.zeta,
            nfp=int(context.static.cfg.nfp),
        )
        return volume[-1]

    def booz_xform_scalar(params):
        state = _state_from_params(params)
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
        return surface.b0 + surface.iota + 1.0e-3 * (surface.b_theta + surface.b_zeta)

    def ntx_transport_proxy(params):
        state = _state_from_params(params)
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
            jnp.asarray([DEFAULT_NU_HAT]),
            er_hat=jnp.asarray([DEFAULT_ER_HAT]),
        )
        return coeffs["D33"][0]

    objectives = {
        "equilibrium_volume": equilibrium_volume,
        "booz_xform_scalar": booz_xform_scalar,
        "ntx_transport_proxy": ntx_transport_proxy,
    }
    objective_payloads: list[dict[str, object]] = []
    summary_max = 0.0
    summary_samples: list[float] = []
    finite_difference_by_id: dict[str, np.ndarray] = {}
    for objective_id, objective in objectives.items():
        value = float(objective(params0))
        direct = np.asarray(jax.jacfwd(objective)(params0), dtype=float)
        finite_difference = np.asarray(
            _finite_difference_gradient(objective, params0, fd_step=fd_step),
            dtype=float,
        )
        mismatch = _relative_error(finite_difference, direct)
        finite_difference_by_id[objective_id] = finite_difference
        objective_payloads.append(
            {
                "id": objective_id,
                "value": value,
                "parameter_names": parameter_names,
                "direct_forward_mode": direct.tolist(),
                "finite_difference": finite_difference.tolist(),
                "relative_mismatch": mismatch.tolist(),
                "status": (
                    "validated"
                    if float(np.max(mismatch)) < 1.0e-3
                    else "closed-not-shipped"
                ),
            }
        )
        summary_max = max(summary_max, float(np.max(mismatch)))
        summary_samples.extend(float(x) for x in mismatch.ravel())

    reverse_booz_fd = finite_difference_by_id["booz_xform_scalar"]
    try:
        reverse_booz = np.asarray(jax.grad(booz_xform_scalar)(params0), dtype=float)
    except Exception as exc:
        reverse_mode_diagnostic: dict[str, object] = {
            "status": "unsupported",
            "objective_id": "booz_xform_scalar",
            "reverse_mode_gradient": None,
            "finite_difference": reverse_booz_fd.tolist(),
            "max_relative_mismatch": None,
            "median_relative_mismatch": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc).splitlines()[0],
        }
    else:
        reverse_booz_mismatch = _relative_error(reverse_booz_fd, reverse_booz)
        reverse_mode_diagnostic = {
            "status": "ok",
            "objective_id": "booz_xform_scalar",
            "reverse_mode_gradient": reverse_booz.tolist(),
            "finite_difference": reverse_booz_fd.tolist(),
            "max_relative_mismatch": float(np.max(reverse_booz_mismatch)),
            "median_relative_mismatch": float(np.median(reverse_booz_mismatch)),
            "error_type": None,
            "error_message": None,
        }

    residual_history = _residual_history(vmec_jax, context, params0)
    residual_contracts = all(
        bool(row.get("contracts", False))
        for row in residual_history
        if row.get("status") == "ok"
    )

    return {
        "benchmark": "implicit_equilibrium_forward_mode_derivative_benchmark",
        "classification": "artifact-backed non-shipping implicit-equilibrium diagnostic",
        "case": {
            "id": "qa_lowres",
            "label": "QA low-res",
            "input_path": str(input_path),
            "geometry_path": "implicit_fixed_boundary_residual_solve",
            "parameter_names": parameter_names,
            "parameter_count": len(parameter_names),
            "inferred_signgs": int(inferred_signgs),
        },
        "implicit_solver": {
            "max_iter": DEFAULT_MAX_ITER,
            "step_size": DEFAULT_STEP_SIZE,
            "vmec_project": False,
            "residual_tangent_mode": "auto",
        },
        "closure_decision": {
            "status": "closed-not-shipped",
            "supported_equilibrium_derivative_path": "explicit_relaxed_fixed_boundary",
            "reason": (
                "The residual-forward implicit path does not provide a validated "
                "surface/transport derivative contract on the committed QA case. "
                "The residual iteration is non-contracting in this diagnostic, "
                "and Boozer/NTX tangent parity fails even though the scalar "
                "volume probe remains small."
            ),
        },
        "grid": {
            "n_theta": grid.n_theta,
            "n_zeta": grid.n_zeta,
            "n_xi": grid.n_xi,
        },
        "fd_step": float(fd_step),
        "objectives": objective_payloads,
        "claim_scope": (
            "This artifact closes the previous implicit-equilibrium derivative "
            "lane as a non-shipping diagnostic. The scalar equilibrium-volume "
            "probe remains near centered finite differences, but the same "
            "residual-forward path does not satisfy the surface/transport "
            "parity needed for optimization claims. The supported differentiable "
            "equilibrium path is the explicit-relaxed fixed-boundary lane."
        ),
        "summary_metrics": {
            "max_relative_mismatch": float(summary_max),
            "median_relative_mismatch": float(np.median(np.asarray(summary_samples))),
            "residual_contracts": bool(residual_contracts),
        },
        "residual_history": residual_history,
        "reverse_mode_diagnostic": reverse_mode_diagnostic,
        "open_work": [
            (
                "restore an implicit-equilibrium derivative lane only after the "
                "backend residual solve contracts and Boozer/NTX tangent parity "
                "passes on the committed QA case"
            ),
            "keep explicit-relaxed fixed-boundary derivatives as the shipping equilibrium path",
        ],
    }


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    _configure_style()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(fd_step=DEFAULT_FD_STEP, grid=DEFAULT_GRID)
    colors = {"forward": "#0072B2", "fd": "#D55E00"}

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    x = np.arange(len(payload["case"]["parameter_names"]))
    width = 0.34
    for axis, objective in zip(axes.flat[:3], payload["objectives"], strict=True):
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
        axis.set_xticks(x, payload["case"]["parameter_names"], rotation=35, ha="right")
        axis.set_title(objective["id"].replace("_", " "))
        axis.set_ylabel("Derivative")
        axis.legend(loc="best")

    reverse = payload["reverse_mode_diagnostic"]
    axes[1, 1].axis("off")
    if reverse["status"] == "ok":
        reverse_text = (
            "Reverse diagnostic (Boozer scalar):\n"
            f"  max mismatch={reverse['max_relative_mismatch']:.2e}\n"
            f"  median mismatch={reverse['median_relative_mismatch']:.2e}\n"
            "  reverse mode currently stays open on this lane"
        )
    else:
        reverse_text = (
            "Reverse diagnostic (Boozer scalar):\n"
            f"  status={reverse['status']}\n"
            f"  {reverse['error_type']}: {reverse['error_message']}"
        )

    axes[1, 1].text(
        0.03,
        0.97,
        (
            f"{payload['case']['label']}\n"
            f"Input: {Path(payload['case']['input_path']).name}\n"
            "Implicit solver:\n"
            f"  iter={payload['implicit_solver']['max_iter']}, "
            f"step={payload['implicit_solver']['step_size']:.1f}, "
            f"tangent={payload['implicit_solver']['residual_tangent_mode']}\n"
            f"Forward max mismatch: {payload['summary_metrics']['max_relative_mismatch']:.2e}\n"
            "Forward median mismatch: "
            f"{payload['summary_metrics']['median_relative_mismatch']:.2e}\n\n"
            f"Closure: {payload['closure_decision']['status']}\n"
            "Objective status:\n"
            f"  equilibrium_volume={payload['objectives'][0]['status']}\n"
            f"  booz_xform_scalar={payload['objectives'][1]['status']}\n"
            f"  ntx_transport_proxy={payload['objectives'][2]['status']}\n\n"
            "Residual contraction:\n"
            f"  contracts={payload['summary_metrics']['residual_contracts']}\n\n"
            f"{reverse_text}\n\n"
            "Objectives:\n"
            "  equilibrium_volume\n"
            "  booz_xform_scalar\n"
            "  ntx_transport_proxy"
        ),
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#d1d5db", "alpha": 0.98},
    )

    fig.suptitle(
        "Implicit-equilibrium derivative diagnostic: closed as non-shipping",
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
