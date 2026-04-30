#!/usr/bin/env python3
"""Audit direct Boozer-file geometry against the validated VMEC-harmonic path.

This script is intentionally diagnostic.  It does not apply a correction or a
fitted bridge.  It dumps the surface metadata, grid geometry, drift source,
operator-channel coefficients, and monoenergetic transport coefficients needed
to decide whether a direct Boozer-file backend is using the same physical and
numerical convention as the VMEC-backed validation path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import (  # noqa: E402
    GridSpec,
    MonoenergeticCase,
    load_boozmn_surface,
    prepare_monoenergetic_system,
    solve_prepared,
    surface_from_vmec_jax_vmec_wout_file,
)
from ntx.operators import OperatorContext, coefficients_for_k, source_modes  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "boozmn_backend_validation_audit"
FIXTURE_WOUT = ROOT / "tests" / "fixtures" / "sample_wout.nc"
FIXTURE_BOOZMN = ROOT / "tests" / "fixtures" / "sample_boozmn.nc"


def _array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=float)


def _scalar(value: Any) -> float:
    return float(np.asarray(jax.device_get(value), dtype=float).reshape(()))


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(item) for item in value]
    return value


def _relative_l2(reference: Any, candidate: Any) -> float:
    ref = _array(reference)
    cand = _array(candidate)
    denom = max(float(np.linalg.norm(ref)), float(np.linalg.norm(cand)), 1.0e-300)
    return float(np.linalg.norm(ref - cand) / denom)


def _relative_abs(reference: Any, candidate: Any) -> float:
    ref = _scalar(reference)
    cand = _scalar(candidate)
    return abs(ref - cand) / max(abs(ref), abs(cand), 1.0e-300)


def _field_stats(value: Any) -> dict[str, float | bool]:
    array = _array(value)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "l2": float(np.linalg.norm(array)),
        "abs_max": float(np.max(np.abs(array))),
        "finite": bool(np.all(np.isfinite(array))),
    }


def _mode_head(surface: Any, count: int = 8) -> dict[str, list[float] | list[int]]:
    return {
        "m": _array(surface.m[:count]).astype(int).tolist(),
        "n": _array(surface.n[:count]).astype(int).tolist(),
        "b_cos": _array(surface.b_cos[:count]).tolist(),
    }


def _surface_metadata(surface: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "type": type(surface).__name__,
        "nfp": int(surface.nfp),
        "iota": _scalar(surface.iota),
        "b0": None if surface.b0 is None else _scalar(surface.b0),
        "psi_p": None if surface.psi_p is None else _scalar(surface.psi_p),
        "mode_count": int(len(surface.m)),
        "mode_head": _mode_head(surface),
    }
    if hasattr(surface, "path"):
        metadata["path"] = str(surface.path)
        metadata["loaded_mode_count"] = int(surface.loaded_mode_count)
        metadata["total_mode_count"] = int(surface.total_mode_count)
        metadata["psi_a_hat"] = _scalar(surface.psi_a_hat)
        metadata["transport_psi_scale"] = _scalar(surface.transport_psi_scale)
    if getattr(surface, "source_path", None) is not None:
        metadata["path"] = str(surface.source_path)
    if hasattr(surface, "b_theta"):
        metadata["b_theta"] = _scalar(surface.b_theta)
        metadata["b_zeta"] = _scalar(surface.b_zeta)
    return metadata


def _read_vmec_edge_psi(path: Path) -> float:
    with Dataset(path, mode="r") as handle:
        phi = np.asarray(handle.variables["phi"][:], dtype=float).reshape(-1)
    return float(abs(phi[-1]) / (2.0 * np.pi))


def _prepared_payload(
    surface: Any,
    grid: GridSpec,
    *,
    nu_hat: float,
    epsi_hat: float,
) -> dict[str, Any]:
    prepared = prepare_monoenergetic_system(surface, grid)
    geom = prepared.geometry
    ctx = OperatorContext(
        surface=surface,
        geometry=geom,
        nu_hat=jnp.asarray(nu_hat, dtype=grid.jax_dtype),
        epsi_hat=jnp.asarray(epsi_hat, dtype=grid.jax_dtype),
    )
    s1, s3 = source_modes(ctx, grid.n_xi)
    lower, diagonal, upper = coefficients_for_k(ctx, 1)
    result = solve_prepared(prepared, MonoenergeticCase(nu_hat=nu_hat, epsi_hat=epsi_hat))

    drift_numerator = geom.b_sub_theta * geom.d_b_dzeta - geom.b_sub_zeta * geom.d_b_dtheta

    return {
        "surface": _surface_metadata(surface),
        "geometry": {
            "b": _field_stats(geom.b),
            "d_b_dtheta": _field_stats(geom.d_b_dtheta),
            "d_b_dzeta": _field_stats(geom.d_b_dzeta),
            "jacobian": _field_stats(geom.jacobian),
            "b_sub_theta": _field_stats(geom.b_sub_theta),
            "b_sub_zeta": _field_stats(geom.b_sub_zeta),
            "b_sup_theta": _field_stats(geom.b_sup_theta),
            "b_sup_zeta": _field_stats(geom.b_sup_zeta),
            "drift_numerator": _field_stats(drift_numerator),
            "radial_drift_spatial": _field_stats(geom.radial_drift_spatial),
            "volume_prime": _scalar(geom.volume_prime),
            "b2_mean": _scalar(geom.b2_mean),
            "b0": _scalar(geom.b0),
            "coefficient_psi_scale": _scalar(geom.coefficient_psi_scale),
            "transport_psi_scale": _scalar(geom.transport_psi_scale),
        },
        "source": {
            "s1_full": _field_stats(s1),
            "s3_full": _field_stats(s3),
            "s1_l0": _field_stats(s1[0]),
            "s1_l2": _field_stats(s1[2]),
            "s3_l1": _field_stats(s3[1]),
        },
        "operator_k1": {
            "lower_theta": _field_stats(lower[0]),
            "lower_zeta": _field_stats(lower[1]),
            "lower_value": _field_stats(lower[2]),
            "diagonal_theta": _field_stats(diagonal[0]),
            "diagonal_zeta": _field_stats(diagonal[1]),
            "diagonal_value": _field_stats(diagonal[2]),
            "upper_theta": _field_stats(upper[0]),
            "upper_zeta": _field_stats(upper[1]),
            "upper_value": _field_stats(upper[2]),
        },
        "arrays": {
            "b": _array(geom.b),
            "jacobian": _array(geom.jacobian),
            "b_sub_theta": _array(geom.b_sub_theta),
            "b_sub_zeta": _array(geom.b_sub_zeta),
            "b_sup_theta": _array(geom.b_sup_theta),
            "b_sup_zeta": _array(geom.b_sup_zeta),
            "radial_drift_spatial": _array(geom.radial_drift_spatial),
            "s1": _array(s1),
            "s3": _array(s3),
            "operator_lower": _array(lower),
            "operator_diagonal": _array(diagonal),
            "operator_upper": _array(upper),
        },
        "transport": result.as_dict(),
    }


def _comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_arrays = reference["arrays"]
    cand_arrays = candidate["arrays"]
    field_names = (
        "b",
        "jacobian",
        "b_sub_theta",
        "b_sub_zeta",
        "b_sup_theta",
        "b_sup_zeta",
        "radial_drift_spatial",
        "s1",
        "s3",
        "operator_lower",
        "operator_diagonal",
        "operator_upper",
    )
    field_relative_l2 = {
        name: _relative_l2(ref_arrays[name], cand_arrays[name]) for name in field_names
    }
    coefficient_names = ("D11", "D31", "D13", "D33", "D33_spitzer")
    coefficient_relative = {
        name: _relative_abs(reference["transport"][name], candidate["transport"][name])
        for name in coefficient_names
    }
    source_drift_relative_l2 = field_relative_l2["radial_drift_spatial"]
    max_transport_relative = max(coefficient_relative.values())
    return {
        "field_relative_l2": field_relative_l2,
        "coefficient_relative_difference": coefficient_relative,
        "max_transport_relative_difference": max_transport_relative,
        "radial_drift_relative_l2": source_drift_relative_l2,
    }


def _strip_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    clean = dict(payload)
    for case in clean["cases"].values():
        case.pop("arrays", None)
    return clean


def build_audit(
    *,
    wout_path: Path,
    boozmn_path: Path,
    rho: float,
    nu_hat: float,
    epsi_hat: float,
    grid: GridSpec,
) -> dict[str, Any]:
    wout_path = wout_path.expanduser().resolve()
    boozmn_path = boozmn_path.expanduser().resolve()
    if not wout_path.exists():
        raise FileNotFoundError(f"VMEC wout file does not exist: {wout_path}")
    if not boozmn_path.exists():
        raise FileNotFoundError(f"Boozer file does not exist: {boozmn_path}")
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must satisfy 0 < rho <= 1")
    if nu_hat <= 0.0:
        raise ValueError("nu_hat must be positive")

    psia = _read_vmec_edge_psi(wout_path)
    vmec_surface = surface_from_vmec_jax_vmec_wout_file(wout_path, s=rho**2)
    boozmn_unit = load_boozmn_surface(boozmn_path, rho=rho, psi_p=1.0).surface
    boozmn_vmec_flux = load_boozmn_surface(boozmn_path, rho=rho, psi_p=psia).surface

    cases = {
        "vmec_harmonic": _prepared_payload(
            vmec_surface,
            grid,
            nu_hat=nu_hat,
            epsi_hat=epsi_hat,
        ),
        "direct_boozer_unit_flux": _prepared_payload(
            boozmn_unit,
            grid,
            nu_hat=nu_hat,
            epsi_hat=epsi_hat,
        ),
        "direct_boozer_vmec_edge_flux": _prepared_payload(
            boozmn_vmec_flux,
            grid,
            nu_hat=nu_hat,
            epsi_hat=epsi_hat,
        ),
    }

    comparisons = {
        name: _comparison(cases["vmec_harmonic"], case)
        for name, case in cases.items()
        if name != "vmec_harmonic"
    }
    best_candidate = min(
        comparisons,
        key=lambda name: comparisons[name]["max_transport_relative_difference"],
    )
    best = comparisons[best_candidate]
    transport_gate = best["max_transport_relative_difference"] < 1.0e-1
    drift_gate = best["radial_drift_relative_l2"] < 1.0e-1

    return {
        "benchmark": "boozmn_backend_validation_audit",
        "classification": "direct Boozer-file backend validation audit",
        "inputs": {
            "wout": str(wout_path),
            "boozmn": str(boozmn_path),
            "rho": rho,
            "s": rho**2,
            "nu_hat": nu_hat,
            "epsi_hat": epsi_hat,
            "grid": {
                "n_theta": grid.n_theta,
                "n_zeta": grid.n_zeta,
                "n_xi": grid.n_xi,
                "dtype": grid.dtype,
                "x64": grid.x64,
            },
            "vmec_edge_psi_over_2pi": psia,
        },
        "summary_metrics": {
            "best_direct_boozer_candidate": best_candidate,
            "best_max_transport_relative_difference": best[
                "max_transport_relative_difference"
            ],
            "best_radial_drift_relative_l2": best["radial_drift_relative_l2"],
            "transport_gate_rtol": 1.0e-1,
            "radial_drift_gate_rtol": 1.0e-1,
            "direct_boozer_backend_closed": bool(transport_gate and drift_gate),
        },
        "interpretation": (
            "The direct Boozer-file backend can be promoted only when both the "
            "transport-coefficient and radial-drift source channels pass on "
            "owned same-coordinate inputs. Failing this audit means the "
            "difference is still a geometry/normalization/convention issue, "
            "not a closure fit target."
        ),
        "cases": cases,
        "comparisons_to_vmec_harmonic": comparisons,
    }


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> Path:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_prefix.with_suffix(".json")
    output_path.write_text(json.dumps(_jsonify(_strip_arrays(payload)), indent=2) + "\n")
    return output_path


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> list[Path]:
    import matplotlib.pyplot as plt

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    comparisons = payload["comparisons_to_vmec_harmonic"]
    cases = payload["cases"]

    labels = {
        "direct_boozer_unit_flux": "unit flux",
        "direct_boozer_vmec_edge_flux": "VMEC edge flux",
    }
    fields = ("b", "jacobian", "radial_drift_spatial", "s1")
    coeffs = ("D11", "D31", "D13", "D33")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    ax_field, ax_coeff, ax_transport, ax_note = axes.ravel()

    x = np.arange(len(fields))
    width = 0.36
    for idx, (name, comparison) in enumerate(comparisons.items()):
        offsets = x + (idx - 0.5) * width
        values = [comparison["field_relative_l2"][field] for field in fields]
        ax_field.bar(offsets, values, width=width, label=labels.get(name, name))
    ax_field.set_xticks(x)
    ax_field.set_xticklabels(("B", "Jacobian", "radial drift", "source s1"), rotation=20)
    ax_field.set_yscale("log")
    ax_field.set_ylabel("relative L2 difference")
    ax_field.set_title("(a) geometry and source channels")
    ax_field.grid(alpha=0.24, lw=0.6, axis="y")
    ax_field.legend(frameon=False)

    x = np.arange(len(coeffs))
    for idx, (name, comparison) in enumerate(comparisons.items()):
        offsets = x + (idx - 0.5) * width
        values = [comparison["coefficient_relative_difference"][coeff] for coeff in coeffs]
        ax_coeff.bar(offsets, values, width=width, label=labels.get(name, name))
    ax_coeff.axhline(1.0e-1, color="black", lw=1.0, ls="--", label="1e-1 gate")
    ax_coeff.set_xticks(x)
    ax_coeff.set_xticklabels(coeffs)
    ax_coeff.set_yscale("log")
    ax_coeff.set_ylabel("relative difference")
    ax_coeff.set_title("(b) transport coefficients")
    ax_coeff.grid(alpha=0.24, lw=0.6, axis="y")

    x = np.arange(len(coeffs))
    vmec = cases["vmec_harmonic"]["transport"]
    unit = cases["direct_boozer_unit_flux"]["transport"]
    psia = cases["direct_boozer_vmec_edge_flux"]["transport"]
    ax_transport.plot(x, [vmec[name] for name in coeffs], "o-", label="VMEC harmonic")
    ax_transport.plot(x, [unit[name] for name in coeffs], "s--", label="Boozer unit flux")
    ax_transport.plot(x, [psia[name] for name in coeffs], "^--", label="Boozer VMEC flux")
    ax_transport.axhline(0.0, color="black", lw=0.8)
    ax_transport.set_xticks(x)
    ax_transport.set_xticklabels(coeffs)
    ax_transport.set_ylabel("coefficient value")
    ax_transport.set_title("(c) signed coefficient values")
    ax_transport.grid(alpha=0.24, lw=0.6)
    ax_transport.legend(frameon=False)

    summary = payload["summary_metrics"]
    ax_note.axis("off")
    ax_note.text(
        0.02,
        0.95,
        "\n".join(
            (
                "(d) audit classification",
                f"best direct path: {summary['best_direct_boozer_candidate']}",
                "max coefficient rdiff: "
                f"{summary['best_max_transport_relative_difference']:.3e}",
                f"radial-drift rdiff: {summary['best_radial_drift_relative_l2']:.3e}",
                "closed: " f"{summary['direct_boozer_backend_closed']}",
                "",
                "A failure localizes the issue to geometry/source convention",
                "before any reduced-closure or current-profile comparison.",
            )
        ),
        va="top",
        ha="left",
        fontsize=10,
    )

    fig.suptitle("Direct Boozer-file backend validation audit", fontsize=13)
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit direct Boozer-file geometry against the VMEC-harmonic path."
    )
    parser.add_argument("--wout", type=Path, default=FIXTURE_WOUT)
    parser.add_argument("--boozmn", type=Path, default=FIXTURE_BOOZMN)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--nu-hat", type=float, default=1.0e-2)
    parser.add_argument("--epsi-hat", type=float, default=0.0)
    parser.add_argument("--n-theta", type=int, default=7)
    parser.add_argument("--n-zeta", type=int, default=7)
    parser.add_argument("--n-xi", type=int, default=8)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_audit(
        wout_path=args.wout,
        boozmn_path=args.boozmn,
        rho=args.rho,
        nu_hat=args.nu_hat,
        epsi_hat=args.epsi_hat,
        grid=GridSpec(args.n_theta, args.n_zeta, args.n_xi),
    )
    json_path = write_payload(payload, args.output_prefix)
    print(f"audit JSON: {json_path}")
    if not args.no_figure:
        figure_paths = build_figure(payload, args.output_prefix)
        for figure_path in figure_paths:
            print(f"audit figure: {figure_path}")
    print(json.dumps(_jsonify(payload["summary_metrics"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
