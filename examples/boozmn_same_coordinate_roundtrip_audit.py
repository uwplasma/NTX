#!/usr/bin/env python3
"""Validate the direct `boozmn` loader on same-coordinate Boozer data.

This audit generates a Boozer file from a VMEC `wout`, reloads the selected
surfaces through `load_boozmn_surface`, and compares them with the in-memory
`vmex -> booz_xform_jax -> NTX` path on the same VMEC half-grid surfaces.
It is a geometry/backend round trip, not a fitted correction.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

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
    surface_from_vmex_wout,
)

DEFAULT_INPUT = Path(
    "/Users/rogeriojorge/local/vmex/examples/data/input.LandremanPaul2021_QA_lowres"
)
DEFAULT_WOUT = Path(
    "/Users/rogeriojorge/local/vmex/examples/data/wout_LandremanPaul2021_QA_lowres.nc"
)
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "boozmn_same_coordinate_roundtrip_audit"


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


def _scaled_abs(reference: Any, candidate: Any, scale: float) -> float:
    ref = _scalar(reference)
    cand = _scalar(candidate)
    return abs(ref - cand) / max(abs(float(scale)), 1.0e-300)


def _transport(surface: Any, grid: GridSpec, *, nu_hat: float, epsi_hat: float) -> dict[str, float]:
    prepared = prepare_monoenergetic_system(surface, grid)
    result = solve_prepared(prepared, MonoenergeticCase(nu_hat=nu_hat, epsi_hat=epsi_hat))
    return {name: _scalar(value) for name, value in result.as_dict().items()}


def _geometry_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    covariant_scale = max(
        abs(_scalar(reference.b_zeta)),
        abs(_scalar(candidate.b_zeta)),
        abs(_scalar(reference.b_theta)),
        abs(_scalar(candidate.b_theta)),
        1.0,
    )
    return {
        "m_equal": float(np.array_equal(_array(reference.m), _array(candidate.m))),
        "n_equal": float(np.array_equal(_array(reference.n), _array(candidate.n))),
        "b_cos_relative_l2": _relative_l2(reference.b_cos, candidate.b_cos),
        "iota_relative": _relative_abs(reference.iota, candidate.iota),
        "b_theta_scaled_difference": _scaled_abs(
            reference.b_theta,
            candidate.b_theta,
            covariant_scale,
        ),
        "b_zeta_relative": _relative_abs(reference.b_zeta, candidate.b_zeta),
        "b0_relative": _relative_abs(reference.b0, candidate.b0),
    }


def _transport_metrics(
    reference: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    return {
        name: abs(reference[name] - candidate[name])
        / max(abs(reference[name]), abs(candidate[name]), 1.0e-300)
        for name in ("D11", "D31", "D13", "D33", "D33_spitzer")
    }


def _read_vmec_edge_psi(path: Path) -> float:
    with Dataset(path, mode="r") as handle:
        phi = np.asarray(handle.variables["phi"][:], dtype=float).reshape(-1)
    return float(abs(phi[-1]) / (2.0 * np.pi))


def _write_boozmn_from_wout(
    *,
    wout_path: Path,
    output_dir: Path,
    surface_indices: tuple[int, ...],
    mboz: int,
    nboz: int,
) -> tuple[Path, np.ndarray]:
    from booz_xform_jax import Booz_xform

    bx = Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(wout_path), flux=True)
    ns_in = int(bx.ns_in)
    invalid = [idx for idx in surface_indices if idx < 0 or idx >= ns_in]
    if invalid:
        raise ValueError(f"surface indices {invalid} are outside [0, {ns_in})")
    bx.compute_surfs = [int(idx) for idx in surface_indices]
    bx.mboz = int(mboz)
    bx.nboz = int(nboz)
    bx.run()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"boozmn_roundtrip_m{mboz}_n{nboz}.nc"
    bx.write_boozmn(str(path))
    return path, np.asarray(bx.s_in, dtype=float)[list(surface_indices)]


def _default_surface_indices(wout_path: Path) -> tuple[int, ...]:
    from booz_xform_jax import Booz_xform

    bx = Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(wout_path), flux=True)
    ns_in = int(bx.ns_in)
    if ns_in < 3:
        return (0,)
    candidates = (max(0, ns_in // 8), max(0, ns_in // 3), max(0, (2 * ns_in) // 3))
    return tuple(dict.fromkeys(min(ns_in - 1, int(idx)) for idx in candidates))


def build_roundtrip_audit(
    *,
    input_path: Path,
    wout_path: Path,
    surface_indices: tuple[int, ...] | None,
    mboz: int,
    nboz: int,
    psi_p: float | None,
    nu_hat: float,
    epsi_hat: float,
    grid: GridSpec,
    output_dir: Path | None = None,
    profile_source: str = "auto",
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    wout_path = wout_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"VMEC input file does not exist: {input_path}")
    if not wout_path.exists():
        raise FileNotFoundError(f"VMEC wout file does not exist: {wout_path}")
    if nu_hat <= 0.0:
        raise ValueError("nu_hat must be positive")
    if surface_indices is None:
        surface_indices = _default_surface_indices(wout_path)
    if len(surface_indices) == 0:
        raise ValueError("at least one surface index is required")

    resolved_psi_p = _read_vmec_edge_psi(wout_path) if psi_p is None else float(psi_p)
    temp_context = (
        tempfile.TemporaryDirectory(prefix="ntx_boozmn_roundtrip_") if output_dir is None else None
    )
    try:
        if temp_context is not None:
            working_dir = Path(temp_context.name)
        else:
            working_dir = output_dir.expanduser().resolve()
        boozmn_path, s_values = _write_boozmn_from_wout(
            wout_path=wout_path,
            output_dir=working_dir,
            surface_indices=surface_indices,
            mboz=mboz,
            nboz=nboz,
        )

        surfaces: list[dict[str, Any]] = []
        max_geometry_relative = 0.0
        max_transport_relative = 0.0
        for packed_index, (surface_index, s_value) in enumerate(
            zip(surface_indices, s_values, strict=True)
        ):
            reference_surface = surface_from_vmex_wout(
                input_path=input_path,
                wout_path=wout_path,
                s=float(s_value),
                mboz=mboz,
                nboz=nboz,
                psi_p=resolved_psi_p,
                profile_source=profile_source,
            )
            by_index = load_boozmn_surface(
                boozmn_path,
                surface_index=packed_index,
                psi_p=resolved_psi_p,
            )
            by_s = load_boozmn_surface(
                boozmn_path,
                s=float(s_value),
                psi_p=resolved_psi_p,
            )
            transport_reference = _transport(
                reference_surface,
                grid,
                nu_hat=nu_hat,
                epsi_hat=epsi_hat,
            )
            transport_index = _transport(
                by_index.surface,
                grid,
                nu_hat=nu_hat,
                epsi_hat=epsi_hat,
            )
            transport_by_s = _transport(
                by_s.surface,
                grid,
                nu_hat=nu_hat,
                epsi_hat=epsi_hat,
            )
            geometry_index = _geometry_metrics(reference_surface, by_index.surface)
            geometry_by_s = _geometry_metrics(reference_surface, by_s.surface)
            transport_index_metrics = _transport_metrics(
                transport_reference,
                transport_index,
            )
            transport_by_s_metrics = _transport_metrics(
                transport_reference,
                transport_by_s,
            )
            geometry_scale = max(
                geometry_index["b_cos_relative_l2"],
                geometry_index["iota_relative"],
                geometry_index["b_theta_scaled_difference"],
                geometry_index["b_zeta_relative"],
                geometry_by_s["b_cos_relative_l2"],
                geometry_by_s["iota_relative"],
                geometry_by_s["b_theta_scaled_difference"],
                geometry_by_s["b_zeta_relative"],
            )
            transport_scale = max(
                max(transport_index_metrics.values()),
                max(transport_by_s_metrics.values()),
            )
            max_geometry_relative = max(max_geometry_relative, geometry_scale)
            max_transport_relative = max(max_transport_relative, transport_scale)
            surfaces.append(
                {
                    "surface_index": int(surface_index),
                    "packed_index": int(packed_index),
                    "s": float(s_value),
                    "rho": float(np.sqrt(max(float(s_value), 0.0))),
                    "loaded_by_s_selected_index": int(by_s.surface_index),
                    "geometry_relative": {
                        "surface_index": geometry_index,
                        "s_selector": geometry_by_s,
                    },
                    "transport": {
                        "reference": transport_reference,
                        "surface_index": transport_index,
                        "s_selector": transport_by_s,
                    },
                    "transport_relative": {
                        "surface_index": transport_index_metrics,
                        "s_selector": transport_by_s_metrics,
                    },
                }
            )

        geometry_gate = 1.0e-7
        transport_gate = 1.0e-6
        return {
            "benchmark": "boozmn_same_coordinate_roundtrip_audit",
            "classification": "same-coordinate VMEC half-grid Boozer-file round trip",
            "inputs": {
                "input": str(input_path),
                "wout": str(wout_path),
                "generated_boozmn": str(boozmn_path),
                "generated_boozmn_retained": output_dir is not None,
                "surface_indices": list(surface_indices),
                "mboz": int(mboz),
                "nboz": int(nboz),
                "nu_hat": float(nu_hat),
                "epsi_hat": float(epsi_hat),
                "psi_p": float(resolved_psi_p),
                "profile_source": str(profile_source),
                "grid": {
                    "n_theta": grid.n_theta,
                    "n_zeta": grid.n_zeta,
                    "n_xi": grid.n_xi,
                    "dtype": grid.dtype,
                    "x64": grid.x64,
                },
            },
            "summary_metrics": {
                "max_geometry_relative_difference": max_geometry_relative,
                "max_transport_relative_difference": max_transport_relative,
                "geometry_gate_rtol": geometry_gate,
                "transport_gate_rtol": transport_gate,
                "roundtrip_closed": bool(
                    max_geometry_relative < geometry_gate
                    and max_transport_relative < transport_gate
                ),
            },
            "interpretation": (
                "Boozer spectra and Boozer radial profiles live on the VMEC "
                "half grid. A passing same-coordinate round trip means the "
                "direct boozmn loader uses s_in/s_b/jlist half-grid metadata "
                "rather than full-grid phi metadata for radial selection and "
                "interpolation."
            ),
            "surfaces": surfaces,
        }
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> Path:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_prefix.with_suffix(".json")
    output_path.write_text(json.dumps(_jsonify(payload), indent=2) + "\n", encoding="utf-8")
    return output_path


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> list[Path]:
    import matplotlib.pyplot as plt

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    surfaces = payload["surfaces"]
    s_values = [surface["s"] for surface in surfaces]
    labels = [f"{value:.3f}" for value in s_values]
    coeffs = ("D11", "D31", "D13", "D33")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    ax_geom, ax_coeff, ax_values, ax_note = axes.ravel()

    geometry_keys = (
        "b_cos_relative_l2",
        "iota_relative",
        "b_theta_scaled_difference",
        "b_zeta_relative",
    )
    for key in geometry_keys:
        ax_geom.plot(
            labels,
            [surface["geometry_relative"]["s_selector"][key] for surface in surfaces],
            "o-",
            label=key.replace("_relative_l2", "").replace("_relative", ""),
        )
    ax_geom.axhline(
        payload["summary_metrics"]["geometry_gate_rtol"],
        color="black",
        lw=1.0,
        ls="--",
        label="geometry gate",
    )
    ax_geom.set_yscale("log")
    ax_geom.set_xlabel("VMEC half-grid s")
    ax_geom.set_ylabel("relative difference")
    ax_geom.set_title("(a) file round-trip geometry")
    ax_geom.grid(alpha=0.24, lw=0.6)
    ax_geom.legend(frameon=False, fontsize=8)

    x = np.arange(len(coeffs))
    width = 0.78 / max(len(surfaces), 1)
    for idx, surface in enumerate(surfaces):
        values = [surface["transport_relative"]["s_selector"][coeff] for coeff in coeffs]
        ax_coeff.bar(x + (idx - (len(surfaces) - 1) / 2.0) * width, values, width=width)
    ax_coeff.axhline(
        payload["summary_metrics"]["transport_gate_rtol"],
        color="black",
        lw=1.0,
        ls="--",
        label="transport gate",
    )
    ax_coeff.set_xticks(x)
    ax_coeff.set_xticklabels(coeffs)
    ax_coeff.set_yscale("log")
    ax_coeff.set_ylabel("relative difference")
    ax_coeff.set_title("(b) coefficient round trip")
    ax_coeff.grid(alpha=0.24, lw=0.6, axis="y")

    for coeff in coeffs:
        ax_values.plot(
            labels,
            [surface["transport"]["reference"][coeff] for surface in surfaces],
            "o-",
            label=f"{coeff} in-memory",
        )
        ax_values.plot(
            labels,
            [surface["transport"]["s_selector"][coeff] for surface in surfaces],
            "x--",
            label=f"{coeff} boozmn",
        )
    ax_values.axhline(0.0, color="black", lw=0.8)
    ax_values.set_xlabel("VMEC half-grid s")
    ax_values.set_ylabel("coefficient value")
    ax_values.set_title("(c) signed transport values")
    ax_values.grid(alpha=0.24, lw=0.6)
    ax_values.legend(frameon=False, fontsize=7, ncols=2)

    summary = payload["summary_metrics"]
    ax_note.axis("off")
    ax_note.text(
        0.02,
        0.95,
        "\n".join(
            (
                "(d) audit classification",
                "same VMEC state, Boozer transform,",
                "half-grid surfaces, and flux scale",
                "",
                f"max geometry rdiff: {summary['max_geometry_relative_difference']:.3e}",
                f"max coefficient rdiff: {summary['max_transport_relative_difference']:.3e}",
                f"closed: {summary['roundtrip_closed']}",
                "",
                "The tested radial coordinate is the VMEC",
                "half grid used by Boozer spectra, not the",
                "full-grid toroidal-flux profile phi_b.",
            )
        ),
        va="top",
        ha="left",
        fontsize=10,
    )

    fig.suptitle("Same-coordinate Boozer-file round-trip audit", fontsize=13)
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _parse_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == "":
        return None
    return tuple(int(item) for item in value.replace(",", " ").split())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate direct boozmn loading against an in-memory same-coordinate path."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--wout", type=Path, default=DEFAULT_WOUT)
    parser.add_argument("--surface-indices", default=None)
    parser.add_argument("--mboz", type=int, default=4)
    parser.add_argument("--nboz", type=int, default=4)
    parser.add_argument("--psi-p", type=float, default=None)
    parser.add_argument(
        "--profile-source",
        choices=("auto", "wout"),
        default="auto",
        help=(
            "Reference VMEC-to-Boozer file path. Both choices use finalized "
            "WOUT magnetic channels with the current vmex API."
        ),
    )
    parser.add_argument("--nu-hat", type=float, default=1.0e-2)
    parser.add_argument("--epsi-hat", type=float, default=0.0)
    parser.add_argument("--n-theta", type=int, default=7)
    parser.add_argument("--n-zeta", type=int, default=7)
    parser.add_argument("--n-xi", type=int, default=8)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_roundtrip_audit(
        input_path=args.input,
        wout_path=args.wout,
        surface_indices=_parse_indices(args.surface_indices),
        mboz=args.mboz,
        nboz=args.nboz,
        psi_p=args.psi_p,
        profile_source=args.profile_source,
        nu_hat=args.nu_hat,
        epsi_hat=args.epsi_hat,
        grid=GridSpec(args.n_theta, args.n_zeta, args.n_xi),
        output_dir=args.output_dir,
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
