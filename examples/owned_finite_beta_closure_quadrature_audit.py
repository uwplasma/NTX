#!/usr/bin/env python3
"""Audit Sonine-order and velocity-quadrature coupling in the finite-beta closure.

The owned finite-beta bootstrap-current stress case is cancellation-sensitive:
small species-flow errors can become order-1 current errors.  This diagnostic
keeps the NTX scan, Redl geometry, profiles, and current normalization fixed,
then varies only the profile-closure Sonine order and velocity quadrature used
by the downstream momentum-restoring solve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from examples.owned_finite_beta_bootstrap_comparison import (  # noqa: E402
    DEFAULT_CASE,
    DEFAULT_MBOZ,
    DEFAULT_NBOZ,
    DEFAULT_REDL_NTHETA,
    EPS,
    ProfileContract,
    _build_species,
    _case_by_id,
    _evaluate_neopax_currents,
    _interp,
    _read_neopax_field,
    _redl_geometry_and_current,
    _relative_error,
    _require_external_stacks,
    _to_jsonable,
    _write_boozmn,
)
from examples.owned_finite_beta_source_channel_audit import (  # noqa: E402
    _load_or_build_scan,
)
from ntx import GridSpec, to_neopax_monoenergetic  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_finite_beta_closure_quadrature_audit"
BOOTSTRAP_JSON = ROOT / "docs" / "_static" / "owned_finite_beta_bootstrap_comparison.json"
WORKDIR = ROOT / "examples" / "outputs" / "owned_finite_beta_closure_quadrature_audit"
PROFILE_CURRENT_GATE = 1.0e-1


def _root_relative(path: Path) -> str:
    path = Path(path)
    resolved = path if path.is_absolute() else ROOT / path
    return str(resolved.resolve().relative_to(ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _contract_from_payload(payload: dict[str, Any]) -> ProfileContract:
    values = dict(payload.get("profile_contract", {}))
    default = ProfileContract()
    return ProfileContract(
        density_core_m3=float(values.get("density_core_m3", default.density_core_m3)),
        density_edge_m3=float(values.get("density_edge_m3", default.density_edge_m3)),
        temperature_core_ev=float(
            values.get("temperature_core_ev", default.temperature_core_ev)
        ),
        temperature_edge_ev=float(
            values.get("temperature_edge_ev", default.temperature_edge_ev)
        ),
        density_power=int(values.get("density_power", default.density_power)),
        temperature_power=int(
            values.get("temperature_power", default.temperature_power)
        ),
        zeff=float(values.get("zeff", default.zeff)),
    )


def _grid_from_payload(payload: dict[str, Any]) -> GridSpec:
    grid = payload.get("inputs", {}).get("ntx_grid", {})
    return GridSpec(
        int(grid.get("n_theta", 25)),
        int(grid.get("n_zeta", 31)),
        int(grid.get("n_xi", 24)),
    )


def _stress_rho_from_reference(payload: dict[str, Any]) -> float:
    comparison = payload["comparison"]
    rho = np.asarray(comparison["rho"], dtype=float)
    error = np.asarray(comparison["relative_error_total_vs_redl"], dtype=float)
    return float(rho[int(np.nanargmax(error))])


def _evaluate_rows(
    *,
    bootstrap_payload: dict[str, Any],
    x_values: tuple[int, ...],
    n_orders: tuple[int, ...],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    *_, NEOPAX = _require_external_stacks()
    inputs = bootstrap_payload["inputs"]
    case = _case_by_id(str(bootstrap_payload.get("case", {}).get("id", DEFAULT_CASE)))
    contract = _contract_from_payload(bootstrap_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    mboz = int(inputs.get("mboz", DEFAULT_MBOZ))
    nboz = int(inputs.get("nboz", DEFAULT_NBOZ))
    boozmn_path = _write_boozmn(case, output_dir, mboz=mboz, nboz=nboz)
    field = _read_neopax_field(int(inputs.get("field_radial_points", 15)), case, boozmn_path)
    species = _build_species(NEOPAX, field, contract)
    scan_rho = np.asarray(inputs["scan_rho"], dtype=float)
    es_values = np.asarray(inputs["Es"], dtype=float)
    grid = _grid_from_payload(bootstrap_payload)
    scan, scan_metadata = _load_or_build_scan(
        bootstrap_payload=bootstrap_payload,
        case=case,
        field=field,
        scan_grid=grid,
        output_dir=output_dir,
    )
    scan_seconds = float(
        scan_metadata.get(
            "scan_build_seconds",
            scan_metadata.get("scan_load_seconds", 0.0),
        )
    )
    database = to_neopax_monoenergetic(
        scan,
        a_b=float(field.a_b),
        d33_mode=str(inputs.get("d33_mode", "spitzer")),
    )
    rho_compare = np.asarray(field.rho_grid, dtype=float)[1:-1]
    redl = _redl_geometry_and_current(
        case,
        rho=rho_compare,
        contract=contract,
        mboz=mboz,
        nboz=nboz,
        redl_ntheta=int(inputs.get("redl_ntheta", DEFAULT_REDL_NTHETA)),
        helicity_n=int(inputs.get("helicity_n", 0)),
    )
    redl_current = np.asarray(redl["current_over_root_fsab2"], dtype=float)
    stress_rho = _stress_rho_from_reference(bootstrap_payload)
    stress_index = int(np.argmin(np.abs(rho_compare - stress_rho)))
    rows: list[dict[str, Any]] = []
    for x_value in x_values:
        for n_order in n_orders:
            start = time.perf_counter()
            closure = _evaluate_neopax_currents(
                NEOPAX,
                species=species,
                field=field,
                database=database,
                neopax_x=int(x_value),
                n_order=int(n_order),
            )
            closure_seconds = float(time.perf_counter() - start)
            total = _interp(
                np.asarray(field.rho_grid, dtype=float),
                np.asarray(closure["current_total_over_root_fsab2"], dtype=float),
                rho_compare,
            )
            nomom = _interp(
                np.asarray(field.rho_grid, dtype=float),
                np.asarray(closure["current_nomom_over_root_fsab2"], dtype=float),
                rho_compare,
            )
            error = _relative_error(redl_current, total)
            rows.append(
                {
                    "neopax_x": int(x_value),
                    "n_order": int(n_order),
                    "x_to_order_ratio": float(x_value / max(n_order, 1)),
                    "stress_rho": float(rho_compare[stress_index]),
                    "stress_relative_error_total_vs_redl": float(error[stress_index]),
                    "stress_current_over_root_fsab2": float(total[stress_index]),
                    "stress_redl_current_over_root_fsab2": float(
                        redl_current[stress_index]
                    ),
                    "max_relative_error_total_vs_redl": float(np.nanmax(error)),
                    "rms_relative_error_total_vs_redl": float(
                        np.sqrt(np.nanmean(error**2))
                    ),
                    "sign_agreement_fraction_total": float(
                        np.mean(np.sign(redl_current) == np.sign(total))
                    ),
                    "nomom_stress_relative_error_vs_redl": float(
                        _relative_error(redl_current, nomom)[stress_index]
                    ),
                    "closure_seconds": closure_seconds,
                }
            )
    metadata = {
        "scan_seconds": scan_seconds,
        "scan_cache": scan_metadata,
        "rho_compare": rho_compare.tolist(),
        "redl_current_over_root_fsab2": redl_current.tolist(),
        "boozmn_path": str(boozmn_path),
        "case": case.as_payload(),
        "inputs": {
            "scan_rho": scan_rho.tolist(),
            "nu_v_count": int(np.asarray(inputs["nu_v"], dtype=float).size),
            "Es": es_values.tolist(),
            "ntx_grid": {
                "n_theta": int(grid.n_theta),
                "n_zeta": int(grid.n_zeta),
                "n_xi": int(grid.n_xi),
            },
            "d33_mode": str(inputs.get("d33_mode", "spitzer")),
            "field_radial_points": int(inputs.get("field_radial_points", 15)),
            "mboz": mboz,
            "nboz": nboz,
            "min_bmn_to_load": float(inputs.get("min_bmn_to_load", 1.0e-5)),
        },
    }
    return rows, metadata


def _row_lookup(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(int(row["neopax_x"]), int(row["n_order"])): row for row in rows}


def _summary_metrics(
    rows: list[dict[str, Any]],
    *,
    bootstrap_payload: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("quadrature audit requires at least one row")
    stress_errors = np.asarray(
        [row["stress_relative_error_total_vs_redl"] for row in rows], dtype=float
    )
    max_errors = np.asarray(
        [row["max_relative_error_total_vs_redl"] for row in rows], dtype=float
    )
    best_stress = rows[int(np.nanargmin(stress_errors))]
    best_max = rows[int(np.nanargmin(max_errors))]
    x_values = sorted({int(row["neopax_x"]) for row in rows})
    n_orders = sorted({int(row["n_order"]) for row in rows})
    lookup = _row_lookup(rows)
    same_order_spreads = []
    for n_order in n_orders:
        subset = [
            lookup[(x_value, n_order)]["stress_relative_error_total_vs_redl"]
            for x_value in x_values
            if (x_value, n_order) in lookup
        ]
        if len(subset) >= 2:
            arr = np.asarray(subset, dtype=float)
            same_order_spreads.append(
                float((np.nanmax(arr) - np.nanmin(arr)) / max(np.nanmin(arr), EPS))
            )
    high_x = x_values[-1]
    high_x_rows = [
        lookup[(high_x, n_order)]
        for n_order in n_orders
        if (high_x, n_order) in lookup
    ]
    high_x_stress = np.asarray(
        [row["stress_relative_error_total_vs_redl"] for row in high_x_rows],
        dtype=float,
    )
    high_x_monotone = bool(
        high_x_stress.size <= 1 or np.all(np.diff(high_x_stress) <= 1.0e-12)
    )
    reference = bootstrap_payload["summary_metrics"]
    reference_error = float(reference["max_relative_error_total_vs_redl_interior"])
    low_quadrature_pass_rows = [
        row
        for row in rows
        if row["stress_relative_error_total_vs_redl"] <= PROFILE_CURRENT_GATE
        and row["x_to_order_ratio"] < 1.0
    ]
    quadrature_stable_pass_rows = [
        row
        for row in rows
        if row["stress_relative_error_total_vs_redl"] <= PROFILE_CURRENT_GATE
        and row["x_to_order_ratio"] >= 1.0
    ]
    return {
        "row_count": int(len(rows)),
        "x_values": x_values,
        "n_orders": n_orders,
        "profile_current_gate": PROFILE_CURRENT_GATE,
        "reference_neopax_x": int(bootstrap_payload["inputs"]["neopax_x"]),
        "reference_n_order": int(bootstrap_payload["inputs"]["n_order"]),
        "reference_stress_relative_error": reference_error,
        "min_stress_relative_error": float(best_stress["stress_relative_error_total_vs_redl"]),
        "min_stress_neopax_x": int(best_stress["neopax_x"]),
        "min_stress_n_order": int(best_stress["n_order"]),
        "min_stress_x_to_order_ratio": float(best_stress["x_to_order_ratio"]),
        "min_stress_gate_pass": bool(
            best_stress["stress_relative_error_total_vs_redl"] <= PROFILE_CURRENT_GATE
        ),
        "min_max_relative_error": float(best_max["max_relative_error_total_vs_redl"]),
        "min_max_neopax_x": int(best_max["neopax_x"]),
        "min_max_n_order": int(best_max["n_order"]),
        "high_x": int(high_x),
        "high_x_stress_error_monotone_nonincreasing_with_pmax": high_x_monotone,
        "high_x_largest_order_stress_relative_error": (
            float(high_x_stress[-1]) if high_x_stress.size else None
        ),
        "max_same_order_stress_spread_over_x": (
            float(np.nanmax(same_order_spreads)) if same_order_spreads else 0.0
        ),
        "underintegrated_gate_pass_count": int(len(low_quadrature_pass_rows)),
        "quadrature_stable_gate_pass_count": int(len(quadrature_stable_pass_rows)),
        "quadrature_stable_current_gate_pass": bool(quadrature_stable_pass_rows),
        "best_stress_pass_rejected_as_underintegrated": bool(
            best_stress["stress_relative_error_total_vs_redl"] <= PROFILE_CURRENT_GATE
            and best_stress["x_to_order_ratio"] < 1.0
        ),
        "quadrature_aliasing_detected": bool(len(low_quadrature_pass_rows) > 0),
        "mean_closure_seconds": float(
            np.nanmean([row["closure_seconds"] for row in rows])
        ),
    }


def build_payload(
    *,
    bootstrap_json: Path = BOOTSTRAP_JSON,
    x_values: tuple[int, ...] = (10, 14, 18),
    n_orders: tuple[int, ...] = (12, 14, 16, 18),
    output_dir: Path = WORKDIR,
    output_prefix: Path = OUTPUT_PREFIX,
) -> dict[str, Any]:
    bootstrap_payload = _load_json(bootstrap_json)
    rows, metadata = _evaluate_rows(
        bootstrap_payload=bootstrap_payload,
        x_values=tuple(int(value) for value in x_values),
        n_orders=tuple(int(value) for value in n_orders),
        output_dir=output_dir,
    )
    metrics = _summary_metrics(rows, bootstrap_payload=bootstrap_payload)
    conclusion = (
        "The finite-beta current gap is sensitive to the coupling between "
        "Sonine order and velocity quadrature.  A low-quadrature setting can "
        "produce an apparent stress-radius pass, but that pass is not accepted "
        "unless it also transfers to higher velocity quadrature.  The current "
        "artifact therefore closes the under-integrated-closure explanation and "
        "keeps finite-beta profile-current parity scoped as an open reduced-"
        "closure lane."
    )
    return _to_jsonable(
        {
            "benchmark": "owned_finite_beta_closure_quadrature_audit",
            "classification": "owned finite-beta closure quadrature audit",
            "claim_scope": (
                "Reuses the owned finite-beta NTX scan and Redl profile contract "
                "while varying only the momentum-closure Sonine order and velocity "
                "quadrature.  It is a numerical/physics consistency audit for the "
                "reduced closure, not a fitted correction and not a bootstrap-"
                "current parity claim."
            ),
            "inputs": {
                "bootstrap_artifact": str(bootstrap_json),
                "x_values": [int(value) for value in x_values],
                "n_orders": [int(value) for value in n_orders],
                "quadrature_note": (
                    "Gauss-Laguerre velocity quadrature must be increased with "
                    "Sonine order; apparent current-gate passes at X < Pmax are "
                    "treated as under-integrated until they transfer to larger X."
                ),
                **metadata["inputs"],
            },
            "rows": rows,
            "metadata": {key: value for key, value in metadata.items() if key != "inputs"},
            "summary_metrics": metrics,
            "conclusion": conclusion,
            "open_work": [
                (
                    "derive or import a quadrature-converged higher-order closure "
                    "before promoting finite-beta bootstrap-current parity"
                ),
                (
                    "require current-gate pass and velocity-quadrature stability "
                    "simultaneously on the finite-beta profile-current stress radius"
                ),
                (
                    "transfer the accepted closure setting to the existing fixed-"
                    "field and integrated W7-X validation artifacts before making "
                    "a broad profile-current claim"
                ),
            ],
            "figure_png": _root_relative(output_prefix.with_suffix(".png")),
            "figure_pdf": _root_relative(output_prefix.with_suffix(".pdf")),
        }
    )


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def _matrix(
    rows: list[dict[str, Any]],
    x_values: list[int],
    n_orders: list[int],
    key: str,
) -> np.ndarray:
    lookup = _row_lookup(rows)
    values = np.full((len(x_values), len(n_orders)), np.nan)
    for ix, x_value in enumerate(x_values):
        for ip, n_order in enumerate(n_orders):
            if (x_value, n_order) in lookup:
                values[ix, ip] = float(lookup[(x_value, n_order)][key])
    return values


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    rows = payload["rows"]
    metrics = payload["summary_metrics"]
    x_values = [int(value) for value in metrics["x_values"]]
    n_orders = [int(value) for value in metrics["n_orders"]]
    stress_matrix = _matrix(
        rows,
        x_values,
        n_orders,
        "stress_relative_error_total_vs_redl",
    )
    max_matrix = _matrix(rows, x_values, n_orders, "max_relative_error_total_vs_redl")
    seconds_matrix = _matrix(rows, x_values, n_orders, "closure_seconds")

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.size": 10.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.1), constrained_layout=True)
    ax_stress, ax_lines, ax_max, ax_runtime = axes.ravel()

    im = ax_stress.imshow(
        stress_matrix,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(float(np.nanmax(stress_matrix)), PROFILE_CURRENT_GATE),
    )
    ax_stress.set_xticks(range(len(n_orders)), labels=[str(value) for value in n_orders])
    ax_stress.set_yticks(range(len(x_values)), labels=[str(value) for value in x_values])
    ax_stress.set_xlabel(r"$P_{max}$")
    ax_stress.set_ylabel("velocity quadrature X")
    ax_stress.set_title("(a) Stress-radius current difference")
    fig.colorbar(im, ax=ax_stress, label="relative difference")
    for ix, x_value in enumerate(x_values):
        for ip, n_order in enumerate(n_orders):
            value = stress_matrix[ix, ip]
            if np.isfinite(value):
                ax_stress.text(
                    ip,
                    ix,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.18 else "black",
                    fontsize=8,
                )
            if x_value < n_order:
                ax_stress.plot(ip, ix, marker="x", color="white", ms=8, mew=1.5)

    for ix, x_value in enumerate(x_values):
        ax_lines.semilogy(
            n_orders,
            stress_matrix[ix],
            marker="o",
            lw=1.8,
            label=f"X={x_value}",
        )
    ax_lines.axhline(PROFILE_CURRENT_GATE, color="0.25", ls="--", lw=1.0)
    ax_lines.set_xlabel(r"$P_{max}$")
    ax_lines.set_ylabel("relative difference")
    ax_lines.set_title("(b) Pmax transfer across quadrature")
    ax_lines.legend(fontsize=8)

    im_max = ax_max.imshow(
        max_matrix,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
    )
    ax_max.set_xticks(range(len(n_orders)), labels=[str(value) for value in n_orders])
    ax_max.set_yticks(range(len(x_values)), labels=[str(value) for value in x_values])
    ax_max.set_xlabel(r"$P_{max}$")
    ax_max.set_ylabel("velocity quadrature X")
    ax_max.set_title("(c) Full-profile max difference")
    fig.colorbar(im_max, ax=ax_max, label="relative difference")

    for ix, x_value in enumerate(x_values):
        ax_runtime.plot(
            n_orders,
            seconds_matrix[ix],
            marker="s",
            lw=1.6,
            label=f"X={x_value}",
        )
    ax_runtime.set_xlabel(r"$P_{max}$")
    ax_runtime.set_ylabel("closure seconds")
    ax_runtime.set_title("(d) Closure runtime")
    ax_runtime.legend(fontsize=8)

    fig.suptitle("Owned finite-beta closure quadrature audit", fontsize=13)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-json", type=Path, default=BOOTSTRAP_JSON)
    parser.add_argument("--x-values", nargs="+", type=int, default=[10, 14, 18])
    parser.add_argument("--n-orders", nargs="+", type=int, default=[12, 14, 16, 18])
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=WORKDIR)
    args = parser.parse_args()

    payload = build_payload(
        bootstrap_json=args.bootstrap_json,
        x_values=tuple(int(value) for value in args.x_values),
        n_orders=tuple(int(value) for value in args.n_orders),
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(json.dumps(payload["summary_metrics"], indent=2))


if __name__ == "__main__":
    main()
