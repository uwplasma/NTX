#!/usr/bin/env python3
"""Audit finite-beta same-grid coefficient sensitivity to resolution.

This diagnostic consumes completed SFINCS-JAX/NTX same-grid artifacts for the
finite-beta QA stress point.  It asks whether the remaining coefficient
difference is removed by increasing angular/pitch resolution or by tightening
the VMEC harmonic cutoff before the bootstrap-current residual is assigned to
the reduced profile-current closure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_finite_beta_sfincs_jax_resolution_audit"
SMOKE_JSON = ROOT / "docs" / "_static" / "owned_finite_beta_sfincs_jax_inputs.json"
PRODUCTION_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_sfincs_jax_production_probe.json"
)
PRODUCTION_TIGHT_HARMONICS_JSON = (
    ROOT
    / "docs"
    / "_static"
    / "owned_finite_beta_sfincs_jax_production_probe_minbmn.json"
)
CONDITIONING_JSON = (
    ROOT / "docs" / "_static" / "owned_finite_beta_current_conditioning_audit.json"
)
DEFAULT_RHO = 1.0 / 7.0
DEFAULT_NU_PRIME = 1.0e-2
DEFAULT_E_STAR = 0.0
EPS = 1.0e-30


@dataclass(frozen=True)
class ProbeRow:
    label: str
    source: str
    rho: float
    nu_prime: float
    e_star: float
    ntheta: int
    nzeta: int
    nxi: int
    solver_tolerance: float
    min_bmn_to_load: float
    seconds: float | None
    status: str
    l13_relative_difference: float
    l31_relative_difference: float
    l33_relative_difference: float
    l33_spitzer_relative_difference: float
    max_transport_relative_difference: float

    def as_payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "source": self.source,
            "rho": self.rho,
            "nu_prime": self.nu_prime,
            "e_star": self.e_star,
            "grid": {
                "Ntheta": self.ntheta,
                "Nzeta": self.nzeta,
                "Nxi": self.nxi,
                "Nx": 1,
            },
            "solverTolerance": self.solver_tolerance,
            "min_Bmn_to_load": self.min_bmn_to_load,
            "seconds": self.seconds,
            "status": self.status,
            "relative_difference": {
                "L13_bridge_vs_sfincs": self.l13_relative_difference,
                "L31_bridge_vs_sfincs": self.l31_relative_difference,
                "L33_bridge_vs_sfincs": self.l33_relative_difference,
                "L33_spitzer_bridge_vs_sfincs": self.l33_spitzer_relative_difference,
            },
            "max_transport_relative_difference": self.max_transport_relative_difference,
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _isclose(a: float, b: float, *, atol: float = 5.0e-7) -> bool:
    return abs(float(a) - float(b)) <= atol


def _extract_row(
    label: str,
    path: Path,
    *,
    rho: float,
    nu_prime: float,
    e_star: float,
) -> ProbeRow:
    payload = _load_json(path)
    candidates: list[dict[str, Any]] = []
    for deck in payload.get("decks", []):
        summary = deck.get("transport_summary")
        if not isinstance(summary, dict) or summary.get("status") != "complete":
            continue
        same_grid = summary.get("ntx_same_grid")
        if not isinstance(same_grid, dict) or same_grid.get("status") != "complete":
            continue
        if not _isclose(float(deck["rho"]), rho):
            continue
        if not _isclose(float(deck["nu_prime"]), nu_prime):
            continue
        if not _isclose(float(deck["e_star"]), e_star):
            continue
        candidates.append(deck)
    if not candidates:
        raise ValueError(
            f"{path} has no completed same-grid deck for "
            f"rho={rho:.8g}, nuPrime={nu_prime:.8g}, EStar={e_star:.8g}"
        )
    deck = candidates[0]
    summary = deck["transport_summary"]
    same_grid = summary["ntx_same_grid"]
    rel = same_grid["relative_difference"]
    channel_values = [
        float(rel["L13_bridge_vs_sfincs"]),
        float(rel["L31_bridge_vs_sfincs"]),
        float(rel["L33_bridge_vs_sfincs"]),
    ]
    grid = payload["inputs"]["grid"]
    return ProbeRow(
        label=label,
        source=str(Path(path)),
        rho=float(deck["rho"]),
        nu_prime=float(deck["nu_prime"]),
        e_star=float(deck["e_star"]),
        ntheta=int(grid["Ntheta"]),
        nzeta=int(grid["Nzeta"]),
        nxi=int(grid["Nxi"]),
        solver_tolerance=float(payload["inputs"]["solverTolerance"]),
        min_bmn_to_load=float(payload["inputs"]["min_Bmn_to_load"]),
        seconds=None if deck.get("seconds") is None else float(deck["seconds"]),
        status=str(deck["status"]),
        l13_relative_difference=float(rel["L13_bridge_vs_sfincs"]),
        l31_relative_difference=float(rel["L31_bridge_vs_sfincs"]),
        l33_relative_difference=float(rel["L33_bridge_vs_sfincs"]),
        l33_spitzer_relative_difference=float(rel["L33_spitzer_bridge_vs_sfincs"]),
        max_transport_relative_difference=float(np.max(channel_values)),
    )


def build_payload(
    *,
    smoke_json: Path = SMOKE_JSON,
    production_json: Path = PRODUCTION_JSON,
    production_tight_harmonics_json: Path = PRODUCTION_TIGHT_HARMONICS_JSON,
    conditioning_json: Path = CONDITIONING_JSON,
    rho: float = DEFAULT_RHO,
    nu_prime: float = DEFAULT_NU_PRIME,
    e_star: float = DEFAULT_E_STAR,
) -> dict[str, object]:
    """Build the coefficient-resolution audit payload."""

    rows = [
        _extract_row("smoke grid", smoke_json, rho=rho, nu_prime=nu_prime, e_star=e_star),
        _extract_row(
            "production grid",
            production_json,
            rho=rho,
            nu_prime=nu_prime,
            e_star=e_star,
        ),
        _extract_row(
            "production grid, tight harmonics",
            production_tight_harmonics_json,
            rho=rho,
            nu_prime=nu_prime,
            e_star=e_star,
        ),
    ]
    conditioning = _load_json(conditioning_json)
    stress = conditioning["stress_radius"]
    required = float(stress["required_coefficient_relative_difference_for_current_gate"])
    smoke = rows[0].max_transport_relative_difference
    production = rows[1].max_transport_relative_difference
    tight = rows[2].max_transport_relative_difference
    return {
        "benchmark": "owned_finite_beta_sfincs_jax_resolution_audit",
        "classification": "finite-beta same-grid coefficient resolution stress audit",
        "claim_scope": (
            "Compares the finite-beta QA stress-radius same-grid transport "
            "coefficients after increasing angular/pitch resolution and after "
            "tightening the VMEC harmonic cutoff. This is a diagnostic for "
            "coefficient normalization and numerical convergence; it is not a "
            "bootstrap-current parity claim."
        ),
        "inputs": {
            "rho": float(rho),
            "nuPrime": float(nu_prime),
            "EStar": float(e_star),
            "smoke_json": str(Path(smoke_json)),
            "production_json": str(Path(production_json)),
            "production_tight_harmonics_json": str(Path(production_tight_harmonics_json)),
            "conditioning_json": str(Path(conditioning_json)),
            "current_gate_relative_error": 1.0e-1,
        },
        "rows": [row.as_payload() for row in rows],
        "summary_metrics": {
            "smoke_max_transport_relative_difference": smoke,
            "production_max_transport_relative_difference": production,
            "production_tight_harmonics_max_transport_relative_difference": tight,
            "production_change_vs_smoke": (production - smoke)
            / max(abs(smoke), EPS),
            "tight_harmonics_change_vs_production": (tight - production)
            / max(abs(production), EPS),
            "required_coefficient_relative_difference_for_current_gate": required,
            "production_precision_gap_to_current_gate": production / max(required, EPS),
            "tight_harmonics_precision_gap_to_current_gate": tight / max(required, EPS),
        },
        "conclusion": (
            "The stress-radius coefficient floor is insensitive to the tested "
            "angular/pitch-resolution increase and to the tighter VMEC harmonic "
            "cutoff. The remaining finite-beta current gap should therefore stay "
            "classified as a monitored profile-current/reduced-closure stress "
            "until a profile-current closure diagnostic, not a scalar fit, closes it."
        ),
        "open_work": [
            (
                "run the same production-grid stress probe at the neighboring "
                "finite-beta radii used by the profile-current observable audit"
            ),
            (
                "isolate the profile-current closure by applying the same "
                "profile drivers directly to the completed same-grid transport "
                "coefficients"
            ),
            (
                "promote finite-beta current parity only after the coefficient "
                "floor and profile-current observable both clear the current "
                "conditioning threshold"
            ),
        ],
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
    }


def write_payload(payload: dict[str, object], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def build_figure(payload: dict[str, object], output_prefix: Path = OUTPUT_PREFIX) -> None:
    rows = payload["rows"]
    metrics = payload["summary_metrics"]
    labels = [
        f"{str(row['label']).replace('production grid, ', '')}\n"
        f"{row['grid']['Ntheta']}x{row['grid']['Nzeta']}x{row['grid']['Nxi']}"
        for row in rows
    ]
    x = np.arange(len(labels), dtype=float)
    width = 0.22
    channels = (
        ("L13", "L13_bridge_vs_sfincs", "#0072b2"),
        ("L31", "L31_bridge_vs_sfincs", "#009e73"),
        ("L33", "L33_bridge_vs_sfincs", "#d55e00"),
    )

    fig, (ax_rel, ax_gap) = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
    for offset, (label, key, color) in zip((-width, 0.0, width), channels, strict=True):
        values = [float(row["relative_difference"][key]) for row in rows]
        ax_rel.bar(x + offset, values, width=width, label=label, color=color)
    required = float(metrics["required_coefficient_relative_difference_for_current_gate"])
    ax_rel.axhline(required, color="#111111", ls="--", lw=1.5, label="current gate need")
    ax_rel.set_yscale("log")
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(labels)
    ax_rel.set_ylabel("same-grid relative difference")
    ax_rel.set_title("(a) Coefficient channels")
    ax_rel.grid(axis="y", alpha=0.25)
    ax_rel.legend(fontsize=8)

    gaps = [
        float(row["max_transport_relative_difference"]) / max(required, EPS)
        for row in rows
    ]
    ax_gap.bar(labels, gaps, color=["#56b4e9", "#0072b2", "#009e73"])
    ax_gap.axhline(1.0, color="#111111", ls="--", lw=1.5)
    ax_gap.set_yscale("log")
    ax_gap.set_ylabel("max coefficient diff / required diff")
    ax_gap.set_title("(b) Current-conditioned precision gap")
    ax_gap.grid(axis="y", alpha=0.25)

    fig.suptitle("Finite-beta same-grid coefficient resolution audit", fontsize=13)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-json", type=Path, default=SMOKE_JSON)
    parser.add_argument("--production-json", type=Path, default=PRODUCTION_JSON)
    parser.add_argument(
        "--production-tight-harmonics-json",
        type=Path,
        default=PRODUCTION_TIGHT_HARMONICS_JSON,
    )
    parser.add_argument("--conditioning-json", type=Path, default=CONDITIONING_JSON)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--nu-prime", type=float, default=DEFAULT_NU_PRIME)
    parser.add_argument("--e-star", type=float, default=DEFAULT_E_STAR)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    args = parser.parse_args()

    payload = build_payload(
        smoke_json=args.smoke_json,
        production_json=args.production_json,
        production_tight_harmonics_json=args.production_tight_harmonics_json,
        conditioning_json=args.conditioning_json,
        rho=float(args.rho),
        nu_prime=float(args.nu_prime),
        e_star=float(args.e_star),
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(
        f"wrote {args.output_prefix.with_suffix('.json')}, "
        f"{args.output_prefix.with_suffix('.png')}, and "
        f"{args.output_prefix.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()
