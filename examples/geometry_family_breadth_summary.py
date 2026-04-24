#!/usr/bin/env python3
"""Summarize committed geometry-family derivative artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "geometry_family_breadth_summary"
STATIC = ROOT / "docs" / "_static"
EPS = 1.0e-16


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (13.2, 6.8),
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
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _load_json(stem: str) -> dict[str, Any]:
    return json.loads((STATIC / f"{stem}.json").read_text(encoding="utf-8"))


def _max(values: list[float] | tuple[float, ...]) -> float:
    return float(max(values)) if values else float("nan")


def _case(
    *,
    case_id: str,
    label: str,
    geometry_path: str,
    observable_scope: str,
    max_relative_mismatch: float,
    median_relative_mismatch: float | None,
    status: str,
    artifact: str,
    notes: str,
) -> dict[str, object]:
    return {
        "id": case_id,
        "label": label,
        "geometry_path": geometry_path,
        "observable_scope": observable_scope,
        "max_relative_mismatch": float(max_relative_mismatch),
        "median_relative_mismatch": (
            None if median_relative_mismatch is None else float(median_relative_mismatch)
        ),
        "status": status,
        "artifact": artifact,
        "notes": notes,
    }


def _objective_mismatch(objective: dict[str, Any]) -> float:
    return _max([float(value) for value in objective["relative_mismatch"]])


def build_payload() -> dict[str, object]:
    geometry = _load_json("geometry_control_derivative_benchmark")
    file_backed = _load_json("file_backed_geometry_control_derivative_benchmark")
    boundary = _load_json("boundary_forward_mode_current_derivative_benchmark")
    explicit = _load_json("explicit_relaxed_boundary_current_derivative_benchmark")
    implicit = _load_json("implicit_equilibrium_forward_mode_derivative_benchmark")

    active_cases: list[dict[str, object]] = [
        _case(
            case_id="analytic_geometry_control",
            label="Analytic control",
            geometry_path="owned analytic Boozer surface",
            observable_scope="D11, D31, D33",
            max_relative_mismatch=geometry["summary_metrics"]["max_relative_mismatch"],
            median_relative_mismatch=geometry["summary_metrics"][
                "median_relative_mismatch"
            ],
            status="stress-pass",
            artifact="docs/_static/geometry_control_derivative_benchmark.json",
            notes="direct geometry-control autodiff versus centered finite differences",
        )
    ]

    for case in file_backed["cases"]:
        active_cases.append(
            _case(
                case_id=f"file_backed_{case['id']}",
                label=str(case["label"]),
                geometry_path=f"file-backed {case['source_kind']} surface",
                observable_scope="D11, D31, D33",
                max_relative_mismatch=case["summary_metrics"][
                    "max_relative_mismatch"
                ],
                median_relative_mismatch=case["summary_metrics"][
                    "median_relative_mismatch"
                ],
                status="stress-pass",
                artifact=(
                    "docs/_static/"
                    "file_backed_geometry_control_derivative_benchmark.json"
                ),
                notes="file-backed geometry-control autodiff versus centered finite differences",
            )
        )

    active_cases.append(
        _case(
            case_id="boundary_projected_current",
            label="Projected boundary",
            geometry_path="boundary-projected VMEC/Boozer path",
            observable_scope="NTX proxy and integrated current",
            max_relative_mismatch=boundary["summary_metrics"]["max_relative_mismatch"],
            median_relative_mismatch=boundary["summary_metrics"][
                "median_relative_mismatch"
            ],
            status="stress-pass",
            artifact=(
                "docs/_static/"
                "boundary_forward_mode_current_derivative_benchmark.json"
            ),
            notes="forward-mode derivatives through the projected boundary path",
        )
    )

    for case in explicit["cases"]:
        active_cases.append(
            _case(
                case_id=f"explicit_relaxed_{case['id']}",
                label=str(case["label"]),
                geometry_path="explicit-relaxed fixed-boundary path",
                observable_scope="Boozer scalar, NTX proxy, and integrated current",
                max_relative_mismatch=case["summary_metrics"][
                    "max_relative_mismatch"
                ],
                median_relative_mismatch=case["summary_metrics"][
                    "median_relative_mismatch"
                ],
                status="stress-pass",
                artifact=(
                    "docs/_static/"
                    "explicit_relaxed_boundary_current_derivative_benchmark.json"
                ),
                notes="self-consistent forward-mode derivatives on committed QA/QH cases",
            )
        )

    implicit_objectives = {objective["id"]: objective for objective in implicit["objectives"]}
    volume = implicit_objectives["equilibrium_volume"]
    active_cases.append(
        _case(
            case_id="implicit_equilibrium_volume",
            label="Implicit volume",
            geometry_path="implicit fixed-boundary equilibrium path",
            observable_scope="equilibrium volume",
            max_relative_mismatch=_objective_mismatch(volume),
            median_relative_mismatch=_objective_mismatch(volume),
            status=str(volume["status"]),
            artifact=(
                "docs/_static/"
                "implicit_equilibrium_forward_mode_derivative_benchmark.json"
            ),
            notes=(
                "scalar volume probe only; implicit surface/transport derivatives "
                "are closed as non-shipping diagnostics"
            ),
        )
    )

    open_cases: list[dict[str, object]] = []
    retired_cases: list[dict[str, object]] = []
    for objective_id, label in (
        ("booz_xform_scalar", "Implicit Boozer"),
        ("ntx_transport_proxy", "Implicit NTX"),
    ):
        objective = implicit_objectives[objective_id]
        retired_cases.append(
            _case(
                case_id=f"implicit_{objective_id}",
                label=label,
                geometry_path="implicit fixed-boundary equilibrium path",
                observable_scope=str(objective_id),
                max_relative_mismatch=_objective_mismatch(objective),
                median_relative_mismatch=None,
                status=str(objective["status"]),
                artifact=(
                    "docs/_static/"
                    "implicit_equilibrium_forward_mode_derivative_benchmark.json"
                ),
                notes=(
                    "closed as non-shipping because residual contraction and "
                    "surface/transport tangent parity do not pass"
                ),
            )
        )

    active_values = [float(case["max_relative_mismatch"]) for case in active_cases]
    open_values = [float(case["max_relative_mismatch"]) for case in open_cases]
    retired_values = [float(case["max_relative_mismatch"]) for case in retired_cases]
    return {
        "benchmark": "geometry_family_breadth_summary",
        "classification": "artifact-backed geometry-breadth stress summary",
        "claim_scope": (
            "Summarizes committed analytic, file-backed, boundary-projected, "
            "explicit-relaxed, and implicit-equilibrium diagnostic artifacts. "
            "This is broader than a single surface, but it is not a full "
            "hidden-symmetry, omnigenous, or broad W7-X/QI validation claim."
        ),
        "literature_anchors": [
            "Paul et al. 2019 adjoint neoclassical optimization",
            "McGreivy 2024 differentiable programming for plasma workflows",
            "Landreman and Paul 2022 precise-QS benchmark family",
            "Plunk, Landreman, and Helander 2019 omnigenous near-axis construction",
            "Rodriguez, Plunk, and Jorge 2025 quasi-isodynamic near-axis construction",
            (
                "Calvo, Velasco, Helander, and Parra 2025 "
                "piecewise-omnigenous low-bootstrap-current motivation"
            ),
        ],
        "source_artifacts": [
            "docs/_static/geometry_control_derivative_benchmark.json",
            "docs/_static/file_backed_geometry_control_derivative_benchmark.json",
            "docs/_static/boundary_forward_mode_current_derivative_benchmark.json",
            "docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.json",
            "docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.json",
        ],
        "active_cases": active_cases,
        "open_cases": open_cases,
        "retired_cases": retired_cases,
        "summary_metrics": {
            "active_case_count": len(active_cases),
            "open_case_count": len(open_cases),
            "retired_case_count": len(retired_cases),
            "max_active_relative_mismatch": _max(active_values),
            "median_active_relative_mismatch": float(np.median(active_values)),
            "max_open_relative_mismatch": _max(open_values) if open_values else 0.0,
            "max_retired_relative_mismatch": _max(retired_values),
            "implicit_validated_objective_count": 1,
            "implicit_open_objective_count": len(open_cases),
            "implicit_retired_objective_count": len(retired_cases),
        },
        "open_work": [
            "broaden committed cases to reusable W7-X EIM/KJM, QI, and omnigenous inputs",
            (
                "add direct D11/D31/D33 parity and convergence ladders before "
                "promoting a full geometry-family claim"
            ),
            (
                "restore implicit-equilibrium derivatives only after residual "
                "contraction and surface/transport tangent parity pass"
            ),
        ],
        "figure_png": "docs/_static/geometry_family_breadth_summary.png",
        "figure_pdf": "docs/_static/geometry_family_breadth_summary.pdf",
    }


def _plot_bars(ax, cases: list[dict[str, object]], *, title: str, color: str) -> None:
    labels = [str(case["label"]).replace(" ", "\n") for case in cases]
    values = np.asarray(
        [max(float(case["max_relative_mismatch"]), EPS) for case in cases],
        dtype=float,
    )
    positions = np.arange(len(cases))
    bars = ax.bar(positions, values, color=color, alpha=0.88, width=0.72)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("max relative AD/FD mismatch")
    y_min = max(values.min() * 0.2, 1.0e-8)
    y_max = max(values.max() * 6.0, 1.0e-3)
    ax.set_ylim(y_min, y_max)
    for reference, label in ((1.0e-4, "1e-4"), (5.0e-4, "5e-4")):
        if not y_min < reference < y_max:
            continue
        ax.axhline(reference, color="0.35", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.text(
            len(cases) - 0.25,
            reference * 1.08,
            label,
            ha="right",
            va="bottom",
            color="0.35",
            fontsize=8.5,
        )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.18,
            f"{value:.1e}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            rotation=80,
        )


def build_figure(payload: dict[str, object], output_prefix: Path = OUTPUT_PREFIX) -> None:
    _configure_style()
    active_cases = list(payload["active_cases"])
    retired_cases = list(payload["retired_cases"])
    fig, (ax_active, ax_open) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.8),
        gridspec_kw={"width_ratios": [2.7, 1.0]},
    )
    _plot_bars(
        ax_active,
        active_cases,
        title="(a) Artifact-backed geometry derivative paths",
        color="#2878b5",
    )
    _plot_bars(
        ax_open,
        retired_cases,
        title="(b) Retired implicit diagnostics",
        color="#c85200",
    )
    summary = payload["summary_metrics"]
    fig.suptitle(
        "Geometry-family derivative breadth summary: "
        f"{summary['active_case_count']} active stress cases, "
        f"{summary['retired_case_count']} retired implicit diagnostics",
        y=0.98,
        fontsize=13.0,
    )
    fig.text(
        0.5,
        0.01,
        "Bars summarize committed JSON artifacts only; broad hidden-symmetry, "
        "omnigenous, and reusable W7-X/QI validation remains planned work.",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="0.25",
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.94))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def write_payload(payload: dict[str, object], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main(output_prefix: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUTPUT_PREFIX,
        help="Output prefix for .png/.pdf/.json artifacts.",
    )
    if output_prefix is None:
        args = parser.parse_args()
        prefix = args.output_prefix
    else:
        prefix = output_prefix
    payload = build_payload()
    write_payload(payload, prefix)
    build_figure(payload, prefix)
    print(f"Wrote {prefix.with_suffix('.png')}")
    print(f"Wrote {prefix.with_suffix('.pdf')}")
    print(f"Wrote {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
