#!/usr/bin/env python3
"""Run a VMEC geometry-family D11/D31/D33 convergence stress diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface  # noqa: E402
from ntx._checkout_paths import (  # noqa: E402
    find_simsopt_root,
    find_stellopt_root,
    find_vmex_root,
    fixture_path,
)
from ntx.config import enable_x64  # noqa: E402
from ntx.solver import prepare_monoenergetic_system, solve_prepared  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "geometry_family_transport_convergence"
COEFFICIENTS = ("D11", "D31", "D33")
REPORTED_COEFFICIENTS = ("D11", "D31", "D13", "D33")
EPS = 1.0e-30
DEFAULT_NU_HAT = 1.0e-3
DEFAULT_ER_HAT = 0.0
DEFAULT_PSI_N = 0.25
DEFAULT_MIN_BMN_TO_LOAD = 1.0e-4
DEFAULT_CONVERGENCE_RTOL = 5.0e-1
DEFAULT_ONSAGER_RELATIVE_RTOL = 5.0e-1

GRID_PRESETS: dict[str, tuple[GridSpec, ...]] = {
    "smoke": (
        GridSpec(5, 5, 4),
        GridSpec(7, 7, 6),
        GridSpec(9, 9, 8),
    ),
    "paper": (
        GridSpec(7, 9, 6),
        GridSpec(9, 11, 8),
        GridSpec(11, 13, 10),
    ),
    "production": (
        GridSpec(29, 31, 28),
        GridSpec(35, 37, 32),
        GridSpec(41, 43, 36),
    ),
}


@dataclass(frozen=True)
class GeometryTransportCase:
    id: str
    label: str
    family: str
    source: str
    path: Path
    psi_n: float = DEFAULT_PSI_N
    min_bmn_to_load: float = DEFAULT_MIN_BMN_TO_LOAD
    notes: str = ""
    promotion_eligible: bool = True

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (13.4, 7.6),
            "figure.dpi": 220,
            "font.size": 10.3,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _add_case(
    cases: list[GeometryTransportCase],
    *,
    root: Path | None,
    relative_path: str,
    case_id: str,
    label: str,
    family: str,
    source: str,
    notes: str,
    promotion_eligible: bool = True,
) -> None:
    if root is None:
        return
    path = root / relative_path
    if path.exists():
        cases.append(
            GeometryTransportCase(
                id=case_id,
                label=label,
                family=family,
                source=source,
                path=path.resolve(),
                notes=notes,
                promotion_eligible=promotion_eligible,
            )
        )


def _add_first_case(
    cases: list[GeometryTransportCase],
    *,
    root: Path | None,
    relative_paths: tuple[str, ...],
    case_id: str,
    label: str,
    family: str,
    source: str,
    notes: str,
    promotion_eligible: bool = True,
) -> None:
    """Add the first available path across supported upstream layouts."""
    if root is None:
        return
    for relative_path in relative_paths:
        if (root / relative_path).exists():
            _add_case(
                cases,
                root=root,
                relative_path=relative_path,
                case_id=case_id,
                label=label,
                family=family,
                source=source,
                notes=notes,
                promotion_eligible=promotion_eligible,
            )
            return


def discover_case_specs(*, include_fixture: bool = False) -> tuple[GeometryTransportCase, ...]:
    """Return reusable local VMEC inputs from the surrounding research stack."""

    cases: list[GeometryTransportCase] = []
    vmex_root = find_vmex_root()
    simsopt_root = find_simsopt_root()
    stellopt_root = find_stellopt_root()

    _add_case(
        cases,
        root=vmex_root,
        relative_path="examples/data/wout_circular_tokamak.nc",
        case_id="circular_tokamak",
        label="Circular tokamak",
        family="tokamak",
        source="vmex examples",
        notes="axisymmetric tokamak baseline",
    )
    _add_case(
        cases,
        root=vmex_root,
        relative_path="examples/data/wout_shaped_tokamak_pressure.nc",
        case_id="shaped_tokamak",
        label="Shaped tokamak",
        family="tokamak",
        source="vmex examples",
        notes="axisymmetric shaped-pressure tokamak baseline",
    )
    _add_first_case(
        cases,
        root=vmex_root,
        relative_paths=(
            "examples/data/wout_nfp2_QA.nc",
            (
                "examples_single_grid/data/"
                "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"
            ),
        ),
        case_id="nfp2_qa",
        label="NFP2 QA",
        family="QA",
        source="vmex examples",
        notes=(
            "near-zero-transform vacuum QA example from the installed vmex "
            "suite; retained as a singular-conditioning diagnostic"
        ),
        promotion_eligible=False,
    )
    _add_case(
        cases,
        root=vmex_root,
        relative_path="examples/data/wout_nfp2_QA_finite_beta.nc",
        case_id="nfp2_qa_finite_beta",
        label="NFP2 QA finite beta",
        family="QA",
        source="vmex examples",
        notes="finite-beta QA example from the installed vmex example suite",
    )
    _add_first_case(
        cases,
        root=vmex_root,
        relative_paths=(
            "examples/data/wout_LandremanPaul2021_QH_reactorScale_lowres.nc",
            (
                "examples_single_grid/data/"
                "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
            ),
        ),
        case_id="precise_qs_qh_reactor",
        label="Precise-QS QH",
        family="QH",
        source="vmex examples",
        notes="precise-QS QH reactor-scale low-resolution reference",
    )
    _add_case(
        cases,
        root=vmex_root,
        relative_path="examples/data/wout_nfp3_QI_fixed_resolution_final.nc",
        case_id="nfp3_qi",
        label="NFP3 QI",
        family="QI",
        source="vmex examples",
        notes="fixed-resolution quasi-isodynamic-style example",
    )
    _add_case(
        cases,
        root=vmex_root,
        relative_path="examples/data/wout_basic_non_stellsym_simsopt.nc",
        case_id="basic_non_stellsym",
        label="Non-stellsym",
        family="non-stellarator-symmetric",
        source="vmex examples",
        notes="non-stellarator-symmetric public example from the local VMEX suite",
    )
    _add_case(
        cases,
        root=simsopt_root,
        relative_path=(
            "tests/test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        ),
        case_id="w7x_eim_ejm_standard",
        label="W7-X EIM/EJM",
        family="W7-X",
        source="SIMSOPT test files",
        notes=(
            "public W7-X standard-configuration reference; local input comments "
            "identify this as EIM geometry corresponding to EJM flux surfaces"
        ),
    )
    _add_case(
        cases,
        root=simsopt_root,
        relative_path=(
            "tests/test_files/"
            "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_"
            "iteratedWithSfincs_reference.nc"
        ),
        case_id="nfp4_qh_reference",
        label="NFP4 QH",
        family="QH",
        source="SIMSOPT test files",
        notes="finite-beta QH reference used in public optimization examples",
    )
    _add_case(
        cases,
        root=simsopt_root,
        relative_path=(
            "tests/test_files/wout_LandremanSengupta2019_section5.4_B2_A80_reference.nc"
        ),
        case_id="high_aspect_qs",
        label="High-aspect QS",
        family="QS",
        source="SIMSOPT test files",
        notes="high-aspect-ratio quasi-symmetric reference",
    )
    _add_case(
        cases,
        root=simsopt_root,
        relative_path=("tests/test_files/wout_LandremanSenguptaPlunk_section5p3_reference.nc"),
        case_id="landreman_sengupta_plunk_non_stellsym",
        label="LSP non-stellsym",
        family="non-stellarator-symmetric",
        source="SIMSOPT test files",
        notes=(
            "section 5.3 Landreman-Sengupta-Plunk public reference; local "
            "README identifies this case as not stellarator symmetric"
        ),
    )
    _add_case(
        cases,
        root=simsopt_root,
        relative_path="tests/test_files/wout_li383_low_res_reference.nc",
        case_id="li383_low_res",
        label="LI383",
        family="NCSX/QS",
        source="SIMSOPT test files",
        notes=("low-resolution LI383 configuration used in public quasisymmetry diagnostic tests"),
    )
    _add_case(
        cases,
        root=simsopt_root,
        relative_path="tests/test_files/wout_n3are_R7.75B5.7_lowres.nc",
        case_id="n3are_lowres",
        label="N3ARE",
        family="stellarator",
        source="SIMSOPT test files",
        notes="public low-resolution three-field-period stellarator-family equilibrium",
    )
    _add_case(
        cases,
        root=stellopt_root,
        relative_path="BENCHMARKS/DIAGNO_TEST/wout_lhd.nc",
        case_id="lhd",
        label="LHD",
        family="heliotron",
        source="STELLOPT benchmarks",
        notes="public LHD diagnostic benchmark equilibrium",
    )
    _add_case(
        cases,
        root=stellopt_root,
        relative_path="BENCHMARKS/DIAGNO_TEST/wout_hsx.nc",
        case_id="hsx_qhs",
        label="HSX",
        family="QHS",
        source="STELLOPT benchmarks",
        notes="public HSX quasi-helically symmetric benchmark equilibrium",
    )
    _add_case(
        cases,
        root=stellopt_root,
        relative_path="BENCHMARKS/DIAGNO_TEST/wout_ncsx.nc",
        case_id="ncsx",
        label="NCSX",
        family="stellarator",
        source="STELLOPT benchmarks",
        notes="public compact-stellarator benchmark equilibrium",
    )
    _add_case(
        cases,
        root=stellopt_root,
        relative_path="BENCHMARKS/DIAGNO_TEST/wout_DIIID_m24n0s99_nfp1.nc",
        case_id="diiid_tokamak",
        label="DIII-D",
        family="tokamak",
        source="STELLOPT benchmarks",
        notes="public axisymmetric DIII-D diagnostic benchmark equilibrium",
    )

    if include_fixture or not cases:
        cases.append(
            GeometryTransportCase(
                id="repo_vmec_fixture",
                label="Repo VMEC fixture",
                family="fixture",
                source="NTX tests",
                path=fixture_path("sample_wout.nc").resolve(),
                notes="small repository-owned VMEC fixture for CI smoke coverage",
            )
        )

    seen: set[str] = set()
    unique_cases: list[GeometryTransportCase] = []
    for case in cases:
        if case.id in seen:
            continue
        seen.add(case.id)
        unique_cases.append(case)
    return tuple(unique_cases)


def _grid_payload(grid: GridSpec) -> dict[str, int]:
    return {
        "n_theta": int(grid.n_theta),
        "n_zeta": int(grid.n_zeta),
        "n_xi": int(grid.n_xi),
    }


def _grid_label(grid: GridSpec) -> str:
    return f"{grid.n_theta}x{grid.n_zeta}x{grid.n_xi}"


def _relative_change(coarse: float, fine: float) -> float:
    return float(abs(coarse - fine) / max(abs(fine), EPS))


def _absolute_change(coarse: float, fine: float) -> float:
    return float(abs(coarse - fine))


def _coefficient_payload(result) -> dict[str, float]:
    return {name: float(getattr(result, name)) for name in REPORTED_COEFFICIENTS}


def _relative_onsager(coefficients: dict[str, float]) -> float:
    d31 = float(coefficients["D31"])
    d13 = float(coefficients["D13"])
    return float(abs(d31 + d13) / max(abs(d31), abs(d13), EPS))


def _solve_case(
    case: GeometryTransportCase,
    *,
    grids: tuple[GridSpec, ...],
    nu_hat: float,
    er_hat: float,
    convergence_rtol: float,
) -> dict[str, object]:
    start = time.perf_counter()
    try:
        surface = load_vmec_surface(
            case.path,
            psi_n=case.psi_n,
            min_bmn_to_load=case.min_bmn_to_load,
        )
    except Exception as exc:  # pragma: no cover - exercised by optional external files.
        return {
            **case.as_payload(),
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }

    load_seconds = time.perf_counter() - start
    grid_results: list[dict[str, object]] = []
    for grid in grids:
        prepare_start = time.perf_counter()
        prepared = prepare_monoenergetic_system(surface, grid)
        prepare_seconds = time.perf_counter() - prepare_start
        solve_start = time.perf_counter()
        result = solve_prepared(prepared, MonoenergeticCase(nu_hat=nu_hat, er_hat=er_hat))
        coefficients = _coefficient_payload(result)
        solve_seconds = time.perf_counter() - solve_start
        finite = bool(
            np.all(np.isfinite([*coefficients.values(), float(result.schur_residual_l2)]))
        )
        relative_onsager = _relative_onsager(coefficients)
        grid_results.append(
            {
                "grid": _grid_payload(grid),
                "grid_label": _grid_label(grid),
                "coefficients": coefficients,
                "schur_residual_l2": float(result.schur_residual_l2),
                "residual_l2": float(result.residual_l2),
                "onsager_residual": float(result.onsager_residual),
                "relative_onsager_residual": float(relative_onsager),
                "D33_spitzer": float(result.D33_spitzer),
                "prepare_seconds": float(prepare_seconds),
                "solve_seconds": float(solve_seconds),
                "finite": finite,
            }
        )

    finest = grid_results[-1]["coefficients"]
    assert isinstance(finest, dict)
    relative_to_finest: list[dict[str, object]] = []
    for grid_result in grid_results[:-1]:
        coefficients = grid_result["coefficients"]
        assert isinstance(coefficients, dict)
        relative_to_finest.append(
            {
                "grid_label": grid_result["grid_label"],
                "relative_change": {
                    name: _relative_change(float(coefficients[name]), float(finest[name]))
                    for name in COEFFICIENTS
                },
            }
        )

    last_relative_change = {
        name: _relative_change(
            float(grid_results[-2]["coefficients"][name]),
            float(grid_results[-1]["coefficients"][name]),
        )
        for name in COEFFICIENTS
    }
    last_absolute_change = {
        name: _absolute_change(
            float(grid_results[-2]["coefficients"][name]),
            float(grid_results[-1]["coefficients"][name]),
        )
        for name in COEFFICIENTS
    }
    max_last_step = max(last_relative_change.values())
    max_to_finest = max(
        (
            max(float(value) for value in item["relative_change"].values())
            for item in relative_to_finest
        ),
        default=0.0,
    )
    all_finite = all(bool(item["finite"]) for item in grid_results)
    max_relative_onsager = max(float(item["relative_onsager_residual"]) for item in grid_results)
    finest_relative_onsager = float(grid_results[-1]["relative_onsager_residual"])
    return {
        **case.as_payload(),
        "status": "stress-pass" if all_finite and max_last_step <= convergence_rtol else "monitor",
        "quality_status": "stress-pass"
        if all_finite and finest_relative_onsager <= DEFAULT_ONSAGER_RELATIVE_RTOL
        else "monitor",
        "surface": {
            "nfp": int(surface.nfp),
            "iota": float(surface.iota),
            "psi_n": float(surface.psi_n),
            "dtype": str(surface.b_cos.dtype),
            "loaded_mode_count": int(surface.loaded_mode_count),
            "total_mode_count": int(surface.total_mode_count),
            "b0": float(surface.b0),
        },
        "load_seconds": float(load_seconds),
        "grid_results": grid_results,
        "relative_to_finest": relative_to_finest,
        "last_step_relative_change": last_relative_change,
        "last_step_absolute_change": last_absolute_change,
        "max_last_step_relative_change": float(max_last_step),
        "max_relative_change_to_finest": float(max_to_finest),
        "max_relative_onsager_residual": float(max_relative_onsager),
        "finest_relative_onsager_residual": float(finest_relative_onsager),
    }


def build_payload(
    *,
    case_specs: tuple[GeometryTransportCase, ...] | None = None,
    grids: tuple[GridSpec, ...] | None = None,
    nu_hat: float = DEFAULT_NU_HAT,
    er_hat: float = DEFAULT_ER_HAT,
    convergence_rtol: float = DEFAULT_CONVERGENCE_RTOL,
    case_limit: int | None = None,
) -> dict[str, object]:
    selected_cases = case_specs if case_specs is not None else discover_case_specs()
    if case_limit is not None and case_limit > 0:
        selected_cases = selected_cases[:case_limit]
    selected_grids = grids if grids is not None else GRID_PRESETS["smoke"]
    if len(selected_grids) < 2:
        raise ValueError("at least two grids are required for convergence diagnostics")
    if any(grid.x64 != selected_grids[0].x64 for grid in selected_grids):
        raise ValueError("all convergence grids must use the same x64 policy")
    enable_x64(selected_grids[0].x64)

    cases = [
        _solve_case(
            case,
            grids=selected_grids,
            nu_hat=nu_hat,
            er_hat=er_hat,
            convergence_rtol=convergence_rtol,
        )
        for case in selected_cases
    ]
    successful_cases = [case for case in cases if case["status"] != "skipped"]
    promotion_cases = [case for case in successful_cases if case["promotion_eligible"]]
    stress_pass_cases = [case for case in successful_cases if case["status"] == "stress-pass"]
    monitored_cases = [case for case in successful_cases if case["status"] == "monitor"]
    quality_pass_cases = [
        case for case in successful_cases if case.get("quality_status") == "stress-pass"
    ]
    quality_monitored_cases = [
        case for case in successful_cases if case.get("quality_status") == "monitor"
    ]
    skipped_cases = [case for case in cases if case["status"] == "skipped"]
    max_last_step = max(
        (float(case["max_last_step_relative_change"]) for case in promotion_cases),
        default=float("nan"),
    )
    max_to_finest = max(
        (float(case["max_relative_change_to_finest"]) for case in promotion_cases),
        default=float("nan"),
    )
    max_residual = max(
        (
            max(float(item["schur_residual_l2"]) for item in case["grid_results"])
            for case in promotion_cases
        ),
        default=float("nan"),
    )
    max_relative_onsager = max(
        (float(case["max_relative_onsager_residual"]) for case in promotion_cases),
        default=float("nan"),
    )
    max_finest_relative_onsager = max(
        (float(case["finest_relative_onsager_residual"]) for case in promotion_cases),
        default=float("nan"),
    )
    return {
        "benchmark": "geometry_family_transport_convergence",
        "classification": "geometry-family monoenergetic transport convergence stress diagnostic",
        "claim_scope": (
            "Runs reusable public VMEC inputs through NTX and reports D11, D31, "
            "and D33 coarse-to-fine changes, with D13 retained for the Onsager "
            "quality check. This is a reduced NTX convergence stress diagnostic, "
            "not an independent-code parity claim."
        ),
        "literature_anchors": [
            "W7-X standard-configuration benchmark workflows",
            "Landreman and Paul 2022 precise quasi-symmetry benchmark family",
            "quasi-isodynamic and omnigenous geometry-family validation literature",
            "VMEC, STELLOPT, and SIMSOPT public equilibrium example suites",
        ],
        "inputs": {
            "nu_hat": float(nu_hat),
            "er_hat": float(er_hat),
            "grids": [_grid_payload(grid) for grid in selected_grids],
            "convergence_rtol": float(convergence_rtol),
            "onsager_relative_rtol": float(DEFAULT_ONSAGER_RELATIVE_RTOL),
            "min_bmn_to_load_default": DEFAULT_MIN_BMN_TO_LOAD,
        },
        "cases": cases,
        "summary_metrics": {
            "case_count": len(cases),
            "successful_case_count": len(successful_cases),
            "promotion_case_count": len(promotion_cases),
            "diagnostic_only_case_count": len(successful_cases) - len(promotion_cases),
            "stress_pass_case_count": len(stress_pass_cases),
            "monitored_case_count": len(monitored_cases),
            "quality_pass_case_count": len(quality_pass_cases),
            "quality_monitored_case_count": len(quality_monitored_cases),
            "skipped_case_count": len(skipped_cases),
            "max_successful_last_step_relative_change": float(max_last_step),
            "max_successful_relative_change_to_finest": float(max_to_finest),
            "max_successful_schur_residual_l2": float(max_residual),
            "max_successful_residual_l2": float(max_residual),
            "max_successful_relative_onsager_residual": float(max_relative_onsager),
            "max_successful_finest_relative_onsager_residual": float(max_finest_relative_onsager),
        },
        "open_work": [
            (
                "resolve the near-zero-transform NFP2 QA conditioning diagnostic "
                "before treating that input as a promoted transport benchmark"
            ),
            (
                "promote only after production-resolution sweeps with independent "
                "reference parity on each family"
            ),
            (
                "add an owned W7-X KJM input once a reusable public reference "
                "input is identified or regenerated"
            ),
            (
                "add radial and collisionality ladders before claiming broad "
                "bootstrap-current-profile validation"
            ),
        ],
        "figure_png": "docs/_static/geometry_family_transport_convergence.png",
        "figure_pdf": "docs/_static/geometry_family_transport_convergence.pdf",
    }


def _case_short_label(case: dict[str, object]) -> str:
    return str(case["label"]).replace(" ", "\n").replace("/", "/\n")


def build_figure(payload: dict[str, object], output_prefix: Path = OUTPUT_PREFIX) -> None:
    _configure_style()
    cases = [case for case in payload["cases"] if case["status"] != "skipped"]
    if not cases:
        raise ValueError("no successful geometry-family transport cases to plot")

    bar_labels = [str(case["label"]) for case in cases]
    heatmap_labels = [_case_short_label(case) for case in cases]
    max_last_step = np.asarray(
        [max(float(case["max_last_step_relative_change"]), 1.0e-16) for case in cases],
        dtype=float,
    )
    heatmap = np.asarray(
        [
            [
                max(float(case["last_step_relative_change"][coefficient]), 1.0e-16)
                for coefficient in COEFFICIENTS
            ]
            for case in cases
        ],
        dtype=float,
    )
    status_colors = {
        "stress-pass": "#2878b5",
        "monitor": "#c85200",
    }
    colors = [status_colors.get(str(case["status"]), "0.45") for case in cases]

    fig, (ax_bar, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(14.2, 7.6),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    positions = np.arange(len(cases))
    bars = ax_bar.bar(positions, max_last_step, color=colors, alpha=0.88, width=0.74)
    ax_bar.set_yscale("log")
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(bar_labels, rotation=52, ha="right", fontsize=7.8)
    ax_bar.set_ylabel("max last-step relative change")
    ax_bar.set_title("(a) D11/D31/D33 convergence stress")
    y_min = max(min(max_last_step) * 0.25, 1.0e-6)
    y_max = max(max(max_last_step) * 8.0, 1.0)
    ax_bar.set_ylim(y_min, y_max)
    for value, label in ((1.0e-1, "1e-1"), (DEFAULT_CONVERGENCE_RTOL, "stress rtol")):
        if y_min < value < y_max:
            ax_bar.axhline(value, color="0.25", linestyle="--", linewidth=0.8, alpha=0.65)
            ax_bar.text(
                len(cases) - 0.25,
                value * 1.08,
                label,
                ha="right",
                va="bottom",
                color="0.25",
                fontsize=8.5,
            )
    for bar, value in zip(bars, max_last_step, strict=True):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.18,
            f"{value:.1e}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            rotation=78,
        )

    log_heatmap = np.log10(heatmap)
    image = ax_heat.imshow(log_heatmap, aspect="auto", cmap="viridis", vmin=-4.0, vmax=1.0)
    ax_heat.set_xticks(np.arange(len(COEFFICIENTS)))
    ax_heat.set_xticklabels([f"${name}$" for name in COEFFICIENTS])
    ax_heat.set_yticks(np.arange(len(cases)))
    ax_heat.set_yticklabels(heatmap_labels)
    ax_heat.set_title("(b) last-step coefficient changes")
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            ax_heat.text(
                col,
                row,
                f"{heatmap[row, col]:.1e}",
                ha="center",
                va="center",
                color="white" if log_heatmap[row, col] < -1.0 else "black",
                fontsize=7.8,
            )
    colorbar = fig.colorbar(image, ax=ax_heat, shrink=0.82, pad=0.02)
    colorbar.set_label(r"$\log_{10}$ relative change")

    summary = payload["summary_metrics"]
    fig.suptitle(
        "VMEC geometry-family NTX transport convergence stress: "
        f"{summary['successful_case_count']} solved cases, "
        f"{summary['stress_pass_case_count']} below stress rtol",
        y=0.98,
        fontsize=13.0,
    )
    fig.text(
        0.5,
        0.01,
        "This figure monitors production-grid convergence across public VMEC examples; "
        "independent-code parity and profile ladders remain separate gates.",
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


def main(
    output_prefix: Path | None = None,
    *,
    case_specs: tuple[GeometryTransportCase, ...] | None = None,
    grids: tuple[GridSpec, ...] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUTPUT_PREFIX,
        help="Output prefix for .png/.pdf/.json artifacts.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(GRID_PRESETS),
        default="smoke",
        help="Grid ladder preset.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=0,
        help="Optional maximum number of discovered cases to run.",
    )
    parser.add_argument(
        "--include-fixture",
        action="store_true",
        help="Include the small repository VMEC fixture even when external examples exist.",
    )
    parser.add_argument("--nu-hat", type=float, default=DEFAULT_NU_HAT)
    parser.add_argument("--er-hat", type=float, default=DEFAULT_ER_HAT)
    parser.add_argument("--convergence-rtol", type=float, default=DEFAULT_CONVERGENCE_RTOL)
    if output_prefix is None:
        args = parser.parse_args()
        prefix = args.output_prefix
        selected_cases = case_specs
        if selected_cases is None:
            selected_cases = discover_case_specs(include_fixture=args.include_fixture)
        selected_grids = grids if grids is not None else GRID_PRESETS[args.preset]
        case_limit = args.case_limit if args.case_limit > 0 else None
        nu_hat = args.nu_hat
        er_hat = args.er_hat
        convergence_rtol = args.convergence_rtol
    else:
        prefix = output_prefix
        selected_cases = case_specs
        selected_grids = grids if grids is not None else GRID_PRESETS["smoke"]
        case_limit = None
        nu_hat = DEFAULT_NU_HAT
        er_hat = DEFAULT_ER_HAT
        convergence_rtol = DEFAULT_CONVERGENCE_RTOL

    payload = build_payload(
        case_specs=selected_cases,
        grids=selected_grids,
        nu_hat=nu_hat,
        er_hat=er_hat,
        convergence_rtol=convergence_rtol,
        case_limit=case_limit,
    )
    write_payload(payload, prefix)
    build_figure(payload, prefix)
    print(f"Wrote {prefix.with_suffix('.png')}")
    print(f"Wrote {prefix.with_suffix('.pdf')}")
    print(f"Wrote {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
