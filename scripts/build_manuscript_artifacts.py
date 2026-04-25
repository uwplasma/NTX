#!/usr/bin/env python3
"""Build manuscript tables and reproducibility metadata from NTX artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def _format_float(value: float, scientific: bool = False) -> str:
    if scientific:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _format_optional_float(value: float | None, scientific: bool = False) -> str:
    if value is None:
        return "unsupported"
    return _format_float(value, scientific=scientific)


def build_payload() -> dict:
    from ntx.validation.benchmark_matrix import benchmark_matrix_payload

    monoenergetic = _load_json(STATIC / "validation_summary.json")
    fixed_field = _load_json(STATIC / "bootstrap_current_fixed_field_validation.json")
    w7x = _load_json(STATIC / "bootstrap_current_reference_audit_w7x.json")
    derivative = _load_json(STATIC / "derivative_path_benchmark.json")
    geometry_derivative = _load_json(STATIC / "geometry_control_derivative_benchmark.json")
    file_backed_geometry_derivative = _load_json(
        STATIC / "file_backed_geometry_control_derivative_benchmark.json"
    )
    boundary_forward_mode = _load_json(
        STATIC / "boundary_forward_mode_current_derivative_benchmark.json"
    )
    implicit_equilibrium_forward_mode = _load_json(
        STATIC / "implicit_equilibrium_forward_mode_derivative_benchmark.json"
    )
    explicit_relaxed_boundary = _load_json(
        STATIC / "explicit_relaxed_boundary_current_derivative_benchmark.json"
    )
    geometry_family_breadth = _load_json(
        STATIC / "geometry_family_breadth_summary.json"
    )
    geometry_family_transport = _load_json(
        STATIC / "geometry_family_transport_convergence.json"
    )
    profile_uncertainty = _load_json(STATIC / "autodiff_profile_uncertainty.json")
    science = _load_json(STATIC / "bootstrap_current_optimization.json")
    cpu = _load_json(STATIC / "performance_scaling_cpu_heavy.json")
    gpu = _load_json(STATIC / "performance_scaling_gpu_heavy.json")
    production_performance = _load_json(STATIC / "performance_scaling_production.json")
    strong_performance = _load_json(STATIC / "performance_strong_scaling_production.json")
    prepared_geometry_reuse = _load_json(
        STATIC / "prepared_geometry_reuse_profile.json"
    )
    figures = _load_json(STATIC / "publication_figure_manifest.json")
    main_text = [
        "validation",
        "closure_validation",
        "w7x_audit",
        "derivative_benchmark",
        "science",
        "performance_production",
        "primitive_transport",
    ]
    supplement = [
        "inverse",
        "profiles",
        "profile_uncertainty",
        "geometry_derivative",
        "file_backed_geometry_derivative",
        "boundary_forward_mode",
        "implicit_equilibrium_forward_mode",
        "boundary_explicit_relaxed",
        "geometry_family_breadth",
        "geometry_family_transport",
        "ambipolar",
        "ambipolar_family",
        "profile_reconstruction",
        "profile_control",
        "profile_basis",
        "profile_transport",
        "bootstrap_proxy",
        "robust_science",
        "performance_smoke",
        "performance_heavy",
        "performance_strong",
        "prepared_geometry_reuse",
    ]
    monoenergetic_metrics = monoenergetic["summary_metrics"]
    fixed_field_case_errors = {
        case_id: case["max_relative_error_vs_sfincs_interior"]
        for case_id, case in fixed_field["cases"].items()
    }
    fixed_field_redl_error = max(
        float(errors["Redl"]) for errors in fixed_field_case_errors.values()
    )
    fixed_field_ntx_neopax_error = max(
        float(errors["NTX+NEOPAX"]) for errors in fixed_field_case_errors.values()
    )
    fine_w7x_error = w7x["bootstrap_current_errors"][-1]["max_relative_error"]
    cpu_best = max(
        row["multiprocess_speedup_vs_serial"] for row in cpu["results"]
    )
    gpu_best = max(
        row["multiprocess_speedup_vs_serial"] for row in gpu["results"]
    )
    implicit_objective_map = {
        objective["id"]: objective for objective in implicit_equilibrium_forward_mode["objectives"]
    }

    return {
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "tables": {
            "monoenergetic_validation": {
                "grid": monoenergetic["grid"],
                "nu_hat": monoenergetic["nu_hat"],
                "er_hat": monoenergetic["er_hat"],
                "summary_metrics": monoenergetic_metrics,
                "literature_anchors": monoenergetic["literature_anchors"],
                "tail_loglog_slopes": {
                    case_id: curves["tail_loglog_slopes"]
                    for case_id, curves in monoenergetic["transport_curves"].items()
                },
            },
            "fixed_field_validation": {
                "case_errors": fixed_field_case_errors,
                "redl_max_interior_relative_error": fixed_field_redl_error,
                "ntx_neopax_max_interior_relative_error": fixed_field_ntx_neopax_error,
                "sfincs_jax_sample_count": fixed_field["sfincs_jax_sample_count"],
                "ntx_neopax_radial_points": fixed_field["ntx_neopax_radial_points"],
                "ntx_neopax_n_order": fixed_field.get("ntx_neopax_n_order"),
                "ntx_neopax_d33_mode": fixed_field.get("ntx_neopax_d33_mode"),
            },
            "validation": {
                "bootstrap_current_reference_scale": w7x["bootstrap_current_reference_scale"],
                "bootstrap_current_errors": w7x["bootstrap_current_errors"],
            },
            "derivatives": {
                "grid": derivative["grid"],
                "nu_hat": derivative["nu_hat"],
                "er_min": derivative["er_min"],
                "er_max": derivative["er_max"],
                "max_relative_mismatch": max(derivative["max_relative_mismatch"]),
                "best_prepared_speedup": max(derivative["speedup_prepared_vs_direct"]),
            },
            "geometry_control_derivatives": {
                "grid": geometry_derivative["grid"],
                "control_modes": geometry_derivative["control_modes"],
                "coefficients": geometry_derivative["coefficients"],
                "summary_metrics": geometry_derivative["summary_metrics"],
                "claim_scope": geometry_derivative["claim_scope"],
            },
            "file_backed_geometry_control_derivatives": {
                "cases": file_backed_geometry_derivative["cases"],
                "summary_metrics": file_backed_geometry_derivative["summary_metrics"],
                "claim_scope": file_backed_geometry_derivative["claim_scope"],
            },
            "boundary_forward_mode_current_derivatives": {
                "case": boundary_forward_mode["case"],
                "summary_metrics": boundary_forward_mode["summary_metrics"],
                "claim_scope": boundary_forward_mode["claim_scope"],
                "objectives": boundary_forward_mode["objectives"],
            },
            "implicit_equilibrium_forward_mode_derivatives": {
                "case": implicit_equilibrium_forward_mode["case"],
                "implicit_solver": implicit_equilibrium_forward_mode["implicit_solver"],
                "summary_metrics": implicit_equilibrium_forward_mode["summary_metrics"],
                "claim_scope": implicit_equilibrium_forward_mode["claim_scope"],
                "objectives": implicit_equilibrium_forward_mode["objectives"],
                "reverse_mode_diagnostic": implicit_equilibrium_forward_mode[
                    "reverse_mode_diagnostic"
                ],
            },
            "explicit_relaxed_boundary_current_derivatives": {
                "cases": explicit_relaxed_boundary["cases"],
                "equilibrium_relaxation": explicit_relaxed_boundary["equilibrium_relaxation"],
                "summary_metrics": explicit_relaxed_boundary["summary_metrics"],
                "claim_scope": explicit_relaxed_boundary["claim_scope"],
                "objective_ids": explicit_relaxed_boundary["objective_ids"],
            },
            "geometry_family_breadth": {
                "active_cases": geometry_family_breadth["active_cases"],
                "open_cases": geometry_family_breadth["open_cases"],
                "retired_cases": geometry_family_breadth.get("retired_cases", []),
                "summary_metrics": geometry_family_breadth["summary_metrics"],
                "claim_scope": geometry_family_breadth["claim_scope"],
                "open_work": geometry_family_breadth["open_work"],
            },
            "geometry_family_transport": {
                "cases": geometry_family_transport["cases"],
                "summary_metrics": geometry_family_transport["summary_metrics"],
                "claim_scope": geometry_family_transport["claim_scope"],
                "inputs": geometry_family_transport["inputs"],
                "open_work": geometry_family_transport["open_work"],
            },
            "profile_uncertainty": {
                "basis_size": profile_uncertainty["basis_size"],
                "sample_count": profile_uncertainty["sample_count"],
                "parameter_std": profile_uncertainty["parameter_std"],
                "max_std_relative_mismatch": profile_uncertainty[
                    "max_std_relative_mismatch"
                ],
                "max_mean_relative_shift": profile_uncertainty[
                    "max_mean_relative_shift"
                ],
                "fisher_eigenvalues": profile_uncertainty["fisher_eigenvalues"],
                "hessian_probe_relative_error": profile_uncertainty[
                    "hessian_probe_relative_error"
                ],
            },
            "performance": {
                "cpu_heavy": cpu,
                "gpu_heavy": gpu,
                "production": production_performance,
                "strong_scaling": strong_performance,
                "prepared_geometry_reuse": prepared_geometry_reuse,
            },
            "science": {
                "wout": science["wout"],
                "harmonic_m": science["harmonic_m"],
                "harmonic_n": science["harmonic_n"],
                "baseline_scale": science["baseline_scale"],
                "optimized_scale": science["optimized_scale"],
                "weighted_gain": science["weighted_gain"],
                "serial_scan_seconds": science["serial_scan_seconds"],
                "parallel_scan_seconds": science["parallel_scan_seconds"],
            },
        },
        "claims": {
            "monoenergetic_dkes_finest_plotted_error": monoenergetic_metrics[
                "dkes_finest_plotted_error"
            ],
            "monoenergetic_vmec_finest_plotted_error": monoenergetic_metrics[
                "vmec_finest_plotted_error"
            ],
            "monoenergetic_dkes_max_onsager_relative": monoenergetic_metrics[
                "dkes_max_onsager_relative"
            ],
            "monoenergetic_vmec_max_onsager_relative": monoenergetic_metrics[
                "vmec_max_onsager_relative"
            ],
            "precise_qs_redl_max_interior_relative_error": fixed_field_redl_error,
            "precise_qs_ntx_neopax_max_interior_relative_error": (
                fixed_field_ntx_neopax_error
            ),
            "precise_qs_ntx_neopax_n_order": fixed_field.get("ntx_neopax_n_order"),
            "precise_qs_ntx_neopax_d33_mode": fixed_field.get("ntx_neopax_d33_mode"),
            "w7x_fine_grid_max_relative_error": fine_w7x_error,
            "derivative_max_relative_mismatch": max(derivative["max_relative_mismatch"]),
            "best_prepared_derivative_speedup": max(derivative["speedup_prepared_vs_direct"]),
            "geometry_control_derivative_max_relative_mismatch": geometry_derivative[
                "summary_metrics"
            ]["max_relative_mismatch"],
            "geometry_control_derivative_median_relative_mismatch": geometry_derivative[
                "summary_metrics"
            ]["median_relative_mismatch"],
            "file_backed_geometry_control_derivative_max_relative_mismatch": (
                file_backed_geometry_derivative["summary_metrics"]["max_relative_mismatch"]
            ),
            "file_backed_geometry_control_derivative_median_relative_mismatch": (
                file_backed_geometry_derivative["summary_metrics"]["median_relative_mismatch"]
            ),
            "boundary_forward_mode_current_derivative_max_relative_mismatch": (
                boundary_forward_mode["summary_metrics"]["max_relative_mismatch"]
            ),
            "boundary_forward_mode_current_derivative_median_relative_mismatch": (
                boundary_forward_mode["summary_metrics"]["median_relative_mismatch"]
            ),
            "implicit_equilibrium_forward_mode_derivative_max_relative_mismatch": (
                implicit_equilibrium_forward_mode["summary_metrics"]["max_relative_mismatch"]
            ),
            "implicit_equilibrium_forward_mode_derivative_median_relative_mismatch": (
                implicit_equilibrium_forward_mode["summary_metrics"]["median_relative_mismatch"]
            ),
            "implicit_equilibrium_volume_relative_mismatch": (
                implicit_objective_map["equilibrium_volume"]["relative_mismatch"][0]
            ),
            "implicit_equilibrium_booz_relative_mismatch": (
                implicit_objective_map["booz_xform_scalar"]["relative_mismatch"][0]
            ),
            "implicit_equilibrium_transport_relative_mismatch": (
                implicit_objective_map["ntx_transport_proxy"]["relative_mismatch"][0]
            ),
            "implicit_equilibrium_reverse_mode_booz_max_relative_mismatch": (
                implicit_equilibrium_forward_mode["reverse_mode_diagnostic"][
                    "max_relative_mismatch"
                ]
            ),
            "explicit_relaxed_boundary_current_derivative_max_relative_mismatch": (
                explicit_relaxed_boundary["summary_metrics"]["max_relative_mismatch"]
            ),
            "explicit_relaxed_boundary_current_derivative_median_relative_mismatch": (
                explicit_relaxed_boundary["summary_metrics"]["median_relative_mismatch"]
            ),
            "explicit_relaxed_boundary_current_volume_relative_difference": (
                explicit_relaxed_boundary["summary_metrics"][
                    "max_ordinary_explicit_volume_relative_difference"
                ]
            ),
            "geometry_family_breadth_active_case_count": (
                geometry_family_breadth["summary_metrics"]["active_case_count"]
            ),
            "geometry_family_breadth_max_active_relative_mismatch": (
                geometry_family_breadth["summary_metrics"][
                    "max_active_relative_mismatch"
                ]
            ),
            "geometry_family_breadth_max_open_relative_mismatch": (
                geometry_family_breadth["summary_metrics"]["max_open_relative_mismatch"]
            ),
            "geometry_family_breadth_max_retired_relative_mismatch": (
                geometry_family_breadth["summary_metrics"].get(
                    "max_retired_relative_mismatch",
                    0.0,
                )
            ),
            "geometry_family_transport_successful_case_count": (
                geometry_family_transport["summary_metrics"]["successful_case_count"]
            ),
            "geometry_family_transport_stress_pass_case_count": (
                geometry_family_transport["summary_metrics"]["stress_pass_case_count"]
            ),
            "geometry_family_transport_max_last_step_relative_change": (
                geometry_family_transport["summary_metrics"][
                    "max_successful_last_step_relative_change"
                ]
            ),
            "profile_uncertainty_basis_size": profile_uncertainty["basis_size"],
            "profile_uncertainty_sample_count": profile_uncertainty["sample_count"],
            "profile_uncertainty_max_std_relative_mismatch": (
                profile_uncertainty["max_std_relative_mismatch"]
            ),
            "profile_uncertainty_max_mean_relative_shift": (
                profile_uncertainty["max_mean_relative_shift"]
            ),
            "profile_uncertainty_min_fisher_eigenvalue": min(
                profile_uncertainty["fisher_eigenvalues"]
            ),
            "profile_uncertainty_max_fisher_eigenvalue": max(
                profile_uncertainty["fisher_eigenvalues"]
            ),
            "profile_uncertainty_hessian_probe_relative_error": (
                profile_uncertainty["hessian_probe_relative_error"]
            ),
            "bootstrap_current_weighted_gain": science["weighted_gain"],
            "cpu_heavy_best_multiprocess_speedup": cpu_best,
            "gpu_heavy_best_multiprocess_speedup": gpu_best,
            "gpu_heavy_healthy_device_count": gpu["healthy_parallel_device_count"],
            "cpu_production_best_device_parallel_speedup": (
                production_performance["cpu"]["best_device_parallel_speedup_vs_serial"]
            ),
            "cpu_production_device_parallel_crossover_cases": (
                production_performance["cpu"]["device_parallel_crossover_cases"]
            ),
            "cpu_production_best_multiprocess_speedup": (
                production_performance["cpu"]["best_multiprocess_speedup_vs_serial"]
            ),
            "gpu_production_best_device_parallel_speedup": (
                production_performance["gpu"]["best_device_parallel_speedup_vs_serial"]
            ),
            "gpu_production_best_multiprocess_speedup": (
                production_performance["gpu"]["best_multiprocess_speedup_vs_serial"]
            ),
            "gpu_production_healthy_device_count": (
                production_performance["gpu"]["healthy_parallel_device_count"]
            ),
            "cpu_strong_best_device_parallel_speedup": (
                strong_performance["cpu"]["best_device_parallel_speedup_vs_serial"]
            ),
            "cpu_strong_best_multiprocess_speedup": (
                strong_performance["cpu"]["best_multiprocess_speedup_vs_serial"]
            ),
            "gpu_strong_best_device_parallel_speedup": (
                strong_performance["gpu"]["best_device_parallel_speedup_vs_serial"]
            ),
            "gpu_strong_best_multiprocess_speedup": (
                strong_performance["gpu"]["best_multiprocess_speedup_vs_serial"]
            ),
            "gpu_strong_healthy_device_count": (
                strong_performance["gpu"]["healthy_parallel_device_count"]
            ),
            "prepared_geometry_reuse_best_compiled_steady_speedup": (
                prepared_geometry_reuse["summary_metrics"][
                    "best_compiled_steady_speedup_vs_direct"
                ]
            ),
            "prepared_geometry_reuse_max_compiled_relative_mismatch": (
                prepared_geometry_reuse["summary_metrics"]["max_compiled_relative_mismatch"]
            ),
        },
        "figures": figures,
        "benchmark_matrix": benchmark_matrix_payload(ROOT),
        "figure_sets": {
            "main_text": main_text,
            "supplement": supplement,
        },
        "commands": {
            "figure_bundle": (
                "python examples/make_publication_figures.py "
                "--figures main_text,supplement"
            ),
            "main_text_figures": (
                "python examples/make_publication_figures.py --figures main_text"
            ),
            "supplement_figures": (
                "python examples/make_publication_figures.py --figures supplement"
            ),
            "tables": "python scripts/build_manuscript_artifacts.py",
            "benchmark_matrix": "python scripts/build_benchmark_matrix.py",
            "validation_subset": (
                "python -m pytest -q "
                "tests/test_w7x_reference_benchmark.py "
                "tests/test_derivative_path_benchmark_example.py "
                "tests/test_bootstrap_current_optimization_example.py "
                "tests/test_manuscript_artifacts_script.py "
                "tests/test_make_publication_figures.py -k "
                "\"subset_writes_manifest or bootstrap_subset_writes_manifest\""
            ),
        },
    }


def build_markdown(payload: dict) -> str:
    validation_rows = payload["tables"]["validation"]["bootstrap_current_errors"]
    monoenergetic_validation = payload["tables"]["monoenergetic_validation"]
    fixed_field_validation = payload["tables"]["fixed_field_validation"]
    cpu_rows = payload["tables"]["performance"]["cpu_heavy"]["results"]
    gpu_rows = payload["tables"]["performance"]["gpu_heavy"]["results"]
    prepared_reuse = payload["tables"]["performance"]["prepared_geometry_reuse"]
    science = payload["tables"]["science"]
    derivatives = payload["tables"]["derivatives"]
    geometry_derivatives = payload["tables"]["geometry_control_derivatives"]
    file_backed_geometry_derivatives = payload["tables"]["file_backed_geometry_control_derivatives"]
    boundary_forward_mode_derivatives = payload["tables"][
        "boundary_forward_mode_current_derivatives"
    ]
    implicit_equilibrium_forward_mode_derivatives = payload["tables"][
        "implicit_equilibrium_forward_mode_derivatives"
    ]
    explicit_relaxed_boundary_derivatives = payload["tables"][
        "explicit_relaxed_boundary_current_derivatives"
    ]
    geometry_family_breadth = payload["tables"]["geometry_family_breadth"]
    geometry_family_transport = payload["tables"]["geometry_family_transport"]
    profile_uncertainty = payload["tables"]["profile_uncertainty"]
    file_backed_max_mismatch = file_backed_geometry_derivatives["summary_metrics"][
        "max_relative_mismatch"
    ]
    file_backed_median_mismatch = file_backed_geometry_derivatives["summary_metrics"][
        "median_relative_mismatch"
    ]
    boundary_forward_max_mismatch = boundary_forward_mode_derivatives["summary_metrics"][
        "max_relative_mismatch"
    ]
    boundary_forward_median_mismatch = boundary_forward_mode_derivatives["summary_metrics"][
        "median_relative_mismatch"
    ]
    implicit_forward_max_mismatch = implicit_equilibrium_forward_mode_derivatives[
        "summary_metrics"
    ]["max_relative_mismatch"]
    implicit_forward_median_mismatch = implicit_equilibrium_forward_mode_derivatives[
        "summary_metrics"
    ]["median_relative_mismatch"]
    implicit_reverse_diagnostic = implicit_equilibrium_forward_mode_derivatives[
        "reverse_mode_diagnostic"
    ]
    implicit_reverse_max_mismatch = implicit_reverse_diagnostic["max_relative_mismatch"]
    implicit_objective_map = {
        objective["id"]: objective for objective in implicit_equilibrium_forward_mode_derivatives[
            "objectives"
        ]
    }
    explicit_relaxed_max_iter = explicit_relaxed_boundary_derivatives["equilibrium_relaxation"][
        "max_iter"
    ]
    explicit_relaxed_step_size = explicit_relaxed_boundary_derivatives[
        "equilibrium_relaxation"
    ]["step_size"]
    explicit_relaxed_case_ids = [
        case["id"] for case in explicit_relaxed_boundary_derivatives["cases"]
    ]
    explicit_relaxed_volume_difference = max(
        case["volume_metrics"]["ordinary_explicit_relative_difference"]
        for case in explicit_relaxed_boundary_derivatives["cases"]
    )
    explicit_relaxed_max_mismatch = explicit_relaxed_boundary_derivatives["summary_metrics"][
        "max_relative_mismatch"
    ]
    explicit_relaxed_median_mismatch = explicit_relaxed_boundary_derivatives[
        "summary_metrics"
    ]["median_relative_mismatch"]
    geometry_family_active = geometry_family_breadth["active_cases"]
    geometry_family_open = geometry_family_breadth["open_cases"]
    geometry_family_retired = geometry_family_breadth.get("retired_cases", [])
    geometry_family_metrics = geometry_family_breadth["summary_metrics"]
    geometry_transport_cases = [
        case for case in geometry_family_transport["cases"] if case["status"] != "skipped"
    ]
    geometry_transport_metrics = geometry_family_transport["summary_metrics"]
    implicit_solver = implicit_equilibrium_forward_mode_derivatives["implicit_solver"]
    implicit_solver_text = (
        f"`iter={implicit_solver['max_iter']}, "
        f"step={implicit_solver['step_size']:.1f}, "
        f"tangent={implicit_solver['residual_tangent_mode']}` |"
    )
    mono_metrics = monoenergetic_validation["summary_metrics"]
    fixed_field_case_errors = fixed_field_validation["case_errors"]
    benchmark_rows = payload["benchmark_matrix"]["entries"]

    lines = [
        "# NTX Manuscript Tables",
        "",
        "## Validation",
        "",
        "| Grid `(N_theta, N_zeta, N_xi)` | Max relative error |",
        "| --- | ---: |",
    ]
    for row in validation_rows:
        grid = tuple(row["grid"])
        lines.append(f"| `{grid}` | {_format_float(row['max_relative_error'], scientific=True)} |")

    lines.extend(
        [
            "",
            "## Monoenergetic Validation Summary",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Grid | `{tuple(monoenergetic_validation['grid'].values())}` |",
            (
                "| DKES-style finest plotted `N_xi` error | "
                f"`{mono_metrics['dkes_finest_plotted_error']:.3e}` |"
            ),
            (
                "| VMEC finest plotted `N_xi` error | "
                f"`{mono_metrics['vmec_finest_plotted_error']:.3e}` |"
            ),
            (
                "| DKES-style max Onsager residual | "
                f"`{mono_metrics['dkes_max_onsager_relative']:.3e}` |"
            ),
            (
                "| VMEC monitored max Onsager residual | "
                f"`{mono_metrics['vmec_max_onsager_relative']:.3e}` |"
            ),
            "",
            "## Fixed-Field Precise-QS Benchmark",
            "",
            "| Case | Redl/SFINCS interior error | NTX+NEOPAX/SFINCS interior stress |",
            "| --- | ---: | ---: |",
        ]
    )
    for case_id, errors in sorted(fixed_field_case_errors.items()):
        lines.append(
            f"| `{case_id}` | `{float(errors['Redl']):.3e}` | "
            f"`{float(errors['NTX+NEOPAX']):.3e}` |"
        )

    lines.extend(
        [
            "",
            "## Benchmark Matrix",
            "",
            "| Benchmark | Lane | Maturity | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in benchmark_rows:
        entry = row["entry"]
        lines.append(
            f"| `{entry['id']}` | `{entry['lane']}` | "
            f"`{entry['maturity']}` | `{row['status']}` |"
        )

    lines.extend(
        [
            "",
            "## Derivatives",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Grid | `{tuple(derivatives['grid'].values())}` |",
            f"| `nu_hat` | `{derivatives['nu_hat']:.3e}` |",
            f"| `E_r` scan | `{derivatives['er_min']:.3e}` to `{derivatives['er_max']:.3e}` |",
            f"| Max relative mismatch | `{derivatives['max_relative_mismatch']:.3e}` |",
            f"| Best prepared speedup | `{derivatives['best_prepared_speedup']:.3f}x` |",
            "",
            "## Geometry-Control Derivatives",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Grid | `{tuple(geometry_derivatives['grid'].values())}` |",
            f"| Controlled modes | `{len(geometry_derivatives['control_modes'])}` |",
            f"| Coefficients | `{', '.join(geometry_derivatives['coefficients'])}` |",
            (
                "| Max AD/centered-FD mismatch | "
                f"`{geometry_derivatives['summary_metrics']['max_relative_mismatch']:.3e}` |"
            ),
            (
                "| Median AD/centered-FD mismatch | "
                f"`{geometry_derivatives['summary_metrics']['median_relative_mismatch']:.3e}` |"
            ),
            "",
            "## Boundary Forward-Mode Current Derivatives",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            (
                "| Controlled parameters | "
                f"`{', '.join(boundary_forward_mode_derivatives['case']['parameter_names'])}` |"
            ),
            (
                "| Max AD/centered-FD mismatch | "
                f"`{boundary_forward_max_mismatch:.3e}` |"
            ),
            (
                "| Median AD/centered-FD mismatch | "
                f"`{boundary_forward_median_mismatch:.3e}` |"
            ),
            "",
            "## Implicit-Equilibrium Forward-Mode Derivatives",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            (
                "| Controlled parameters | "
                "`"
                + ", ".join(
                    implicit_equilibrium_forward_mode_derivatives["case"]["parameter_names"]
                )
                + "` |"
            ),
            (
                "| Implicit solver | "
                f"{implicit_solver_text}"
            ),
            (
                "| Max AD/centered-FD mismatch | "
                f"`{implicit_forward_max_mismatch:.3e}` |"
            ),
            (
                "| Median AD/centered-FD mismatch | "
                f"`{implicit_forward_median_mismatch:.3e}` |"
            ),
            (
                "| Reverse-mode Boozer max mismatch | "
                "`"
                + _format_optional_float(implicit_reverse_max_mismatch, scientific=True)
                + "` |"
            ),
            (
                "| Reverse-mode Boozer status | "
                f"`{implicit_reverse_diagnostic['status']}` |"
            ),
            (
                "| Equilibrium-volume mismatch | "
                f"`{implicit_objective_map['equilibrium_volume']['relative_mismatch'][0]:.3e}` |"
            ),
            (
                "| Boozer-scalar mismatch | "
                f"`{implicit_objective_map['booz_xform_scalar']['relative_mismatch'][0]:.3e}` |"
            ),
            (
                "| NTX transport mismatch | "
                f"`{implicit_objective_map['ntx_transport_proxy']['relative_mismatch'][0]:.3e}` |"
            ),
            "",
            "## Explicit-Relaxed Boundary Current Derivatives",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            (
                "| Cases | "
                f"`{', '.join(explicit_relaxed_case_ids)}` |"
            ),
            (
                "| Explicit relaxation | "
                f"`iter={explicit_relaxed_max_iter}, step={explicit_relaxed_step_size:.1e}` |"
            ),
            (
                "| Ordinary/explicit volume rel. diff. | "
                f"`{explicit_relaxed_volume_difference:.3e}` |"
            ),
            (
                "| Max AD/centered-FD mismatch | "
                f"`{explicit_relaxed_max_mismatch:.3e}` |"
            ),
            (
                "| Median AD/centered-FD mismatch | "
                f"`{explicit_relaxed_median_mismatch:.3e}` |"
            ),
            "",
            "## File-Backed Geometry-Control Derivatives",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            (
                "| Cases | `"
                + ", ".join(case["id"] for case in file_backed_geometry_derivatives["cases"])
                + "` |"
            ),
            (
                "| Max AD/centered-FD mismatch | "
                f"`{file_backed_max_mismatch:.3e}` |"
            ),
            (
                "| Median AD/centered-FD mismatch | "
                f"`{file_backed_median_mismatch:.3e}` |"
            ),
            "",
            "## Geometry-Family Breadth Summary",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            (
                "| Active artifact-backed cases | "
                f"`{geometry_family_metrics['active_case_count']}` |"
            ),
            (
                "| Open implicit objectives | "
                f"`{geometry_family_metrics['open_case_count']}` |"
            ),
            (
                "| Retired implicit diagnostics | "
                f"`{geometry_family_metrics.get('retired_case_count', 0)}` |"
            ),
            (
                "| Active case ids | `"
                + ", ".join(case["id"] for case in geometry_family_active)
                + "` |"
            ),
            (
                "| Open case ids | `"
                + ", ".join(case["id"] for case in geometry_family_open)
                + "` |"
            ),
            (
                "| Retired implicit ids | `"
                + ", ".join(case["id"] for case in geometry_family_retired)
                + "` |"
            ),
            (
                "| Max active AD/centered-FD mismatch | "
                f"`{geometry_family_metrics['max_active_relative_mismatch']:.3e}` |"
            ),
            (
                "| Max retired implicit mismatch | "
                f"`{geometry_family_metrics.get('max_retired_relative_mismatch', 0.0):.3e}` |"
            ),
            "",
            "## Geometry-Family Transport Convergence",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            (
                "| Solved VMEC cases | "
                f"`{geometry_transport_metrics['successful_case_count']}` |"
            ),
            (
                "| Below smoke convergence rtol | "
                f"`{geometry_transport_metrics['stress_pass_case_count']}` |"
            ),
            (
                "| Max last-step relative change | "
                f"`{geometry_transport_metrics['max_successful_last_step_relative_change']:.3e}` |"
            ),
            (
                "| Max relative change to finest grid | "
                f"`{geometry_transport_metrics['max_successful_relative_change_to_finest']:.3e}` |"
            ),
            (
                "| Solved case ids | `"
                + ", ".join(case["id"] for case in geometry_transport_cases)
                + "` |"
            ),
            "",
            "## Profile Uncertainty",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Radial electric-field basis size | `{profile_uncertainty['basis_size']}` |",
            f"| Monte Carlo samples | `{profile_uncertainty['sample_count']}` |",
            (
                "| Max linearized/Monte-Carlo std mismatch | "
                f"`{profile_uncertainty['max_std_relative_mismatch']:.3e}` |"
            ),
            (
                "| Max Monte-Carlo mean shift | "
                f"`{profile_uncertainty['max_mean_relative_shift']:.3e}` |"
            ),
            (
                "| Fisher eigenvalue range | "
                f"`{min(profile_uncertainty['fisher_eigenvalues']):.3e}` to "
                f"`{max(profile_uncertainty['fisher_eigenvalues']):.3e}` |"
            ),
            (
                "| Hessian-vector/Fisher probe mismatch | "
                f"`{profile_uncertainty['hessian_probe_relative_error']:.3e}` |"
            ),
            "",
            "## Bootstrap-Current Optimization",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Harmonic `(m, n)` | `({science['harmonic_m']}, {science['harmonic_n']})` |",
            f"| Baseline scale | `{science['baseline_scale']:.3f}` |",
            f"| Optimized scale | `{science['optimized_scale']:.3f}` |",
            f"| Weighted current gain | `{science['weighted_gain']:.3f}x` |",
            f"| Serial scan time | `{science['serial_scan_seconds']:.3f} s` |",
            f"| Parallel scan time | `{science['parallel_scan_seconds']:.3f} s` |",
            "",
            "## Performance",
            "",
            "### CPU heavy-grid scaling",
            "",
            "| Cases | Serial [s] | Multiprocess [s] | Speedup |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cpu_rows:
        lines.append(
            f"| {row['num_cases']} | {_format_float(row['serial_seconds'])} | "
            f"{_format_float(row['multiprocess_seconds'])} | "
            f"{_format_float(row['multiprocess_speedup_vs_serial'])}x |"
        )

    lines.extend(
        [
            "",
            "### GPU heavy-grid scaling",
            "",
            "| Cases | Serial [s] | Multiprocess [s] | Speedup | Healthy devices |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in gpu_rows:
        lines.append(
            f"| {row['num_cases']} | {_format_float(row['serial_seconds'])} | "
            f"{_format_float(row['multiprocess_seconds'])} | "
            f"{_format_float(row['multiprocess_speedup_vs_serial'])}x | "
            f"{payload['tables']['performance']['gpu_heavy']['healthy_parallel_device_count']} |"
        )

    lines.extend(
        [
            "",
            "### Prepared-geometry reuse",
            "",
            (
                "| Cases | Direct [s] | Prepared total [s] | Compiled steady [s] | "
                "Compiled speedup |"
            ),
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in prepared_reuse["results"]:
        lines.append(
            f"| {row['num_cases']} | {_format_float(row['direct_seconds'])} | "
            f"{_format_float(row['prepared_total_seconds'])} | "
            f"{_format_float(row['compiled_steady_seconds'])} | "
            f"{_format_float(row['compiled_steady_speedup_vs_direct'])}x |"
        )

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "| Key | Value |",
            "| --- | --- |",
            f"| Commit | `{payload['git']['commit']}` |",
            f"| Branch | `{payload['git']['branch']}` |",
            f"| Python | `{payload['environment']['python']}` |",
            f"| JAX | `{payload['environment']['jax']}` |",
            f"| NumPy | `{payload['environment']['numpy']}` |",
            f"| Platform | `{payload['environment']['platform']}` |",
            f"| Figure bundle | `{payload['commands']['figure_bundle']}` |",
            f"| Main-text figures | `{payload['commands']['main_text_figures']}` |",
            f"| Supplement figures | `{payload['commands']['supplement_figures']}` |",
            f"| Artifact tables | `{payload['commands']['tables']}` |",
            f"| Benchmark matrix | `{payload['commands']['benchmark_matrix']}` |",
            f"| Validation subset | `{payload['commands']['validation_subset']}` |",
            "",
        ]
    )
    return "\n".join(lines)


def build_claims_markdown(payload: dict) -> str:
    claims = payload["claims"]
    explicit_relaxed_max_mismatch = claims[
        "explicit_relaxed_boundary_current_derivative_max_relative_mismatch"
    ]
    explicit_relaxed_median_mismatch = claims[
        "explicit_relaxed_boundary_current_derivative_median_relative_mismatch"
    ]
    explicit_relaxed_volume_difference = claims[
        "explicit_relaxed_boundary_current_volume_relative_difference"
    ]
    return "\n".join(
        [
            "# NTX Manuscript Claims",
            "",
            "These are the current paper-facing technical claims derived directly from the",
            "validated NTX artifacts.",
            "",
            (
                "- The monoenergetic validation-summary gate keeps the committed "
                "DKES-style and VMEC finest plotted `N_xi` convergence errors at "
                f"`{claims['monoenergetic_dkes_finest_plotted_error']:.3e}` and "
                f"`{claims['monoenergetic_vmec_finest_plotted_error']:.3e}`; "
                "the DKES-style max Onsager residual is "
                f"`{claims['monoenergetic_dkes_max_onsager_relative']:.3e}`, "
                "while the VMEC Onsager residual is retained as a monitored "
                "finite-resolution stress metric at "
                f"`{claims['monoenergetic_vmec_max_onsager_relative']:.3e}`."
            ),
            (
                "- The fixed-field precise-QS benchmark keeps the Redl/SFINCS "
                "interior maximum relative error at "
                f"`{claims['precise_qs_redl_max_interior_relative_error']:.3e}`; "
                "the corresponding `NTX+NEOPAX` total-current closure stress "
                "comparison uses "
                f"`d33_mode={claims['precise_qs_ntx_neopax_d33_mode']}` and "
                f"`n_order={claims['precise_qs_ntx_neopax_n_order']}`, reaching "
                f"`{claims['precise_qs_ntx_neopax_max_interior_relative_error']:.3e}`, "
                "while species-current parity remains out of scope."
            ),
            (
                "- W7-X imported-workflow bootstrap-current convergence reaches "
                "a maximum relative error of "
                f"`{claims['w7x_fine_grid_max_relative_error']:.3e}` on the fine "
                "`25 x 25 x 64` grid."
            ),
            (
                "- The prepared implicit-adjoint derivative path matches direct "
                "reverse-mode with a maximum relative mismatch of "
                f"`{claims['derivative_max_relative_mismatch']:.3e}` on the "
                "committed derivative benchmark."
            ),
            (
                "- The prepared derivative path reaches a best observed speedup "
                "of "
                f"`{claims['best_prepared_derivative_speedup']:.3f}x` on the "
                "benchmarked electric-field scan."
            ),
            (
                "- The three-harmonic geometry-control derivative stress "
                "benchmark matches centered finite differences with a maximum "
                "relative mismatch of "
                f"`{claims['geometry_control_derivative_max_relative_mismatch']:.3e}` "
                "and a median mismatch of "
                f"`{claims['geometry_control_derivative_median_relative_mismatch']:.3e}`."
            ),
            (
                "- The file-backed Boozer and VMEC geometry-control derivative "
                "stress benchmark matches centered finite differences with a "
                "maximum relative mismatch of "
                f"`{claims['file_backed_geometry_control_derivative_max_relative_mismatch']:.3e}` "
                "and a median mismatch of "
                f"`{claims['file_backed_geometry_control_derivative_median_relative_mismatch']:.3e}`."
            ),
            (
                "- The boundary-projected `vmec_jax -> booz_xform_jax -> NTX` "
                "and `NTX+NEOPAX` forward-mode stress benchmark matches centered "
                "finite differences with a maximum relative mismatch of "
                f"`{claims['boundary_forward_mode_current_derivative_max_relative_mismatch']:.3e}` "
                "and a median mismatch of "
                f"`{claims['boundary_forward_mode_current_derivative_median_relative_mismatch']:.3e}`."
            ),
            (
                "- The implicit fixed-boundary `vmec_jax -> booz_xform_jax -> NTX` "
                "diagnostic is closed as non-shipping on the committed QA case: "
                "the equilibrium-volume "
                "derivative matches centered finite differences with relative mismatch "
                f"`{claims['implicit_equilibrium_volume_relative_mismatch']:.3e}`, "
                "while the Boozer scalar and NTX transport observables fail the "
                "surface/transport parity contract at "
                f"`{claims['implicit_equilibrium_booz_relative_mismatch']:.3e}` and "
                f"`{claims['implicit_equilibrium_transport_relative_mismatch']:.3e}`."
            ),
            (
                "- The matching reverse-mode Boozer-scalar diagnostic on the "
                "non-shipping implicit-equilibrium diagnostic remains unavailable because the "
                "current JAX transform rejects the implicit dynamic-loop solve "
                "on that path."
            ),
            (
                "- The explicit-relaxed `vmec_jax -> booz_xform_jax -> NTX` "
                "and `NTX+NEOPAX` boundary-to-current QA/QH stress benchmark matches "
                "centered finite differences with a maximum relative mismatch of "
                f"`{explicit_relaxed_max_mismatch:.3e}` "
                "and a median mismatch of "
                f"`{explicit_relaxed_median_mismatch:.3e}`, "
                "while the ordinary and explicit-relaxed primal volumes agree "
                "to "
                f"`{explicit_relaxed_volume_difference:.3e}` "
                "on the committed QA/QH family cases."
            ),
            (
                "- The artifact-backed geometry-family breadth summary now covers "
                f"`{claims['geometry_family_breadth_active_case_count']}` active "
                "analytic, file-backed, boundary-projected, explicit-relaxed, "
                "and implicit-volume stress cases with maximum active mismatch "
                f"`{claims['geometry_family_breadth_max_active_relative_mismatch']:.3e}`. "
                "The implicit Boozer and NTX transport objectives are closed as "
                "non-shipping diagnostics with maximum mismatch "
                f"`{claims['geometry_family_breadth_max_retired_relative_mismatch']:.3e}` "
                "and are excluded from promoted geometry-family claims."
            ),
            (
                "- The geometry-family transport convergence stress diagnostic "
                f"solves `{claims['geometry_family_transport_successful_case_count']}` "
                "public VMEC-family cases, with "
                f"`{claims['geometry_family_transport_stress_pass_case_count']}` "
                "below the smoke-grid convergence tolerance and maximum last-step "
                "relative D11/D31/D33 change "
                f"`{claims['geometry_family_transport_max_last_step_relative_change']:.3e}`. "
                "It is a reduced NTX convergence diagnostic, not an "
                "independent-code parity claim."
            ),
            (
                "- The profile uncertainty stress benchmark now uses a "
                f"`{claims['profile_uncertainty_basis_size']}`-term radial "
                "electric-field basis and "
                f"`{claims['profile_uncertainty_sample_count']}` Monte Carlo "
                "samples; the local combined-residual Hessian-vector probe "
                "matches the Fisher/Gauss-Newton product to relative error "
                f"`{claims['profile_uncertainty_hessian_probe_relative_error']:.3e}`. "
                "The current D33-only propagated-standard-deviation comparison "
                "is retained as a stress metric at "
                f"`{claims['profile_uncertainty_max_std_relative_mismatch']:.3e}` "
                "rather than a promoted profile-UQ claim."
            ),
            (
                "- The differentiable bootstrap-current optimization example "
                "improves the weighted current proxy by "
                f"`{claims['bootstrap_current_weighted_gain']:.3f}x` on the "
                "committed W7-X study."
            ),
            (
                "- On the heavy CPU benchmark, multiprocess execution reaches "
                "a best observed speedup of "
                f"`{claims['cpu_heavy_best_multiprocess_speedup']:.3f}x`."
            ),
            (
                "- On the heavy GPU benchmark, the current multiprocess path "
                "reaches a best observed speedup of "
                f"`{claims['gpu_heavy_best_multiprocess_speedup']:.3f}x` with "
                f"`{claims['gpu_heavy_healthy_device_count']}` healthy parallel "
                "GPU device(s), so the current paper should frame GPU "
                "multiprocess as a characterized execution mode rather than a "
                "throughput win."
            ),
            (
                "- The production-grid CPU performance map shows the "
                "single-process device-parallel lane crossing serial at "
                f"`{claims['cpu_production_device_parallel_crossover_cases']}` "
                "cases and reaching a best observed speedup of "
                f"`{claims['cpu_production_best_device_parallel_speedup']:.3f}x`; "
                "the same production-grid multiprocess lane remains below "
                f"`{claims['cpu_production_best_multiprocess_speedup']:.3f}x`."
            ),
            (
                "- The production-grid GPU map uses "
                f"`{claims['gpu_production_healthy_device_count']}` healthy "
                "parallel GPU device(s) on the tested two-GPU workstation; "
                "device-parallel timing is characterized, but multiprocess "
                "throughput remains below serial at "
                f"`{claims['gpu_production_best_multiprocess_speedup']:.3f}x`."
            ),
            (
                "- The fixed-workload CPU strong-scaling map reaches best "
                "observed device-parallel and multiprocess speedups of "
                f"`{claims['cpu_strong_best_device_parallel_speedup']:.3f}x` "
                "and "
                f"`{claims['cpu_strong_best_multiprocess_speedup']:.3f}x`, "
                "respectively."
            ),
            (
                "- The fixed-workload GPU strong-scaling map reports "
                f"`{claims['gpu_strong_healthy_device_count']}` healthy "
                "parallel GPU device(s), with best observed device-parallel "
                "and multiprocess speedups of "
                f"`{claims['gpu_strong_best_device_parallel_speedup']:.3f}x` "
                "and "
                f"`{claims['gpu_strong_best_multiprocess_speedup']:.3f}x`."
            ),
            (
                "- On the prepared-geometry reuse profile, the compiled steady "
                "solver reaches a best observed speedup of "
                f"`{claims['prepared_geometry_reuse_best_compiled_steady_speedup']:.3f}x` "
                "against direct repeated solves with maximum coefficient mismatch "
                f"`{claims['prepared_geometry_reuse_max_compiled_relative_mismatch']:.3e}`."
            ),
            "",
            "These claims should be used consistently in the manuscript text, captions, and",
            "response-to-reviewer notes.",
            "",
        ]
    )


def main() -> None:
    payload = build_payload()
    markdown = build_markdown(payload)
    claims = build_claims_markdown(payload)
    (STATIC / "manuscript_artifacts.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (STATIC / "manuscript_tables.md").write_text(markdown, encoding="utf-8")
    (STATIC / "manuscript_claims.md").write_text(claims, encoding="utf-8")
    print(f"Wrote {STATIC / 'manuscript_artifacts.json'}")
    print(f"Wrote {STATIC / 'manuscript_tables.md'}")
    print(f"Wrote {STATIC / 'manuscript_claims.md'}")


if __name__ == "__main__":
    main()
