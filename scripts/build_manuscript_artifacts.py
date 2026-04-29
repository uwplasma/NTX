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
    owned_geometry_neopax = _load_json(STATIC / "owned_geometry_neopax_dataset.json")
    owned_finite_beta_sfincs = _load_json(
        STATIC / "owned_finite_beta_sfincs_jax_inputs.json"
    )
    owned_finite_beta_resolution = _load_json(
        STATIC / "owned_finite_beta_sfincs_jax_resolution_audit.json"
    )
    owned_finite_beta_production_ladder = _load_json(
        STATIC / "owned_finite_beta_sfincs_jax_production_ladder_audit.json"
    )
    owned_finite_beta_bootstrap = _load_json(
        STATIC / "owned_finite_beta_bootstrap_comparison.json"
    )
    owned_finite_beta_closure = _load_json(
        STATIC / "owned_finite_beta_closure_localization.json"
    )
    owned_finite_beta_observable = _load_json(
        STATIC / "owned_finite_beta_profile_current_observable_audit.json"
    )
    owned_finite_beta_conditioning = _load_json(
        STATIC / "owned_finite_beta_current_conditioning_audit.json"
    )
    owned_finite_beta_quadrature = _load_json(
        STATIC / "owned_finite_beta_closure_quadrature_audit.json"
    )
    owned_finite_beta_source_channel = _load_json(
        STATIC / "owned_finite_beta_source_channel_audit.json"
    )
    owned_finite_beta_source_response_profile = _load_json(
        STATIC / "owned_finite_beta_source_response_profile_audit.json"
    )
    owned_finite_beta_closure_target = _load_json(
        STATIC / "owned_finite_beta_closure_target_audit.json"
    )
    owned_finite_beta_radial_interpolation = _load_json(
        STATIC / "owned_finite_beta_radial_interpolation_audit.json"
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
        "owned_geometry_neopax",
        "owned_finite_beta_sfincs_jax_inputs",
        "owned_finite_beta_sfincs_jax_resolution_audit",
        "owned_finite_beta_sfincs_jax_production_ladder",
        "owned_finite_beta_bootstrap_comparison",
        "owned_finite_beta_closure_localization",
        "owned_finite_beta_profile_current_observable",
        "owned_finite_beta_current_conditioning",
        "owned_finite_beta_closure_quadrature",
        "owned_finite_beta_source_channel",
        "owned_finite_beta_source_response_profile",
        "owned_finite_beta_closure_target",
        "owned_finite_beta_radial_interpolation",
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
            "owned_geometry_neopax": {
                "cases": owned_geometry_neopax["cases"],
                "summary_metrics": owned_geometry_neopax["summary_metrics"],
                "claim_scope": owned_geometry_neopax["claim_scope"],
                "comparison_policy": owned_geometry_neopax["comparison_policy"],
                "normalization_contract": owned_geometry_neopax["normalization_contract"],
                "open_work": owned_geometry_neopax["open_work"],
            },
            "owned_finite_beta_sfincs_jax_inputs": {
                "summary_metrics": owned_finite_beta_sfincs["summary_metrics"],
                "claim_scope": owned_finite_beta_sfincs["claim_scope"],
                "normalization_contract": owned_finite_beta_sfincs[
                    "normalization_contract"
                ],
                "open_work": owned_finite_beta_sfincs["open_work"],
            },
            "owned_finite_beta_sfincs_jax_resolution_audit": {
                "summary_metrics": owned_finite_beta_resolution["summary_metrics"],
                "claim_scope": owned_finite_beta_resolution["claim_scope"],
                "conclusion": owned_finite_beta_resolution["conclusion"],
                "rows": owned_finite_beta_resolution["rows"],
                "open_work": owned_finite_beta_resolution["open_work"],
            },
            "owned_finite_beta_sfincs_jax_production_ladder": {
                "summary_metrics": owned_finite_beta_production_ladder[
                    "summary_metrics"
                ],
                "claim_scope": owned_finite_beta_production_ladder["claim_scope"],
                "conclusion": owned_finite_beta_production_ladder["conclusion"],
                "stress_row": owned_finite_beta_production_ladder["stress_row"],
                "open_work": owned_finite_beta_production_ladder["open_work"],
            },
            "owned_finite_beta_bootstrap_comparison": {
                "case": owned_finite_beta_bootstrap["case"],
                "inputs": owned_finite_beta_bootstrap["inputs"],
                "summary_metrics": owned_finite_beta_bootstrap["summary_metrics"],
                "comparison": {
                    "momentum_order_scan": owned_finite_beta_bootstrap.get(
                        "comparison",
                        {},
                    ).get("momentum_order_scan", {}),
                },
                "claim_scope": owned_finite_beta_bootstrap["claim_scope"],
                "normalization_contract": owned_finite_beta_bootstrap[
                    "normalization_contract"
                ],
                "open_work": owned_finite_beta_bootstrap["open_work"],
            },
            "owned_finite_beta_closure_localization": {
                "summary_metrics": owned_finite_beta_closure["summary_metrics"],
                "claim_scope": owned_finite_beta_closure["claim_scope"],
                "conclusion": owned_finite_beta_closure["conclusion"],
                "matched_radii": owned_finite_beta_closure["matched_radii"],
                "open_work": owned_finite_beta_closure["open_work"],
            },
            "owned_finite_beta_profile_current_observable": {
                "summary_metrics": owned_finite_beta_observable["summary_metrics"],
                "claim_scope": owned_finite_beta_observable["claim_scope"],
                "conclusion": owned_finite_beta_observable["conclusion"],
                "stress_radius": owned_finite_beta_observable["stress_radius"],
                "momentum_order_at_stress_radius": owned_finite_beta_observable[
                    "momentum_order_at_stress_radius"
                ],
                "open_work": owned_finite_beta_observable["open_work"],
            },
            "owned_finite_beta_current_conditioning": {
                "summary_metrics": owned_finite_beta_conditioning["summary_metrics"],
                "claim_scope": owned_finite_beta_conditioning["claim_scope"],
                "conclusion": owned_finite_beta_conditioning["conclusion"],
                "stress_radius": owned_finite_beta_conditioning["stress_radius"],
                "open_work": owned_finite_beta_conditioning["open_work"],
            },
            "owned_finite_beta_closure_quadrature": {
                "summary_metrics": owned_finite_beta_quadrature["summary_metrics"],
                "claim_scope": owned_finite_beta_quadrature["claim_scope"],
                "conclusion": owned_finite_beta_quadrature["conclusion"],
                "rows": owned_finite_beta_quadrature["rows"],
                "open_work": owned_finite_beta_quadrature["open_work"],
            },
            "owned_finite_beta_source_channel": {
                "summary_metrics": owned_finite_beta_source_channel["summary_metrics"],
                "claim_scope": owned_finite_beta_source_channel["claim_scope"],
                "conclusion": owned_finite_beta_source_channel["conclusion"],
                "rows": owned_finite_beta_source_channel["rows"],
                "open_work": owned_finite_beta_source_channel["open_work"],
            },
            "owned_finite_beta_source_response_profile": {
                "summary_metrics": owned_finite_beta_source_response_profile[
                    "summary_metrics"
                ],
                "claim_scope": owned_finite_beta_source_response_profile[
                    "claim_scope"
                ],
                "conclusion": owned_finite_beta_source_response_profile["conclusion"],
                "rows": owned_finite_beta_source_response_profile["rows"],
                "open_work": owned_finite_beta_source_response_profile["open_work"],
            },
            "owned_finite_beta_closure_target": {
                "summary_metrics": owned_finite_beta_closure_target["summary_metrics"],
                "claim_scope": owned_finite_beta_closure_target["claim_scope"],
                "rows": owned_finite_beta_closure_target["rows"],
                "correlations": owned_finite_beta_closure_target["correlations"],
                "linear_diagnostics": owned_finite_beta_closure_target[
                    "linear_diagnostics"
                ],
                "closure_requirements": owned_finite_beta_closure_target[
                    "closure_requirements"
                ],
            },
            "owned_finite_beta_radial_interpolation": {
                "summary_metrics": owned_finite_beta_radial_interpolation[
                    "summary_metrics"
                ],
                "claim_scope": owned_finite_beta_radial_interpolation["claim_scope"],
                "conclusion": owned_finite_beta_radial_interpolation["conclusion"],
                "radial_contract": owned_finite_beta_radial_interpolation[
                    "radial_contract"
                ],
                "open_work": owned_finite_beta_radial_interpolation["open_work"],
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
            "owned_finite_beta_bootstrap_max_relative_error": (
                owned_finite_beta_bootstrap["summary_metrics"][
                    "max_relative_error_total_vs_redl_interior"
                ]
            ),
            "owned_finite_beta_bootstrap_n_order": (
                owned_finite_beta_bootstrap["inputs"]["n_order"]
            ),
            "owned_finite_beta_bootstrap_d33_mode": (
                owned_finite_beta_bootstrap["inputs"]["d33_mode"]
            ),
            "owned_finite_beta_bootstrap_nu_v_count": (
                len(owned_finite_beta_bootstrap["inputs"]["nu_v"])
            ),
            "owned_finite_beta_bootstrap_psi_p": (
                owned_finite_beta_bootstrap["ntx_neopax"]["booz_xform_psi_p"]
            ),
            "owned_finite_beta_sfincs_completed_transport_count": (
                owned_finite_beta_sfincs["summary_metrics"][
                    "completed_transport_matrix_count"
                ]
            ),
            "owned_finite_beta_sfincs_ntx_same_grid_count": (
                owned_finite_beta_sfincs["summary_metrics"][
                    "completed_ntx_same_grid_comparison_count"
                ]
            ),
            "owned_finite_beta_sfincs_max_transport_relative_difference": (
                owned_finite_beta_sfincs["summary_metrics"][
                    "max_ntx_same_grid_transport_relative_difference"
                ]
            ),
            "owned_finite_beta_resolution_production_precision_gap": (
                owned_finite_beta_resolution["summary_metrics"][
                    "production_precision_gap_to_current_gate"
                ]
            ),
            "owned_finite_beta_resolution_tight_harmonics_precision_gap": (
                owned_finite_beta_resolution["summary_metrics"][
                    "tight_harmonics_precision_gap_to_current_gate"
                ]
            ),
            "owned_finite_beta_resolution_production_change_vs_smoke": (
                owned_finite_beta_resolution["summary_metrics"][
                    "production_change_vs_smoke"
                ]
            ),
            "owned_finite_beta_resolution_tight_harmonics_change_vs_production": (
                owned_finite_beta_resolution["summary_metrics"][
                    "tight_harmonics_change_vs_production"
                ]
            ),
            "owned_finite_beta_production_ladder_count": (
                owned_finite_beta_production_ladder["summary_metrics"][
                    "completed_production_ladder_count"
                ]
            ),
            "owned_finite_beta_production_ladder_max_transport_error": (
                owned_finite_beta_production_ladder["summary_metrics"][
                    "max_production_transport_relative_difference"
                ]
            ),
            "owned_finite_beta_production_ladder_precision_gap": (
                owned_finite_beta_production_ladder["summary_metrics"][
                    "max_production_precision_gap_to_current_gate"
                ]
            ),
            "owned_finite_beta_production_ladder_profile_current_error": (
                owned_finite_beta_production_ladder["summary_metrics"][
                    "max_profile_current_relative_difference_on_ladder"
                ]
            ),
            "owned_finite_beta_bootstrap_rms_relative_error": (
                owned_finite_beta_bootstrap["summary_metrics"][
                    "rms_relative_error_total_vs_redl_interior"
                ]
            ),
            "owned_finite_beta_bootstrap_sign_agreement": (
                owned_finite_beta_bootstrap["summary_metrics"][
                    "sign_agreement_fraction_total"
                ]
            ),
            "owned_finite_beta_closure_inner_gap_coefficient_error": (
                owned_finite_beta_closure["summary_metrics"][
                    "inner_gap_coefficient_relative_difference"
                ]
            ),
            "owned_finite_beta_closure_inner_gap_current_error": (
                owned_finite_beta_closure["summary_metrics"][
                    "inner_gap_bootstrap_relative_difference"
                ]
            ),
            "owned_finite_beta_closure_inner_gap_error_ratio": (
                owned_finite_beta_closure["summary_metrics"][
                    "inner_gap_current_to_coefficient_error_ratio"
                ]
            ),
            "owned_finite_beta_closure_coefficient_gate_pass": (
                owned_finite_beta_closure["summary_metrics"]["coefficient_gate_pass"]
            ),
            "owned_finite_beta_closure_profile_gate_pass": (
                owned_finite_beta_closure["summary_metrics"]["profile_current_gate_pass"]
            ),
            "owned_finite_beta_observable_stress_rho": (
                owned_finite_beta_observable["summary_metrics"]["stress_rho"]
            ),
            "owned_finite_beta_observable_applied_over_needed": (
                owned_finite_beta_observable["summary_metrics"][
                    "stress_applied_over_needed_correction"
                ]
            ),
            "owned_finite_beta_observable_residual_over_needed": (
                owned_finite_beta_observable["summary_metrics"][
                    "stress_residual_after_correction_over_needed"
                ]
            ),
            "owned_finite_beta_observable_cancellation_amplification": (
                owned_finite_beta_observable["summary_metrics"][
                    "stress_species_correction_cancellation_amplification"
                ]
            ),
            "owned_finite_beta_observable_residual_over_species_l1": (
                owned_finite_beta_observable["summary_metrics"][
                    "stress_residual_after_correction_over_species_correction_l1"
                ]
            ),
            "owned_finite_beta_observable_pmax_error_reduction": (
                owned_finite_beta_observable["summary_metrics"][
                    "pmax_stress_error_reduction"
                ]
            ),
            "owned_finite_beta_observable_correction_sign_agreement": (
                owned_finite_beta_observable["summary_metrics"][
                    "correction_sign_agreement_fraction"
                ]
            ),
            "owned_finite_beta_conditioning_stress_condition_number": (
                owned_finite_beta_conditioning["summary_metrics"][
                    "stress_current_condition_number"
                ]
            ),
            "owned_finite_beta_conditioning_required_coefficient_error": (
                owned_finite_beta_conditioning["summary_metrics"][
                    "stress_required_coefficient_relative_difference_for_current_gate"
                ]
            ),
            "owned_finite_beta_conditioning_coefficient_precision_gap": (
                owned_finite_beta_conditioning["summary_metrics"][
                    "stress_coefficient_precision_gap_to_current_gate"
                ]
            ),
            "owned_finite_beta_conditioning_coefficient_bound": (
                owned_finite_beta_conditioning["summary_metrics"][
                    "stress_coefficient_limited_current_relative_error_bound"
                ]
            ),
            "owned_finite_beta_quadrature_underintegrated_gate_pass_count": (
                owned_finite_beta_quadrature["summary_metrics"][
                    "underintegrated_gate_pass_count"
                ]
            ),
            "owned_finite_beta_quadrature_stable_gate_pass_count": (
                owned_finite_beta_quadrature["summary_metrics"][
                    "quadrature_stable_gate_pass_count"
                ]
            ),
            "owned_finite_beta_quadrature_stable_current_gate_pass": (
                owned_finite_beta_quadrature["summary_metrics"][
                    "quadrature_stable_current_gate_pass"
                ]
            ),
            "owned_finite_beta_quadrature_min_stress_error": (
                owned_finite_beta_quadrature["summary_metrics"][
                    "min_stress_relative_error"
                ]
            ),
            "owned_finite_beta_quadrature_min_stress_x": (
                owned_finite_beta_quadrature["summary_metrics"]["min_stress_neopax_x"]
            ),
            "owned_finite_beta_quadrature_min_stress_pmax": (
                owned_finite_beta_quadrature["summary_metrics"]["min_stress_n_order"]
            ),
            "owned_finite_beta_quadrature_high_x_largest_order_stress_error": (
                owned_finite_beta_quadrature["summary_metrics"][
                    "high_x_largest_order_stress_relative_error"
                ]
            ),
            "owned_finite_beta_quadrature_max_same_order_spread": (
                owned_finite_beta_quadrature["summary_metrics"][
                    "max_same_order_stress_spread_over_x"
                ]
            ),
            "owned_finite_beta_source_channel_reconstruction_residual": (
                owned_finite_beta_source_channel["summary_metrics"][
                    "max_source_channel_superposition_relative_residual"
                ]
            ),
            "owned_finite_beta_source_channel_gate_pass": (
                owned_finite_beta_source_channel["summary_metrics"][
                    "source_channel_superposition_gate_pass"
                ]
            ),
            "owned_finite_beta_source_channel_high_stable_error": (
                owned_finite_beta_source_channel["summary_metrics"][
                    "high_stable_public_relative_error_vs_redl"
                ]
            ),
            "owned_finite_beta_source_channel_high_stable_dominant": (
                owned_finite_beta_source_channel["summary_metrics"][
                    "high_stable_dominant_effective_channel"
                ]
            ),
            "owned_finite_beta_source_channel_temperature_fraction": (
                owned_finite_beta_source_channel["summary_metrics"][
                    "high_stable_effective_temperature_fraction_of_total"
                ]
            ),
            "owned_finite_beta_source_channel_density_fraction": (
                owned_finite_beta_source_channel["summary_metrics"][
                    "high_stable_density_electric_fraction_of_total"
                ]
            ),
            "owned_finite_beta_source_channel_parallel_fraction": (
                owned_finite_beta_source_channel["summary_metrics"][
                    "high_stable_parallel_electric_fraction_of_total"
                ]
            ),
            "owned_finite_beta_source_channel_cancellation_factor": (
                owned_finite_beta_source_channel["summary_metrics"][
                    "high_stable_species_cancellation_factor"
                ]
            ),
            "owned_finite_beta_source_channel_temperature_response_multiplier": (
                owned_finite_beta_source_channel["summary_metrics"].get(
                    "high_stable_effective_temperature_response_multiplier_to_redl"
                )
            ),
            "owned_finite_beta_source_channel_temperature_response_error": (
                owned_finite_beta_source_channel["summary_metrics"].get(
                    "high_stable_effective_temperature_channel_relative_error_vs_redl"
                )
            ),
            "owned_finite_beta_source_channel_redl_temperature_fraction": (
                owned_finite_beta_source_channel["summary_metrics"].get(
                    "high_stable_redl_effective_temperature_fraction_of_total"
                )
            ),
            "owned_finite_beta_source_response_profile_radius_count": (
                owned_finite_beta_source_response_profile["summary_metrics"][
                    "radius_count"
                ]
            ),
            "owned_finite_beta_source_response_profile_max_error": (
                owned_finite_beta_source_response_profile["summary_metrics"][
                    "high_order_max_public_relative_error_vs_redl"
                ]
            ),
            "owned_finite_beta_source_response_profile_multiplier_min": (
                owned_finite_beta_source_response_profile["summary_metrics"].get(
                    "high_order_temperature_response_multiplier_min"
                )
            ),
            "owned_finite_beta_source_response_profile_multiplier_median": (
                owned_finite_beta_source_response_profile["summary_metrics"].get(
                    "high_order_temperature_response_multiplier_median"
                )
            ),
            "owned_finite_beta_source_response_profile_multiplier_max": (
                owned_finite_beta_source_response_profile["summary_metrics"].get(
                    "high_order_temperature_response_multiplier_max"
                )
            ),
            "owned_finite_beta_source_response_profile_multiplier_span": (
                owned_finite_beta_source_response_profile["summary_metrics"].get(
                    "high_order_temperature_response_multiplier_span"
                )
            ),
            "owned_finite_beta_source_response_profile_stress_rho": (
                owned_finite_beta_source_response_profile["summary_metrics"][
                    "high_order_stress_rho"
                ]
            ),
            "owned_finite_beta_source_response_profile_nu_correlation": (
                owned_finite_beta_source_response_profile["summary_metrics"].get(
                    "temperature_response_correlation_with_log10_nu_e_star"
                )
            ),
            "owned_finite_beta_closure_target_best_driver": (
                owned_finite_beta_closure_target["summary_metrics"][
                    "best_single_physics_driver"
                ]
            ),
            "owned_finite_beta_closure_target_best_driver_abs_pearson": (
                owned_finite_beta_closure_target["summary_metrics"][
                    "best_single_physics_driver_abs_pearson"
                ]
            ),
            "owned_finite_beta_closure_target_best_model": (
                owned_finite_beta_closure_target["summary_metrics"][
                    "best_leave_one_out_model"
                ]
            ),
            "owned_finite_beta_closure_target_best_model_loo_rmse": (
                owned_finite_beta_closure_target["summary_metrics"][
                    "best_leave_one_out_rmse"
                ]
            ),
            "owned_finite_beta_closure_target_improvement_over_constant": (
                owned_finite_beta_closure_target["summary_metrics"][
                    "best_leave_one_out_improvement_over_constant"
                ]
            ),
            "owned_finite_beta_closure_target_runtime_correction_applied": (
                owned_finite_beta_closure_target["summary_metrics"][
                    "runtime_correction_applied"
                ]
            ),
            "owned_finite_beta_radial_interpolation_baseline_max_error": (
                owned_finite_beta_radial_interpolation["summary_metrics"][
                    "baseline_max_relative_error_total_vs_redl"
                ]
            ),
            "owned_finite_beta_radial_interpolation_matched_max_error": (
                owned_finite_beta_radial_interpolation["summary_metrics"][
                    "field_radius_matched_max_relative_error_total_vs_redl"
                ]
            ),
            "owned_finite_beta_radial_interpolation_baseline_stress_rho": (
                owned_finite_beta_radial_interpolation["summary_metrics"][
                    "baseline_stress_rho"
                ]
            ),
            "owned_finite_beta_radial_interpolation_baseline_stress_error": (
                owned_finite_beta_radial_interpolation["summary_metrics"][
                    "baseline_stress_relative_error"
                ]
            ),
            "owned_finite_beta_radial_interpolation_matched_stress_error": (
                owned_finite_beta_radial_interpolation["summary_metrics"][
                    "field_radius_matched_error_at_baseline_stress_rho"
                ]
            ),
            "owned_finite_beta_radial_interpolation_gate_pass": (
                owned_finite_beta_radial_interpolation["summary_metrics"][
                    "field_radius_matched_current_gate_pass"
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
    owned_finite_beta_bootstrap = payload["tables"][
        "owned_finite_beta_bootstrap_comparison"
    ]
    owned_finite_beta_bootstrap_inputs = owned_finite_beta_bootstrap["inputs"]
    owned_finite_beta_bootstrap_order_scan = owned_finite_beta_bootstrap[
        "comparison"
    ].get("momentum_order_scan", {})
    owned_finite_beta_bootstrap_metrics = owned_finite_beta_bootstrap["summary_metrics"]
    owned_finite_beta_closure = payload["tables"][
        "owned_finite_beta_closure_localization"
    ]
    owned_finite_beta_closure_metrics = owned_finite_beta_closure["summary_metrics"]
    closure_inner_coefficient_error = owned_finite_beta_closure_metrics[
        "inner_gap_coefficient_relative_difference"
    ]
    closure_inner_current_error = owned_finite_beta_closure_metrics[
        "inner_gap_bootstrap_relative_difference"
    ]
    closure_inner_error_ratio = owned_finite_beta_closure_metrics[
        "inner_gap_current_to_coefficient_error_ratio"
    ]
    owned_finite_beta_observable = payload["tables"][
        "owned_finite_beta_profile_current_observable"
    ]
    owned_finite_beta_observable_metrics = owned_finite_beta_observable[
        "summary_metrics"
    ]
    observable_applied_over_needed = owned_finite_beta_observable_metrics[
        "stress_applied_over_needed_correction"
    ]
    observable_residual_over_needed = owned_finite_beta_observable_metrics[
        "stress_residual_after_correction_over_needed"
    ]
    observable_cancellation_amplification = owned_finite_beta_observable_metrics[
        "stress_species_correction_cancellation_amplification"
    ]
    observable_residual_over_species_l1 = owned_finite_beta_observable_metrics[
        "stress_residual_after_correction_over_species_correction_l1"
    ]
    observable_pmax_error_reduction = owned_finite_beta_observable_metrics[
        "pmax_stress_error_reduction"
    ]
    owned_finite_beta_conditioning = payload["tables"][
        "owned_finite_beta_current_conditioning"
    ]
    owned_finite_beta_conditioning_metrics = owned_finite_beta_conditioning[
        "summary_metrics"
    ]
    conditioning_current_condition = owned_finite_beta_conditioning_metrics[
        "stress_current_condition_number"
    ]
    conditioning_required_coefficient_error = owned_finite_beta_conditioning_metrics[
        "stress_required_coefficient_relative_difference_for_current_gate"
    ]
    conditioning_precision_gap = owned_finite_beta_conditioning_metrics[
        "stress_coefficient_precision_gap_to_current_gate"
    ]
    conditioning_coefficient_bound = owned_finite_beta_conditioning_metrics[
        "stress_coefficient_limited_current_relative_error_bound"
    ]
    owned_finite_beta_quadrature = payload["tables"][
        "owned_finite_beta_closure_quadrature"
    ]
    owned_finite_beta_quadrature_metrics = owned_finite_beta_quadrature[
        "summary_metrics"
    ]
    quadrature_underintegrated_passes = owned_finite_beta_quadrature_metrics[
        "underintegrated_gate_pass_count"
    ]
    quadrature_stable_passes = owned_finite_beta_quadrature_metrics[
        "quadrature_stable_gate_pass_count"
    ]
    quadrature_stable_gate_pass = owned_finite_beta_quadrature_metrics[
        "quadrature_stable_current_gate_pass"
    ]
    quadrature_min_stress_error = owned_finite_beta_quadrature_metrics[
        "min_stress_relative_error"
    ]
    quadrature_min_stress_x = owned_finite_beta_quadrature_metrics[
        "min_stress_neopax_x"
    ]
    quadrature_min_stress_pmax = owned_finite_beta_quadrature_metrics[
        "min_stress_n_order"
    ]
    quadrature_high_x_error = owned_finite_beta_quadrature_metrics[
        "high_x_largest_order_stress_relative_error"
    ]
    quadrature_max_same_order_spread = owned_finite_beta_quadrature_metrics[
        "max_same_order_stress_spread_over_x"
    ]
    owned_finite_beta_source_channel = payload["tables"][
        "owned_finite_beta_source_channel"
    ]
    owned_finite_beta_source_channel_metrics = owned_finite_beta_source_channel[
        "summary_metrics"
    ]
    source_channel_reconstruction_residual = owned_finite_beta_source_channel_metrics[
        "max_source_channel_superposition_relative_residual"
    ]
    source_channel_gate_pass = owned_finite_beta_source_channel_metrics[
        "source_channel_superposition_gate_pass"
    ]
    source_channel_high_stable_error = owned_finite_beta_source_channel_metrics[
        "high_stable_public_relative_error_vs_redl"
    ]
    source_channel_high_stable_dominant = owned_finite_beta_source_channel_metrics[
        "high_stable_dominant_effective_channel"
    ]
    source_channel_temperature_fraction = owned_finite_beta_source_channel_metrics[
        "high_stable_effective_temperature_fraction_of_total"
    ]
    source_channel_density_fraction = owned_finite_beta_source_channel_metrics[
        "high_stable_density_electric_fraction_of_total"
    ]
    source_channel_parallel_fraction = owned_finite_beta_source_channel_metrics[
        "high_stable_parallel_electric_fraction_of_total"
    ]
    source_channel_cancellation_factor = owned_finite_beta_source_channel_metrics[
        "high_stable_species_cancellation_factor"
    ]
    source_channel_temperature_response_multiplier = (
        owned_finite_beta_source_channel_metrics.get(
            "high_stable_effective_temperature_response_multiplier_to_redl"
        )
    )
    source_channel_temperature_response_error = (
        owned_finite_beta_source_channel_metrics.get(
            "high_stable_effective_temperature_channel_relative_error_vs_redl"
        )
    )
    source_channel_redl_temperature_fraction = (
        owned_finite_beta_source_channel_metrics.get(
            "high_stable_redl_effective_temperature_fraction_of_total"
        )
    )
    owned_finite_beta_source_response_profile = payload["tables"][
        "owned_finite_beta_source_response_profile"
    ]
    source_response_profile_metrics = owned_finite_beta_source_response_profile[
        "summary_metrics"
    ]
    source_response_profile_radius_count = source_response_profile_metrics[
        "radius_count"
    ]
    source_response_profile_max_error = source_response_profile_metrics[
        "high_order_max_public_relative_error_vs_redl"
    ]
    source_response_profile_multiplier_min = source_response_profile_metrics.get(
        "high_order_temperature_response_multiplier_min"
    )
    source_response_profile_multiplier_median = source_response_profile_metrics.get(
        "high_order_temperature_response_multiplier_median"
    )
    source_response_profile_multiplier_max = source_response_profile_metrics.get(
        "high_order_temperature_response_multiplier_max"
    )
    source_response_profile_multiplier_span = source_response_profile_metrics.get(
        "high_order_temperature_response_multiplier_span"
    )
    source_response_profile_stress_rho = source_response_profile_metrics[
        "high_order_stress_rho"
    ]
    source_response_profile_nu_correlation = source_response_profile_metrics.get(
        "temperature_response_correlation_with_log10_nu_e_star"
    )
    owned_finite_beta_closure_target = payload["tables"][
        "owned_finite_beta_closure_target"
    ]
    closure_target_metrics = owned_finite_beta_closure_target["summary_metrics"]
    closure_target_best_driver = closure_target_metrics[
        "best_single_physics_driver"
    ]
    closure_target_best_driver_abs_pearson = closure_target_metrics[
        "best_single_physics_driver_abs_pearson"
    ]
    closure_target_best_model = closure_target_metrics["best_leave_one_out_model"]
    closure_target_best_model_loo_rmse = closure_target_metrics[
        "best_leave_one_out_rmse"
    ]
    closure_target_improvement_over_constant = closure_target_metrics[
        "best_leave_one_out_improvement_over_constant"
    ]
    closure_target_runtime_correction_applied = closure_target_metrics[
        "runtime_correction_applied"
    ]
    owned_finite_beta_resolution = payload["tables"][
        "owned_finite_beta_sfincs_jax_resolution_audit"
    ]
    owned_finite_beta_resolution_metrics = owned_finite_beta_resolution[
        "summary_metrics"
    ]
    resolution_production_gap = owned_finite_beta_resolution_metrics[
        "production_precision_gap_to_current_gate"
    ]
    resolution_tight_harmonics_gap = owned_finite_beta_resolution_metrics[
        "tight_harmonics_precision_gap_to_current_gate"
    ]
    owned_finite_beta_production_ladder = payload["tables"][
        "owned_finite_beta_sfincs_jax_production_ladder"
    ]
    owned_finite_beta_production_ladder_metrics = (
        owned_finite_beta_production_ladder["summary_metrics"]
    )
    production_ladder_count = owned_finite_beta_production_ladder_metrics[
        "completed_production_ladder_count"
    ]
    production_ladder_max_error = owned_finite_beta_production_ladder_metrics[
        "max_production_transport_relative_difference"
    ]
    production_ladder_precision_gap = owned_finite_beta_production_ladder_metrics[
        "max_production_precision_gap_to_current_gate"
    ]
    finite_beta_bootstrap_max_error = owned_finite_beta_bootstrap_metrics[
        "max_relative_error_total_vs_redl_interior"
    ]
    finite_beta_bootstrap_rms_error = owned_finite_beta_bootstrap_metrics[
        "rms_relative_error_total_vs_redl_interior"
    ]
    finite_beta_bootstrap_sign_fraction = owned_finite_beta_bootstrap_metrics[
        "sign_agreement_fraction_total"
    ]
    finite_beta_bootstrap_order_summary = ", ".join(
        (
            f"P={entry['n_order']}: "
            f"{entry['max_relative_error_total_vs_redl']:.2e}/"
            f"{entry['rms_relative_error_total_vs_redl']:.2e}"
        )
        for _, entry in sorted(
            owned_finite_beta_bootstrap_order_scan.items(),
            key=lambda item: int(item[0]),
        )
    )
    if not finite_beta_bootstrap_order_summary:
        finite_beta_bootstrap_order_summary = "not recorded"
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
            "## Owned Finite-Beta Bootstrap-Current Stress",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Case | `{owned_finite_beta_bootstrap['case']['id']}` |",
            (
                "| Closure configuration | "
                f"`P={owned_finite_beta_bootstrap_inputs['n_order']}`, "
                f"`D33={owned_finite_beta_bootstrap_inputs['d33_mode']}`, "
                f"`nu/v points={len(owned_finite_beta_bootstrap_inputs['nu_v'])}` |"
            ),
            (
                "| Boozer psi_p | "
                f"`{payload['claims']['owned_finite_beta_bootstrap_psi_p']:.6e}` |"
            ),
            (
                "| Max total-current relative difference vs Redl | "
                f"`{finite_beta_bootstrap_max_error:.3e}` |"
            ),
            (
                "| RMS total-current relative difference vs Redl | "
                f"`{finite_beta_bootstrap_rms_error:.3e}` |"
            ),
            (
                "| Sign-agreement fraction | "
                f"`{finite_beta_bootstrap_sign_fraction:.3f}` |"
            ),
            (
                "| Inner-gap same-grid coefficient relative difference | "
                f"`{closure_inner_coefficient_error:.3e}` |"
            ),
            (
                "| Inner-gap profile-current relative difference | "
                f"`{closure_inner_current_error:.3e}` |"
            ),
            (
                "| Inner-gap current/coefficient error ratio | "
                f"`{closure_inner_error_ratio:.3e}` |"
            ),
            (
                "| Stress-radius applied/needed correction | "
                f"`{observable_applied_over_needed:.3f}` |"
            ),
            (
                "| Stress-radius residual/needed correction | "
                f"`{observable_residual_over_needed:.3f}` |"
            ),
            (
                "| Stress-radius species-correction cancellation amplification | "
                f"`{observable_cancellation_amplification:.3f}` |"
            ),
            (
                "| Stress-radius residual/species-correction L1 | "
                f"`{observable_residual_over_species_l1:.3e}` |"
            ),
            (
                "| Stress-radius current condition number | "
                f"`{conditioning_current_condition:.3e}` |"
            ),
            (
                "| Required coefficient error for `1e-1` current gate | "
                f"`{conditioning_required_coefficient_error:.3e}` |"
            ),
            (
                "| Coefficient precision gap to current gate | "
                f"`{conditioning_precision_gap:.3f}x` |"
            ),
            (
                "| Production-grid coefficient precision gap | "
                f"`{resolution_production_gap:.3f}x` |"
            ),
            (
                "| Tight-harmonic coefficient precision gap | "
                f"`{resolution_tight_harmonics_gap:.3f}x` |"
            ),
            (
                "| Production radial/collisionality ladder count | "
                f"`{production_ladder_count}` |"
            ),
            (
                "| Production ladder max coefficient difference | "
                f"`{production_ladder_max_error:.3e}` |"
            ),
            (
                "| Production ladder precision gap | "
                f"`{production_ladder_precision_gap:.3f}x` |"
            ),
            (
                "| Coefficient-conditioned current-error bound | "
                f"`{conditioning_coefficient_bound:.3e}` |"
            ),
            (
                "| Under-integrated closure current-gate passes | "
                f"`{quadrature_underintegrated_passes}` |"
            ),
            (
                "| Quadrature-stable closure current-gate passes | "
                f"`{quadrature_stable_passes}` |"
            ),
            (
                "| Quadrature-stable current gate | "
                f"`{quadrature_stable_gate_pass}` |"
            ),
            (
                "| Best stress-radius closure setting | "
                f"`P={quadrature_min_stress_pmax}, X={quadrature_min_stress_x}, "
                f"error={quadrature_min_stress_error:.3e}` |"
            ),
            (
                "| Highest-X largest-order stress error | "
                f"`{quadrature_high_x_error:.3e}` |"
            ),
            (
                "| Max same-order stress spread over X | "
                f"`{quadrature_max_same_order_spread:.3e}` |"
            ),
            (
                "| Source-channel reconstruction residual | "
                f"`{source_channel_reconstruction_residual:.3e}` |"
            ),
            (
                "| Source-channel reconstruction gate | "
                f"`{source_channel_gate_pass}` |"
            ),
            (
                "| High-order source-channel stress error | "
                f"`{source_channel_high_stable_error:.3e}` |"
            ),
            (
                "| Dominant high-order source channel | "
                f"`{source_channel_high_stable_dominant}` |"
            ),
            (
                "| High-order temperature/density/parallel fractions | "
                f"`{source_channel_temperature_fraction:.3e}` / "
                f"`{source_channel_density_fraction:.3e}` / "
                f"`{source_channel_parallel_fraction:.3e}` |"
            ),
            (
                "| Source-channel species-cancellation factor | "
                f"`{source_channel_cancellation_factor:.3e}` |"
            ),
            (
                "| Redl temperature response multiplier at high order | "
                f"`{source_channel_temperature_response_multiplier:.3e}` |"
            )
            if source_channel_temperature_response_multiplier is not None
            else "",
            (
                "| Redl temperature-channel relative difference at high order | "
                f"`{source_channel_temperature_response_error:.3e}` |"
            )
            if source_channel_temperature_response_error is not None
            else "",
            (
                "| Redl temperature-channel fraction of target current | "
                f"`{source_channel_redl_temperature_fraction:.3e}` |"
            )
            if source_channel_redl_temperature_fraction is not None
            else "",
            (
                "| Profile source-response radii | "
                f"`{source_response_profile_radius_count}` |"
            ),
            (
                "| Profile source-response max current stress | "
                f"`{source_response_profile_max_error:.3e}` at "
                f"`rho={source_response_profile_stress_rho:.3f}` |"
            ),
            (
                "| Profile temperature response multiplier min/median/max | "
                f"`{source_response_profile_multiplier_min:.3e}` / "
                f"`{source_response_profile_multiplier_median:.3e}` / "
                f"`{source_response_profile_multiplier_max:.3e}` |"
            )
            if (
                source_response_profile_multiplier_min is not None
                and source_response_profile_multiplier_median is not None
                and source_response_profile_multiplier_max is not None
            )
            else "",
            (
                "| Profile temperature response multiplier span | "
                f"`{source_response_profile_multiplier_span:.3e}` |"
            )
            if source_response_profile_multiplier_span is not None
            else "",
            (
                "| Temperature response correlation with log10(nu_e*) | "
                f"`{source_response_profile_nu_correlation:.3e}` |"
            )
            if source_response_profile_nu_correlation is not None
            else "",
            (
                "| Closure-target best physics driver | "
                f"`{closure_target_best_driver}` "
                f"(`|r|={closure_target_best_driver_abs_pearson:.3e}`) |"
            ),
            (
                "| Closure-target best diagnostic model | "
                f"`{closure_target_best_model}` "
                f"(`LOO RMSE={closure_target_best_model_loo_rmse:.3e}`) |"
            )
            if closure_target_best_model_loo_rmse is not None
            else "",
            (
                "| Closure-target improvement over constant response | "
                f"`{closure_target_improvement_over_constant:.3e}` |"
            )
            if closure_target_improvement_over_constant is not None
            else "",
            (
                "| Closure-target runtime correction applied | "
                f"`{closure_target_runtime_correction_applied}` |"
            ),
            (
                "| Stress-radius Pmax error reduction | "
                f"`{observable_pmax_error_reduction:.3f}x` |"
            ),
            (
                "| Sonine-order max/RMS relative differences | "
                f"`{finite_beta_bootstrap_order_summary}` |"
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
    finite_beta_resolution_tight_change = claims[
        "owned_finite_beta_resolution_tight_harmonics_change_vs_production"
    ]
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
                "- The owned finite-beta SFINCS-JAX generation lane now has "
                f"`{claims['owned_finite_beta_sfincs_completed_transport_count']}` "
                "completed same-grid transport-matrix output(s) and "
                f"`{claims['owned_finite_beta_sfincs_ntx_same_grid_count']}` "
                "coefficient-level NTX comparison(s). The current maximum "
                "same-grid `L13/L31/L33` relative difference is "
                f"`{claims['owned_finite_beta_sfincs_max_transport_relative_difference']:.3e}`; "
                "this is a transport-coefficient stress diagnostic, not yet a "
                "profile-current parity claim."
            ),
            (
                "- The owned finite-beta bootstrap-current stress audit now runs "
                "Redl and `NTX+NEOPAX` on the same VMEC wout, Boozer transform, "
                "analytic profile contract, radial grid, and current normalization. "
                "The Boozer-coordinate path passes physical "
                f"`psi_p={claims['owned_finite_beta_bootstrap_psi_p']:.3e}`, "
                "the profile convolution uses "
                f"`{claims['owned_finite_beta_bootstrap_nu_v_count']}` adaptive "
                "`nu/v` support points, and the reported closure uses "
                f"`P={claims['owned_finite_beta_bootstrap_n_order']}` with "
                f"`D33={claims['owned_finite_beta_bootstrap_d33_mode']}`. "
                "The production-resolution reduced-closure total-current gap "
                "remains open at max/RMS "
                f"`{claims['owned_finite_beta_bootstrap_max_relative_error']:.3e}`/"
                f"`{claims['owned_finite_beta_bootstrap_rms_relative_error']:.3e}` "
                "with sign-agreement fraction "
                f"`{claims['owned_finite_beta_bootstrap_sign_agreement']:.3f}`, "
                "so this artifact is a stress diagnostic rather than a promoted "
                "finite-beta parity claim."
            ),
            (
                "- The owned finite-beta closure-localization sidecar compares "
                "the same-grid coefficient ladder with the profile-current "
                "stress artifact at the inner gap. The coefficient-side "
                "relative difference is "
                f"`{claims['owned_finite_beta_closure_inner_gap_coefficient_error']:.3e}` "
                "while the profile-current relative difference is "
                f"`{claims['owned_finite_beta_closure_inner_gap_current_error']:.3e}`, "
                "a ratio of "
                f"`{claims['owned_finite_beta_closure_inner_gap_error_ratio']:.3e}`. "
                "This keeps the remaining work in the reduced "
                "momentum/profile-current observable rather than relabeling it "
                "as a coefficient-normalization failure."
            ),
            (
                "- The owned finite-beta profile-current observable audit shows "
                "that the stress-radius momentum correction has unit sign "
                "agreement and applies "
                f"`{claims['owned_finite_beta_observable_applied_over_needed']:.3f}` "
                "of the correction needed to match the Redl target, leaving "
                f"`{claims['owned_finite_beta_observable_residual_over_needed']:.3f}` "
                "of the needed correction as residual. At the same radius the "
                "oppositely signed species corrections amplify the net-current "
                "observable by "
                f"`{claims['owned_finite_beta_observable_cancellation_amplification']:.3f}x`, "
                "and the remaining residual is only "
                f"`{claims['owned_finite_beta_observable_residual_over_species_l1']:.3e}` "
                "of the species-correction L1 scale. The Pmax sidecar reduces "
                "the stress error by "
                f"`{claims['owned_finite_beta_observable_pmax_error_reduction']:.3f}x`, "
                "so the remaining gap is an amplitude/observable closure issue, "
                "not a correction-sign failure."
            ),
            (
                "- The owned finite-beta current-conditioning audit shows that "
                "the same stress radius has species-flow L1 divided by Redl net "
                "current "
                f"`{claims['owned_finite_beta_conditioning_stress_condition_number']:.3e}`. "
                "A `1e-1` net-current gate therefore requires coefficient "
                "precision "
                f"`{claims['owned_finite_beta_conditioning_required_coefficient_error']:.3e}`, "
                "while the current same-grid smoke ladder is looser by "
                f"`{claims['owned_finite_beta_conditioning_coefficient_precision_gap']:.3f}x`. "
                "This keeps the next finite-beta step on production same-grid "
                "coefficient/profile-current diagnostics before changing the "
                "reduced closure."
            ),
            (
                "- The owned finite-beta production-resolution coefficient "
                "probe raises the stress-radius SFINCS-JAX/NTX grid from "
                "`25 x 31 x 32` to `35 x 43 x 48` and also tightens the VMEC "
                "harmonic cutoff. The production coefficient floor changes by "
                f"`{claims['owned_finite_beta_resolution_production_change_vs_smoke']:.3e}` "
                "relative to the smoke ladder, while the tight-harmonic probe "
                "changes by "
                f"`{finite_beta_resolution_tight_change:.3e}` "
                "relative to the production probe. The resulting precision "
                "gaps remain "
                f"`{claims['owned_finite_beta_resolution_production_precision_gap']:.3f}x`/"
                f"`{claims['owned_finite_beta_resolution_tight_harmonics_precision_gap']:.3f}x` "
                "above the current-conditioned target, so the remaining work is "
                "not closed by angular resolution or harmonic truncation."
            ),
            (
                "- The owned finite-beta production radial/collisionality "
                "ladder completes "
                f"`{claims['owned_finite_beta_production_ladder_count']}` "
                "same-grid SFINCS-JAX/NTX points. Its maximum coefficient "
                "difference is "
                f"`{claims['owned_finite_beta_production_ladder_max_transport_error']:.3e}`, "
                "still below the order-`1e-1` coefficient gate, but the "
                "current-conditioned precision gap remains "
                f"`{claims['owned_finite_beta_production_ladder_precision_gap']:.3f}x` "
                "at the inner stress radius. This closes the finite-beta "
                "production coefficient-ladder lane and keeps the open mismatch "
                "at the profile-current closure layer."
            ),
            (
                "- The owned finite-beta closure-quadrature audit shows that "
                f"`{claims['owned_finite_beta_quadrature_underintegrated_gate_pass_count']}` "
                "stress-radius current-gate pass occurs only when the velocity "
                "quadrature is lower than the Sonine truncation, while "
                f"`{claims['owned_finite_beta_quadrature_stable_gate_pass_count']}` "
                "quadrature-stable pass is found. The best apparent "
                "stress-radius setting is "
                f"`P={claims['owned_finite_beta_quadrature_min_stress_pmax']}`, "
                f"`X={claims['owned_finite_beta_quadrature_min_stress_x']}` with "
                "relative difference "
                f"`{claims['owned_finite_beta_quadrature_min_stress_error']:.3e}`, "
                "but the highest-X largest-order stress difference remains "
                f"`{claims['owned_finite_beta_quadrature_high_x_largest_order_stress_error']:.3e}` "
                "and same-order stress values vary over X by "
                f"`{claims['owned_finite_beta_quadrature_max_same_order_spread']:.3e}`. "
                "This closes the under-integrated apparent-pass route and keeps "
                "the finite-beta bootstrap-current gap assigned to a "
                "quadrature-converged reduced-closure lane."
            ),
            (
                "- The owned finite-beta source-channel audit freezes the same "
                "stress-radius matrix and solves one physical RHS channel at a "
                "time. The summed channels reconstruct the full corrected "
                "current with relative residual "
                f"`{claims['owned_finite_beta_source_channel_reconstruction_residual']:.3e}` "
                "and gate status "
                f"`{claims['owned_finite_beta_source_channel_gate_pass']}`. "
                "At the quadrature-stable high-order setting the current "
                "relative difference is "
                f"`{claims['owned_finite_beta_source_channel_high_stable_error']:.3e}`, "
                "the dominant source is "
                f"`{claims['owned_finite_beta_source_channel_high_stable_dominant']}`, "
                "and the temperature/density/parallel source fractions are "
                f"`{claims['owned_finite_beta_source_channel_temperature_fraction']:.3e}`/"
                f"`{claims['owned_finite_beta_source_channel_density_fraction']:.3e}`/"
                f"`{claims['owned_finite_beta_source_channel_parallel_fraction']:.3e}`. "
                + (
                    "The Redl temperature-channel target would require a "
                    "high-order response multiplier of "
                    "`"
                    f"{claims['owned_finite_beta_source_channel_temperature_response_multiplier']:.3e}"
                    "` "
                    "relative to the frozen corrected source solve, with "
                    "channel relative difference "
                    "`"
                    f"{claims['owned_finite_beta_source_channel_temperature_response_error']:.3e}"
                    "`. "
                    if claims.get(
                        "owned_finite_beta_source_channel_temperature_response_multiplier"
                    )
                    is not None
                    else ""
                )
                + "This keeps the remaining finite-beta closure work on a "
                "physics-derived source-channel response, not on fitted "
                "thresholds or hidden normalization constants."
            ),
            (
                "- The owned finite-beta profile source-response audit extends "
                "that source-channel measurement over "
                f"`{claims['owned_finite_beta_source_response_profile_radius_count']}` "
                "profile radii. The high-order current stress reaches "
                f"`{claims['owned_finite_beta_source_response_profile_max_error']:.3e}` "
                "at "
                f"`rho={claims['owned_finite_beta_source_response_profile_stress_rho']:.3f}`, "
                "and the Redl/NTX effective-temperature response multiplier "
                "spans "
                f"`{claims['owned_finite_beta_source_response_profile_multiplier_min']:.3e}`/"
                f"`{claims['owned_finite_beta_source_response_profile_multiplier_median']:.3e}`/"
                f"`{claims['owned_finite_beta_source_response_profile_multiplier_max']:.3e}` "
                "over min/median/max. This is kept as a profile-wide "
                "physics-localization map before any reduced-closure change is "
                "accepted."
            )
            if (
                claims.get("owned_finite_beta_source_response_profile_multiplier_min")
                is not None
                and claims.get(
                    "owned_finite_beta_source_response_profile_multiplier_median"
                )
                is not None
                and claims.get("owned_finite_beta_source_response_profile_multiplier_max")
                is not None
            )
            else "",
            (
                "- The finite-beta closure-target audit converts that "
                "profile-response map into a driver-identification artifact. "
                "Its strongest single local driver is "
                f"`{claims['owned_finite_beta_closure_target_best_driver']}` "
                "with absolute Pearson correlation "
                f"`{claims['owned_finite_beta_closure_target_best_driver_abs_pearson']:.3e}`; "
                "the best leave-one-out diagnostic model is "
                f"`{claims['owned_finite_beta_closure_target_best_model']}` "
                "with RMSE "
                f"`{claims['owned_finite_beta_closure_target_best_model_loo_rmse']:.3e}` "
                "and improvement over a constant response of "
                f"`{claims['owned_finite_beta_closure_target_improvement_over_constant']:.3e}`. "
                "No runtime correction is applied by this artifact."
            )
            if (
                claims.get("owned_finite_beta_closure_target_best_model_loo_rmse")
                is not None
                and claims.get(
                    "owned_finite_beta_closure_target_improvement_over_constant"
                )
                is not None
            )
            else "",
            (
                "- The radial-interpolation audit rebuilds the same finite-beta "
                "database on the exact field radii used by the profile-current "
                "observable. The previous stress point at "
                f"`rho={claims['owned_finite_beta_radial_interpolation_baseline_stress_rho']:.3f}` "
                "changes from "
                f"`{claims['owned_finite_beta_radial_interpolation_baseline_stress_error']:.3e}` "
                "to "
                f"`{claims['owned_finite_beta_radial_interpolation_matched_stress_error']:.3e}`, "
                "but the field-radius-matched profile maximum remains "
                f"`{claims['owned_finite_beta_radial_interpolation_matched_max_error']:.3e}` "
                "and gate status is "
                f"`{claims['owned_finite_beta_radial_interpolation_gate_pass']}`. "
                "This keeps the result as an interpolation sensitivity "
                "diagnostic, not a promoted runtime policy."
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
