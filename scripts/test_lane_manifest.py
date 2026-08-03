#!/usr/bin/env python3
"""Print and validate the maintained NTX CI test-lane manifest."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Literal

Lane = Literal[
    "core_foundation",
    "core_cli_workflows",
    "core_io_workflows",
    "core_parallel_workflows",
    "core_neopax_workflows",
    "core_profile_audit_workflow",
    "core_profile_basic_workflows",
    "core_profile_optimization_workflows",
    "core_profile_transport_workflows",
    "core_autodiff_uncertainty_workflow",
    "core_robust_bootstrap_workflow",
    "core_validation",
    "integration_examples",
    "heavy_examples_profiles",
    "heavy_examples_derivatives",
    "heavy_examples_boundary",
    "heavy_examples_publication",
]

ROOT = Path(__file__).resolve().parents[1]

HEAVY_EXAMPLES_PROFILES: tuple[str, ...] = (
    "tests/test_ambipolar_profile_example.py",
    "tests/test_ambipolar_profile_family_example.py",
    "tests/test_primitive_profile_transport_example.py",
    "tests/test_profile_basis_optimization_example.py",
    "tests/test_profile_control_optimization_example.py",
    "tests/test_profile_transport_loop_example.py",
)

HEAVY_EXAMPLES_DERIVATIVES: tuple[str, ...] = (
    "tests/test_derivative_audit_example.py",
    "tests/test_derivative_path_benchmark_example.py",
    "tests/test_file_backed_geometry_control_derivative_benchmark_example.py",
    "tests/test_geometry_control_derivative_benchmark_example.py",
)

HEAVY_EXAMPLES_BOUNDARY: tuple[str, ...] = (
    "tests/test_boundary_forward_mode_current_derivative_benchmark_example.py",
    "tests/test_explicit_relaxed_boundary_current_derivative_benchmark_example.py",
    "tests/test_implicit_equilibrium_forward_mode_derivative_benchmark_example.py",
)

HEAVY_EXAMPLES_PUBLICATION: tuple[str, ...] = (
    "tests/test_bootstrap_current_optimization_example.py",
    "tests/test_make_publication_figures.py",
    "tests/test_manuscript_artifacts_script.py",
    "tests/test_performance_scaling_example.py",
    "tests/test_prepared_geometry_reuse_profile_example.py",
    "tests/test_validation_summary_example.py",
)

INTEGRATION_EXAMPLES: tuple[str, ...] = (
    "tests/test_bootstrap_current_reference_audit_w7x.py",
    "tests/test_bootstrap_current_vmec_or_boozmn_example.py",
    "tests/test_bootstrap_current_with_neopax_example.py",
    "tests/test_differentiable_neopax_field.py",
    "tests/test_plot_output_npz_example.py",
    "tests/test_precise_qs_redl_sfincs_audit.py",
    "tests/test_profile_fixed_field_workflow_script.py",
    "tests/test_profile_w7x_integrated_workflow_script.py",
)

CORE_FOUNDATION_TESTS: tuple[str, ...] = (
    "tests/test_autodiff.py",
    "tests/test_boozmn.py",
    "tests/test_config.py",
    "tests/test_convergence.py",
    "tests/test_database.py",
    "tests/test_documentation_contract.py",
    "tests/test_geometry.py",
    "tests/test_grids.py",
    "tests/test_io_unit.py",
    "tests/test_multiprocess_parallel.py",
    "tests/test_operators.py",
    "tests/test_parallel.py",
    "tests/test_parallel_unit.py",
    "tests/test_profiles_unit.py",
    "tests/test_resolution.py",
    "tests/test_source_map.py",
    "tests/test_suite_leaves_tree_clean.py",
    "tests/test_solver.py",
    "tests/test_solver_derivative_audit.py",
    "tests/test_solver_residuals.py",
    "tests/test_solvax_integration.py",
    "tests/test_exact_window_adjoint.py",
    "tests/test_certified_adjoint_window.py",
    "tests/test_vmec.py",
    "tests/test_vmex_backend.py",
    "tests/test_vmex_vmec.py",
    "tests/test_vmec_physics.py",
)

CORE_CLI_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_cli.py",
    "tests/test_cli_unit.py",
    "tests/test_distribution_size.py",
    "tests/test_examples.py",
    "tests/test_namespace_imports.py",
    "tests/test_packaging.py",
)

CORE_IO_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_inputfiles.py",
    "tests/test_inputfiles_unit.py",
    "tests/test_interp.py",
    "tests/test_profile_script.py",
    "tests/test_vmec_scan.py",
)

CORE_PARALLEL_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_gpu_scripts.py",
    "tests/test_gpu_smoke.py",
    "tests/test_multiprocess_script.py",
    "tests/test_parallel_script.py",
)

CORE_NEOPAX_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_neopax_adapter.py",
    "tests/test_neopax_arrays.py",
    "tests/test_neopax_qi.py",
)

CORE_PROFILE_AUDIT_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_profile_force_reconstruction_audit_example.py",
)

CORE_PROFILE_BASIC_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_profiles_workflows.py::test_ambipolar_profile_solver_returns_finite_result_and_reduces_loss",
    "tests/test_profiles_workflows.py::test_ambipolar_residual_and_solver_are_differentiable",
    "tests/test_profiles_workflows.py::test_ambipolar_residual_profile_has_expected_shape",
    "tests/test_profiles_workflows.py::test_profile_family_solver_and_bootstrap_objective_return_finite_results",
    "tests/test_profiles_workflows.py::test_profile_family_solver_defaults_control_index",
)

CORE_PROFILE_OPTIMIZATION_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_profiles_workflows.py::test_profile_control_application_and_optimization_return_finite_results",
    "tests/test_profiles_workflows.py::test_profile_control_optimization_supports_unbounded_control",
    "tests/test_profiles_workflows.py::test_profile_basis_control_application_and_optimization_return_finite_results",
    "tests/test_profiles_workflows.py::test_profile_basis_optimization_supports_unbounded_control",
)

CORE_PROFILE_TRANSPORT_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_profiles_workflows.py::test_profile_transport_loop_returns_finite_histories",
    "tests/test_profiles_workflows.py::test_profile_transport_closure_shape_mismatch_raises",
    "tests/test_profiles_workflows.py::test_profile_transport_loop_handles_rejected_backtracking",
    "tests/test_profiles_workflows.py::test_advance_profile_transport_rejects_species_shape_mismatch",
    "tests/test_profiles_workflows.py::test_primitive_profile_transport_loop_returns_finite_histories",
    "tests/test_profiles_workflows.py::test_primitive_profile_transport_update_preserves_positive_density_temperature",
    "tests/test_profiles_workflows.py::test_primitive_profile_transport_loop_handles_rejected_backtracking",
)

CORE_AUTODIFF_UNCERTAINTY_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_autodiff_profile_uncertainty_example.py",
)

CORE_ROBUST_BOOTSTRAP_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_bootstrap_current_robust_optimization_example.py",
)

CORE_VALIDATION_TESTS: tuple[str, ...] = (
    "tests/test_angular_oversampling.py",
    "tests/test_benchmark_matrix.py",
    "tests/test_benchmark_scaling_script.py",
    "tests/test_boozmn_backend_validation_audit.py",
    "tests/test_boozmn_same_coordinate_roundtrip_audit.py",
    "tests/test_build_coverage_report_script.py",
    "tests/test_checkout_paths.py",
    "tests/test_ci_lane_manifest.py",
    "tests/test_closure_validation_report_script.py",
    "tests/test_coverage_edges.py",
    "tests/test_fixed_field_momentum_correction_diagnostic.py",
    "tests/test_fixed_field_parallel_flow_audit.py",
    "tests/test_fixed_field_transport_matrix_audit.py",
    "tests/test_fixed_field_validation_metrics.py",
    "tests/test_geometry_family_breadth_summary.py",
    "tests/test_geometry_family_transport_convergence.py",
    "tests/test_momentum_correction_mapping_audit.py",
    "tests/test_owned_finite_beta_bootstrap_comparison.py",
    "tests/test_owned_finite_beta_closure_localization.py",
    "tests/test_owned_finite_beta_current_conditioning_audit.py",
    "tests/test_owned_finite_beta_closure_quadrature_audit.py",
    "tests/test_owned_finite_beta_source_channel_audit.py",
    "tests/test_owned_finite_beta_source_response_profile_audit.py",
    "tests/test_owned_finite_beta_closure_target_audit.py",
    "tests/test_owned_finite_beta_radial_interpolation_audit.py",
    "tests/test_owned_finite_beta_profile_current_observable_audit.py",
    "tests/test_owned_finite_beta_sfincs_jax_production_ladder_audit.py",
    "tests/test_owned_finite_beta_sfincs_jax_resolution_audit.py",
    "tests/test_owned_finite_beta_sfincs_jax_inputs.py",
    "tests/test_owned_finite_beta_sfincs_jax_profile_current_audit.py",
    "tests/test_owned_finite_beta_sfincs_jax_profile_current_resolution_audit.py",
    "tests/test_owned_geometry_neopax_dataset.py",
    "tests/test_physics_gates.py",
    "tests/test_repository_size.py",
    "tests/test_w7x_reference_benchmark.py",
)

LANES: dict[Lane, tuple[str, ...]] = {
    "core_foundation": CORE_FOUNDATION_TESTS,
    "core_cli_workflows": CORE_CLI_WORKFLOW_TESTS,
    "core_io_workflows": CORE_IO_WORKFLOW_TESTS,
    "core_parallel_workflows": CORE_PARALLEL_WORKFLOW_TESTS,
    "core_neopax_workflows": CORE_NEOPAX_WORKFLOW_TESTS,
    "core_profile_audit_workflow": CORE_PROFILE_AUDIT_WORKFLOW_TESTS,
    "core_profile_basic_workflows": CORE_PROFILE_BASIC_WORKFLOW_TESTS,
    "core_profile_optimization_workflows": CORE_PROFILE_OPTIMIZATION_WORKFLOW_TESTS,
    "core_profile_transport_workflows": CORE_PROFILE_TRANSPORT_WORKFLOW_TESTS,
    "core_autodiff_uncertainty_workflow": CORE_AUTODIFF_UNCERTAINTY_WORKFLOW_TESTS,
    "core_robust_bootstrap_workflow": CORE_ROBUST_BOOTSTRAP_WORKFLOW_TESTS,
    "core_validation": CORE_VALIDATION_TESTS,
    "integration_examples": INTEGRATION_EXAMPLES,
    "heavy_examples_profiles": HEAVY_EXAMPLES_PROFILES,
    "heavy_examples_derivatives": HEAVY_EXAMPLES_DERIVATIVES,
    "heavy_examples_boundary": HEAVY_EXAMPLES_BOUNDARY,
    "heavy_examples_publication": HEAVY_EXAMPLES_PUBLICATION,
}


def all_manifest_tests() -> tuple[str, ...]:
    return tuple(path for lane in LANES.values() for path in lane)


def selection_file(selection: str) -> str:
    """Return the file path for a pytest file or node-id selection."""

    return selection.split("::", 1)[0]


def discovered_tests(root: Path = ROOT) -> tuple[str, ...]:
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in (root / "tests").glob("test_*.py"))
    )


def discovered_test_nodes(root: Path = ROOT) -> dict[str, tuple[str, ...]]:
    """Return top-level pytest node ids discoverable without importing tests."""

    nodes_by_file: dict[str, tuple[str, ...]] = {}
    for file_path in discovered_tests(root):
        path = root / file_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=file_path)
        nodes: list[str] = []
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith(
                "test_"
            ):
                nodes.append(f"{file_path}::{item.name}")
            if isinstance(item, ast.ClassDef) and item.name.startswith("Test"):
                for method in item.body:
                    if isinstance(
                        method, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and method.name.startswith("test_"):
                        nodes.append(f"{file_path}::{item.name}::{method.name}")
        nodes_by_file[file_path] = tuple(sorted(nodes))
    return nodes_by_file


def validate_manifest(root: Path = ROOT) -> tuple[str, ...]:
    """Return validation errors for the maintained manifest."""

    errors: list[str] = []
    manifest_tests = all_manifest_tests()
    duplicate_paths = sorted(
        {path for path in manifest_tests if manifest_tests.count(path) > 1}
    )
    if duplicate_paths:
        errors.append("duplicate test-lane selections: " + ", ".join(duplicate_paths))

    discovered = set(discovered_tests(root))
    manifest_files = {selection_file(selection) for selection in manifest_tests}
    missing_from_manifest = sorted(discovered - manifest_files)
    missing_on_disk = sorted(manifest_files - discovered)
    if missing_from_manifest:
        errors.append("unclassified test files: " + ", ".join(missing_from_manifest))
    if missing_on_disk:
        errors.append("manifest paths missing on disk: " + ", ".join(missing_on_disk))

    whole_file_selections = {selection for selection in manifest_tests if "::" not in selection}
    node_selection_files = {
        selection_file(selection) for selection in manifest_tests if "::" in selection
    }
    mixed_selection_files = sorted(whole_file_selections & node_selection_files)
    if mixed_selection_files:
        errors.append(
            "files cannot mix whole-file and node-id lane selections: "
            + ", ".join(mixed_selection_files)
        )

    nodes_by_file = discovered_test_nodes(root)
    for file_path in sorted(node_selection_files):
        declared_nodes = {
            selection for selection in manifest_tests if selection_file(selection) == file_path
        }
        discovered_nodes = set(nodes_by_file.get(file_path, ()))
        unknown_nodes = sorted(declared_nodes - discovered_nodes)
        missing_nodes = sorted(discovered_nodes - declared_nodes)
        if unknown_nodes:
            errors.append("unknown test node selections: " + ", ".join(unknown_nodes))
        if missing_nodes:
            errors.append("unclassified test nodes: " + ", ".join(missing_nodes))
    return tuple(errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lane",
        nargs="?",
        choices=tuple(LANES),
        help="Lane to print, one test path per line.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that every tests/test_*.py file is assigned to exactly one lane.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.check:
        errors = validate_manifest(root)
        if errors:
            raise SystemExit("\n".join(errors))
    if args.lane is not None:
        for path in LANES[args.lane]:
            print(path)


if __name__ == "__main__":
    main()
