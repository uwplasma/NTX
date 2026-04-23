#!/usr/bin/env python3
"""Print and validate the maintained NTX CI test-lane manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

Lane = Literal[
    "core_foundation",
    "core_workflows",
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
    "tests/test_database.py",
    "tests/test_geometry.py",
    "tests/test_grids.py",
    "tests/test_io_unit.py",
    "tests/test_multiprocess_parallel.py",
    "tests/test_operators.py",
    "tests/test_parallel.py",
    "tests/test_parallel_unit.py",
    "tests/test_profiles_unit.py",
    "tests/test_solver.py",
    "tests/test_vmec.py",
    "tests/test_vmec_jax_backend.py",
    "tests/test_vmec_jax_vmec.py",
    "tests/test_vmec_physics.py",
)

CORE_WORKFLOW_TESTS: tuple[str, ...] = (
    "tests/test_autodiff_profile_uncertainty_example.py",
    "tests/test_bootstrap_current_robust_optimization_example.py",
    "tests/test_cli.py",
    "tests/test_cli_unit.py",
    "tests/test_examples.py",
    "tests/test_gpu_scripts.py",
    "tests/test_gpu_smoke.py",
    "tests/test_inputfiles.py",
    "tests/test_inputfiles_unit.py",
    "tests/test_multiprocess_script.py",
    "tests/test_namespace_imports.py",
    "tests/test_neopax_adapter.py",
    "tests/test_neopax_arrays.py",
    "tests/test_neopax_qi.py",
    "tests/test_packaging.py",
    "tests/test_parallel_script.py",
    "tests/test_profile_force_reconstruction_audit_example.py",
    "tests/test_profile_script.py",
    "tests/test_profiles_workflows.py",
    "tests/test_vmec_scan.py",
)

CORE_VALIDATION_TESTS: tuple[str, ...] = (
    "tests/test_benchmark_matrix.py",
    "tests/test_benchmark_scaling_script.py",
    "tests/test_build_coverage_report_script.py",
    "tests/test_checkout_paths.py",
    "tests/test_ci_lane_manifest.py",
    "tests/test_closure_validation_report_script.py",
    "tests/test_coverage_edges.py",
    "tests/test_fixed_field_momentum_correction_diagnostic.py",
    "tests/test_fixed_field_parallel_flow_audit.py",
    "tests/test_fixed_field_transport_matrix_audit.py",
    "tests/test_momentum_correction_mapping_audit.py",
    "tests/test_physics_gates.py",
    "tests/test_w7x_reference_benchmark.py",
)

LANES: dict[Lane, tuple[str, ...]] = {
    "core_foundation": CORE_FOUNDATION_TESTS,
    "core_workflows": CORE_WORKFLOW_TESTS,
    "core_validation": CORE_VALIDATION_TESTS,
    "integration_examples": INTEGRATION_EXAMPLES,
    "heavy_examples_profiles": HEAVY_EXAMPLES_PROFILES,
    "heavy_examples_derivatives": HEAVY_EXAMPLES_DERIVATIVES,
    "heavy_examples_boundary": HEAVY_EXAMPLES_BOUNDARY,
    "heavy_examples_publication": HEAVY_EXAMPLES_PUBLICATION,
}


def all_manifest_tests() -> tuple[str, ...]:
    return tuple(path for lane in LANES.values() for path in lane)


def discovered_tests(root: Path = ROOT) -> tuple[str, ...]:
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in (root / "tests").glob("test_*.py"))
    )


def validate_manifest(root: Path = ROOT) -> tuple[str, ...]:
    """Return validation errors for the maintained manifest."""

    errors: list[str] = []
    manifest_tests = all_manifest_tests()
    duplicate_paths = sorted(
        {path for path in manifest_tests if manifest_tests.count(path) > 1}
    )
    if duplicate_paths:
        errors.append("duplicate test-lane paths: " + ", ".join(duplicate_paths))

    discovered = set(discovered_tests(root))
    manifest = set(manifest_tests)
    missing_from_manifest = sorted(discovered - manifest)
    missing_on_disk = sorted(manifest - discovered)
    if missing_from_manifest:
        errors.append("unclassified test files: " + ", ".join(missing_from_manifest))
    if missing_on_disk:
        errors.append("manifest paths missing on disk: " + ", ".join(missing_on_disk))
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
