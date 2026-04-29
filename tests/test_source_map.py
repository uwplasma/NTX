from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTERNAL_MODULES_REQUIRING_SOURCE_MAP = (
    "src/ntx/_autodiff_bootstrap.py",
    "src/ntx/_autodiff_bootstrap_common.py",
    "src/ntx/_autodiff_bootstrap_deterministic.py",
    "src/ntx/_autodiff_bootstrap_robust.py",
    "src/ntx/_autodiff_derivatives.py",
    "src/ntx/_autodiff_helpers.py",
    "src/ntx/_autodiff_inverse.py",
    "src/ntx/_autodiff_profile.py",
    "src/ntx/_autodiff_types.py",
    "src/ntx/_autodiff_workflows.py",
    "src/ntx/_geometry_eval.py",
    "src/ntx/_geometry_types.py",
    "src/ntx/_inputfiles_model.py",
    "src/ntx/_inputfiles_output.py",
    "src/ntx/_inputfiles_reporting.py",
    "src/ntx/_inputfiles_run.py",
    "src/ntx/_neopax_bridge.py",
    "src/ntx/_neopax_field.py",
    "src/ntx/_neopax_field_utils.py",
    "src/ntx/_neopax_fluxes.py",
    "src/ntx/_neopax_io.py",
    "src/ntx/_neopax_scan.py",
    "src/ntx/_neopax_scan_coefficients.py",
    "src/ntx/_neopax_scan_fields.py",
    "src/ntx/_neopax_types.py",
    "src/ntx/_neopax_vmec_jax_boozer.py",
    "src/ntx/_neopax_vmec_jax_field.py",
    "src/ntx/_neopax_vmec_jax_profiles.py",
    "src/ntx/_profiles_controls.py",
    "src/ntx/_profiles_ambipolar_types.py",
    "src/ntx/_profiles_channels.py",
    "src/ntx/_profiles_control_types.py",
    "src/ntx/_profiles_control_basis.py",
    "src/ntx/_profiles_control_scalar.py",
    "src/ntx/_profiles_eval.py",
    "src/ntx/_profiles_primitives.py",
    "src/ntx/_profiles_radial.py",
    "src/ntx/_profiles_species_types.py",
    "src/ntx/_profiles_transport.py",
    "src/ntx/_profiles_transport_closure.py",
    "src/ntx/_profiles_transport_terms.py",
    "src/ntx/_profiles_transport_types.py",
    "src/ntx/_profiles_types.py",
    "src/ntx/_solver_adjoint.py",
    "src/ntx/_solver_context.py",
    "src/ntx/_solver_core.py",
    "src/ntx/_solver_factorization.py",
    "src/ntx/_solver_prepared.py",
    "src/ntx/_solver_scan.py",
    "src/ntx/_solver_types.py",
    "src/ntx/_vmec_jax_boundary.py",
    "src/ntx/_vmec_jax_boozer.py",
    "src/ntx/_vmec_jax_surfaces.py",
    "src/ntx/validation/_benchmark_matrix_autodiff.py",
    "src/ntx/validation/_benchmark_matrix_autodiff_derivatives.py",
    "src/ntx/validation/_benchmark_matrix_autodiff_design.py",
    "src/ntx/validation/_benchmark_matrix_bootstrap.py",
    "src/ntx/validation/_benchmark_matrix_geometry.py",
    "src/ntx/validation/_benchmark_matrix_geometry_finite_beta.py",
    "src/ntx/validation/_benchmark_matrix_integrated.py",
    "src/ntx/validation/_benchmark_matrix_monoenergetic.py",
    "src/ntx/validation/_benchmark_matrix_performance.py",
    "src/ntx/validation/_benchmark_matrix_profiles.py",
    "src/ntx/validation/_benchmark_matrix_types.py",
    "src/ntx/validation/_finite_beta_closure_target.py",
    "src/ntx/validation/_physics_gate_analytical.py",
    "src/ntx/validation/_physics_gate_artifact_eval.py",
    "src/ntx/validation/_physics_gate_artifact_registry.py",
    "src/ntx/validation/_physics_gate_artifact_registry_finite_beta.py",
    "src/ntx/validation/_physics_gate_artifacts.py",
    "src/ntx/validation/_physics_gate_artifacts_finite_beta.py",
    "src/ntx/validation/_physics_gate_registry.py",
    "src/ntx/validation/_physics_gate_types.py",
)


def test_source_map_mentions_split_internal_modules() -> None:
    text = (ROOT / "docs" / "source-map.md").read_text(encoding="utf-8")

    missing = [
        module_path
        for module_path in INTERNAL_MODULES_REQUIRING_SOURCE_MAP
        if f"`{module_path}`" not in text
    ]

    assert missing == []
