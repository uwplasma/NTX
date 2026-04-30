from __future__ import annotations

from pathlib import Path

from ._physics_gate_artifact_eval import _append_summary_metric_gate
from ._physics_gate_types import PhysicsGateResult


def append_finite_beta_artifact_gates(
    results: list[PhysicsGateResult],
    static_root: Path,
) -> None:
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_same_grid_coefficient_stress",
        path=static_root / "owned_finite_beta_closure_localization.json",
        metric_key="max_same_grid_coefficient_relative_difference",
        details=(
            "same-grid finite-beta transport-matrix coefficient comparison; "
            "this isolates normalization/interpolation before profile closure"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_profile_current_observable_stress",
        path=static_root / "owned_finite_beta_profile_current_observable_audit.json",
        metric_key="stress_relative_error_total_vs_redl",
        details=(
            "monitored finite-beta profile-current stress metric; not a parity "
            "claim while the same-grid profile-current closure comparison is open"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_species_cancellation_stress",
        path=static_root / "owned_finite_beta_profile_current_observable_audit.json",
        metric_key="stress_residual_after_correction_over_species_correction_l1",
        details=(
            "monitored species-current cancellation scale at the finite-beta "
            "stress radius"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_current_conditioning_stress",
        path=static_root / "owned_finite_beta_current_conditioning_audit.json",
        metric_key="max_coefficient_precision_gap_to_current_gate",
        details=(
            "monitored maximum coefficient precision gap after species-current "
            "cancellation conditioning; values above one mean sensitive radii "
            "need a tighter same-grid coefficient ladder before coefficient "
            "uncertainty can be ruled out"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_resolution_floor_stress",
        path=static_root / "owned_finite_beta_sfincs_jax_resolution_audit.json",
        metric_key="production_precision_gap_to_current_gate",
        details=(
            "monitored finite-beta production-grid coefficient floor compared "
            "with the current-conditioned precision target; values above one "
            "mean resolution and harmonic-cutoff probes still do not clear the "
            "net-current gate"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_production_ladder_stress",
        path=(
            static_root
            / "owned_finite_beta_sfincs_jax_production_ladder_audit.json"
        ),
        metric_key="max_production_precision_gap_to_current_gate",
        details=(
            "monitored finite-beta production radial/collisionality coefficient "
            "ladder compared with the current-conditioned precision target; "
            "values above one keep bootstrap-current parity open at the "
            "profile-current closure layer"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_closure_quadrature_stress",
        path=static_root / "owned_finite_beta_closure_quadrature_audit.json",
        metric_key="underintegrated_gate_pass_count",
        details=(
            "monitored finite-beta closure quadrature aliasing count; nonzero "
            "means a current-gate pass was observed only where velocity "
            "quadrature was lower than the Sonine truncation"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_source_channel_reconstruction",
        path=static_root / "owned_finite_beta_source_channel_audit.json",
        metric_key="max_source_channel_superposition_relative_residual",
        details=(
            "finite-beta stress-radius source-channel decomposition of the "
            "same momentum-restoring linear system; the one-channel solves "
            "must reconstruct the full corrected current before the dominant "
            "drive channel is interpreted"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_temperature_source_response_stress",
        path=static_root / "owned_finite_beta_source_channel_audit.json",
        metric_key="high_stable_effective_temperature_response_multiplier_to_redl",
        details=(
            "monitored ratio between the Redl effective-temperature target "
            "channel and the frozen high-order NTX+NEOPAX corrected source "
            "channel; this localizes the closure gap without adding a fitted "
            "runtime correction"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_profile_source_response_stress",
        path=static_root / "owned_finite_beta_source_response_profile_audit.json",
        metric_key="high_order_temperature_response_multiplier_span",
        details=(
            "monitored radial span of the high-order Redl/NTX effective-"
            "temperature source-response multiplier; this maps the finite-beta "
            "closure gap across the profile without applying a fitted runtime "
            "correction"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_closure_target_driver_stress",
        path=static_root / "owned_finite_beta_closure_target_audit.json",
        metric_key="best_single_physics_driver_abs_pearson",
        details=(
            "monitored finite-beta closure-target driver ranking; this checks "
            "whether the source-response target follows local neoclassical "
            "drivers before any runtime closure change is proposed"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_radial_interpolation_stress",
        path=static_root / "owned_finite_beta_radial_interpolation_audit.json",
        metric_key="field_radius_matched_max_relative_error_total_vs_redl",
        details=(
            "monitored finite-beta profile-current sensitivity to rebuilding "
            "the monoenergetic database on the exact field radii; this separates "
            "radial interpolation sensitivity from closure physics without "
            "changing the runtime default"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_field_radius_matched_quadrature_stress",
        path=(
            static_root
            / "owned_finite_beta_field_radius_matched_closure_quadrature_audit.json"
        ),
        metric_key="quadrature_stable_gate_pass_count",
        details=(
            "monitored matched-radius finite-beta closure quadrature pass count; "
            "zero means no current-gate pass survives the velocity-quadrature "
            "stability requirement after the sparse radial interpolation layer "
            "is removed"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_field_radius_matched_source_reconstruction",
        path=(
            static_root
            / "owned_finite_beta_field_radius_matched_source_channel_audit.json"
        ),
        metric_key="max_source_channel_superposition_relative_residual",
        details=(
            "matched-radius finite-beta source-channel decomposition of the "
            "same momentum-restoring solve; the one-channel RHS solves must "
            "reconstruct the full corrected current before source-response "
            "interpretation"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_field_radius_matched_temperature_response_stress",
        path=(
            static_root
            / "owned_finite_beta_field_radius_matched_source_channel_audit.json"
        ),
        metric_key="high_stable_effective_temperature_response_multiplier_to_redl",
        details=(
            "monitored matched-radius ratio between the Redl effective-"
            "temperature target channel and the high-order corrected source "
            "channel; this localizes the remaining closure response without "
            "applying a fitted runtime correction"
        ),
    )


__all__ = ["append_finite_beta_artifact_gates"]
