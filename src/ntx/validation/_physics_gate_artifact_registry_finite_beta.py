from __future__ import annotations

from ._physics_gate_types import PhysicsGate

FINITE_BETA_ARTIFACT_GATES: tuple[PhysicsGate, ...] = (
    PhysicsGate(
        name="owned_finite_beta_same_grid_coefficient_stress",
        category="independent",
        metric="max same-grid finite-beta SFINCS-JAX vs NTX coefficient difference",
        relation="<=",
        threshold=1.0e-1,
        source="docs/_static/owned_finite_beta_closure_localization.json",
        rationale=(
            "The finite-beta bootstrap-current stress audit should not be "
            "interpreted until the same-radius, same-collisionality "
            "transport-matrix coefficients are normalized consistently against "
            "the independently generated finite-beta runs."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_profile_current_observable_stress",
        category="stress",
        metric="stress-radius finite-beta profile-current relative difference",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_profile_current_observable_audit.json",
        rationale=(
            "The finite-beta profile-current observable is deliberately kept as "
            "a monitored reduced-closure stress diagnostic until production "
            "same-grid profile-current diagnostics pass; it is not a parity "
            "gate."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_species_cancellation_stress",
        category="stress",
        metric="stress-radius residual divided by species-correction L1 scale",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_profile_current_observable_audit.json",
        rationale=(
            "The finite-beta net current is cancellation-dominated at the "
            "stress radius, so species-current imbalance must be tracked "
            "separately from the net-current relative error before any broader "
            "closure change is promoted."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_current_conditioning_stress",
        category="stress",
        metric="max coefficient precision gap for the 1e-1 current gate",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_current_conditioning_audit.json",
        rationale=(
            "The finite-beta net-current observable can amplify small species-flow "
            "or coefficient errors.  This monitor reports how much tighter the "
            "same-grid coefficient ladder must be before the remaining current "
            "gap can be assigned to the reduced profile-current closure rather "
            "than coefficient uncertainty."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_resolution_floor_stress",
        category="stress",
        metric="production same-grid coefficient precision gap for the 1e-1 current gate",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_sfincs_jax_resolution_audit.json",
        rationale=(
            "The finite-beta stress-radius coefficient floor must be separated "
            "from angular/pitch resolution and VMEC harmonic truncation before "
            "the remaining net-current gap is assigned to the profile-current "
            "closure."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_production_ladder_stress",
        category="stress",
        metric="max production-ladder coefficient precision gap for the 1e-1 current gate",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_sfincs_jax_production_ladder_audit.json",
        rationale=(
            "The finite-beta QA production radial/collisionality ladder should "
            "show whether the coefficient floor is localized or a whole-profile "
            "resolution failure before the remaining current residual is assigned "
            "to the profile-current closure."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_sfincs_jax_profile_current_stress",
        category="stress",
        metric="max RHSMode=1 SFINCS-JAX finite-beta profile-current difference",
        relation="monitor",
        threshold=None,
        source=(
            "docs/_static/"
            "owned_finite_beta_sfincs_jax_profile_current_audit.json"
        ),
        rationale=(
            "The owned finite-beta SFINCS-JAX profile-current diagnostic runs "
            "RHSMode=1 on the same VMEC/profile contract as Redl and "
            "NTX+NEOPAX. It remains a monitored stress lane, not a parity gate, "
            "until pitch, velocity, radial, and collisionality-normalization "
            "ladders are complete."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_closure_quadrature_stress",
        category="stress",
        metric="under-integrated finite-beta closure gate-pass count",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_closure_quadrature_audit.json",
        rationale=(
            "Higher Sonine order in the finite-beta profile-current closure "
            "must transfer to higher velocity quadrature before any apparent "
            "current-gate pass is interpreted as physical convergence."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_source_channel_reconstruction",
        category="stress",
        metric="max source-channel superposition relative residual",
        relation="<=",
        threshold=1.0e-8,
        source="docs/_static/owned_finite_beta_source_channel_audit.json",
        rationale=(
            "The finite-beta source-channel diagnostic must reconstruct the "
            "same momentum-restoring linear solve from one-channel RHS solves "
            "before its density/electric, temperature-gradient, and "
            "parallel-electric decomposition is interpreted physically."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_temperature_source_response_stress",
        category="stress",
        metric="high-order Redl/NTX effective-temperature source response multiplier",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_source_channel_audit.json",
        rationale=(
            "The finite-beta profile-current closure gap is localized to the "
            "effective temperature-gradient source channel. The Redl target "
            "response ratio is tracked as a physics diagnostic, not as a "
            "runtime fit or acceptance gate."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_profile_source_response_stress",
        category="stress",
        metric="high-order profile span of Redl/NTX temperature-source response",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_source_response_profile_audit.json",
        rationale=(
            "The finite-beta source-response mismatch should be mapped across "
            "the profile and compared with physical drivers before any "
            "reduced-closure change is promoted. The radial response span is a "
            "diagnostic, not a fitted correction."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_closure_target_driver_stress",
        category="stress",
        metric="best single-driver correlation with finite-beta temperature-source response",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_closure_target_audit.json",
        rationale=(
            "Before any finite-beta profile-current closure change is promoted, "
            "the measured source-response target should be compared with local "
            "neoclassical drivers such as trapped fraction, geometry factor, "
            "and collisionality. This gate tracks model identifiability only; "
            "it is not a runtime fit."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_radial_interpolation_stress",
        category="stress",
        metric="field-radius-matched finite-beta current-profile relative difference",
        relation="monitor",
        threshold=None,
        source="docs/_static/owned_finite_beta_radial_interpolation_audit.json",
        rationale=(
            "Finite-beta bootstrap-current closure diagnostics must separate "
            "radial database interpolation from the reduced momentum/profile "
            "closure.  This monitor tracks the same profile-current observable "
            "after rebuilding the database on the exact field radii; it is not "
            "a promoted runtime interpolation policy."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_field_radius_matched_quadrature_stress",
        category="stress",
        metric="field-radius-matched quadrature-stable closure gate-pass count",
        relation="monitor",
        threshold=None,
        source=(
            "docs/_static/"
            "owned_finite_beta_field_radius_matched_closure_quadrature_audit.json"
        ),
        rationale=(
            "After removing the sparse-radius interpolation layer, finite-beta "
            "closure improvements must still transfer to velocity quadrature "
            "at least as large as the Sonine truncation. This monitor separates "
            "field-radius interpolation sensitivity from under-integrated "
            "closure aliasing."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_field_radius_matched_source_reconstruction",
        category="stress",
        metric="field-radius-matched source-channel superposition relative residual",
        relation="<=",
        threshold=1.0e-8,
        source=(
            "docs/_static/"
            "owned_finite_beta_field_radius_matched_source_channel_audit.json"
        ),
        rationale=(
            "After removing the sparse-radius interpolation layer, the "
            "field-radius-matched source-channel diagnostic must still "
            "reconstruct the same momentum-restoring solve from one-channel "
            "RHS solves before its physical channel split is interpreted."
        ),
    ),
    PhysicsGate(
        name="owned_finite_beta_field_radius_matched_temperature_response_stress",
        category="stress",
        metric="field-radius-matched Redl/NTX effective-temperature source response multiplier",
        relation="monitor",
        threshold=None,
        source=(
            "docs/_static/"
            "owned_finite_beta_field_radius_matched_source_channel_audit.json"
        ),
        rationale=(
            "The matched-radius finite-beta profile-current gap should be "
            "localized to physical source-channel response after interpolation "
            "is removed. The temperature response ratio is tracked as a "
            "diagnostic, not as a fitted runtime correction."
        ),
    ),
)

__all__ = ["FINITE_BETA_ARTIFACT_GATES"]
