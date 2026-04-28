from __future__ import annotations

from ._physics_gate_types import PhysicsGate

ARTIFACT_GATES: tuple[PhysicsGate, ...] = (
    PhysicsGate(
        name="monoenergetic_validation_summary",
        category="analytical",
        metric="max finest plotted Legendre-convergence error",
        relation="<=",
        threshold=2.5e-1,
        source="docs/_static/validation_summary.json",
        rationale=(
            "The repository-owned DKES-style and VMEC validation surfaces must "
            "show bounded Legendre convergence for the promoted monoenergetic "
            "coefficient benchmark before broader literature comparisons are "
            "interpreted."
        ),
    ),
    PhysicsGate(
        name="w7x_integrated_rebuild_raw",
        category="transfer",
        metric="best W7-X imported-workflow max relative error",
        relation="<=",
        threshold=2.0e-2,
        source="docs/_static/bootstrap_current_reference_audit_w7x.json",
        rationale=(
            "The rebuilt raw-branch integrated workflow must stay aligned with "
            "the frozen W7-X reference profile."
        ),
    ),
    PhysicsGate(
        name="prepared_derivative_path_consistency",
        category="analytical",
        metric="max relative prepared-vs-direct derivative mismatch",
        relation="<=",
        threshold=1.0e-4,
        source="docs/_static/derivative_path_benchmark.json",
        rationale=(
            "The prepared custom-VJP derivative path supports sensitivity, "
            "inverse-design, and uncertainty workflows, so it must agree with "
            "direct JAX differentiation on the committed scalar benchmark."
        ),
    ),
    PhysicsGate(
        name="geometry_control_derivative_stress",
        category="stress",
        metric="max relative direct-AD vs finite-difference mismatch",
        relation="<=",
        threshold=2.0e-4,
        source="docs/_static/geometry_control_derivative_benchmark.json",
        rationale=(
            "Owned analytic geometry-control derivatives should agree with "
            "centered finite differences before the same workflow is used for "
            "sensitivity, inverse-design, or uncertainty studies."
        ),
    ),
    PhysicsGate(
        name="file_backed_geometry_control_derivative_stress",
        category="stress",
        metric="max relative direct-AD vs finite-difference mismatch",
        relation="<=",
        threshold=5.0e-4,
        source="docs/_static/file_backed_geometry_control_derivative_benchmark.json",
        rationale=(
            "The geometry-control derivative audit must transfer from the "
            "owned analytic surface to repository-owned file-backed Boozer and "
            "VMEC sample surfaces."
        ),
    ),
    PhysicsGate(
        name="boundary_forward_mode_current_derivative_stress",
        category="stress",
        metric="max relative forward-mode vs finite-difference mismatch",
        relation="<=",
        threshold=1.0e-5,
        source="docs/_static/boundary_forward_mode_current_derivative_benchmark.json",
        rationale=(
            "The boundary-projected geometry path through optional JAX "
            "geometry backends, NTX, and the integrated-current objective must "
            "stay differentiable on the committed sample case."
        ),
    ),
    PhysicsGate(
        name="explicit_relaxed_boundary_current_derivative_stress",
        category="stress",
        metric="max relative forward-mode vs finite-difference mismatch",
        relation="<=",
        threshold=1.0e-4,
        source=(
            "docs/_static/"
            "explicit_relaxed_boundary_current_derivative_benchmark.json"
        ),
        rationale=(
            "The explicit-relaxed boundary-to-current family should preserve "
            "forward-mode agreement with centered finite differences on the "
            "committed QA/QH cases."
        ),
    ),
    PhysicsGate(
        name="implicit_equilibrium_derivative_nonshipping_diagnostic",
        category="stress",
        metric="max relative forward-mode vs finite-difference mismatch",
        relation="monitor",
        threshold=None,
        source=(
            "docs/_static/"
            "implicit_equilibrium_forward_mode_derivative_benchmark.json"
        ),
        rationale=(
            "The implicit-equilibrium derivative diagnostic is kept visible but "
            "closed as a non-shipping lane: residual contraction and Boozer/NTX "
            "transport tangent parity do not yet pass."
        ),
    ),
    PhysicsGate(
        name="geometry_family_transport_convergence_stress",
        category="stress",
        metric="max last-step relative D11/D31/D33 change across solved VMEC families",
        relation="monitor",
        threshold=None,
        source="docs/_static/geometry_family_transport_convergence.json",
        rationale=(
            "Broad VMEC example families should at least produce finite "
            "transport coefficients with visible grid-convergence behavior "
            "before they are promoted to independent-code parity claims."
        ),
    ),
    PhysicsGate(
        name="bootstrap_current_optimization_gain",
        category="stress",
        metric="weighted optimized-current gain",
        relation=">=",
        threshold=1.0,
        source="docs/_static/bootstrap_current_optimization.json",
        rationale=(
            "The differentiable bootstrap-current optimization figure should "
            "remain an actual improvement over the committed baseline before "
            "the manuscript cites the weighted-gain number."
        ),
    ),
    PhysicsGate(
        name="precise_qs_redl_vs_sfincs",
        category="independent",
        metric="interior max relative error of Redl vs archived SFINCS",
        relation="<=",
        threshold=1.0e-1,
        source="docs/_static/bootstrap_current_fixed_field_validation.json",
        rationale=(
            "The precise-QS benchmark should reproduce the established Redl "
            "agreement with archived SFINCS on the interior window."
        ),
    ),
    PhysicsGate(
        name="precise_qs_ntx_neopax_closure_stress",
        category="stress",
        metric="interior max relative error of NTX+NEOPAX vs archived SFINCS",
        relation="<=",
        threshold=1.0e-1,
        source="docs/_static/bootstrap_current_fixed_field_validation.json",
        rationale=(
            "The reduced fixed-field closure must reproduce the archived "
            "precise-QS total bootstrap-current profile within the documented "
            "interior-window stress tolerance without fitted constants."
        ),
    ),
    PhysicsGate(
        name="pmax_convergence_precise_qs",
        category="stress",
        metric="max relative change between successive Pmax levels",
        relation="monitor",
        threshold=None,
        source="docs/_static/closure_pmax_convergence.json",
        rationale=(
            "Higher-order closure work must show controlled convergence in "
            "Pmax on the precise-QS QA/QH stress family."
        ),
    ),
    PhysicsGate(
        name="w7x_pmax_transfer_regression",
        category="transfer",
        metric="integrated W7-X max relative error under higher-order closure",
        relation="monitor",
        threshold=None,
        source="docs/_static/closure_pmax_convergence.json",
        rationale=(
            "Any higher-order closure extension must transfer to the "
            "integrated W7-X workflow without regressing the imported path."
        ),
    ),
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
)

__all__ = ["ARTIFACT_GATES"]
