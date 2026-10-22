"""Producing physics-gate artifacts and evaluating them into a pass or fail.

The writers that produce each artifact at zero and finite beta, and the
evaluation that turns one into a gate result.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._physics_gate import _gate_by_name
from ._physics_gate_types import GateStatus, PhysicsGate, PhysicsGateResult

__all__ = [
    "_append_missing_artifact_gate",
    "_append_summary_metric_gate",
    "_evaluate_scalar_gate",
    "append_finite_beta_artifact_gates",
    "evaluate_artifact_gates",
]


# --- _physics_gate_artifact_eval ---
# Evaluates one artifact-backed gate against its committed record.


def _append_summary_metric_gate(
    results: list[PhysicsGateResult],
    *,
    gate_name: str,
    path: Path,
    metric_key: str,
    details: str,
) -> None:
    """Evaluate one gate against a metric in a summary artifact.

    A missing artifact is recorded as a missing-artifact result rather than
    skipped, so an absent file fails the gate instead of quietly passing it.
    """
    gate = _gate_by_name(gate_name)
    if path.exists():
        payload = json.loads(path.read_text())
        value = float(payload["summary_metrics"][metric_key])
        results.append(_evaluate_scalar_gate(gate, value, details=details))
    else:
        _append_missing_artifact_gate(results, gate, path)


def _append_missing_artifact_gate(
    results: list[PhysicsGateResult],
    gate: PhysicsGate,
    path: Path,
) -> None:
    """Record a gate whose backing artifact is absent."""
    results.append(
        PhysicsGateResult(
            gate=gate,
            value=None,
            status="missing",
            details=f"missing artifact: {path}",
        )
    )


def _evaluate_scalar_gate(
    gate: PhysicsGate,
    value: float,
    *,
    details: str = "",
) -> PhysicsGateResult:
    """Compare a value against its gate threshold."""
    if gate.relation == "<=":
        assert gate.threshold is not None
        status: GateStatus = "pass" if value <= gate.threshold else "fail"
    elif gate.relation == ">=":
        assert gate.threshold is not None
        status = "pass" if value >= gate.threshold else "fail"
    elif gate.relation == "monitor":
        status = "monitor"
    else:
        status = "monitor"
    return PhysicsGateResult(gate=gate, value=value, status=status, details=details)


# --- _physics_gate_artifacts_finite_beta ---
# Artifact-backed physics gates for the finite-beta lane.


def append_finite_beta_artifact_gates(
    results: list[PhysicsGateResult],
    static_root: Path,
) -> None:
    """Evaluate every finite-beta artifact-backed gate."""
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
        details=("monitored species-current cancellation scale at the finite-beta stress radius"),
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
        path=(static_root / "owned_finite_beta_sfincs_jax_production_ladder_audit.json"),
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
        gate_name="owned_finite_beta_sfincs_jax_profile_current_stress",
        path=static_root / "owned_finite_beta_sfincs_jax_profile_current_audit.json",
        metric_key="max_sfincs_jax_relative_error_vs_redl",
        details=(
            "monitored finite-beta RHSMode=1 SFINCS-JAX profile-current "
            "diagnostic on the same owned VMEC/profile contract as Redl and "
            "NTX+NEOPAX; this is not a promoted parity gate until the direct "
            "profile-current convergence and normalization ladder is complete"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="owned_finite_beta_sfincs_jax_profile_current_pitch_resolution_stress",
        path=(static_root / "owned_finite_beta_sfincs_jax_profile_current_resolution_audit.json"),
        metric_key="tail_even_odd_relative_gap",
        details=(
            "accepted finite-beta RHSMode=1 SFINCS-JAX pitch Legendre "
            "truncation stress metric; adjacent high-Nxi parity branches "
            "are below the 1.5e-1 reduced-closure stress tolerance"
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
        path=(static_root / "owned_finite_beta_field_radius_matched_closure_quadrature_audit.json"),
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
        path=(static_root / "owned_finite_beta_field_radius_matched_source_channel_audit.json"),
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
        path=(static_root / "owned_finite_beta_field_radius_matched_source_channel_audit.json"),
        metric_key="high_stable_effective_temperature_response_multiplier_to_redl",
        details=(
            "monitored matched-radius ratio between the Redl effective-"
            "temperature target channel and the high-order corrected source "
            "channel; this localizes the remaining closure response without "
            "applying a fitted runtime correction"
        ),
    )


# --- _physics_gate_artifacts ---
# Artifact-backed physics gates: committed records against thresholds.


def evaluate_artifact_gates(root: Path) -> list[PhysicsGateResult]:
    """Evaluate committed artifact-backed gates below a repository root.

    Missing artifacts produce ``missing`` results rather than exceptions so
    validation reports can distinguish absent evidence from failed thresholds.
    """

    root = Path(root)
    static_root = root / "docs" / "_static"
    results: list[PhysicsGateResult] = []

    validation_gate = _gate_by_name("monoenergetic_validation_summary")
    validation_path = static_root / "validation_summary.json"
    if validation_path.exists():
        payload = json.loads(validation_path.read_text())
        metrics = payload["summary_metrics"]
        finest_error = max(
            float(metrics["dkes_finest_plotted_error"]),
            float(metrics["vmec_finest_plotted_error"]),
        )
        results.append(
            PhysicsGateResult(
                gate=validation_gate,
                value=finest_error,
                status="pass"
                if finest_error <= float(validation_gate.threshold or 0.0)
                else "fail",
                details=(
                    "max of DKES-style and VMEC finest plotted N_xi errors "
                    "against the finest validation-summary reference"
                ),
            )
        )
    else:
        results.append(
            PhysicsGateResult(
                gate=validation_gate,
                value=None,
                status="missing",
                details=f"missing artifact: {validation_path}",
            )
        )

    w7x_gate = _gate_by_name("w7x_integrated_rebuild_raw")
    w7x_path = static_root / "bootstrap_current_reference_audit_w7x.json"
    if w7x_path.exists():
        payload = json.loads(w7x_path.read_text())
        best_error = min(
            float(item["max_relative_error"]) for item in payload["bootstrap_current_errors"]
        )
        results.append(_evaluate_scalar_gate(w7x_gate, best_error))
    else:
        results.append(
            PhysicsGateResult(
                gate=w7x_gate,
                value=None,
                status="missing",
                details=f"missing artifact: {w7x_path}",
            )
        )

    derivative_gate = _gate_by_name("prepared_derivative_path_consistency")
    derivative_path = static_root / "derivative_path_benchmark.json"
    if derivative_path.exists():
        payload = json.loads(derivative_path.read_text())
        max_mismatch = max(float(item) for item in payload["max_relative_mismatch"])
        speedups = [float(item) for item in payload["speedup_prepared_vs_direct"]]
        results.append(
            PhysicsGateResult(
                gate=derivative_gate,
                value=max_mismatch,
                status="pass"
                if max_mismatch <= float(derivative_gate.threshold or 0.0)
                else "fail",
                details=(
                    "prepared derivative path compared with direct reverse-mode; "
                    f"minimum reported speedup={min(speedups):.3g}"
                ),
            )
        )
    else:
        results.append(
            PhysicsGateResult(
                gate=derivative_gate,
                value=None,
                status="missing",
                details=f"missing artifact: {derivative_path}",
            )
        )

    _append_summary_metric_gate(
        results,
        gate_name="geometry_control_derivative_stress",
        path=static_root / "geometry_control_derivative_benchmark.json",
        metric_key="max_relative_mismatch",
        details=(
            "owned analytic geometry-control direct AD compared with centered finite differences"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="file_backed_geometry_control_derivative_stress",
        path=static_root / "file_backed_geometry_control_derivative_benchmark.json",
        metric_key="max_relative_mismatch",
        details=(
            "file-backed Boozer/VMEC geometry-control direct AD compared with "
            "centered finite differences"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="boundary_forward_mode_current_derivative_stress",
        path=static_root / "boundary_forward_mode_current_derivative_benchmark.json",
        metric_key="max_relative_mismatch",
        details=(
            "boundary-projected forward-mode derivatives compared with centered finite differences"
        ),
    )

    explicit_gate = _gate_by_name("explicit_relaxed_boundary_current_derivative_stress")
    explicit_path = static_root / "explicit_relaxed_boundary_current_derivative_benchmark.json"
    if explicit_path.exists():
        payload = json.loads(explicit_path.read_text())
        metrics = payload["summary_metrics"]
        max_mismatch = float(metrics["max_relative_mismatch"])
        volume_difference = float(metrics["max_ordinary_explicit_volume_relative_difference"])
        results.append(
            _evaluate_scalar_gate(
                explicit_gate,
                max_mismatch,
                details=(
                    "explicit-relaxed forward-mode derivatives compared with "
                    "centered finite differences; max ordinary-vs-explicit "
                    f"volume relative difference={volume_difference:.3g}"
                ),
            )
        )
    else:
        _append_missing_artifact_gate(results, explicit_gate, explicit_path)

    _append_summary_metric_gate(
        results,
        gate_name="implicit_equilibrium_derivative_nonshipping_diagnostic",
        path=static_root / "implicit_equilibrium_forward_mode_derivative_benchmark.json",
        metric_key="max_relative_mismatch",
        details=(
            "monitored implicit-equilibrium diagnostic closed as non-shipping; "
            "the explicit-relaxed path remains the supported equilibrium "
            "derivative route"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="geometry_family_transport_convergence_stress",
        path=static_root / "geometry_family_transport_convergence.json",
        metric_key="max_successful_last_step_relative_change",
        details=(
            "production-grid D11/D31/D33 last-step convergence across reusable "
            "VMEC geometry families; not an independent-code parity gate"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="angular_oversampling_convergence_stress",
        path=static_root / "angular_oversampling_audit.json",
        metric_key="max_recommended_relative_error",
        details=(
            "measured D11/D31/D33 error at the warning-level angular "
            "oversampling recommendation relative to the finest audit grid"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="boozmn_same_coordinate_roundtrip",
        path=static_root / "boozmn_same_coordinate_roundtrip_audit.json",
        metric_key="max_transport_relative_difference",
        details=(
            "same-coordinate VMEC half-grid Boozer-file round trip compared "
            "with the in-memory vmex/booz_xform_jax path"
        ),
    )
    _append_summary_metric_gate(
        results,
        gate_name="boozmn_finite_beta_wout_roundtrip",
        path=static_root / "boozmn_finite_beta_wout_roundtrip_audit.json",
        metric_key="max_transport_relative_difference",
        details=(
            "finite-beta finalized-wout magnetic-channel Boozer transform "
            "round trip compared on the same VMEC half-grid surfaces"
        ),
    )

    optimization_gate = _gate_by_name("bootstrap_current_optimization_gain")
    optimization_path = static_root / "bootstrap_current_optimization.json"
    if optimization_path.exists():
        payload = json.loads(optimization_path.read_text())
        weighted_gain = float(payload["weighted_gain"])
        results.append(
            _evaluate_scalar_gate(
                optimization_gate,
                weighted_gain,
                details=(
                    "optimized weighted reduced bootstrap-current response divided by baseline"
                ),
            )
        )
    else:
        _append_missing_artifact_gate(results, optimization_gate, optimization_path)

    fixed_gate_redl = _gate_by_name("precise_qs_redl_vs_sfincs")
    fixed_gate_closure = _gate_by_name("precise_qs_ntx_neopax_closure_stress")
    fixed_path = static_root / "bootstrap_current_fixed_field_validation.json"
    if fixed_path.exists():
        payload = json.loads(fixed_path.read_text())
        redl_error = max(
            float(case["max_relative_error_vs_sfincs_interior"]["Redl"])
            for case in payload["cases"].values()
        )
        closure_error = max(
            float(case["max_relative_error_vs_sfincs_interior"]["NTX+NEOPAX"])
            for case in payload["cases"].values()
        )
        results.append(_evaluate_scalar_gate(fixed_gate_redl, redl_error))
        results.append(
            _evaluate_scalar_gate(
                fixed_gate_closure,
                closure_error,
                details=(
                    "fixed-field reduced-closure total-current stress metric; "
                    "not an independent species-current parity gate"
                ),
            )
        )
    else:
        for gate in (fixed_gate_redl, fixed_gate_closure):
            results.append(
                PhysicsGateResult(
                    gate=gate,
                    value=None,
                    status="missing",
                    details=f"missing artifact: {fixed_path}",
                )
            )

    convergence_path = static_root / "closure_pmax_convergence.json"
    for gate in (
        _gate_by_name("pmax_convergence_precise_qs"),
        _gate_by_name("w7x_pmax_transfer_regression"),
    ):
        if convergence_path.exists():
            payload = json.loads(convergence_path.read_text())
            key = (
                "precise_qs_max_successive_change"
                if gate.name == "pmax_convergence_precise_qs"
                else "w7x_max_relative_error"
            )
            results.append(
                PhysicsGateResult(
                    gate=gate,
                    value=float(payload[key]),
                    status="monitor",
                    details="tracked for higher-order closure development",
                )
            )
        else:
            results.append(
                PhysicsGateResult(
                    gate=gate,
                    value=None,
                    status="missing",
                    details=f"missing artifact: {convergence_path}",
                )
            )

    append_finite_beta_artifact_gates(results, static_root)

    return results
