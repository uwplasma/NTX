from __future__ import annotations

import json
from pathlib import Path

from ._physics_gate_registry import _gate_by_name
from ._physics_gate_types import GateStatus, PhysicsGate, PhysicsGateResult


def evaluate_artifact_gates(root: Path) -> list[PhysicsGateResult]:
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
            float(item["max_relative_error"])
            for item in payload["bootstrap_current_errors"]
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
            "owned analytic geometry-control direct AD compared with centered "
            "finite differences"
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
            "boundary-projected forward-mode derivatives compared with "
            "centered finite differences"
        ),
    )

    explicit_gate = _gate_by_name("explicit_relaxed_boundary_current_derivative_stress")
    explicit_path = (
        static_root / "explicit_relaxed_boundary_current_derivative_benchmark.json"
    )
    if explicit_path.exists():
        payload = json.loads(explicit_path.read_text())
        metrics = payload["summary_metrics"]
        max_mismatch = float(metrics["max_relative_mismatch"])
        volume_difference = float(
            metrics["max_ordinary_explicit_volume_relative_difference"]
        )
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
        gate_name="implicit_equilibrium_derivative_open_stress",
        path=static_root / "implicit_equilibrium_forward_mode_derivative_benchmark.json",
        metric_key="max_relative_mismatch",
        details=(
            "monitored implicit-equilibrium diagnostic; volume derivative "
            "closes while Boozer-space and NTX transport observables remain open"
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
                details="optimized weighted bootstrap-current proxy divided by baseline",
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
            PhysicsGateResult(
                gate=fixed_gate_closure,
                value=closure_error,
                status="monitor",
                details="tracked as a closure-model stress metric, not a parity gate",
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

    return results


def _append_summary_metric_gate(
    results: list[PhysicsGateResult],
    *,
    gate_name: str,
    path: Path,
    metric_key: str,
    details: str,
) -> None:
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


__all__ = [
    "_append_missing_artifact_gate",
    "_append_summary_metric_gate",
    "_evaluate_scalar_gate",
    "evaluate_artifact_gates",
]
