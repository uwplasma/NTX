"""Evaluates one artifact-backed gate against its committed record."""

from __future__ import annotations

import json
from pathlib import Path

from ._physics_gate_registry import _gate_by_name
from ._physics_gate_types import GateStatus, PhysicsGate, PhysicsGateResult


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
]
