"""The physics gates: analytical limits and artifact thresholds, evaluated.

Public entry point for the gate registry and its evaluation. Analytical gates
compare against a closed-form limit; artifact gates compare a committed record
against a declared threshold. Both report pass, fail, monitor or missing, and a
missing artifact is never silently a pass.
"""

from __future__ import annotations

from ._physics_gate import (
    ANALYTICAL_GATES,
    ARTIFACT_GATES,
    _gate_by_name,
    physics_gate_registry,
)
from ._physics_gate_artifacts import (
    _append_missing_artifact_gate,
    _append_summary_metric_gate,
    _evaluate_scalar_gate,
    evaluate_artifact_gates,
)
from ._physics_gate_types import (
    GateCategory,
    GateRelation,
    GateStatus,
    PhysicsGate,
    PhysicsGateResult,
)

__all__ = [
    "ANALYTICAL_GATES",
    "ARTIFACT_GATES",
    "GateCategory",
    "GateRelation",
    "GateStatus",
    "PhysicsGate",
    "PhysicsGateResult",
    "_append_missing_artifact_gate",
    "_append_summary_metric_gate",
    "_evaluate_scalar_gate",
    "_gate_by_name",
    "evaluate_artifact_gates",
    "physics_gate_registry",
]
