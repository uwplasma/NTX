"""Compatibility facade for NTX physics-gate validation helpers."""

from .validation.physics_gates import (
    ANALYTICAL_GATES,
    ARTIFACT_GATES,
    GateCategory,
    GateRelation,
    GateStatus,
    PhysicsGate,
    PhysicsGateResult,
    evaluate_artifact_gates,
    physics_gate_registry,
)

__all__ = [
    "ANALYTICAL_GATES",
    "ARTIFACT_GATES",
    "GateCategory",
    "GateRelation",
    "GateStatus",
    "PhysicsGate",
    "PhysicsGateResult",
    "evaluate_artifact_gates",
    "physics_gate_registry",
]
