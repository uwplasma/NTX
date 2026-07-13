"""Validation registries and benchmark metadata for NTX."""

from ._angular_oversampling import (
    AngularOversamplingAudit,
    AngularOversamplingPoint,
    audit_angular_oversampling,
)
from .benchmark_matrix import (
    BenchmarkEntry,
    BenchmarkEvaluation,
    BenchmarkPathStatus,
    benchmark_matrix,
    evaluate_benchmark_matrix,
    write_benchmark_matrix_json,
)
from .physics_gates import (
    ANALYTICAL_GATES,
    ARTIFACT_GATES,
    PhysicsGate,
    PhysicsGateResult,
    evaluate_artifact_gates,
    physics_gate_registry,
)

__all__ = [
    "ANALYTICAL_GATES",
    "ARTIFACT_GATES",
    "AngularOversamplingAudit",
    "AngularOversamplingPoint",
    "BenchmarkEntry",
    "BenchmarkEvaluation",
    "BenchmarkPathStatus",
    "PhysicsGate",
    "PhysicsGateResult",
    "benchmark_matrix",
    "audit_angular_oversampling",
    "evaluate_artifact_gates",
    "evaluate_benchmark_matrix",
    "physics_gate_registry",
    "write_benchmark_matrix_json",
]
