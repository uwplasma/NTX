"""Validation registries and benchmark metadata for NTX."""

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
    "BenchmarkEntry",
    "BenchmarkEvaluation",
    "BenchmarkPathStatus",
    "PhysicsGate",
    "PhysicsGateResult",
    "benchmark_matrix",
    "evaluate_artifact_gates",
    "evaluate_benchmark_matrix",
    "physics_gate_registry",
    "write_benchmark_matrix_json",
]
