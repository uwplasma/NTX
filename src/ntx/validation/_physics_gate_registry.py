from __future__ import annotations

from ._physics_gate_analytical import ANALYTICAL_GATES
from ._physics_gate_artifact_registry import ARTIFACT_GATES
from ._physics_gate_types import PhysicsGate


def physics_gate_registry() -> tuple[PhysicsGate, ...]:
    """Return analytical and artifact-backed physics gates in stable order."""

    return ANALYTICAL_GATES + ARTIFACT_GATES


def _gate_by_name(name: str) -> PhysicsGate:
    for gate in physics_gate_registry():
        if gate.name == name:
            return gate
    raise KeyError(name)


__all__ = [
    "ANALYTICAL_GATES",
    "ARTIFACT_GATES",
    "_gate_by_name",
    "physics_gate_registry",
]
