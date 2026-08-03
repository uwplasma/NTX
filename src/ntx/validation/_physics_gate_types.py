"""Types for physics gates: category, relation and status.

A gate is a claim plus the comparison that decides it. Keeping the vocabulary
here means a new relation or status is constrained at every use site.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

GateCategory = Literal["analytical", "independent", "transfer", "stress"]
GateRelation = Literal["<=", ">=", "monitor", "test"]
GateStatus = Literal["pass", "fail", "monitor", "missing"]


@dataclass(frozen=True)
class PhysicsGate:
    name: str
    category: GateCategory
    metric: str
    relation: GateRelation
    threshold: float | None
    source: str
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicsGateResult:
    gate: PhysicsGate
    value: float | None
    status: GateStatus
    details: str = ""

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "gate": self.gate.as_dict(),
            "value": self.value,
            "status": self.status,
        }
        if self.details:
            payload["details"] = self.details
        return payload


__all__ = [
    "GateCategory",
    "GateRelation",
    "GateStatus",
    "PhysicsGate",
    "PhysicsGateResult",
]
