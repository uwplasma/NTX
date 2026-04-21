from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
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
        payload = {
            "gate": self.gate.as_dict(),
            "value": self.value,
            "status": self.status,
        }
        if self.details:
            payload["details"] = self.details
        return payload


ANALYTICAL_GATES: tuple[PhysicsGate, ...] = (
    PhysicsGate(
        name="onsager_symmetry",
        category="analytical",
        metric="|D13 + D31|",
        relation="test",
        threshold=None,
        source="tests/test_solver.py and examples/validation_summary.py",
        rationale=(
            "The monoenergetic solve must preserve the Onsager symmetry expected "
            "for the source split and the Legendre-space discretization."
        ),
    ),
    PhysicsGate(
        name="p2_projection_exact_recovery",
        category="analytical",
        metric="generated Sonine/Hankel P=2 recovery",
        relation="<=",
        threshold=1.0e-12,
        source="local imported closure tests/test_moment_projection.py",
        rationale=(
            "Any higher-order closure work must reduce exactly to the present "
            "three-moment system at P=2 before new physics is introduced."
        ),
    ),
    PhysicsGate(
        name="observable_map_fixed",
        category="analytical",
        metric="U_parallel = n c0",
        relation="test",
        threshold=None,
        source="closure derivation in the manuscript and fixed-field audits",
        rationale=(
            "The parallel-flow observable is fixed by the Sonine basis and must "
            "not be changed to fit a benchmark."
        ),
    ),
)


ARTIFACT_GATES: tuple[PhysicsGate, ...] = (
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
        relation="monitor",
        threshold=None,
        source="docs/_static/bootstrap_current_fixed_field_validation.json",
        rationale=(
            "This benchmark is retained as a closure-model stress test. It is "
            "monitored continuously but is not a solver-side release gate."
        ),
    ),
)


def physics_gate_registry() -> tuple[PhysicsGate, ...]:
    return ANALYTICAL_GATES + ARTIFACT_GATES


def evaluate_artifact_gates(root: Path) -> list[PhysicsGateResult]:
    root = Path(root)
    static_root = root / "docs" / "_static"
    results: list[PhysicsGateResult] = []

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

    return results


def _gate_by_name(name: str) -> PhysicsGate:
    for gate in physics_gate_registry():
        if gate.name == name:
            return gate
    raise KeyError(name)


def _evaluate_scalar_gate(gate: PhysicsGate, value: float) -> PhysicsGateResult:
    if gate.relation == "<=":
        status: GateStatus = "pass" if value <= float(gate.threshold) else "fail"
    elif gate.relation == ">=":
        status = "pass" if value >= float(gate.threshold) else "fail"
    elif gate.relation == "monitor":
        status = "monitor"
    else:
        status = "monitor"
    return PhysicsGateResult(gate=gate, value=value, status=status)
