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
        payload: dict[str, object] = {
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
        name="low_order_collision_block_recovery",
        category="analytical",
        metric="generated low-order momentum-conserving collision blocks",
        relation="test",
        threshold=None,
        source="standard low-order moment equations and local closure tests",
        rationale=(
            "The active low-order collisional blocks must be reproducible from "
            "the standard momentum-conserving moment equations, up to the "
            "heat-flow basis convention used by the runtime."
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
    PhysicsGate(
        name="intrinsic_ambipolarity_symmetric_limit",
        category="analytical",
        metric="symmetric-limit ambipolar structure preserved",
        relation="test",
        threshold=None,
        source=(
            "Sugama–Nishimura finite-order moment-equation requirements and "
            "tests/test_physics_gates.py"
        ),
        rationale=(
            "At every finite truncation, the projected closure must preserve the "
            "intrinsic ambipolar-diffusion structure in symmetric limits."
        ),
    ),
    PhysicsGate(
        name="spitzer_inverse_collisionality_limit",
        category="analytical",
        metric="constant-field D33_spitzer proportional to 1/nu_hat",
        relation="test",
        threshold=None,
        source="tests/test_physics_gates.py",
        rationale=(
            "In the constant-field limit the drift source vanishes and the "
            "remaining parallel-conductivity branch should reduce to the "
            "Spitzer-like inverse-collisionality normalization used by NTX."
        ),
    ),
    PhysicsGate(
        name="operator_parameter_derivative_consistency",
        category="analytical",
        metric="dD_k/dnu_hat and dD_k/depsi_hat match operator autodiff",
        relation="test",
        threshold=None,
        source="tests/test_operators.py",
        rationale=(
            "The implicit-adjoint path differentiates through hand-coded "
            "parameter-derivative blocks, so those blocks must be exactly the "
            "derivatives of the assembled Legendre-space operator with respect "
            "to collisionality and radial-electric-field normalization."
        ),
    ),
    PhysicsGate(
        name="momentum_conservation_null_mode",
        category="analytical",
        metric="common-flow collisional null mode preserved",
        relation="test",
        threshold=None,
        source="momentum-restoring closure derivation and local closure tests",
        rationale=(
            "The higher-order collisional blocks must conserve total parallel "
            "momentum, so a common-flow null mode remains present."
        ),
    ),
    PhysicsGate(
        name="particle_conservation_invariant",
        category="analytical",
        metric="collisional particle invariant preserved",
        relation="test",
        threshold=None,
        source="linearized collision-operator moment-equation constraints",
        rationale=(
            "The projected collision model must not generate a spurious particle "
            "source at any truncation."
        ),
    ),
    PhysicsGate(
        name="energy_conservation_invariant",
        category="analytical",
        metric="collisional energy invariant preserved",
        relation="test",
        threshold=None,
        source="linearized collision-operator moment-equation constraints",
        rationale=(
            "The collisional blocks must preserve the energy invariant in the "
            "same projected basis used for higher-order closure."
        ),
    ),
    PhysicsGate(
        name="collision_operator_self_adjointness",
        category="analytical",
        metric="weighted collisional form is self-adjoint",
        relation="test",
        threshold=None,
        source="finite-order Laguerre/Sonine Coulomb-operator literature",
        rationale=(
            "The finite-order collisional operator should preserve the "
            "self-adjoint structure underlying Onsager symmetry and the H-theorem."
        ),
    ),
    PhysicsGate(
        name="entropy_production_nonnegative",
        category="analytical",
        metric="symmetric collisional form is positive semidefinite",
        relation="test",
        threshold=None,
        source="Sugama–Horton entropy-production constraints",
        rationale=(
            "The finite-order collision model must not violate the "
            "non-negativity of entropy production."
        ),
    ),
)


ARTIFACT_GATES: tuple[PhysicsGate, ...] = (
    PhysicsGate(
        name="monoenergetic_validation_summary",
        category="analytical",
        metric="max finest plotted Legendre-convergence error",
        relation="<=",
        threshold=2.5e-1,
        source="docs/_static/validation_summary.json",
        rationale=(
            "The repository-owned DKES-style and VMEC validation surfaces must "
            "show bounded Legendre convergence for the promoted monoenergetic "
            "coefficient benchmark before broader literature comparisons are "
            "interpreted."
        ),
    ),
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
    PhysicsGate(
        name="pmax_convergence_precise_qs",
        category="stress",
        metric="max relative change between successive Pmax levels",
        relation="monitor",
        threshold=None,
        source="future closure_pmax_convergence.json artifact",
        rationale=(
            "Higher-order closure work must show controlled convergence in "
            "Pmax on the precise-QS QA/QH stress family."
        ),
    ),
    PhysicsGate(
        name="w7x_pmax_transfer_regression",
        category="transfer",
        metric="integrated W7-X max relative error under higher-order closure",
        relation="monitor",
        threshold=None,
        source="future closure_pmax_convergence.json artifact",
        rationale=(
            "Any higher-order closure extension must transfer to the "
            "integrated W7-X workflow without regressing the imported path."
        ),
    ),
)


def physics_gate_registry() -> tuple[PhysicsGate, ...]:
    return ANALYTICAL_GATES + ARTIFACT_GATES


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


def _gate_by_name(name: str) -> PhysicsGate:
    for gate in physics_gate_registry():
        if gate.name == name:
            return gate
    raise KeyError(name)


def _evaluate_scalar_gate(gate: PhysicsGate, value: float) -> PhysicsGateResult:
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
    return PhysicsGateResult(gate=gate, value=value, status=status)
