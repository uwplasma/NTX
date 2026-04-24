from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ntx.physics_gates import physics_gate_registry
from ntx.validation.benchmark_matrix import (
    BenchmarkEntry,
    BenchmarkEvaluation,
    BenchmarkPathStatus,
    benchmark_matrix,
    benchmark_matrix_payload,
    evaluate_benchmark_matrix,
    write_benchmark_matrix_json,
)

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_matrix_has_unique_ids_and_expected_lanes():
    entries = benchmark_matrix()
    ids = [entry.id for entry in entries]
    assert len(ids) == len(set(ids))
    assert "monoenergetic_validation_summary" in ids
    assert "w7x_integrated_transfer" in ids
    assert "fixed_field_ntx_neopax_closure_stress" in ids
    assert "geometry_control_derivative_benchmark" in ids
    assert "file_backed_geometry_control_derivative_benchmark" in ids
    assert "boundary_forward_mode_current_derivative_benchmark" in ids
    assert "implicit_equilibrium_forward_mode_derivative_benchmark" in ids
    assert "explicit_relaxed_boundary_current_derivative_benchmark" in ids
    assert "bootstrap_current_optimization" in ids
    assert "geometry_breadth_hidden_symmetry" in ids
    assert "full_monoenergetic_geometry_family" in ids
    assert "large_geometry_control_autodiff" in ids


def test_benchmark_matrix_paths_exist_for_active_lanes():
    evaluations = evaluate_benchmark_matrix(ROOT)
    by_id = {evaluation.entry.id: evaluation for evaluation in evaluations}

    for benchmark_id, evaluation in by_id.items():
        if evaluation.entry.maturity == "planned-lane":
            assert evaluation.status == "planned", benchmark_id
            assert evaluation.entry.open_work
        else:
            assert evaluation.status == "complete", benchmark_id

    stress_entries = [
        evaluation.entry for evaluation in evaluations if evaluation.entry.maturity == "stress-gate"
    ]
    assert stress_entries
    assert all(entry.open_work for entry in stress_entries)


def test_artifact_gate_sources_are_represented_in_benchmark_matrix():
    matrix_artifacts = {
        artifact for entry in benchmark_matrix() for artifact in entry.artifacts
    }
    artifact_gate_sources = {
        gate.source
        for gate in physics_gate_registry()
        if gate.source.startswith("docs/_static/") and gate.source.endswith(".json")
    }

    assert artifact_gate_sources <= matrix_artifacts


def test_build_benchmark_matrix_script_writes_machine_readable_artifact(tmp_path):
    output_json = tmp_path / "benchmark_matrix.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_benchmark_matrix.py"),
            "--output-json",
            str(output_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["summary"]["incomplete"] == 0
    assert payload["summary"]["planned"] >= 1
    assert any(
        entry["entry"]["id"] == "prepared_derivative_path" for entry in payload["entries"]
    )


def test_benchmark_evaluation_status_and_payload_branches(tmp_path):
    entry = BenchmarkEntry(
        id="unit_gate",
        lane="monoenergetic",
        maturity="positive-gate",
        title="Unit gate",
        claim_scope="unit-test only",
        literature_anchors=("unit",),
        scripts=("missing_script.py",),
        tests=(),
        artifacts=(),
        manuscript_figures=(),
        docs=(),
    )
    missing = BenchmarkPathStatus(kind="script", path="missing_script.py", exists=False)
    evaluation = BenchmarkEvaluation(entry=entry, path_status=(missing,))

    assert missing.as_dict() == {
        "kind": "script",
        "path": "missing_script.py",
        "exists": False,
    }
    assert evaluation.missing_required_paths == ("missing_script.py",)
    assert evaluation.status == "incomplete"
    payload = evaluation.as_dict()
    assert payload["entry"]["id"] == "unit_gate"
    assert payload["status"] == "incomplete"
    assert payload["missing_required_paths"] == ["missing_script.py"]

    planned_entry = BenchmarkEntry(
        id="planned_unit_gate",
        lane="geometry-breadth",
        maturity="planned-lane",
        title="Planned gate",
        claim_scope="planned only",
        literature_anchors=(),
        scripts=("missing_planned_script.py",),
        tests=(),
        artifacts=(),
        manuscript_figures=(),
        docs=(),
        open_work=("add owned inputs",),
    )
    planned = BenchmarkEvaluation(entry=planned_entry, path_status=(missing,))
    assert planned.missing_required_paths == ()
    assert planned.status == "planned"

    output_json = tmp_path / "nested" / "benchmark_matrix.json"
    write_benchmark_matrix_json(ROOT, output_json)
    written = json.loads(output_json.read_text(encoding="utf-8"))
    direct = benchmark_matrix_payload(ROOT)
    assert written["summary"] == direct["summary"]
