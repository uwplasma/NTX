from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from ._benchmark_matrix_entries import benchmark_matrix
from ._benchmark_matrix_types import (
    BenchmarkEntry,
    BenchmarkEvaluation,
    BenchmarkLane,
    BenchmarkMaturity,
    BenchmarkPathStatus,
)

__all__ = [
    "BenchmarkEntry",
    "BenchmarkEvaluation",
    "BenchmarkLane",
    "BenchmarkMaturity",
    "BenchmarkPathStatus",
    "benchmark_matrix",
    "benchmark_matrix_payload",
    "evaluate_benchmark_matrix",
    "write_benchmark_matrix_json",
]


def evaluate_benchmark_matrix(root: Path) -> tuple[BenchmarkEvaluation, ...]:
    root = Path(root)
    evaluations: list[BenchmarkEvaluation] = []
    for entry in benchmark_matrix():
        statuses: list[BenchmarkPathStatus] = []
        path_groups: tuple[
            tuple[Literal["script", "test", "artifact", "doc"], tuple[str, ...]],
            ...,
        ] = (
            ("script", entry.scripts),
            ("test", entry.tests),
            ("artifact", entry.artifacts),
            ("doc", entry.docs),
        )
        for kind, paths in path_groups:
            statuses.extend(
                BenchmarkPathStatus(kind=kind, path=path, exists=(root / path).exists())
                for path in paths
            )
        evaluations.append(BenchmarkEvaluation(entry=entry, path_status=tuple(statuses)))
    return tuple(evaluations)


def benchmark_matrix_payload(root: Path) -> dict[str, object]:
    evaluations = evaluate_benchmark_matrix(root)
    return {
        "entries": [evaluation.as_dict() for evaluation in evaluations],
        "summary": {
            "complete": sum(evaluation.status == "complete" for evaluation in evaluations),
            "incomplete": sum(evaluation.status == "incomplete" for evaluation in evaluations),
            "planned": sum(evaluation.status == "planned" for evaluation in evaluations),
        },
    }


def write_benchmark_matrix_json(root: Path, output_path: Path) -> None:
    payload = benchmark_matrix_payload(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
