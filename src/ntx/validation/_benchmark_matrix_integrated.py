"""Benchmark-matrix entries for end-to-end integrated workflows."""

from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


def integrated_workflow_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="w7x_integrated_transfer",
            lane="integrated-workflow",
            maturity="positive-gate",
            title="Imported W7-X workflow transfer",
            claim_scope=(
                "The rebuilt raw-branch NTX database transfers through the "
                "imported W7-X workflow without exceeding the current release gate."
            ),
            literature_anchors=(
                "W7-X benchmark profile used by the imported workflow",
                "neoclassical bootstrap-current profile validation practice",
            ),
            scripts=("examples/bootstrap_current_reference_audit_w7x.py",),
            tests=(
                "tests/test_bootstrap_current_reference_audit_w7x.py",
                "tests/test_w7x_reference_benchmark.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_reference_audit_w7x.png",
                "docs/_static/bootstrap_current_reference_audit_w7x.pdf",
                "docs/_static/bootstrap_current_reference_audit_w7x.json",
            ),
            manuscript_figures=("bootstrap_current_reference_audit_w7x",),
            docs=("docs/physics-gates.md", "docs/validation.md", "docs/manuscript.md"),
        ),
    )


__all__ = ["integrated_workflow_benchmark_entries"]
