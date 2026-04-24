from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


def profile_workflow_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="profile_force_reconstruction",
            lane="profile-workflow",
            maturity="stress-gate",
            title="Primitive profile to force reconstruction",
            claim_scope=(
                "The primitive-profile builder is audited on the precise-QS "
                "profile family and kept as a monitored profile stress benchmark."
            ),
            literature_anchors=(
                "Landreman and Paul 2022 precise-QS density and temperature profiles",
                "thermodynamic-force formulation of neoclassical transport",
            ),
            scripts=("examples/profile_force_reconstruction_audit.py",),
            tests=("tests/test_profile_force_reconstruction_audit_example.py",),
            artifacts=(
                "docs/_static/profile_force_reconstruction_audit.png",
                "docs/_static/profile_force_reconstruction_audit.pdf",
                "docs/_static/profile_force_reconstruction_audit.json",
            ),
            manuscript_figures=("profile_force_reconstruction_audit",),
            docs=("docs/profiles.md",),
            open_work=("tighten profile-level physical diagnostics for long-radius studies",),
        ),
    )


__all__ = ["profile_workflow_benchmark_entries"]
