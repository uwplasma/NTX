"""Benchmark-matrix entries for profile and closure workflows."""

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
        BenchmarkEntry(
            id="profile_basis_optimization",
            lane="profile-workflow",
            maturity="stress-gate",
            title="Radial-basis profile-control optimization",
            claim_scope=(
                "A three-function radial basis control is optimized through the "
                "profile closure, with objective, residual, and control metrics "
                "stored as a monitored workflow artifact."
            ),
            literature_anchors=(
                "profile-level neoclassical transport sensitivity workflows",
                "differentiable programming verification by controlled objectives",
            ),
            scripts=("examples/profile_basis_optimization.py",),
            tests=("tests/test_profile_basis_optimization_example.py",),
            artifacts=(
                "docs/_static/profile_basis_optimization.png",
                "docs/_static/profile_basis_optimization.pdf",
                "docs/_static/profile_basis_optimization.json",
            ),
            manuscript_figures=("profile_basis_optimization",),
            docs=("docs/profiles.md", "docs/manuscript.md"),
            open_work=(
                "broaden the radial basis to reusable physics-motivated profile families",
                "promote only after cross-geometry profile/current validation",
            ),
        ),
    )


__all__ = ["profile_workflow_benchmark_entries"]
