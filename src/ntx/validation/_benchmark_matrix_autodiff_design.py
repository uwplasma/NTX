from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


def autodiff_design_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="autodiff_inverse_problem",
            lane="autodiff",
            maturity="stress-gate",
            title="Synthetic inverse-design recovery",
            claim_scope=(
                "A differentiable profile/geometry control can recover a generated "
                "target on a small owned inverse problem."
            ),
            literature_anchors=(
                "adjoint stellarator-optimization validation practice",
                "differentiable programming verification by generated targets",
            ),
            scripts=("examples/autodiff_inverse_problem.py",),
            tests=("tests/test_autodiff.py", "tests/test_make_publication_figures.py"),
            artifacts=(
                "docs/_static/autodiff_inverse_problem.png",
                "docs/_static/autodiff_inverse_problem.pdf",
            ),
            manuscript_figures=("autodiff_inverse_problem",),
            docs=("docs/autodiff.md",),
            open_work=("extend from scalar controls to larger geometry-control families",),
        ),
        BenchmarkEntry(
            id="autodiff_profile_uncertainty",
            lane="autodiff",
            maturity="stress-gate",
            title="Radial-basis profile uncertainty and Fisher/HVP audit",
            claim_scope=(
                "Linearized covariance propagation, Monte Carlo sampling, and "
                "a Fisher/Gauss-Newton versus Hessian-vector-product probe are "
                "checked on the same differentiable radial-basis profile map."
            ),
            literature_anchors=(
                "differentiable programming uncertainty propagation",
                "profile-level neoclassical workflow sensitivity analysis",
            ),
            scripts=("examples/autodiff_profile_uncertainty.py",),
            tests=(
                "tests/test_autodiff.py",
                "tests/test_autodiff_profile_uncertainty_example.py",
            ),
            artifacts=(
                "docs/_static/autodiff_profile_uncertainty.png",
                "docs/_static/autodiff_profile_uncertainty.pdf",
                "docs/_static/autodiff_profile_uncertainty.json",
            ),
            manuscript_figures=("autodiff_profile_uncertainty",),
            docs=("docs/autodiff.md",),
            open_work=(
                "broaden from the committed radial-basis audit to cross-geometry profile families",
            ),
        ),
        BenchmarkEntry(
            id="bootstrap_current_optimization",
            lane="autodiff",
            maturity="stress-gate",
            title="Differentiable bootstrap-current optimization",
            claim_scope=(
                "The differentiable bootstrap-current objective improves a "
                "weighted current proxy on the committed W7-X study while "
                "remaining scoped below a broad stellarator-design claim."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "direct neoclassical ion-transport optimization literature",
                "differentiable programming verification by generated objectives",
            ),
            scripts=("examples/bootstrap_current_optimization.py",),
            tests=(
                "tests/test_autodiff.py",
                "tests/test_bootstrap_current_optimization_example.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_optimization.png",
                "docs/_static/bootstrap_current_optimization.pdf",
                "docs/_static/bootstrap_current_optimization.json",
            ),
            manuscript_figures=("bootstrap_current_optimization",),
            docs=("docs/autodiff.md", "docs/examples.md", "docs/manuscript.md"),
            open_work=(
                "promote only after broader geometry-family controls are added",
                "tie future optimization claims to reusable derivative-audit gates",
            ),
        ),
        BenchmarkEntry(
            id="robust_bootstrap_current_optimization",
            lane="autodiff",
            maturity="stress-gate",
            title="Robust bootstrap-current optimization under control uncertainty",
            claim_scope=(
                "The robust-design objective improves under a prescribed Gaussian "
                "control uncertainty on a deterministic owned benchmark."
            ),
            literature_anchors=(
                "adjoint stellarator optimization",
                "direct neoclassical ion-transport optimization",
                "robust optimization under parameter uncertainty",
            ),
            scripts=("examples/bootstrap_current_robust_optimization.py",),
            tests=(
                "tests/test_autodiff.py",
                "tests/test_bootstrap_current_robust_optimization_example.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_robust_optimization.png",
                "docs/_static/bootstrap_current_robust_optimization.pdf",
                "docs/_static/bootstrap_current_robust_optimization.json",
            ),
            manuscript_figures=("bootstrap_current_robust_optimization",),
            docs=("docs/autodiff.md",),
            open_work=(
                "promote only after testing on broader geometry controls and profile families",
            ),
        ),
    )


def autodiff_design_planned_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="large_geometry_control_autodiff",
            lane="autodiff",
            maturity="planned-lane",
            title="Large geometry-control autodiff validation",
            claim_scope=(
                "The scalar derivative and optimization examples should be "
                "extended to larger geometry-control families before promoting "
                "optimization-grade stellarator-design claims."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "direct neoclassical transport optimization literature",
            ),
            scripts=(),
            tests=(),
            artifacts=(),
            manuscript_figures=(),
            docs=("docs/autodiff.md", "docs/research-roadmap.md"),
            open_work=(
                (
                    "broaden the current analytic and file-backed audits into "
                    "reusable geometry families"
                ),
                "compare direct autodiff, implicit-adjoint, and finite-difference probes",
                "measure memory and factorization reuse under real scan workloads",
                "add publication-ready derivative and optimization figures",
            ),
        ),
    )


__all__ = [
    "autodiff_design_benchmark_entries",
    "autodiff_design_planned_benchmark_entries",
]
