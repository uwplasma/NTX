from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


def autodiff_active_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="prepared_derivative_path",
            lane="autodiff",
            maturity="positive-gate",
            title="Prepared implicit-adjoint derivative path",
            claim_scope=(
                "Prepared derivatives agree with direct reverse-mode and give a "
                "measured speedup on the committed electric-field scan."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
            ),
            scripts=("examples/derivative_path_benchmark.py",),
            tests=("tests/test_derivative_path_benchmark_example.py",),
            artifacts=(
                "docs/_static/derivative_path_benchmark.png",
                "docs/_static/derivative_path_benchmark.pdf",
                "docs/_static/derivative_path_benchmark.json",
            ),
            manuscript_figures=("derivative_path_benchmark",),
            docs=("docs/autodiff.md", "docs/manuscript.md"),
            open_work=(
                "reduce memory and factorization overhead on larger geometry-control scans",
            ),
        ),
        BenchmarkEntry(
            id="geometry_control_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="Three-harmonic geometry-control derivative audit",
            claim_scope=(
                "Direct geometry-control autodiff agrees with centered finite "
                "differences for three controlled Boozer harmonics on an owned "
                "analytic surface."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "Escoto thesis monoenergetic formulation",
            ),
            scripts=("examples/geometry_control_derivative_benchmark.py",),
            tests=(
                "tests/test_geometry_control_derivative_benchmark_example.py",
                "tests/test_make_publication_figures.py",
            ),
            artifacts=(
                "docs/_static/geometry_control_derivative_benchmark.png",
                "docs/_static/geometry_control_derivative_benchmark.pdf",
                "docs/_static/geometry_control_derivative_benchmark.json",
            ),
            manuscript_figures=("geometry_control_derivative_benchmark",),
            docs=("docs/autodiff.md", "docs/manuscript.md"),
            open_work=(
                "transfer this audit to reusable VMEC/Boozer geometry-control families",
                "compare geometry pullbacks with an implicit-adjoint path once implemented",
            ),
        ),
        BenchmarkEntry(
            id="file_backed_geometry_control_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="File-backed geometry-control derivative audit",
            claim_scope=(
                "Direct geometry-control autodiff agrees with centered finite "
                "differences on repository-owned Boozer and VMEC sample surfaces."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "Escoto thesis monoenergetic formulation",
            ),
            scripts=("examples/file_backed_geometry_control_derivative_benchmark.py",),
            tests=("tests/test_file_backed_geometry_control_derivative_benchmark_example.py",),
            artifacts=(
                "docs/_static/file_backed_geometry_control_derivative_benchmark.png",
                "docs/_static/file_backed_geometry_control_derivative_benchmark.pdf",
                "docs/_static/file_backed_geometry_control_derivative_benchmark.json",
            ),
            manuscript_figures=("file_backed_geometry_control_derivative_benchmark",),
            docs=("docs/autodiff.md", "docs/manuscript.md"),
            open_work=(
                "extend from repository sample files to broader reusable geometry families",
                "compare geometry pullbacks with an implicit-adjoint path once implemented",
                "measure memory and factorization reuse on larger geometry-control scans",
            ),
        ),
        BenchmarkEntry(
            id="boundary_forward_mode_current_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="Boundary-to-output forward-mode derivative audit",
            claim_scope=(
                "Low-dimensional boundary controls propagate through "
                "boundary-projected vmec_jax geometry, booz_xform_jax, NTX "
                "coefficients, and an NTX+NEOPAX integrated-current objective "
                "under forward-mode autodiff."
            ),
            literature_anchors=(
                "vmec_jax explicit differentiable boundary workflows",
                "booz_xform_jax JAX-native Boozer transform",
                "McGreivy 2024 differentiable programming for plasma workflows",
            ),
            scripts=("examples/boundary_forward_mode_current_derivative_benchmark.py",),
            tests=("tests/test_boundary_forward_mode_current_derivative_benchmark_example.py",),
            artifacts=(
                "docs/_static/boundary_forward_mode_current_derivative_benchmark.png",
                "docs/_static/boundary_forward_mode_current_derivative_benchmark.pdf",
                "docs/_static/boundary_forward_mode_current_derivative_benchmark.json",
            ),
            manuscript_figures=("boundary_forward_mode_current_derivative_benchmark",),
            docs=("docs/autodiff.md", "docs/research-roadmap.md"),
            open_work=(
                (
                    "keep this fast projected-geometry lane as a low-cost precursor "
                    "to the equilibrium-relaxed benchmark"
                ),
                (
                    "validate broader non-axisymmetric benchmark families "
                    "beyond the repository sample input"
                ),
                (
                    "determine whether reverse-mode can be repaired or whether "
                    "the boundary-control lane should stay forward-mode only"
                ),
            ),
        ),
        BenchmarkEntry(
            id="implicit_equilibrium_forward_mode_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="Implicit-equilibrium forward-mode derivative audit",
            claim_scope=(
                "The implicit fixed-boundary vmec_jax residual solve reaches "
                "Boozer geometry and an NTX transport observable under "
                "forward-mode autodiff on the committed QA case, but only the "
                "equilibrium-volume derivative currently matches centered "
                "finite differences on this lane."
            ),
            literature_anchors=(
                "vmec_jax implicit fixed-boundary differentiation",
                "booz_xform_jax JAX-native Boozer transform",
                "McGreivy 2024 differentiable programming for plasma workflows",
            ),
            scripts=("examples/implicit_equilibrium_forward_mode_derivative_benchmark.py",),
            tests=("tests/test_implicit_equilibrium_forward_mode_derivative_benchmark_example.py",),
            artifacts=(
                "docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.png",
                "docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.pdf",
                "docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.json",
            ),
            manuscript_figures=("implicit_equilibrium_forward_mode_derivative_benchmark",),
            docs=("docs/autodiff.md", "docs/research-roadmap.md", "docs/manuscript.md"),
            open_work=(
                "recover Boozer-scalar parity on the implicit geometry path",
                "recover NTX transport parity on the implicit geometry path",
                (
                    "extend the implicit forward-mode lane from NTX transport to "
                    "NTX+NEOPAX integrated current"
                ),
                "broaden beyond the committed QA case to additional geometry families",
                "repair reverse mode through the implicit vmec_jax -> booz_xform_jax path",
            ),
        ),
        BenchmarkEntry(
            id="explicit_relaxed_boundary_current_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="Explicit-relaxed boundary-to-current derivative audit",
            claim_scope=(
                "Low-dimensional boundary controls propagate through an "
                "explicitly relaxed fixed-boundary vmec_jax solve, "
                "booz_xform_jax, NTX coefficients, and an NTX+NEOPAX "
                "integrated-current objective under forward-mode autodiff on "
                "committed QA and QH family cases, while preserving the "
                "ordinary primal volume."
            ),
            literature_anchors=(
                "vmec_jax explicit differentiable boundary workflows",
                "booz_xform_jax JAX-native Boozer transform",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "Landreman and Paul 2022 precise-QS benchmark family",
            ),
            scripts=("examples/explicit_relaxed_boundary_current_derivative_benchmark.py",),
            tests=("tests/test_explicit_relaxed_boundary_current_derivative_benchmark_example.py",),
            artifacts=(
                "docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.png",
                "docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.pdf",
                "docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.json",
            ),
            manuscript_figures=("explicit_relaxed_boundary_current_derivative_benchmark",),
            docs=("docs/autodiff.md", "docs/research-roadmap.md", "docs/manuscript.md"),
            open_work=(
                (
                    "widen beyond the committed QA and QH cases to additional "
                    "geometry families"
                ),
                (
                    "establish whether the implicit-equilibrium path can recover "
                    "the same boundary sensitivities"
                ),
                (
                    "determine whether reverse-mode can be repaired on the "
                    "equilibrium-relaxed boundary-control lane"
                ),
            ),
        ),
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
            title="Linearized profile uncertainty against Monte Carlo",
            claim_scope=(
                "Linearized covariance propagation is checked against a small "
                "Monte Carlo ensemble on the same differentiable profile map."
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
            open_work=("add Fisher or Hessian-vector probes on a larger profile basis",),
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


def autodiff_planned_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
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
    "autodiff_active_benchmark_entries",
    "autodiff_planned_benchmark_entries",
]
